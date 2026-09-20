#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include "native/providers/gdn/sm121_mtp_prompt_cache.h"
#include "native/providers/gdn/sm121_mtp_projection.h"

namespace fs = std::filesystem;
namespace mtp = qrt_sm121_mtp;
void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
template<class T> std::vector<T> read(const fs::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(T) || file.tellg() > 128u * 1024u * 1024u)
        throw std::runtime_error("invalid or oversized original input file: " + path.string());
    std::vector<T> values(static_cast<size_t>(file.tellg()) / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(T));
    if (!file) throw std::runtime_error("short original input file");
    return values;
}
struct Storage {
    std::vector<void*> pointers;
    ~Storage() {
        if (hipDeviceSynchronize() != hipSuccess) return;
        for (void* p : pointers) (void)hipFree(p);
    }
    template<class T> T* allocate(size_t count) {
        T* pointer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer), count * sizeof(T)));
        pointers.push_back(pointer);
        return pointer;
    }
    template<class T> T* upload(const std::vector<T>& values) {
        T* pointer = allocate<T>(values.size());
        check(hipMemcpy(pointer, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
        return pointer;
    }
};
struct ProjectionChecks {
    std::vector<uint16_t> fusion_input, fusion, normalized_input, kv;
    unsigned int calls = 0;
    size_t input_mismatches = 0, output_mismatches = 0, compared_elements = 0;
    static hipError_t run(void* context, const uint16_t* weights, const uint16_t* input,
        uint16_t* output, unsigned int output_features, unsigned int input_features,
        unsigned int tokens, hipStream_t stream) {
        auto& self = *static_cast<ProjectionChecks*>(context);
        const bool fusion_stage = self.calls == 0;
        if (self.calls > 1 || output_features != (fusion_stage ? 2048u : 1024u) ||
            input_features != (fusion_stage ? 4096u : 2048u)) return hipErrorInvalidValue;
        const auto& expected_input = fusion_stage ? self.fusion_input : self.normalized_input;
        const auto& expected_output = fusion_stage ? self.fusion : self.kv;
        if (expected_input.size() != size_t(tokens) * input_features ||
            expected_output.size() != size_t(tokens) * output_features) return hipErrorInvalidValue;
        // Only compare copies of computed values. Expected buffers are never
        // uploaded or supplied to the projection or prompt-cache kernels.
        hipError_t status = hipStreamSynchronize(stream);
        if (status != hipSuccess) return status;
        std::vector<uint16_t> actual(expected_input.size());
        status = hipMemcpy(actual.data(), input, actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
        if (status != hipSuccess) return status;
        for (size_t i = 0; i < actual.size(); ++i) self.input_mismatches += actual[i] != expected_input[i];
        self.compared_elements += actual.size();
        status = mtp::launch_projection(weights, input, output, output_features, input_features, tokens, 1024u, stream);
        if (status != hipSuccess) return status;
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) return status;
        actual.resize(expected_output.size());
        status = hipMemcpy(actual.data(), output, actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
        if (status != hipSuccess) return status;
        for (size_t i = 0; i < actual.size(); ++i) self.output_mismatches += actual[i] != expected_output[i];
        self.compared_elements += actual.size();
        ++self.calls;
        return hipSuccess;
    }
};

int main(int argc, char** argv) try {
    if (argc != 5) throw std::runtime_error("original_case_directory original_weight_directory rsqrt_table rope_table");
    const fs::path case_dir(argv[1]), weight_dir(argv[2]);
    const auto full = [&](const char* label) { return read<uint16_t>(case_dir / (std::string("full-mtp-") + label + ".bin")); };
    const auto hidden = full("target-hidden"), embeddings = full("embedding");
    const auto ids = read<uint32_t>(case_dir / "full-mtp-shifted-input-ids.bin");
    const auto expected_k = full("k-rope"), expected_v = full("v");
    const auto rsqrt = read<unsigned char>(argv[3]);
    const auto rope = read<uint16_t>(argv[4]);
    const size_t tokens = ids.size();
    if ((tokens != 7169u && tokens != 8192u) || hidden.size() != tokens * 2048u ||
        embeddings.size() != hidden.size() || expected_k.size() != tokens * 512u ||
        expected_v.size() != expected_k.size() || rope.size() != size_t(262144u) * 64u ||
        !qrt_sm121_rsqrt::valid_layout(rsqrt.data(), rsqrt.size()))
        throw std::runtime_error("full original MTP control shape or table layout changed");
    for (uint32_t id : ids) if (id >= 248320u) throw std::runtime_error("actual shifted ID outside vocabulary");
    ProjectionChecks projections;
    projections.fusion_input = full("fusion-input");
    projections.fusion = full("fusion");
    projections.normalized_input = full("input-norm");
    projections.kv = full("kv-projection");
    Storage storage;
    auto* embedding_table = storage.allocate<uint16_t>(size_t(248320u) * 2048u);
    std::unordered_map<uint32_t, size_t> unique;
    for (size_t row = 0; row < tokens; ++row) {
        const auto inserted = unique.emplace(ids[row], row);
        if (!inserted.second) {
            if (!std::equal(embeddings.begin() + row * 2048u, embeddings.begin() + (row + 1u) * 2048u,
                            embeddings.begin() + inserted.first->second * 2048u))
                throw std::runtime_error("same actual ID has different original embeddings");
        } else check(hipMemcpy(embedding_table + size_t(ids[row]) * 2048u,
            embeddings.data() + row * 2048u, 2048u * sizeof(uint16_t), hipMemcpyHostToDevice));
    }
    const auto weight = [&](const char* name, size_t elements) {
        const auto values = read<uint16_t>(weight_dir / (std::string(name) + ".bf16.bin"));
        if (values.size() != elements) throw std::runtime_error("original MTP weight shape changed");
        return storage.upload(values);
    };
    mtp::PromptWeights weights;
    weights.embeddings = embedding_table;
    weights.embedding_norm = weight("pre-fc-embedding-norm", 2048u);
    weights.hidden_norm = weight("pre-fc-hidden-norm", 2048u);
    weights.fusion = weight("fc", size_t(2048u) * 4096u);
    weights.input_norm = weight("input-norm", 2048u);
    weights.kv_projection = weight("kv-projection", size_t(1024u) * 2048u);
    weights.key_norm = weight("key-norm", 256u);
    weights.split1024_pre_fc_norm = true; // Original full-capture S4L4... launcher.
    const auto* device_hidden = storage.upload(hidden);
    const auto* device_ids = storage.upload(ids);
    const auto* device_rsqrt = storage.upload(rsqrt);
    const auto* device_rope = storage.upload(rope);
    mtp::PromptCache cache;
    check(cache.reserve(262144u, static_cast<unsigned int>(tokens)));
    check(hipMemset(const_cast<uint16_t*>(cache.data()), 0xa5, size_t(262144u) * 1024u * sizeof(uint16_t)));
    const auto start = std::chrono::steady_clock::now();
    const auto step = cache.append(weights, device_hidden, device_ids, device_rsqrt, device_rope,
        262144u, 0u, static_cast<unsigned int>(tokens), &ProjectionChecks::run, &projections);
    const double completed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    if (step.status != hipSuccess || step.completion_unknown) {
        std::cerr << "prompt cache failed at " << step.stage << '\n';
        check(step.status);
        throw std::runtime_error("prompt cache completion unknown");
    }
    if (projections.calls != 2 || cache.retained_tokens() != tokens || cache.quarantined())
        throw std::runtime_error("prompt cache did not publish its complete first batch");
    // Check the entire cache in bounded host windows, including every cell
    // beyond the published prefix. This is observation of the actual owner.
    std::vector<uint16_t> window(8192u * 1024u);
    size_t cache_mismatches = 0, untouched_mismatches = 0;
    for (size_t first = 0; first < 262144u; first += 8192u) {
        check(hipMemcpy(window.data(), cache.data() + first * 1024u,
            window.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        for (size_t i = 0; i < window.size(); ++i) {
            const size_t row = first + i / 1024u, feature = i % 1024u;
            if (row >= tokens) untouched_mismatches += window[i] != 0xa5a5u;
            else cache_mismatches += window[i] != (feature < 512u
                ? expected_k[row * 512u + feature] : expected_v[row * 512u + feature - 512u]);
        }
    }
    const bool pass = !projections.input_mismatches && !projections.output_mismatches &&
                      !cache_mismatches && !untouched_mismatches;
    std::cout << "{\"kind\":\"original_full_mtp_prompt_cache_native_replay\",\"rows\":" << tokens
        << ",\"projection_calls\":" << projections.calls << ",\"projection_compared_elements\":" << projections.compared_elements
        << ",\"projection_input_mismatches\":" << projections.input_mismatches
        << ",\"projection_output_mismatches\":" << projections.output_mismatches
        << ",\"cache_compared_elements\":" << tokens * 1024u << ",\"cache_bf16_mismatches\":" << cache_mismatches
        << ",\"untouched_compared_elements\":" << (262144u - tokens) * 1024u
        << ",\"untouched_mismatches\":" << untouched_mismatches << ",\"retained_tokens\":" << cache.retained_tokens()
        << ",\"cache_owner_bytes\":" << cache.allocated_bytes() << ",\"projection_workspace_bytes\":0"
        << ",\"completed_with_observations_ms\":" << completed_ms << ",\"pass\":" << (pass ? "true" : "false")
        << ",\"native_inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return pass ? 0 : 1;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
