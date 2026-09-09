// Standalone captured-input diagnostic. Never linked into the model runtime.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(hipError_t code) {
    if (code != hipSuccess) throw std::runtime_error(hipGetErrorString(code));
}
struct Buffer {
    void* data = nullptr;
    ~Buffer() { if (data) (void)hipFree(data); }
    void allocate(size_t bytes) { check(hipMalloc(&data, bytes)); }
    template<class T> void upload(const std::vector<T>& source) {
        allocate(source.size() * sizeof(T));
        check(hipMemcpy(data, source.data(), source.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    template<class T> T* at(size_t offset = 0) { return static_cast<T*>(data) + offset; }
};
struct Module { hipModule_t handle = nullptr; ~Module() { if (handle) (void)hipModuleUnload(handle); } };
struct Stream { hipStream_t handle = nullptr; ~Stream() { if (handle) (void)hipStreamDestroy(handle); } };
struct Event { hipEvent_t handle = nullptr; ~Event() { if (handle) (void)hipEventDestroy(handle); } };
template<class T> std::vector<T> read(const std::string& path, size_t count, size_t padded) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(count * sizeof(T))) {
        throw std::runtime_error("capture size mismatch: " + path);
    }
    std::vector<T> result(padded);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (!input) throw std::runtime_error("capture read failed: " + path);
    return result;
}
float value(uint16_t bits) {
    const uint32_t raw = uint32_t(bits) << 16;
    float result; std::memcpy(&result, &raw, 4); return result;
}
uint16_t bf16(float input) {
    uint32_t bits; std::memcpy(&bits, &input, 4);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
unsigned int number(const char* argument, unsigned int maximum) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(argument, &end, 10);
    if (!*argument || !end || *end || parsed > maximum) throw std::runtime_error("invalid bounded integer");
    return static_cast<unsigned int>(parsed);
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 8) {
        std::cerr << "usage: fla-output-capture-replay <hsaco> <symbol> <threads> <shared> <capture-dir> <tokens> <q64|real>\n";
        return 2;
    }
    const unsigned int threads = number(argv[3], 1024), shared = number(argv[4], 65536);
    const unsigned int tokens = number(argv[6], 8192);
    if (!tokens || !threads || threads % 32) throw std::runtime_error("invalid shape or workgroup");
    const bool real = std::string(argv[7]) == "real";
    if ((!real && std::string(argv[7]) != "q64") || (!real && tokens != 64)) throw std::runtime_error("invalid capture layout");
    if (std::string(argv[2]) != "_fla_chunk_output_kernel") throw std::runtime_error("not the captured output-stage ABI");
    const unsigned int padded = (tokens + 63u) / 64u * 64u;
    constexpr size_t key_features = 16u * 128u, value_features = 32u * 128u;
    constexpr size_t state_elements = 32u * 128u * 128u;
    const std::string directory = argv[5], prefix = real ? "full-" : "";
    auto path = [&](const char* name) { return directory + "/" + prefix + name + ".bin"; };
    auto q = read<uint16_t>(path("q-normalized-bf16"), tokens * key_features, padded * key_features);
    auto k = read<uint16_t>(path("k-normalized-bf16"), tokens * key_features, padded * key_features);
    auto v = read<uint16_t>(path("v-new-bf16"), tokens * value_features, padded * value_features);
    auto h = read<uint16_t>(path("chunk-state-bf16"), padded / 64u * state_elements, padded / 64u * state_elements);
    auto g = read<float>(path("g-cumsum-f32"), tokens * 32u, padded * 32u);
    auto reference = read<uint16_t>(path(real ? "native-output-bf16" : "output-bf16"), tokens * value_features, tokens * value_features);
    for (const auto* surface : {&q, &k, &v, &h, &reference}) {
        for (uint16_t x : *surface) if (!std::isfinite(value(x))) throw std::runtime_error("nonfinite BF16 input");
    }
    for (float x : g) if (!std::isfinite(x)) throw std::runtime_error("nonfinite gate input");
    for (unsigned int t = tokens; t < padded; ++t) {
        for (unsigned int head = 0; head < 32; ++head) g[t * 32u + head] = g[(tokens - 1u) * 32u + head];
    }
    const size_t allocation_bytes = (q.size() + k.size() + v.size() + h.size()) * 2u
                                  + g.size() * 4u + padded * value_features * 4u;
    if (allocation_bytes > 512u * 1024u * 1024u) throw std::runtime_error("diagnostic allocation ceiling exceeded");
    check(hipSetDevice(0));
    hipDeviceProp_t properties{};
    check(hipGetDeviceProperties(&properties, 0));
    if (std::string(properties.gcnArchName).find("gfx1151") != 0) throw std::runtime_error("requires gfx1151");
    size_t free_bytes = 0, total_bytes = 0;
    check(hipMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < allocation_bytes + 512u * 1024u * 1024u) throw std::runtime_error("insufficient device memory reserve");
    Stream stream; Module module; Event begin, end;
    check(hipStreamCreate(&stream.handle));
    check(hipEventCreate(&begin.handle)); check(hipEventCreate(&end.handle));
    check(hipModuleLoad(&module.handle, argv[1]));
    hipFunction_t function = nullptr;
    check(hipModuleGetFunction(&function, module.handle, argv[2]));
    Buffer dq, dk, dv, dh, dg, output;
    dq.upload(q); dk.upload(k); dv.upload(v); dh.upload(h); dg.upload(g);
    output.allocate(padded * value_features * 4u);
    float maximum_ms = 0;
    unsigned int segments = 0;
    for (unsigned int offset = 0; offset < padded; offset += 1024u) {
        int32_t count = static_cast<int32_t>(std::min(1024u, padded - offset));
        auto* pq = dq.at<uint16_t>(offset * key_features);
        auto* pk = dk.at<uint16_t>(offset * key_features);
        auto* pv = dv.at<uint16_t>(offset * value_features);
        auto* ph = dh.at<uint16_t>(offset / 64u * state_elements);
        auto* pg = dg.at<float>(offset * 32u);
        auto* po = output.at<float>(offset * value_features);
        void* arguments[] = {&pq, &pk, &pv, &ph, &pg, &po, &count};
        check(hipEventRecord(begin.handle, stream.handle));
        check(hipModuleLaunchKernel(function, 4u, static_cast<unsigned int>(count) / 64u, 32u,
                                    threads, 1u, 1u, shared, stream.handle, arguments, nullptr));
        check(hipEventRecord(end.handle, stream.handle));
        check(hipEventSynchronize(end.handle));
        float milliseconds = 0;
        check(hipEventElapsedTime(&milliseconds, begin.handle, end.handle));
        if (!(milliseconds <= 100.0f)) throw std::runtime_error("100 ms dispatch admission exceeded; no next segment");
        maximum_ms = std::max(maximum_ms, milliseconds); ++segments;
    }
    std::vector<float> result(tokens * value_features);
    check(hipMemcpy(result.data(), output.data, result.size() * 4u, hipMemcpyDeviceToHost));
    uint64_t mismatch = 0, nonfinite = 0;
    double error2 = 0, norm2 = 0, maximum_error = 0;
    int64_t first = -1; float first_actual = 0, first_expected = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i])) { ++nonfinite; continue; }
        const float expected = value(reference[i]);
        const double delta = double(result[i]) - expected;
        error2 += delta * delta; norm2 += double(expected) * expected;
        if (bf16(result[i]) != reference[i] || result[i] != value(bf16(result[i]))) {
            ++mismatch; maximum_error = std::max(maximum_error, std::abs(delta));
            if (first < 0) { first = static_cast<int64_t>(i); first_actual = result[i]; first_expected = expected; }
        }
    }
    std::cout << std::setprecision(17)
              << "{\"kind\":\"native_captured_output_stage\",\"model_loaded\":false,\"inference_acceptance\":false"
              << ",\"tokens\":" << tokens << ",\"elements\":" << result.size()
              << ",\"segments\":" << segments << ",\"allocation_bytes\":" << allocation_bytes
              << ",\"maximum_dispatch_ms\":" << maximum_ms << ",\"mismatch_count\":" << mismatch
              << ",\"nonfinite_count\":" << nonfinite << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1.0e-300))
              << ",\"maximum_absolute_error\":" << maximum_error << ",\"first_index\":" << first
              << ",\"first_actual\":" << first_actual << ",\"first_expected\":" << first_expected << "}\n";
    return mismatch || nonfinite ? 3 : 0;
} catch (const std::exception& error) {
    std::cerr << "fla_output_capture_replay error=" << error.what() << '\n';
    return 2;
}
