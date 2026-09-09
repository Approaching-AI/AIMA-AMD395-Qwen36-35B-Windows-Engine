// Isolated captured-input diagnostics; never linked into the inference runtime.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <hip/hip_runtime.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
constexpr size_t key_features = 2048, value_features = 4096, matrix_features = 2048;
constexpr size_t state_elements = 32u * 128u * 128u;
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
float value(float x) { return x; }
float value(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
uint16_t bf16(float x) {
    uint32_t bits; std::memcpy(&bits, &x, 4);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
unsigned int number(const char* text, unsigned int limit) {
    char* end = nullptr; const unsigned long x = std::strtoul(text, &end, 10);
    if (!*text || !end || *end || x > limit) throw std::runtime_error("invalid bounded integer");
    return static_cast<unsigned int>(x);
}
template<class T> std::vector<T> read_range(const std::string& path, size_t total,
                                           size_t start, size_t count, size_t padded) {
    if (start > total || count > total - start || padded < count) throw std::runtime_error("invalid capture slice");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(total * sizeof(T))) throw std::runtime_error("capture size mismatch: " + path);
    std::vector<T> result(padded);
    file.seekg(start * sizeof(T)); file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("capture read failed: " + path);
    for (T x : result) if (!std::isfinite(value(x))) throw std::runtime_error("nonfinite capture: " + path);
    return result;
}
struct Buffer {
    void* data = nullptr;
    Buffer() = default; Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (data) (void)hipFree(data); }
    void allocate(size_t bytes) { check(hipMalloc(&data, bytes)); }
    template<class T> void upload(const std::vector<T>& host) {
        allocate(host.size() * sizeof(T)); check(hipMemcpy(data, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    template<class T> T* at(size_t offset = 0) { return static_cast<T*>(data) + offset; }
};
template<class T> std::vector<T> download(const T* pointer, size_t count) {
    std::vector<T> result(count); check(hipMemcpy(result.data(), pointer, count * sizeof(T), hipMemcpyDeviceToHost)); return result;
}
struct Event { hipEvent_t handle = nullptr; ~Event() { if (handle) (void)hipEventDestroy(handle); } };
struct Stream { hipStream_t handle = nullptr; ~Stream() { if (handle) (void)hipStreamDestroy(handle); } };
struct Module { hipModule_t handle = nullptr; ~Module() { if (handle) (void)hipModuleUnload(handle); } };
struct Launcher {
    Stream stream; Module module; Event begin, end; hipFunction_t function = nullptr;
    unsigned int threads, shared, segments = 0; float maximum_ms = 0;
    Launcher(const char* file, const char* symbol, unsigned int block, unsigned int lds, size_t allocation)
        : threads(block), shared(lds) {
        if (allocation > 512u * 1024u * 1024u) throw std::runtime_error("512 MiB allocation ceiling exceeded");
        std::cerr << "UPSTREAM_REPLAY phase=hip_preflight\n" << std::flush;
        check(hipSetDevice(0)); hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
        if (std::string(device.gcnArchName).find("gfx1151") != 0) throw std::runtime_error("requires gfx1151");
        size_t free_bytes = 0, total_bytes = 0; check(hipMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes < allocation + 512u * 1024u * 1024u) throw std::runtime_error("device memory reserve failed");
        check(hipStreamCreate(&stream.handle)); check(hipEventCreate(&begin.handle)); check(hipEventCreate(&end.handle));
        check(hipModuleLoad(&module.handle, file)); check(hipModuleGetFunction(&function, module.handle, symbol));
    }
    template<size_t SourceArguments, class... Arguments>
    void launch(unsigned int x, unsigned int y, unsigned int offset, Arguments... parameters) {
        static_assert(sizeof...(parameters) == SourceArguments, "source ABI parameter count");
        void* global_scratch = nullptr; void* profile_scratch = nullptr;
        std::array<void*, SourceArguments + 2> args{parameters..., &global_scratch, &profile_scratch};
        std::cerr << "UPSTREAM_REPLAY phase=dispatch offset=" << offset << " abi_slots=" << args.size() << '\n' << std::flush;
        check(hipEventRecord(begin.handle, stream.handle));
        check(hipModuleLaunchKernel(function, x, y, 1u, threads, 1u, 1u, shared, stream.handle, args.data(), nullptr));
        check(hipEventRecord(end.handle, stream.handle)); check(hipEventSynchronize(end.handle));
        float milliseconds = 0; check(hipEventElapsedTime(&milliseconds, begin.handle, end.handle));
        if (!(milliseconds <= 100.0f)) throw std::runtime_error("100 ms dispatch admission exceeded; no next segment");
        maximum_ms = std::max(maximum_ms, milliseconds); ++segments;
    }
};
struct Stats {
    uint64_t count = 0, mismatches = 0, bit_mismatches = 0, nonfinite = 0;
    double error2 = 0, norm2 = 0, maximum = 0; int64_t first = -1;
    float first_actual = 0, first_expected = 0;
    template<class T> void add(T a, T b) {
        const float x = value(a), y = value(b); const size_t index = count++;
        bit_mismatches += std::memcmp(&a, &b, sizeof(T)) != 0;
        if (!std::isfinite(x) || !std::isfinite(y)) { ++nonfinite; return; }
        const double delta = double(x) - y; error2 += delta * delta; norm2 += double(y) * y;
        if (x != y) {
            ++mismatches; maximum = std::max(maximum, std::abs(delta));
            if (first < 0) { first = static_cast<int64_t>(index); first_actual = x; first_expected = y; }
        }
    }
    void print(const std::string& name) const {
        std::cout << '"' << name << "\":{\"elements\":" << count << ",\"mismatch_count\":" << mismatches
                  << ",\"bit_mismatch_count\":" << bit_mismatches << ",\"nonfinite_count\":" << nonfinite
                  << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1.0e-300))
                  << ",\"maximum_absolute_error\":" << maximum << ",\"first_index\":" << first
                  << ",\"first_actual\":" << first_actual << ",\"first_expected\":" << first_expected << '}';
    }
};
template<class T> Stats compare(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) throw std::runtime_error("comparison size mismatch");
    Stats stats; for (size_t i = 0; i < a.size(); ++i) stats.add(a[i], b[i]); return stats;
}
template<class T> void dump_q64(const std::string& stage, const std::string& name, const std::vector<T>& values) {
    const char* directory = std::getenv("QRT_FLA_UPSTREAM_DUMP_Q64_DIR");
    if (!directory || !*directory) return;
    if (values.size() * sizeof(T) > 2u * 1024u * 1024u) throw std::runtime_error("q64 dump exceeds 2 MiB");
    const std::string path = std::string(directory) + "/" + stage + "-" + name + ".bin";
    if (std::ifstream(path).good()) throw std::runtime_error("refusing capture overwrite");
    std::ofstream file(path, std::ios::binary); file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!file) throw std::runtime_error("capture dump failed");
}
}  // namespace

int main(int argc, char** argv) try {
    if (argc != 9) {
        std::cerr << "usage: fla-upstream-capture-replay <solve|wu|state> <hsaco> <symbol> <threads> <shared> <capture-dir> <source-tokens> <tokens>\n";
        return 2;
    }
    const std::string stage = argv[1], directory = argv[6];
    const unsigned int threads = number(argv[4], 1024), shared = number(argv[5], 65536);
    const unsigned int source_tokens = number(argv[7], 8192), tokens = number(argv[8], 8192);
    if (!tokens || !source_tokens || tokens > source_tokens || (tokens != source_tokens && tokens % 64) || !threads || threads % 32)
        throw std::runtime_error("invalid source, view, or workgroup shape");
    const char* dump = std::getenv("QRT_FLA_UPSTREAM_DUMP_Q64_DIR");
    if (dump && *dump && tokens != 64) throw std::runtime_error("binary stage capture is restricted to a q64 view");
    const std::string symbol = stage == "solve" ? "_fla_solve_tril_64_kernel" :
                               stage == "wu" ? "_fla_recompute_w_u_kernel" :
                               stage == "state" ? "_fla_chunk_state_kernel" : "";
    if (symbol.empty() || symbol != argv[3]) throw std::runtime_error("stage and kernel ABI mismatch");
    const unsigned int padded = (tokens + 63u) / 64u * 64u;
    const size_t chunks = padded / 64u, source_chunks = (source_tokens + 63u) / 64u;
    auto path = [&](const char* name) { return directory + "/full-" + name + ".bin"; };
    auto bf_input = [&](const char* name, size_t features) {
        return read_range<uint16_t>(path(name), source_tokens * features, 0, tokens * features, padded * features);
    };
    auto bf_reference = [&](const char* name, size_t features) {
        return read_range<uint16_t>(path(name), source_tokens * features, 0, tokens * features, tokens * features);
    };
    size_t allocation = 0;
    std::vector<std::pair<std::string, Stats>> surfaces;
    unsigned int segments = 0; float maximum_ms = 0;
    if (stage == "solve") {
        auto a = read_range<float>(path("a-f32"), source_tokens * matrix_features, 0, tokens * matrix_features, padded * matrix_features);
        auto expected = bf_reference("a-inverse-bf16", matrix_features);
        allocation = padded * matrix_features * 6u;
        Launcher launch(argv[2], argv[3], threads, shared, allocation); Buffer input, output;
        input.upload(a); output.allocate(padded * matrix_features * 2u);
        // solve_tril only writes the lower 16x16 block triangle. Upper blocks
        // must be initialized exactly as in the production provider.
        check(hipMemset(output.data, 0, padded * matrix_features * 2u));
        for (unsigned int offset = 0; offset < padded; offset += 1024u) {
            int32_t count = static_cast<int32_t>(std::min(1024u, padded - offset));
            auto* pa = input.at<float>(offset * matrix_features); auto* pi = output.at<uint16_t>(offset * matrix_features);
            launch.launch<3>(static_cast<unsigned int>(count) / 64u, 32u, offset, &pa, &pi, &count);
        }
        auto actual = download(output.at<uint16_t>(), tokens * matrix_features);
        surfaces.emplace_back("a-inverse-bf16", compare(actual, expected)); dump_q64(stage, "a-inverse-bf16", actual);
        segments = launch.segments; maximum_ms = launch.maximum_ms;
    } else {
        auto k = bf_input("k-normalized-bf16", key_features);
        auto g = read_range<float>(path("g-cumsum-f32"), source_tokens * 32u, 0, tokens * 32u, padded * 32u);
        for (unsigned int t = tokens; t < padded; ++t) for (unsigned int h = 0; h < 32; ++h) g[t * 32u + h] = g[(tokens - 1u) * 32u + h];
        if (stage == "wu") {
            auto v = bf_input("v-bf16", value_features), beta = bf_input("beta-bf16", 32u);
            auto inverse = bf_input("a-inverse-bf16", matrix_features);
            auto expected_w = bf_reference("w-bf16", value_features), expected_u = bf_reference("u-bf16", value_features);
            allocation = (k.size() + v.size() + beta.size() + inverse.size() + 2u * padded * value_features) * 2u + g.size() * 4u;
            Launcher launch(argv[2], argv[3], threads, shared, allocation); Buffer dk, dv, db, da, dg, dw, du;
            dk.upload(k); dv.upload(v); db.upload(beta); da.upload(inverse); dg.upload(g);
            dw.allocate(padded * value_features * 2u); du.allocate(padded * value_features * 2u);
            for (unsigned int offset = 0; offset < padded; offset += 1024u) {
                int32_t count = static_cast<int32_t>(std::min(1024u, padded - offset));
                auto* pk = dk.at<uint16_t>(offset * key_features); auto* pv = dv.at<uint16_t>(offset * value_features);
                auto* pb = db.at<uint16_t>(offset * 32u); auto* pa = da.at<uint16_t>(offset * matrix_features);
                auto* pg = dg.at<float>(offset * 32u); auto* pw = dw.at<uint16_t>(offset * value_features); auto* pu = du.at<uint16_t>(offset * value_features);
                launch.launch<8>(static_cast<unsigned int>(count) / 64u, 32u, offset, &pk, &pv, &pb, &pw, &pu, &pa, &pg, &count);
            }
            auto w = download(dw.at<uint16_t>(), tokens * value_features), u = download(du.at<uint16_t>(), tokens * value_features);
            surfaces.emplace_back("w-bf16", compare(w, expected_w)); surfaces.emplace_back("u-bf16", compare(u, expected_u));
            dump_q64(stage, "w-bf16", w); dump_q64(stage, "u-bf16", u);
            segments = launch.segments; maximum_ms = launch.maximum_ms;
        } else {
            auto w = bf_input("w-bf16", value_features), u = bf_input("u-bf16", value_features);
            auto seed = read_range<float>(path("initial_state-f32"), state_elements, 0, state_elements, state_elements);
            auto expected_v = bf_reference("v-new-bf16", value_features);
            auto expected_h = read_range<uint16_t>(path("chunk-state-bf16"), source_chunks * state_elements, 0, chunks * state_elements, chunks * state_elements);
            std::vector<float> expected_terminal;
            std::vector<uint16_t> expected_next;
            if (tokens == source_tokens)
                expected_terminal = read_range<float>(path("native-final-state-f32"), state_elements, 0, state_elements, state_elements);
            else
                expected_next = read_range<uint16_t>(path("chunk-state-bf16"), source_chunks * state_elements, chunks * state_elements, state_elements, state_elements);
            allocation = (k.size() + w.size() + u.size() + padded * value_features + chunks * state_elements) * 2u + (g.size() + state_elements * 2u) * 4u;
            Launcher launch(argv[2], argv[3], threads, shared, allocation); Buffer dk, dw, du, dg, dh, dv, state_a, state_b;
            dk.upload(k); dw.upload(w); du.upload(u); dg.upload(g); state_a.upload(seed); state_b.allocate(state_elements * 4u);
            dh.allocate(chunks * state_elements * 2u); dv.allocate(padded * value_features * 2u);
            float* initial = state_a.at<float>(); float* final = state_b.at<float>();
            for (unsigned int offset = 0; offset < padded; offset += 1024u) {
                int32_t count = static_cast<int32_t>(std::min(1024u, padded - offset));
                auto* pk = dk.at<uint16_t>(offset * key_features); auto* pw = dw.at<uint16_t>(offset * value_features);
                auto* pu = du.at<uint16_t>(offset * value_features); auto* pv = dv.at<uint16_t>(offset * value_features);
                auto* pg = dg.at<float>(offset * 32u); auto* ph = dh.at<uint16_t>(offset / 64u * state_elements);
                launch.launch<9>(8u, 32u, offset, &pk, &pu, &pw, &pv, &pg, &initial, &ph, &final, &count);
                std::swap(initial, final);  // The launch has completed before buffer reuse.
            }
            auto v = download(dv.at<uint16_t>(), tokens * value_features), h = download(dh.at<uint16_t>(), chunks * state_elements);
            auto terminal = download(initial, state_elements);
            surfaces.emplace_back("v-new-bf16", compare(v, expected_v)); surfaces.emplace_back("chunk-state-bf16", compare(h, expected_h));
            if (tokens == source_tokens) {
                surfaces.emplace_back("final-state-f32", compare(terminal, expected_terminal));
            } else {
                std::vector<uint16_t> rounded(state_elements); for (size_t i = 0; i < state_elements; ++i) rounded[i] = bf16(terminal[i]);
                surfaces.emplace_back("next-chunk-state-bf16", compare(rounded, expected_next));
            }
            dump_q64(stage, "v-new-bf16", v); dump_q64(stage, "chunk-state-bf16", h); dump_q64(stage, "terminal-state-f32", terminal);
            segments = launch.segments; maximum_ms = launch.maximum_ms;
        }
    }
    std::cerr << "UPSTREAM_REPLAY phase=readback_complete\n" << std::flush;
    std::cout << std::setprecision(17) << "{\"kind\":\"native_captured_upstream_stage\",\"stage\":\"" << stage
              << "\",\"source_tokens\":" << source_tokens << ",\"tokens\":" << tokens << ",\"model_loaded\":false,\"inference_acceptance\":false"
              << ",\"allocation_bytes\":" << allocation << ",\"segments\":" << segments << ",\"maximum_dispatch_ms\":" << maximum_ms << ",\"surfaces\":{";
    bool mismatch = false;
    for (size_t i = 0; i < surfaces.size(); ++i) {
        if (i) std::cout << ','; surfaces[i].second.print(surfaces[i].first);
        mismatch |= surfaces[i].second.mismatches != 0 || surfaces[i].second.nonfinite != 0;
    }
    std::cout << "}}\n"; return mismatch ? 3 : 0;
} catch (const std::exception& error) {
    std::cerr << "fla_upstream_capture_replay error=" << error.what() << '\n'; return 2;
}
