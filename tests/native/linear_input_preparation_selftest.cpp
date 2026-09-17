#include <hip/hip_runtime.h>
#include "../../native/providers/gdn/linear_input_preparation.h"
#include "../../native/providers/gdn/blackwell_l2norm.h"
// Generated from the pinned whole provider, with every function unchanged.
#include "linear_input_control.generated.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace p = qrt_linear_input_preparation;
constexpr size_t guard = 64u;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
template<class T> std::vector<T> read_file(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f) && f.tellg() == std::streamoff(count * sizeof(T)), "input file size");
    std::vector<T> out(count); f.seekg(0); f.read(reinterpret_cast<char*>(out.data()), count * sizeof(T));
    require(bool(f), "input file read"); return out;
}
template<class T> struct Device {
    T* base = nullptr;
    size_t count;
    std::vector<T> initial;
    explicit Device(size_t cells) : count(cells), initial(cells + 2u * guard) {
        std::memset(initial.data(), 0xa5, initial.size() * sizeof(T));
        check(hipMalloc(reinterpret_cast<void**>(&base), initial.size() * sizeof(T))); reset();
    }
    explicit Device(const std::vector<T>& source) : Device(source.size()) {
        std::memcpy(initial.data() + guard, source.data(), count * sizeof(T)); reset();
    }
    ~Device() { if (base) (void)hipFree(base); }
    Device(const Device&) = delete;
    T* data() { return base + guard; }
    void reset() { check(hipMemcpy(base, initial.data(), initial.size() * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> read() {
        std::vector<T> out(initial.size()); check(hipMemcpy(out.data(), base, out.size() * sizeof(T), hipMemcpyDeviceToHost));
        require(!std::memcmp(out.data(), initial.data(), guard * sizeof(T)) &&
            !std::memcmp(out.data() + guard + count, initial.data() + guard + count, guard * sizeof(T)), "device redzone");
        return out;
    }
    void immutable() { const auto x = read(); require(!std::memcmp(x.data(), initial.data(), x.size() * sizeof(T)), "immutable input changed"); }
};
struct Stream {
    hipStream_t value = nullptr;
    Stream() { check(hipStreamCreateWithFlags(&value, hipStreamNonBlocking)); }
    ~Stream() { if (value) (void)hipStreamDestroy(value); }
    void finish() {
        hipEvent_t event; check(hipEventCreateWithFlags(&event, hipEventDisableTiming)); check(hipEventRecord(event, value));
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const auto status = hipEventQuery(event); if (status == hipSuccess) break;
            require(status == hipErrorNotReady && std::chrono::steady_clock::now() < deadline, "completion deadline");
            std::this_thread::yield();
        }
        check(hipEventDestroy(event));
    }
};
__global__ void compact_v(const float* raw, uint16_t* v, unsigned tokens) {
    const size_t cell = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell < size_t(tokens) * 4096u) v[cell] = qrt_fla_blackwell::to_bf16(raw[(cell / 4096u) * 8192u + 4096u + cell % 4096u]);
}
uint32_t mix(uint32_t x) { x ^= x << 13u; x ^= x >> 17u; return x ^ (x << 5u); }
float f32(uint32_t bits) { float x; std::memcpy(&x, &bits, 4u); return x; }
uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4u); return b; }
uint16_t conv_round(float x) { const auto b = bits(x); return uint16_t((b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u); }
uint16_t norm_round(float x) {
    const auto b = bits(x); return (b & 0x7fffffffu) > 0x7f800000u ? uint16_t((b | 0x400000u) >> 16u) : conv_round(x);
}
float host_conv(const std::vector<float>& input, const std::vector<uint16_t>& weights,
    const std::vector<unsigned char>& silu, unsigned token, unsigned feature) {
    float sum = 0.0f;
    for (unsigned tap = 0u; tap < 4u; ++tap) if (token + tap >= 3u) {
        const float x = f32(uint32_t(conv_round(input[size_t(token + tap - 3u) * 8192u + feature])) << 16u);
        const float w = f32(uint32_t(weights[feature * 4u + tap]) << 16u);
        sum += f32(uint32_t(conv_round(x * w)) << 16u);
    }
    return f32(uint32_t(qrt_sm121_silu::evaluate(silu.data(), sum)) << 16u);
}
size_t cpu_check(const std::vector<float>& input, const std::vector<uint16_t>& weights,
    const std::vector<unsigned char>& silu, const std::vector<unsigned char>& rsqrt,
    const std::vector<float>& raw, const std::vector<uint16_t>& q, const std::vector<uint16_t>& k,
    unsigned first, unsigned count) {
    size_t checked = 0u;
    const unsigned positions[] = {0u, count / 2u, count - 1u};
    for (unsigned local : positions) for (unsigned head : {0u, 7u, 15u}) for (unsigned side = 0u; side < 2u; ++side) {
        float values[128], sums[16];
        for (unsigned i = 0u; i < 128u; ++i) {
            const unsigned feature = side * 2048u + head * 128u + i;
            values[i] = host_conv(input, weights, silu, first + local, feature);
            require(bits(values[i]) == bits(raw[guard + size_t(local) * 8192u + feature]), "CPU convolution mismatch");
        }
        for (unsigned lane = 0u; lane < 16u; ++lane) {
            const auto* v = values + lane * 8u; float sum = std::fma(v[0], v[0], v[1] * v[1]);
            for (unsigned i = 2u; i < 8u; ++i) sum = std::fma(v[i], v[i], sum);
            sums[lane] = sum;
        }
        for (unsigned delta = 8u; delta; delta >>= 1u) {
            float prior[16]; std::memcpy(prior, sums, sizeof(sums));
            for (unsigned lane = 0u; lane < 16u; ++lane) sums[lane] = prior[lane] + prior[lane ^ delta];
        }
        const auto& output = side ? k : q;
        for (unsigned i = 0u; i < 128u; ++i) {
            const float inverse = qrt_sm121_rsqrt::evaluate(rsqrt.data(), sums[i / 8u] + 1.0e-6f);
            require(norm_round(values[i] * inverse) == output[guard + size_t(local) * 2048u + head * 128u + i], "CPU normalization mismatch");
            ++checked;
        }
    }
    return checked;
}

void run(const std::vector<float>& input, const std::vector<uint16_t>& weights,
    const std::vector<unsigned char>& silu, const std::vector<unsigned char>& rsqrt,
    Device<unsigned char>& ds, Device<unsigned char>& dr, unsigned first, unsigned count,
    unsigned mode, bool timing, const std::string& gb10 = "") {
    const unsigned tokens = unsigned(input.size() / 8192u);
    Device<float> dx(input), raw(size_t(count) * 8192u);
    Device<uint16_t> dw(weights), q(size_t(count) * 2048u), k(size_t(count) * 2048u), v(size_t(count) * 4096u);
    std::vector<uint32_t> window(size_t(count) * 4u);
    for (unsigned t = 0u; t < count; ++t) for (unsigned tap = 0u; tap < 4u; ++tap)
        window[t * 4u + tap] = first + t + tap >= 3u ? first + t + tap - 3u : UINT32_MAX;
    Device<uint32_t> dwindows(window);
    p::Inputs in{dx.data(), input.size(), dw.data(), weights.size(), ds.data(), silu.size(), dr.data(), rsqrt.size(), tokens};
    p::Outputs out{q.data(), k.data(), v.data(), q.count, k.count, v.count};
    Stream stream;
    std::vector<float> original_raw;
    std::vector<uint16_t> original_q, original_k, original_v;
    double samples[2][3]{}; size_t cpu = 0u, gb_cells = 0u;
    const unsigned attempts = timing ? 4u : 1u;
    for (unsigned attempt = 0u; attempt < attempts; ++attempt) for (unsigned position = 0u; position < 2u; ++position) {
        const unsigned variant = (position + attempt) % 2u;
        q.reset(); k.reset(); v.reset(); raw.reset(); stream.finish();
        const auto start = std::chrono::steady_clock::now();
        if (!variant) {
            hipLaunchKernelGGL(selected_conv_qkv_window_kernel, dim3(32u, count), dim3(256u), 0u, stream.value,
                dx.data(), dw.data(), first ? dwindows.data() : nullptr, raw.data(), count, 3u, nullptr, nullptr, 0u, ds.data());
            check(hipGetLastError()); check(qrt_fla_blackwell_norm::normalize(raw.data(), q.data(), k.data(), count, stream.value));
            hipLaunchKernelGGL(compact_v, dim3((v.count + 255u) / 256u), dim3(256u), 0u, stream.value, raw.data(), v.data(), count);
            check(hipGetLastError());
        } else check(p::launch(in, out, first, count, stream.value));
        stream.finish();
        if (attempt) samples[variant][attempt - 1u] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto actual_q = q.read(), actual_k = k.read(), actual_v = v.read(); const auto actual_raw = raw.read();
        if (!attempt && !variant) {
            original_raw = actual_raw; original_q = actual_q; original_k = actual_k; original_v = actual_v;
            if (mode != 2u) cpu = cpu_check(input, weights, silu, rsqrt, original_raw, original_q, original_k, first, count);
            if (!gb10.empty()) {
                const auto gq = read_file<uint16_t>(gb10 + "/full-q-bf16.bin", 7169u * 2048u);
                const auto gk = read_file<uint16_t>(gb10 + "/full-k-bf16.bin", 7169u * 2048u);
                const auto gv = read_file<uint16_t>(gb10 + "/full-v-bf16.bin", 7169u * 4096u);
                require(first == 0u && count >= 7169u, "GB10 capture span");
                for (unsigned t = 0u; t < 7169u; ++t) for (unsigned f = 0u; f < 8192u; ++f) {
                    const uint16_t expected = f < 2048u ? gq[size_t(t) * 2048u + f] : f < 4096u ? gk[size_t(t) * 2048u + f - 2048u] : gv[size_t(t) * 4096u + f - 4096u];
                    require(conv_round(actual_raw[guard + size_t(t) * 8192u + f]) == expected, "GB10 raw convolution mismatch"); ++gb_cells;
                }
            }
        }
        require(actual_q == original_q && actual_k == original_k && actual_v == original_v, "complete compact output mismatch");
        if (variant) require(!std::memcmp(actual_raw.data(), raw.initial.data(), actual_raw.size() * 4u), "production candidate wrote optional raw storage");
        else require(!std::memcmp(actual_raw.data(), original_raw.data(), actual_raw.size() * 4u), "control convolution changed");
        dx.immutable(); dw.immutable(); dwindows.immutable();
    }
    // Audit is separate from timing and must preserve every production output.
    q.reset(); k.reset(); v.reset(); raw.reset(); out.raw = raw.data(); out.raw_cells = raw.count;
    check(p::launch(in, out, first, count, stream.value)); stream.finish();
    const auto captured_raw = raw.read();
    require(!std::memcmp(captured_raw.data(), original_raw.data(), captured_raw.size() * 4u), "audit convolution mismatch");
    require(q.read() == original_q && k.read() == original_k && v.read() == original_v, "audit/production compact mismatch");
    dx.immutable(); dw.immutable(); ds.immutable(); dr.immutable(); dwindows.immutable();
    for (unsigned variant = 0u; variant < 2u; ++variant) {
        std::array<double, 3> ordered{samples[variant][0], samples[variant][1], samples[variant][2]}; std::sort(ordered.begin(), ordered.end());
        std::printf("{\"kind\":\"linear_input_preparation_component\",\"variant\":%u,\"source_tokens\":%u,\"first\":%u,\"tokens\":%u,\"input_mode\":%u,\"compact_cells\":%zu,\"cpu_norm_cells\":%zu,\"gb10_raw_convolution_cells\":%zu,\"raw_bit_mismatches\":0,\"complete_operator_ms\":%.7f,\"samples_ms\":[%.7f,%.7f,%.7f],\"measured_attempts\":%u,\"warmups\":1,\"all_attempts_verified\":true,\"production_diagnostic_parity\":true,\"redzones_pass\":true,\"immutable_inputs_and_tables\":true,\"fused_raw_storage_bytes\":0,\"control_raw_storage_bytes\":%zu,\"source_control_extracted_unchanged\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            variant, tokens, first, count, mode, size_t(count) * 8192u, cpu, gb_cells, ordered[1], samples[variant][0], samples[variant][1], samples[variant][2], timing ? 3u : 0u, size_t(count) * 8192u * 4u);
    }
    std::fflush(stdout);
}

int main(int argc, char** argv) {
    try {
        require(argc == 4 || argc == 7, "SILU RSQRT safety|capture [PROJECTED_BF16 WEIGHTS_BF16 GB10_DIRECTORY]");
        const auto silu = read_file<unsigned char>(argv[1], qrt_sm121_silu::table_bytes);
        const auto rsqrt = read_file<unsigned char>(argv[2], qrt_sm121_rsqrt::table_bytes);
        require(qrt_sm121_silu::valid_layout(silu.data(), silu.size()) && qrt_sm121_rsqrt::valid_layout(rsqrt.data(), rsqrt.size()), "table layout");
        _putenv_s("QRT_FLA_GDN_SM121_RSQRT_TABLE", argv[2]); check(qrt_fla_blackwell_norm::prepare_table());
        Device<unsigned char> ds(silu), dr(rsqrt);
        if (std::string(argv[3]) == "safety") {
            require(argc == 4, "safety arguments");
            for (unsigned tokens : {1u, 2u, 3u, 4u, 5u, 63u, 65u, 1025u}) for (unsigned mode = 0u; mode < 3u; ++mode) {
                std::vector<float> input(size_t(tokens) * 8192u); std::vector<uint16_t> weights(8192u * 4u);
                const uint16_t special[] = {0u, 0x8000u, 1u, 0x807fu, 0x0080u, 0x7f7fu, 0xff7fu, 0x7f80u, 0xff80u, 0x7fc1u, 0xffffu, 0x3f80u};
                for (size_t i = 0u; i < input.size(); ++i) {
                    const auto x = mix(uint32_t(i) ^ 0x3958192u);
                    input[i] = mode == 1u ? f32((x & 1u) << 31u) : mode == 2u ? f32(uint32_t(special[x % 12u]) << 16u) : f32((x & 0x807fffffu) | ((116u + x % 17u) << 23u));
                }
                for (size_t i = 0u; i < weights.size(); ++i) {
                    const auto x = mix(uint32_t(i) ^ 0x8192395u);
                    weights[i] = mode == 2u ? special[x % 12u] : uint16_t((x & 0x807fu) | ((118u + x % 12u) << 7u));
                }
                run(input, weights, silu, rsqrt, ds, dr, 0u, tokens, mode, false);
                if (tokens > 3u) run(input, weights, silu, rsqrt, ds, dr, 3u, tokens - 3u, mode, false);
            }
        } else {
            require(std::string(argv[3]) == "capture" && argc == 7, "capture arguments");
            const auto projected = read_file<uint16_t>(argv[4], 7169u * 8192u);
            const auto weights = read_file<uint16_t>(argv[5], 8192u * 4u);
            for (unsigned tokens : {7169u, 8192u}) {
                std::vector<float> input(size_t(tokens) * 8192u);
                for (size_t i = 0u; i < input.size(); ++i) input[i] = f32(uint32_t(projected[i % projected.size()]) << 16u);
                run(input, weights, silu, rsqrt, ds, dr, 0u, tokens, 3u, true, argv[6]);
            }
        }
        std::vector<unsigned char> original(rsqrt.size());
        check(hipMemcpy(original.data(), qrt_fla_blackwell_norm::table_device(), original.size(), hipMemcpyDeviceToHost));
        require(original == rsqrt, "original normalization table changed");
        qrt_fla_blackwell_norm::release_table(); return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
