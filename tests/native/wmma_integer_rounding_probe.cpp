// Isolated hardware diagnostic. No runtime dispatch or integer repair uses it.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace {
constexpr unsigned guard = 64u, families = 8u, cases = 64u, permutations = 4u;
constexpr unsigned tiles = families * cases * permutations, cells = tiles * 256u;
uint32_t bits(float value) { uint32_t result; std::memcpy(&result, &value, 4u); return result; }
uint32_t hash(uint32_t x) { x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; return x ^ (x >> 16u); }
unsigned permute(unsigned permutation, unsigned k) {
    return permutation == 0u ? k : permutation == 1u ? 15u - k :
        permutation == 2u ? (k % 8u) * 2u + k / 8u : (k * 5u + 3u) % 16u;
}
int operand(unsigned family, unsigned sample, unsigned row, unsigned k, bool right) {
    const unsigned index = family == 7u ? k / 2u : k;
    const unsigned random = hash(0x3958192u ^ sample * 7919u ^ row * 104729u ^ index * 65537u ^ unsigned(right) * 99991u);
    if ((family == 3u && k >= 2u) || (family == 4u && k >= 4u) || (family == 5u && k >= 8u)) return 0;
    if (family == 0u) return int(random & 255u);
    if (family == 2u) return right ? int(random & 255u) : -int(random & 255u);
    if (family >= 6u) return right && (k & 1u) ? -int(random & 255u) : int(random & 255u);
    return int(random & 511u) - 256;
}
uint16_t encode(int value, bool bf16) {
    if (bf16) return uint16_t(bits(float(value)) >> 16u);
    const unsigned magnitude = unsigned(value < 0 ? -value : value);
    unsigned exponent = 0u;
    for (unsigned copy = magnitude; copy > 1u; copy >>= 1u) ++exponent;
    return magnitude ? uint16_t((value < 0 ? 0x8000u : 0u) | ((exponent + 15u) << 10u) |
        ((magnitude << (10u - exponent)) & 1023u)) : 0u;
}
float decode(uint16_t value, bool bf16) {
    if (bf16) { const uint32_t raw = uint32_t(value) << 16u; float result; std::memcpy(&result, &raw, 4u); return result; }
    const int sign = value & 0x8000u ? -1 : 1;
    const unsigned exponent = (value >> 10u) & 31u;
    return exponent ? float(sign) * std::ldexp(float(1024u + (value & 1023u)), int(exponent) - 25) : 0.0f;
}
struct Inputs {
    std::vector<uint16_t> left, right;
    std::vector<int64_t> expected;
    explicit Inputs(bool bf16) : left(cells + 2u * guard, 0x5a5au), right(left), expected(cells) {
        for (int value = -256; value <= 256; ++value)
            if (decode(encode(value, bf16), bf16) != float(value)) throw std::runtime_error("integer encoding changed value");
        for (unsigned tile = 0u; tile < tiles; ++tile) {
            const unsigned permutation = tile % permutations, sample = tile / permutations % cases, family = tile / (permutations * cases);
            for (unsigned row = 0u; row < 16u; ++row) for (unsigned k = 0u; k < 16u; ++k) {
                const unsigned original_k = permute(permutation, k);
                left[guard + tile * 256u + row * 16u + k] = encode(operand(family, sample, row, original_k, false), bf16);
                right[guard + tile * 256u + row * 16u + k] = encode(operand(family, sample, row, original_k, true), bf16);
            }
            for (unsigned row = 0u; row < 16u; ++row) for (unsigned column = 0u; column < 16u; ++column) {
                int64_t total = 0;
                for (unsigned k = 0u; k < 16u; ++k)
                    total += int64_t(operand(family, sample, row, k, false)) * operand(family, sample, column, k, true);
                if (total < -1048576 || total > 1048576 || int64_t(float(total)) != total)
                    throw std::runtime_error("independent integer oracle is not exact FP32");
                expected[tile * 256u + row * 16u + column] = total;
                int64_t encoded_total = 0;
                for (unsigned k = 0u; k < 16u; ++k)
                    encoded_total += int64_t(decode(left[guard + tile * 256u + row * 16u + k], bf16)) *
                        int64_t(decode(right[guard + tile * 256u + column * 16u + k], bf16));
                if (encoded_total != total) throw std::runtime_error("encoded or permuted inputs changed integer dot");
            }
        }
    }
};

#if defined(__HIPCC__)
using U16x16 = unsigned short __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
template<bool Bf16>
__global__ void intrinsic_control(const uint16_t* left, const uint16_t* right, float* output, float bias) {
    const unsigned lane = threadIdx.x, tile = blockIdx.x;
    U16x16 a{}, b{};
#pragma unroll
    for (unsigned k = 0u; k < 16u; ++k) {
        a[k] = left[tile * 256u + (lane % 16u) * 16u + k];
        b[k] = right[tile * 256u + (lane % 16u) * 16u + k];
    }
    F32x8 zero{};
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) zero[element] = bias;
    F32x8 result;
    if constexpr (Bf16) result = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, zero);
    else {
        using F16x16 = _Float16 __attribute__((ext_vector_type(16)));
        F16x16 fa, fb; __builtin_memcpy(&fa, &a, sizeof(a)); __builtin_memcpy(&fb, &b, sizeof(b));
        result = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(fa, fb, zero);
    }
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element)
        output[tile * 256u + (2u * element + lane / 16u) * 16u + lane % 16u] = result[element];
}
template<unsigned Mode, bool Bf16>
__global__ void probe(const uint16_t* left, const uint16_t* right, float* output, uint32_t* metadata) {
    const unsigned lane = threadIdx.x, tile = blockIdx.x;
    U16x16 a{}, b{};
#pragma unroll
    for (unsigned k = 0u; k < 16u; ++k) {
        a[k] = left[tile * 256u + (lane % 16u) * 16u + k];
        b[k] = right[tile * 256u + (lane % 16u) * 16u + k];
    }
    const F32x8 zero{};
    F32x8 result;
    unsigned before, during, after;
    float positive, negative;
    const float one = 1.0f, half_ulp = 0x1p-24f;
    // Keep set/use/restore in one assembly block. Early-clobber constraints
    // prevent D from overlapping A/B. Only this wave's four round bits change;
    // denorm and exception state are untouched, and the original bits return.
#define QRT_MODE_PROBE(OP) asm volatile( \
        "s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 4)\n\t" \
        "s_round_mode %9\n\t" \
        "s_getreg_b32 %1, hwreg(HW_REG_MODE, 0, 4)\n\t" \
        "v_add_f32_e64 %3, %10, %11\n\t" \
        "v_add_f32_e64 %4, -%10, -%11\n\t" \
        OP " %5, %6, %7, %8\n\t" \
        "s_setreg_b32 hwreg(HW_REG_MODE, 0, 4), %0\n\t" \
        "s_getreg_b32 %2, hwreg(HW_REG_MODE, 0, 4)\n\t" \
        : "=&s"(before), "=&s"(during), "=&s"(after), "=&v"(positive), "=&v"(negative), "=&v"(result) \
        : "v"(a), "v"(b), "v"(zero), "n"(Mode * 5u), "v"(one), "v"(half_ulp) : "memory")
    if constexpr (Bf16) { QRT_MODE_PROBE("v_wmma_f32_16x16x16_bf16"); }
    else { QRT_MODE_PROBE("v_wmma_f32_16x16x16_f16"); }
#undef QRT_MODE_PROBE
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element)
        output[tile * 256u + (2u * element + lane / 16u) * 16u + lane % 16u] = result[element];
    if (!lane) {
        metadata[tile * 5u] = before; metadata[tile * 5u + 1u] = during; metadata[tile * 5u + 2u] = after;
        metadata[tile * 5u + 3u] = __float_as_uint(positive); metadata[tile * 5u + 4u] = __float_as_uint(negative);
    }
}
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
struct Device {
    void* data = nullptr;
    explicit Device(size_t bytes) { check(hipMalloc(&data, bytes)); }
    ~Device() { if (data && hipFree(data) != hipSuccess) std::abort(); }
    template<class T> T* as() { return static_cast<T*>(data); }
};
void finish() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error("WMMA probe completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
void run(bool bf16) {
    Inputs input(bf16);
    Device left(input.left.size() * 2u), right(input.right.size() * 2u), output((cells + 2u * guard) * 4u), metadata((tiles * 5u + 2u * guard) * 4u);
    check(hipMemcpy(left.data, input.left.data(), input.left.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(right.data, input.right.data(), input.right.size() * 2u, hipMemcpyHostToDevice));
    std::vector<float> result(cells + 2u * guard), nearest, intrinsic(result.size());
    std::vector<uint32_t> modes(tiles * 5u + 2u * guard);
    check(hipMemset(output.data, 0xa5, result.size() * 4u));
    if (bf16) hipLaunchKernelGGL((intrinsic_control<true>), dim3(tiles), dim3(32u), 0u, nullptr,
        left.as<uint16_t>() + guard, right.as<uint16_t>() + guard, output.as<float>() + guard, 0.0f);
    else hipLaunchKernelGGL((intrinsic_control<false>), dim3(tiles), dim3(32u), 0u, nullptr,
        left.as<uint16_t>() + guard, right.as<uint16_t>() + guard, output.as<float>() + guard, 0.0f);
    check(hipGetLastError()); finish();
    check(hipMemcpy(intrinsic.data(), output.data, intrinsic.size() * 4u, hipMemcpyDeviceToHost));
    for (unsigned mode = 0u; mode < 4u; ++mode) {
        check(hipMemset(output.data, 0xa5, result.size() * 4u)); check(hipMemset(metadata.data, 0xa5, modes.size() * 4u));
#define QRT_LAUNCH_MODE(M, B) hipLaunchKernelGGL((probe<M, B>), dim3(tiles), dim3(32u), 0u, nullptr, left.as<uint16_t>() + guard, right.as<uint16_t>() + guard, output.as<float>() + guard, metadata.as<uint32_t>() + guard)
#define QRT_DISPATCH_MODE(M) case M: if (bf16) { QRT_LAUNCH_MODE(M, true); } else { QRT_LAUNCH_MODE(M, false); } break
        switch (mode) { QRT_DISPATCH_MODE(0u); QRT_DISPATCH_MODE(1u); QRT_DISPATCH_MODE(2u); QRT_DISPATCH_MODE(3u); }
#undef QRT_DISPATCH_MODE
#undef QRT_LAUNCH_MODE
        check(hipGetLastError()); finish();
        check(hipMemcpy(result.data(), output.data, result.size() * 4u, hipMemcpyDeviceToHost));
        check(hipMemcpy(modes.data(), metadata.data, modes.size() * 4u, hipMemcpyDeviceToHost));
        for (unsigned i = 0u; i < guard; ++i)
            if (bits(result[i]) != 0xa5a5a5a5u || bits(result[result.size() - 1u - i]) != 0xa5a5a5a5u ||
                modes[i] != 0xa5a5a5a5u || modes[modes.size() - 1u - i] != 0xa5a5a5a5u) throw std::runtime_error("probe output guard changed");
        for (unsigned tile = 0u; tile < tiles; ++tile) {
            const auto* observed = modes.data() + guard + tile * 5u;
            if (observed[0] != observed[2] || observed[1] != mode * 5u ||
                observed[3] != (mode == 1u ? 0x3f800001u : 0x3f800000u) ||
                observed[4] != (mode == 2u ? 0xbf800001u : 0xbf800000u)) throw std::runtime_error("round state or scalar control failed");
        }
        if (!mode) {
            if (std::memcmp(intrinsic.data(), result.data(), result.size() * 4u))
                throw std::runtime_error("isolated assembly changed intrinsic output or guard");
            nearest = result;
        }
        for (unsigned family = 0u; family < families; ++family) {
            unsigned mismatch = 0u, fractional = 0u, changed = 0u, repaired = 0u, first = cells;
            double maximum_error = 0.0;
            const unsigned begin = family * cases * permutations * 256u, end = begin + cases * permutations * 256u;
            for (unsigned i = begin; i < end; ++i) {
                const float value = result[guard + i];
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite small-integer dot");
                const bool different = value != float(input.expected[i]);
                mismatch += unsigned(different); fractional += unsigned(std::trunc(value) != value);
                changed += unsigned(bits(value) != bits(nearest[guard + i]));
                repaired += unsigned(std::nearbyint(value) != float(input.expected[i]));
                maximum_error = std::max(maximum_error, std::fabs(double(value) - double(input.expected[i])));
                if (different && first == cells) first = i;
            }
            std::printf("{\"kind\":\"wmma_integer_rounding_diagnostic\",\"dtype\":\"%s\",\"round_mode\":%u,\"family\":%u,\"cells\":%u,\"integer_value_mismatches\":%u,\"fractional_results\":%u,\"raw_changes_from_nearest\":%u,\"nearest_integer_mismatches_diagnostic_only\":%u,\"maximum_absolute_error\":%.12g,\"round_state_restored\":true,\"scalar_round_control_pass\":true,\"redzones_pass\":true,\"integer_repair_implemented\":false,\"inference_acceptance\":false}\n",
                bf16 ? "bf16" : "fp16", mode, family, end - begin, mismatch, fractional, changed, repaired, maximum_error);
            if (!mode && first < cells) {
                const unsigned tile = first / 256u, row = first % 256u / 16u, column = first % 16u;
                std::printf("{\"kind\":\"wmma_integer_counterexample\",\"dtype\":\"%s\",\"family\":%u,\"tile\":%u,\"row\":%u,\"column\":%u,\"expected_integer\":%lld,\"actual\":%.12g,\"actual_bits\":%u,\"left\":[", bf16 ? "bf16" : "fp16", family, tile, row, column, static_cast<long long>(input.expected[first]), double(result[guard + first]), bits(result[guard + first]));
                for (unsigned k = 0u; k < 16u; ++k) std::printf("%s%.0f", k ? "," : "", double(decode(input.left[guard + tile * 256u + row * 16u + k], bf16)));
                std::printf("],\"right\":[");
                for (unsigned k = 0u; k < 16u; ++k) std::printf("%s%.0f", k ? "," : "", double(decode(input.right[guard + tile * 256u + column * 16u + k], bf16)));
                std::printf("]}\n");
            }
        }
        std::fflush(stdout);
    }
    std::vector<uint16_t> copy(input.left.size());
    check(hipMemcpy(copy.data(), left.data, copy.size() * 2u, hipMemcpyDeviceToHost));
    if (copy != input.left) throw std::runtime_error("left input or guard changed");
    check(hipMemcpy(copy.data(), right.data, copy.size() * 2u, hipMemcpyDeviceToHost));
    if (copy != input.right) throw std::runtime_error("right input or guard changed");
    std::printf("{\"kind\":\"wmma_integer_probe_inputs\",\"dtype\":\"%s\",\"immutable_inputs\":true,\"redzones_pass\":true}\n", bf16 ? "bf16" : "fp16");
}
void run_bias(bool bf16) {
    Inputs input(bf16);
    Device left(input.left.size() * 2u), right(input.right.size() * 2u), output((cells + 2u * guard) * 4u);
    check(hipMemcpy(left.data, input.left.data(), input.left.size() * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(right.data, input.right.data(), input.right.size() * 2u, hipMemcpyHostToDevice));
    std::vector<float> result(cells + 2u * guard);
    for (int bias : {0, 262144, 1048576, 4194304, 12582912, -262144, -1048576, -4194304, -12582912}) {
        check(hipMemset(output.data, 0xa5, result.size() * 4u));
        if (bf16) hipLaunchKernelGGL((intrinsic_control<true>), dim3(tiles), dim3(32u), 0u, nullptr,
            left.as<uint16_t>() + guard, right.as<uint16_t>() + guard, output.as<float>() + guard, float(bias));
        else hipLaunchKernelGGL((intrinsic_control<false>), dim3(tiles), dim3(32u), 0u, nullptr,
            left.as<uint16_t>() + guard, right.as<uint16_t>() + guard, output.as<float>() + guard, float(bias));
        check(hipGetLastError()); finish();
        check(hipMemcpy(result.data(), output.data, result.size() * 4u, hipMemcpyDeviceToHost));
        for (unsigned i = 0u; i < guard; ++i)
            if (bits(result[i]) != 0xa5a5a5a5u || bits(result[result.size() - 1u - i]) != 0xa5a5a5a5u)
                throw std::runtime_error("bias output guard changed");
        for (unsigned family = 0u; family < families; ++family) {
            unsigned mismatches = 0u, fractional = 0u, first = cells;
            double maximum_error = 0.0;
            const unsigned begin = family * cases * permutations * 256u, end = begin + cases * permutations * 256u;
            for (unsigned i = begin; i < end; ++i) {
                const float value = result[guard + i];
                if (!std::isfinite(value)) throw std::runtime_error("nonfinite biased integer dot");
                const int64_t expected_with_bias = input.expected[i] + bias;
                if (int64_t(float(expected_with_bias)) != expected_with_bias) throw std::runtime_error("biased oracle loses integer precision");
                // FP64 subtraction is exact for these small FP32 values. No
                // nearest-integer conversion participates in this comparison.
                const double restored = double(value) - double(bias);
                const bool different = restored != double(input.expected[i]);
                mismatches += unsigned(different); fractional += unsigned(std::trunc(restored) != restored);
                maximum_error = std::max(maximum_error, std::fabs(restored - double(input.expected[i])));
                if (different && first == cells) first = i;
            }
            std::printf("{\"kind\":\"wmma_integer_bias_diagnostic\",\"dtype\":\"%s\",\"bias\":%d,\"family\":%u,\"cells\":%u,\"integer_value_mismatches\":%u,\"fractional_results\":%u,\"maximum_absolute_error\":%.12g,\"redzones_pass\":true,\"integer_rounding_used\":false,\"product_dispatch_changed\":false,\"inference_acceptance\":false}\n", bf16 ? "bf16" : "fp16", bias, family, end - begin, mismatches, fractional, maximum_error);
            if (first < cells) std::printf("{\"kind\":\"wmma_integer_bias_counterexample\",\"dtype\":\"%s\",\"bias\":%d,\"family\":%u,\"cell\":%u,\"expected_integer\":%lld,\"biased_result_bits\":%u,\"restored\":%.12g}\n", bf16 ? "bf16" : "fp16", bias, family, first, static_cast<long long>(input.expected[first]), bits(result[guard + first]), double(result[guard + first]) - double(bias));
        }
        std::fflush(stdout);
    }
    std::vector<uint16_t> copy(input.left.size());
    check(hipMemcpy(copy.data(), left.data, copy.size() * 2u, hipMemcpyDeviceToHost));
    if (copy != input.left) throw std::runtime_error("bias left input or guard changed");
    check(hipMemcpy(copy.data(), right.data, copy.size() * 2u, hipMemcpyDeviceToHost));
    if (copy != input.right) throw std::runtime_error("bias right input or guard changed");
    std::printf("{\"kind\":\"wmma_integer_probe_inputs\",\"dtype\":\"%s\",\"immutable_inputs\":true,\"redzones_pass\":true}\n", bf16 ? "bf16" : "fp16");
}
#endif
}
int main(int argc, char** argv) try {
#if defined(__HIPCC__)
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0));
    if (std::strncmp(device.gcnArchName, "gfx1151", 7u)) throw std::runtime_error("requires gfx1151");
    if (argc == 2 && std::strcmp(argv[1], "--bias") == 0) { run_bias(true); run_bias(false); }
    else if (argc == 1) { run(true); run(false); }
    else throw std::runtime_error("usage: wmma probe [--bias]");
#else
    (void)argc; (void)argv;
    const Inputs bf16(true), fp16(false);
    if (bf16.expected != fp16.expected) throw std::runtime_error("dtype oracle changed");
    for (unsigned i = 0u; i < cells; ++i)
        if (bf16.expected[i] != bf16.expected[(i / 1024u) * 1024u + i % 256u]) throw std::runtime_error("permutation changed oracle");
    std::printf("{\"kind\":\"wmma_integer_probe_host_inputs\",\"unique_integer_dots\":%u,\"permutations\":4,\"encoded_cells_per_dtype\":%u,\"native_wmma_checked\":false}\n", cells / permutations, cells * 2u);
#endif
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
