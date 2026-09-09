#ifndef QRT_FLA_BLACKWELL_ACCUMULATOR_H
#define QRT_FLA_BLACKWELL_ACCUMULATOR_H

#include "../moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"

// Reuse the characterized Blackwell K16/26-bit accumulator. Each 16-lane
// subgroup owns one dot cell; lane zero carries its K16 groups without
// splitting the K128 reduction into separately rounded K64 partials.
namespace qrt_fla_blackwell {
constexpr unsigned int kGroup = 16;
constexpr int16_t kZeroExponent = -133;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kChunk = 64;

__device__ __forceinline__ qrt_q1_moe_hawkeye::Value normalize(
    int64_t sum, int maximum_exponent
) {
    const bool negative = sum < 0;
    uint64_t magnitude = negative ? static_cast<uint64_t>(-sum) : static_cast<uint64_t>(sum);
    const unsigned int width = qrt_q1_moe_hawkeye::bit_width_u64(magnitude);
    if (!width) return {0u, kZeroExponent, negative};
    int exponent = maximum_exponent + static_cast<int>(width) - 26;
    magnitude = width > 26 ? magnitude >> (width - 26) : magnitude << (26 - width);
    if (exponent < -126) {
        const unsigned int shift = static_cast<unsigned int>(-126 - exponent);
        magnitude = shift >= 64 ? 0u : magnitude >> shift;
        exponent = -126;
    }
    magnitude >>= 2;
    return magnitude ? qrt_q1_moe_hawkeye::Value{static_cast<uint32_t>(magnitude), static_cast<int16_t>(exponent), negative}
                     : qrt_q1_moe_hawkeye::Value{0u, kZeroExponent, negative};
}

__device__ __forceinline__ qrt_q1_moe_hawkeye::Value accumulate(
    qrt_q1_moe_hawkeye::Value accumulator, uint16_t left, uint16_t right,
    unsigned int lane
) {
    const auto product = qrt_q1_moe_hawkeye::multiply_bf16(left, right, kZeroExponent);
    const auto significand = __shfl(accumulator.significand, 0, kGroup);
    const int exponent = __shfl(static_cast<int>(accumulator.exponent), 0, kGroup);
    const int negative = __shfl(static_cast<int>(accumulator.negative), 0, kGroup);
    int maximum = static_cast<int>(product.exponent) > exponent ? product.exponent : exponent;
    for (unsigned int delta = 8; delta; delta >>= 1) {
        const int other = __shfl_xor(maximum, delta, kGroup);
        maximum = other > maximum ? other : maximum;
    }
    const unsigned int shift = static_cast<unsigned int>(maximum - product.exponent);
    const uint64_t aligned = shift >= 32 ? 0u : (uint64_t(product.significand) << 2) >> shift;
    int64_t sum = product.negative ? -static_cast<int64_t>(aligned) : static_cast<int64_t>(aligned);
    if (lane == 0) {
        const unsigned int acc_shift = static_cast<unsigned int>(maximum - exponent);
        const uint64_t acc_aligned = acc_shift >= 32 ? 0u : (uint64_t(significand) << 2) >> acc_shift;
        sum += negative ? -static_cast<int64_t>(acc_aligned) : static_cast<int64_t>(acc_aligned);
    }
    for (unsigned int delta = 8; delta; delta >>= 1) sum += __shfl_down(sum, delta, kGroup);
    if (lane == 0) accumulator = normalize(sum, maximum);
    return accumulator;
}

__device__ __forceinline__ float from_bf16(uint16_t bits) {
    union { uint32_t u; float f; } value;
    value.u = uint32_t(bits) << 16;
    return value.f;
}

__device__ __forceinline__ uint16_t to_bf16(float input) {
    union { float f; uint32_t u; } value;
    value.f = input;
    if ((value.u & 0x7fffffffU) > 0x7f800000U) return uint16_t((value.u | 0x00400000U) >> 16);
    return uint16_t((value.u + 0x7fffU + ((value.u >> 16) & 1U)) >> 16);
}


}  // namespace qrt_fla_blackwell
#endif
