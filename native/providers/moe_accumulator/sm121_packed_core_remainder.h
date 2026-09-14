#pragma once
#include "sm121_integer_core.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_CORE_REMAINDER_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CORE_REMAINDER_INLINE inline
#endif

// Exact compensation of integer matrix products. The original BF16 words
// remain available for the few signed16 encoding exceptions. No numerical
// certificate or output tolerance is used by the packed compensation path.
namespace qrt_sm121_core_remainder {
using Row = qrt_sm121_integer_core::Row;
using Value = qrt_q1_moe_hawkeye::Value;
using AlignedSum = qrt_sm121_group16::AlignedSum;
struct Prepared { uint32_t magnitudes[8]; uint32_t negative; };

QRT_CORE_REMAINDER_INLINE Prepared prepare(const Row& row) {
    Prepared result{};
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t core = qrt_sm121_integer_core::encode(row.original[i], row.unit);
        const uint16_t magnitude = core & 0x8000u ? uint16_t(0u - core) : core;
        result.magnitudes[i / 2u] |= uint32_t(magnitude) << (i % 2u * 16u);
        result.negative |= uint32_t((row.original[i] & 0x8000u) != 0u) << i;
    }
    return result;
}

QRT_CORE_REMAINDER_INLINE uint32_t multiply_low16(uint32_t left, uint32_t right) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    uint32_t result;
    asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(result) : "v"(left), "v"(right));
    return result;
#else
    return (((left & 65535u) * (right & 65535u)) & 65535u) |
        (((left >> 16u) * (right >> 16u)) << 16u);
#endif
}

QRT_CORE_REMAINDER_INLINE int32_t remainder_pair(const Prepared& left,
    const Prepared& right, unsigned even_index, uint32_t mask) {
    const uint32_t products = multiply_low16(left.magnitudes[even_index / 2u], right.magnitudes[even_index / 2u]);
    const unsigned signs = (left.negative ^ right.negative) >> even_index;
    const int32_t low = int32_t(products & mask), high = int32_t((products >> 16u) & mask);
    return ((signs & 1u) ? -low : low) + ((signs & 2u) ? -high : high);
}

template<bool Sparse, bool FastFirst>
QRT_CORE_REMAINDER_INLINE bool sum(Value carry, const Row& left, const Row& right,
    const Prepared& left_prepared, const Prepared& right_prepared, int64_t mathematical,
    AlignedSum* output, unsigned* path = nullptr) {
    if (path) *path = 0u;
    if (left.unit < 0 || right.unit < 0) return false;
    const uint32_t exceptions = (left.exceptions | right.exceptions) & left.nonzero & right.nonzero;
    if constexpr (FastFirst) {
        if (!exceptions && qrt_sm121_integer_parts::sum_exact_integer_product(carry, mathematical,
            left.unit, left.maximum, right.unit, right.maximum, output, left.trailing, right.trailing)) {
            if (path) *path = 1u;
            return true;
        }
    }
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    if (left.maximum + right.maximum - 254 > maximum)
        maximum = qrt_sm121_integer_core::paired_maximum(left, right, maximum);
    const int shift = maximum - (left.unit + right.unit - 254) - 11;
    if (shift < -25 || shift > 16) {
        const bool supported = qrt_sm121_integer_core::sum_integer_product(carry,left,right,mathematical,output);
        if (path && supported) *path = 3u;
        return supported;
    }
    int32_t discarded = 0;
    if (shift > 0) {
        const uint32_t mask = (1u << shift) - 1u;
        if constexpr (Sparse) {
            uint32_t pending = qrt_sm121_integer_core::remainder_mask(left,right,unsigned(shift));
            while (pending) {
                const unsigned index = qrt_sm121_integer_core::first_bit(pending) & ~1u;
                pending &= ~(3u << index);
                discarded += remainder_pair(left_prepared,right_prepared,index,mask);
            }
        } else {
#pragma unroll
            for (unsigned index = 0u; index < 16u; index += 2u)
                discarded += remainder_pair(left_prepared,right_prepared,index,mask);
        }
    }
    // Signed low-bit remainders are exact through shift16, so this dividend
    // is divisible by 2^shift. Its sign-magnitude shift is the original
    // toward-zero alignment, without a hardware integer division.
    const int64_t compensated = mathematical - discarded;
    const int64_t products = shift <= 0 ? compensated * (int64_t(1) << (-shift)) :
        compensated < 0 ? -int64_t(uint64_t(-compensated) >> shift) : int64_t(uint64_t(compensated) >> shift);
    int64_t corrections = 0;
    uint32_t pending = exceptions;
    while (pending) {
        const unsigned index = qrt_sm121_integer_core::first_bit(pending);
        pending &= pending - 1u;
        const uint16_t a = left.original[index], b = right.original[index];
        const auto original = qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
        const unsigned original_shift = unsigned(maximum - original.exponent);
        const uint32_t original_aligned = original_shift >= 32u ? 0u : (original.significand << 2u) >> original_shift;
        const unsigned half = (index & 1u) * 16u;
        const uint32_t ac = (left_prepared.magnitudes[index / 2u] >> half) & 65535u;
        const uint32_t bc = (right_prepared.magnitudes[index / 2u] >> half) & 65535u;
        const uint64_t magnitude = uint64_t(ac) * bc;
        const uint64_t core_aligned = shift <= 0 ? magnitude << (-shift) : magnitude >> shift;
        const int64_t difference = int64_t(original_aligned) - int64_t(core_aligned);
        corrections += ((a ^ b) & 0x8000u) ? -difference : difference;
    }
    const unsigned carry_shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned_carry = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    const int64_t total = products + corrections + (carry.negative ? -int64_t(aligned_carry) : int64_t(aligned_carry));
    if (total < -int64_t(UINT32_MAX) || total > int64_t(UINT32_MAX)) return false;
    *output = {{uint32_t(total < 0 ? -total : total),total < 0},maximum};
    if (path) *path = 2u;
    return true;
}
} // namespace qrt_sm121_core_remainder
#undef QRT_CORE_REMAINDER_INLINE
