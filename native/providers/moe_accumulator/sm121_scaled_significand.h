#ifndef QRT_SM121_SCALED_SIGNIFICAND_H
#define QRT_SM121_SCALED_SIGNIFICAND_H
#include "sm121_float_alignment.h"

#if defined(__HIPCC__)
#define QRT_SCALED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SCALED_INLINE inline
#endif

namespace qrt_sm121_scaled_significand {
using Value = qrt_q1_moe_hawkeye::Value;
struct Pair { uint32_t left, right; int exponent; };

QRT_SCALED_INLINE bool eligible(uint16_t word) {
    const unsigned exponent = (word >> 7u) & 255u;
    return !(word & 0x7fffu) || (exponent != 0u && exponent != 255u);
}

// Decode only the original normal BF16 significands and exponent sum. The
// actual product may be far below/above the normal FP32 range; it is never
// materialized at that scale. Subnormal/special source words use the caller's
// original integer fallback. Signed zeros contribute an ordinary zero term.
QRT_SCALED_INLINE Pair prepare(uint16_t left, uint16_t right) {
    const bool zero = !(left & 0x7fffu) || !(right & 0x7fffu);
    return {zero ? 0u : ((uint32_t(left & 0x807fu) << 16u) | 0x3f800000u),
        uint32_t(right & 0x807fu) << 16u,
        zero ? -133 : int((left >> 7u) & 255u) + int((right >> 7u) & 255u) - 254};
}

// Original aligned magnitude is (sigA*sigB <<11) >> delta. At delta>=27
// every possible product is zero. Otherwise make FP32 operands with exponents
// 127 and152-delta: their product is precisely sigA*sigB*2^(11-delta).
// It has at most16 significant bits, is finite/normal (or zero), and fits
// signed32. One scalar multiplication and RTZ conversion therefore reproduce
// the integer alignment even for original BF16 products outside FP32 range.
QRT_SCALED_INLINE uint32_t aligned(Pair pair, int maximum) {
    const unsigned delta = unsigned(maximum - pair.exponent);
    if (delta >= 27u) return 0u;
    const float left = qrt_sm121_float_alignment::from_bits(pair.left);
    const float right = qrt_sm121_float_alignment::from_bits(pair.right | ((152u - delta) << 23u));
    return uint32_t(int32_t(left * right));
}

QRT_SCALED_INLINE bool sum(Value carry, const uint16_t* left, const uint16_t* right,
    qrt_sm121_group16::AlignedSum* output) {
    Pair pairs[16]; int maximum = carry.exponent > -133 ? carry.exponent : -133;
    for (unsigned i = 0u; i < 16u; ++i) {
        if (!eligible(left[i]) || !eligible(right[i])) return false;
        pairs[i] = prepare(left[i], right[i]);
        maximum = pairs[i].exponent > maximum ? pairs[i].exponent : maximum;
    }
    uint32_t modulo = 0u;
    for (unsigned i = 0u; i < 16u; ++i) modulo += aligned(pairs[i], maximum);
    const unsigned shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned_carry = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
    modulo += carry.negative ? 0u - aligned_carry : aligned_carry;
    *output = {qrt_sm121_group16::decode_modulo_sum(modulo, ((left[0] ^ right[0]) & 0x8000u) != 0u), maximum};
    return true;
}
} // namespace qrt_sm121_scaled_significand
#undef QRT_SCALED_INLINE
#endif
