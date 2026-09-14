#pragma once
#include "sm121_integer_core.h"
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_INTEGER_INTERVAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_INTEGER_INTERVAL_INLINE inline
#endif

// Component certificate. Exact signed matrix products may still determine the
// original normalized result when per-product alignment discards low bits.
// Unproved endpoints retain the original K16 fallback; no empirical tolerance.
namespace qrt_sm121_integer_interval {
using Value = qrt_q1_moe_hawkeye::Value;
using Row = qrt_sm121_integer_core::Row;
struct Bounds { int64_t lower, upper; int maximum; unsigned discarded_pairs; };

QRT_INTEGER_INTERVAL_INLINE int64_t floor_shift(int64_t value, unsigned shift) {
    const uint64_t magnitude = uint64_t(value < 0 ? -value : value);
    const uint64_t whole = magnitude >> shift, remainder = magnitude & ((uint64_t(1) << shift) - 1u);
    return value < 0 ? -int64_t(whole) - int64_t(remainder != 0u) : int64_t(whole);
}
QRT_INTEGER_INTERVAL_INLINE int64_t ceil_shift(int64_t value, unsigned shift) {
    return -floor_shift(-value, shift);
}
QRT_INTEGER_INTERVAL_INLINE bool bounds(Value carry, const Row& left, const Row& right,
    int64_t mathematical, Bounds* output) {
    if (left.unit < 0 || right.unit < 0 ||
        ((left.exceptions | right.exceptions) & left.nonzero & right.nonzero)) return false;
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    if (left.maximum + right.maximum - 254 > maximum)
        maximum = qrt_sm121_integer_core::paired_maximum(left, right, maximum);
    const int shift = maximum - (left.unit + right.unit - 254) - 11;
    int64_t lower = 0, upper = 0;
    unsigned discarded = 0u;
    if (shift <= 0) {
        if (shift < -25) { if (mathematical) return false; }
        else lower = upper = mathematical * (int64_t(1) << (-shift));
    } else if (shift < 30) {
        const unsigned s = unsigned(shift);
        const uint32_t loss = qrt_sm121_integer_core::remainder_mask(left, right, s);
        uint32_t negative = 0u;
        for (unsigned word = 0u; word < 4u; ++word) {
            const uint32_t signs = (uint32_t(left.high[word]) ^ uint32_t(right.high[word])) & 0x80808080u;
            const uint32_t nibble = (((signs >> 7u) * 0x01020408u) >> 24u) & 15u;
            negative |= nibble << (word * 4u);
        }
        const unsigned neg = unsigned(__builtin_popcount(loss & negative));
        const unsigned pos = unsigned(__builtin_popcount(loss & ~negative));
        discarded = neg + pos;
        const int64_t largest_remainder = (int64_t(1) << s) - 1u;
        // Every marked absolute remainder is in [1, 2^s-1]. Their signs are
        // known from encoded operands. The compensated sum is an integer, so
        // directed division tightens the inclusive interval without reading
        // or multiplying any individual operand pair.
        const int64_t remainder_low = int64_t(pos) - int64_t(neg) * largest_remainder;
        const int64_t remainder_high = int64_t(pos) * largest_remainder - int64_t(neg);
        lower = ceil_shift(mathematical - remainder_high, s);
        upper = floor_shift(mathematical - remainder_low, s);
    }
    // At shifts >=30 every signed16 core product is individually zero.
    const unsigned carry_shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    const int64_t signed_carry = carry.negative ? -int64_t(aligned) : int64_t(aligned);
    lower += signed_carry; upper += signed_carry;
    if (lower > upper || lower < -int64_t(UINT32_MAX) || upper > int64_t(UINT32_MAX)) return false;
    *output = {lower, upper, maximum, discarded};
    return true;
}
QRT_INTEGER_INTERVAL_INLINE Value normalize(int64_t sum, int maximum) {
    return qrt_sm121_canonical::normalize(uint32_t(sum < 0 ? -sum : sum), sum < 0, maximum);
}
QRT_INTEGER_INTERVAL_INLINE bool same(Value a, Value b) {
    return a.significand == b.significand && a.exponent == b.exponent && a.negative == b.negative;
}
QRT_INTEGER_INTERVAL_INLINE bool accumulate(Value carry, const Row& left, const Row& right,
    int64_t mathematical, Value* output) {
    Bounds interval;
    if (!bounds(carry, left, right, mathematical, &interval)) return false;
    const Value low = normalize(interval.lower, interval.maximum), high = normalize(interval.upper, interval.maximum);
    if (!same(low, high)) return false;
    *output = low;
    return true;
}
} // namespace qrt_sm121_integer_interval
#undef QRT_INTEGER_INTERVAL_INLINE
