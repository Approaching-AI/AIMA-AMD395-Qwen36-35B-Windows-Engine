#ifndef QRT_SM121_INTEGER_CORE_H
#define QRT_SM121_INTEGER_CORE_H

#include "sm121_integer_parts.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_CORE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CORE_INLINE inline
#endif

namespace qrt_sm121_integer_core {

// Four IU8 matrix products cover the signed sixteen-bit cores. The original
// BF16 words and an exception mask retain every bit removed from those cores;
// this is an exact decomposition, not a change to the model representation.
struct Row {
    uint16_t original[18];
    int high[4], low[4];
    uint32_t trailing[4], exponents[4];
    int unit;
    uint32_t exceptions, nonzero;
    int maximum;
};
static_assert(sizeof(Row) == 116u);  // Odd dword stride in shared memory.

QRT_CORE_INLINE uint16_t encode(uint16_t value, int unit) {
    if (unit < 0 || !(value & 0x7fffu)) return 0u;
    const int shift = int((value >> 7u) & 255u) - unit;
    const uint32_t significand = 128u | (value & 127u);
    const uint32_t magnitude = shift <= -8 ? 0u : shift < 0
        ? significand >> (-shift) : significand << shift;
    return static_cast<uint16_t>((value & 0x8000u) ? 0u - magnitude : magnitude);
}

QRT_CORE_INLINE void prepare(Row& row) {
    int maximum = 0;
    bool valid = true;
    row.nonzero = row.exceptions = 0u;
    for (unsigned i = 0u; i < 16u; ++i) {
        if (!(row.original[i] & 0x7fffu)) continue;
        row.nonzero |= 1u << i;
        const int exponent = (row.original[i] >> 7u) & 255u;
        valid = valid && exponent != 0 && exponent != 255;
        maximum = exponent > maximum ? exponent : maximum;
    }
    row.maximum = row.nonzero ? maximum : 127;
    row.unit = !valid ? -1 : !row.nonzero ? 127 : maximum > 8 ? maximum - 7 : 1;
    for (unsigned word = 0u; word < 4u; ++word) {
        uint32_t high = 0u, low = 0u, trailing = 0u, exponents = 0u;
        for (unsigned byte = 0u; byte < 4u; ++byte) {
            const unsigned i = word * 4u + byte;
            const uint16_t value = row.original[i], core = encode(value, row.unit);
            const int exponent = (value >> 7u) & 255u;
            const int shift = exponent - row.unit;
            const unsigned significand = 128u | (value & 127u);
            if ((value & 0x7fffu) && row.unit >= 0 && shift < 0 &&
                (-shift >= 8 || (significand & ((1u << (-shift)) - 1u))))
                row.exceptions |= 1u << i;
            high |= uint32_t(core >> 8u) << (byte * 8u);
            low |= uint32_t(core & 255u) << (byte * 8u);
            trailing |= qrt_sm121_integer_parts::trailing_bits(core) << (byte * 8u);
            exponents |= uint32_t(exponent) << (byte * 8u);
        }
        row.high[word] = int(high); row.low[word] = int(low);
        row.trailing[word] = trailing; row.exponents[word] = exponents;
    }
}

// Byte sums cannot overflow: trailing counts are at most31, and shift<30.
// Gather one high bit per byte into a four-bit nibble. This identifies the
// individual core products that need truncation compensation, rather than
// falling back to all sixteen original products when just one loses bits.
QRT_CORE_INLINE uint32_t remainder_mask(const Row& left, const Row& right, unsigned shift) {
    if (!shift || shift >= 30u) return 0u;
    const uint32_t threshold = shift * 0x01010101u;
    uint32_t mask = 0u;
    for (unsigned word = 0u; word < 4u; ++word) {
        const uint32_t high = ~(left.trailing[word] + right.trailing[word] +
            0x80808080u - threshold) & 0x80808080u;
        const uint32_t nibble = (((high >> 7u) * 0x01020408u) >> 24u) & 15u;
        mask |= nibble << (word * 4u);
    }
    return mask;
}

// Two independent sixteen-bit additions sum four pairs of byte exponents.
// Nonzero masks suppress a zero operand's arbitrary exponent. The exact
// paired maximum is required whenever alignment discards product bits.
QRT_CORE_INLINE int paired_maximum(const Row& left, const Row& right, int maximum) {
    const uint32_t nonzero = left.nonzero & right.nonzero;
    for (unsigned word = 0u; word < 4u; ++word) {
        const uint32_t a = left.exponents[word], b = right.exponents[word];
        const uint32_t even = (a & 0x00ff00ffu) + (b & 0x00ff00ffu);
        const uint32_t odd = ((a >> 8u) & 0x00ff00ffu) + ((b >> 8u) & 0x00ff00ffu);
        const uint32_t active = (nonzero >> (word * 4u)) & 15u;
        const int e0 = active & 1u ? int(even & 65535u) - 254 : -133;
        const int e1 = active & 2u ? int(odd & 65535u) - 254 : -133;
        const int e2 = active & 4u ? int(even >> 16u) - 254 : -133;
        const int e3 = active & 8u ? int(odd >> 16u) - 254 : -133;
        maximum = e0 > maximum ? e0 : maximum;
        maximum = e1 > maximum ? e1 : maximum;
        maximum = e2 > maximum ? e2 : maximum;
        maximum = e3 > maximum ? e3 : maximum;
    }
    return maximum;
}

QRT_CORE_INLINE unsigned first_bit(uint32_t bits) {
    return qrt_sm121_integer_parts::trailing_bits(bits);
}

QRT_CORE_INLINE int signed_core(uint16_t value, int unit) {
    const uint16_t encoded = encode(value, unit);
    return encoded & 0x8000u ? int(encoded) - 65536 : int(encoded);
}

QRT_CORE_INLINE bool sum(qrt_q1_moe_hawkeye::Value carry,
    const Row& left, const Row& right, const int32_t (&partials)[4],
    qrt_sm121_group16::AlignedSum* output, unsigned* replayed_pairs = nullptr) {
    if (left.unit < 0 || right.unit < 0) return false;
    const uint32_t exceptions = (left.exceptions | right.exceptions) & left.nonzero & right.nonzero;
    if (!exceptions && qrt_sm121_integer_parts::sum_exact_range(carry, partials,
            left.unit, left.maximum, right.unit, right.maximum, output,
            left.trailing, right.trailing)) {
        if (replayed_pairs) *replayed_pairs = 0u;
        return true;
    }
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    if (left.maximum + right.maximum - 254 > maximum)
        maximum = paired_maximum(left, right, maximum);
    const int shift = maximum - (left.unit + right.unit - 254) - 11;
    const int64_t mathematical = int64_t(partials[0]) * 65536 +
        (int64_t(partials[1]) + partials[2]) * 256 + partials[3];
    // A nonzero encoded product has at most thirty bits. If it survives,
    // its original exponent also bounds any required left shift by25.
    if (shift < -25 && mathematical) return false;
    const uint32_t loss = shift > 0 ? remainder_mask(left, right, unsigned(shift)) : 0u;
    uint32_t pending = loss | exceptions;
    int64_t discarded = 0, corrections = 0;
    unsigned replays = 0u;
    while (pending) {
        const unsigned i = first_bit(pending);
        pending &= pending - 1u;
        ++replays;
        const uint16_t a = left.original[i], b = right.original[i];
        const int ac = signed_core(a, left.unit), bc = signed_core(b, right.unit);
        const uint32_t magnitude = uint32_t(ac < 0 ? -ac : ac) * uint32_t(bc < 0 ? -bc : bc);
        const bool negative = ((a ^ b) & 0x8000u) != 0u;
        if (loss & (1u << i)) {
            const uint32_t remainder = magnitude & ((1u << shift) - 1u);
            discarded += negative ? -int64_t(remainder) : int64_t(remainder);
        }
        if (exceptions & (1u << i)) {
            const auto original = qrt_q1_moe_hawkeye::multiply_bf16(a, b, -133);
            const unsigned original_shift = unsigned(maximum - original.exponent);
            const uint32_t original_aligned = original_shift >= 32u ? 0u :
                (original.significand << 2u) >> original_shift;
            const uint32_t core_aligned = !magnitude ? 0u : shift >= 30 ? 0u : shift > 0
                ? magnitude >> shift : uint32_t(uint64_t(magnitude) << (-shift));
            const int64_t difference = int64_t(original_aligned) - core_aligned;
            corrections += negative ? -difference : difference;
        }
    }
    const int64_t compensated = mathematical - discarded;
    const int64_t products = !compensated || shift >= 30 ? 0 : shift <= 0
        ? compensated * (int64_t(1) << (-shift))
        : compensated < 0 ? -int64_t(uint64_t(-compensated) >> shift)
                          : int64_t(uint64_t(compensated) >> shift);
    const unsigned carry_shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned_carry = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    const int64_t total = products + corrections +
        (carry.negative ? -int64_t(aligned_carry) : int64_t(aligned_carry));
    *output = {{uint32_t(total < 0 ? -total : total), total < 0}, maximum};
    if (replayed_pairs) *replayed_pairs = replays;
    return true;
}

}  // namespace qrt_sm121_integer_core
#undef QRT_CORE_INLINE
#endif
