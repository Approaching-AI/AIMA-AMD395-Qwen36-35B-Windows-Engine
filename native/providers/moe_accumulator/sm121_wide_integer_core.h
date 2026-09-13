#ifndef QRT_SM121_WIDE_INTEGER_CORE_H
#define QRT_SM121_WIDE_INTEGER_CORE_H
#include "sm121_integer_core.h"
#include "sm121_positive_karatsuba.h"
#if defined(__HIPCC__)
#define QRT_WIDE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_WIDE_INLINE inline
#endif

namespace qrt_sm121_wide_core {
// An 18-bit signed core covers two more exponent bits. Two positive FP16
// digits replace packed IU8 digits; redundant exponent bytes are read from
// original BF16 pairs only when needed. The odd 33-dword row is smaller than
// the previous 148-byte prepared three-part experiment.
struct Row {
    uint16_t original[18], high[16], low[16];
    uint32_t trailing[4];
    int unit, maximum;
    uint32_t exceptions, nonzero;
};
static_assert(sizeof(Row) == 132u);
constexpr uint32_t mask = 0x3ffffu, center = 0x20000u;

QRT_WIDE_INLINE uint32_t encode(uint16_t value, int unit) {
    if (unit < 0 || !(value & 0x7fffu)) return 0u;
    const int shift = int((value >> 7u) & 255u) - unit;
    const uint32_t significand = 128u | (value & 127u);
    const uint32_t magnitude = shift <= -8 ? 0u : shift < 0 ? significand >> (-shift) : significand << shift;
    return ((value & 0x8000u) ? 0u - magnitude : magnitude) & mask;
}
QRT_WIDE_INLINE int signed_core(uint16_t value, int unit) {
    const uint32_t core = encode(value, unit);
    return core & center ? int(core) - int(mask + 1u) : int(core);
}
QRT_WIDE_INLINE uint32_t unsigned_row_sum(const Row& row) {
    return uint32_t(row.original[16]) | (uint32_t(row.original[17]) << 16u);
}
QRT_WIDE_INLINE void prepare(Row& row) {
    int maximum = 0; bool valid = true;
    row.nonzero = row.exceptions = 0u;
    for (unsigned i = 0u; i < 16u; ++i) {
        if (!(row.original[i] & 0x7fffu)) continue;
        row.nonzero |= 1u << i;
        const int exponent = (row.original[i] >> 7u) & 255u;
        valid = valid && exponent != 0 && exponent != 255;
        maximum = exponent > maximum ? exponent : maximum;
    }
    row.maximum = row.nonzero ? maximum : 127;
    row.unit = !valid ? -1 : !row.nonzero ? 127 : maximum > 10 ? maximum - 9 : 1;
    uint32_t total = 0u;
    for (unsigned word = 0u; word < 4u; ++word) {
        uint32_t trailing = 0u;
        for (unsigned byte = 0u; byte < 4u; ++byte) {
            const unsigned i = word * 4u + byte;
            const uint16_t value = row.original[i];
            const uint32_t core = encode(value, row.unit), positive = core ^ center;
            const int shift = int((value >> 7u) & 255u) - row.unit;
            const unsigned significand = 128u | (value & 127u);
            if ((value & 0x7fffu) && row.unit >= 0 && shift < 0 &&
                (-shift >= 8 || (significand & ((1u << (-shift)) - 1u)))) row.exceptions |= 1u << i;
            row.high[i] = qrt_sm121_positive_karatsuba::positive_half_bits(positive >> 9u);
            row.low[i] = qrt_sm121_positive_karatsuba::positive_half_bits(positive & 511u);
            total += positive;
            trailing |= qrt_sm121_integer_parts::trailing_bits(core) << (byte * 8u);
        }
        row.trailing[word] = trailing;
    }
    row.original[16] = uint16_t(total); row.original[17] = uint16_t(total >> 16u);
}

// All three matrix partials are nonnegative integers. Their largest possible
// absolute sum is 16*1022^2=16711744, below 2^24. Centering is undone in int64.
QRT_WIDE_INLINE int64_t reconstruct(int high, int low, int combined,
                                    uint32_t left_sum, uint32_t right_sum) {
    const int64_t unsigned_product = int64_t(high) * 262144 +
        (int64_t(combined) - high - low) * 512 + low;
    return unsigned_product - int64_t(left_sum + right_sum) * center + int64_t(16) * center * center;
}

QRT_WIDE_INLINE int paired_maximum(const Row& left, const Row& right, int maximum) {
    const uint32_t nonzero = left.nonzero & right.nonzero;
    for (unsigned pair = 0u; pair < 8u; ++pair) {
        const uint32_t a = uint32_t(left.original[2u * pair]) | (uint32_t(left.original[2u * pair + 1u]) << 16u);
        const uint32_t b = uint32_t(right.original[2u * pair]) | (uint32_t(right.original[2u * pair + 1u]) << 16u);
        const uint32_t sum = ((a >> 7u) & 0x00ff00ffu) + ((b >> 7u) & 0x00ff00ffu);
        const int first = (nonzero & (1u << (2u * pair))) ? int(sum & 65535u) - 254 : -133;
        const int second = (nonzero & (2u << (2u * pair))) ? int(sum >> 16u) - 254 : -133;
        maximum = first > maximum ? first : maximum;
        maximum = second > maximum ? second : maximum;
    }
    return maximum;
}
QRT_WIDE_INLINE uint32_t remainder_mask(const Row& left, const Row& right, unsigned shift) {
    if (!shift || shift >= 34u) return 0u;
    const uint32_t threshold = shift * 0x01010101u;
    uint32_t bits = 0u;
    for (unsigned word = 0u; word < 4u; ++word) {
        const uint32_t high = ~(left.trailing[word] + right.trailing[word] + 0x80808080u - threshold) & 0x80808080u;
        bits |= ((((high >> 7u) * 0x01020408u) >> 24u) & 15u) << (word * 4u);
    }
    return bits;
}

QRT_WIDE_INLINE bool sum_integer_product(qrt_q1_moe_hawkeye::Value carry,
    const Row& left, const Row& right, int64_t mathematical,
    qrt_sm121_group16::AlignedSum* output, unsigned* replayed_pairs = nullptr) {
    if (left.unit < 0 || right.unit < 0) return false;
    const uint32_t exceptions = (left.exceptions | right.exceptions) & left.nonzero & right.nonzero;
    if (!exceptions && qrt_sm121_integer_parts::sum_exact_integer_product(carry, mathematical,
        left.unit, left.maximum, right.unit, right.maximum, output, left.trailing, right.trailing)) {
        if (replayed_pairs) *replayed_pairs = 0u;
        return true;
    }
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    if (left.maximum + right.maximum - 254 > maximum) maximum = paired_maximum(left, right, maximum);
    const int shift = maximum - (left.unit + right.unit - 254) - 11;
    // Any nonzero encoded pair implies both original exponents are at least
    // unit-7, hence shift>=-25. Below that bound all core pairs are zero;
    // original BF16 exception terms can still reconstruct the group safely.
    if (shift < -25 && mathematical) return false;
    const uint32_t loss = shift > 0 ? remainder_mask(left, right, unsigned(shift)) : 0u;
    uint32_t pending = loss | exceptions; unsigned replays = 0u;
    int64_t discarded = 0, corrections = 0;
    while (pending) {
        const unsigned i = qrt_sm121_integer_parts::trailing_bits(pending);
        pending &= pending - 1u; ++replays;
        const uint16_t a = left.original[i], b = right.original[i];
        const int ac = signed_core(a, left.unit), bc = signed_core(b, right.unit);
        const uint64_t magnitude = uint64_t(ac < 0 ? -ac : ac) * uint64_t(bc < 0 ? -bc : bc);
        const bool negative = ((a ^ b) & 0x8000u) != 0u;
        if (loss & (1u << i)) {
            const uint64_t remainder = magnitude & ((uint64_t(1) << shift) - 1u);
            discarded += negative ? -int64_t(remainder) : int64_t(remainder);
        }
        if (exceptions & (1u << i)) {
            const auto original = qrt_q1_moe_hawkeye::multiply_bf16(a, b, -133);
            const unsigned original_shift = unsigned(maximum - original.exponent);
            const uint32_t original_aligned = original_shift >= 32u ? 0u : (original.significand << 2u) >> original_shift;
            const uint32_t core_aligned = !magnitude || shift >= 34 ? 0u : shift > 0 ?
                uint32_t(magnitude >> shift) : uint32_t(magnitude << (-shift));
            const int64_t difference = int64_t(original_aligned) - core_aligned;
            corrections += negative ? -difference : difference;
        }
    }
    const int64_t compensated = mathematical - discarded;
    const int64_t products = !compensated || shift >= 34 ? 0 : shift <= 0 ?
        compensated * (int64_t(1) << (-shift)) : compensated < 0 ?
        -int64_t(uint64_t(-compensated) >> shift) : int64_t(uint64_t(compensated) >> shift);
    const unsigned carry_shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    const int64_t total = products + corrections + (carry.negative ? -int64_t(aligned) : int64_t(aligned));
    *output = {{uint32_t(total < 0 ? -total : total), total < 0}, maximum};
    if (replayed_pairs) *replayed_pairs = replays;
    return true;
}

#if defined(__HIPCC__)
using F16x16 = _Float16 __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
struct Parts { F32x8 high, low, combined; };
__device__ __forceinline__ Parts products(const Row& left, const Row& right) {
    F16x16 ah{}, al{}, bh{}, bl{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        ah[i] = __builtin_bit_cast(_Float16, left.high[i]); al[i] = __builtin_bit_cast(_Float16, left.low[i]);
        bh[i] = __builtin_bit_cast(_Float16, right.high[i]); bl[i] = __builtin_bit_cast(_Float16, right.low[i]);
    }
    const F16x16 as = ah + al, bs = bh + bl;
    const F32x8 zero{};
    return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, zero),
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bl, zero),
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(as, bs, zero)};
}
#endif
}  // namespace qrt_sm121_wide_core
#undef QRT_WIDE_INLINE
#endif
