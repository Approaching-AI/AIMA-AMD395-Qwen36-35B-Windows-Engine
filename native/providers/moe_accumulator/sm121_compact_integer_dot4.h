#pragma once
#include "sm121_scalar_integer_core.h"
#include "sm121_prepared_integer_pairs.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_COMPACT_DOT4_INLINE __host__ __device__ __forceinline__
#else
#define QRT_COMPACT_DOT4_INLINE inline
#endif

// Isolated lossless encoding. The older scalar-dot4 row uses156 bytes of
// original words, byte digits, exponents, trailing counts and remainder data.
// Here signed16 coefficients reconstruct every supported BF16 word. One
// unsupported word makes the entire row retain its original32 bytes instead.
namespace qrt_sm121_compact_integer_dot4 {
using Value = qrt_q1_moe_hawkeye::Value;
struct Row {
    uint32_t pairs[8];
    uint32_t exponents[4];
    uint32_t control;
};
static_assert(sizeof(Row) == 52u); // Odd thirteen-dword LDS stride.

QRT_COMPACT_DOT4_INLINE unsigned unit(const Row& row) { return row.control & 255u; }
QRT_COMPACT_DOT4_INLINE unsigned nonzero(const Row& row) { return row.control >> 16u; }
QRT_COMPACT_DOT4_INLINE uint16_t word(const Row& row, unsigned i) {
    return uint16_t(row.pairs[i / 2u] >> ((i & 1u) * 16u));
}
QRT_COMPACT_DOT4_INLINE uint16_t original(const Row& row, unsigned i) {
    const uint16_t bits = word(row, i);
    if (!unit(row)) return bits;
    const bool negative = (bits & 0x8000u) != 0u;
    const unsigned magnitude = negative ? unsigned(uint16_t(0u - bits)) : bits;
    if (!magnitude) return 0u;
    const unsigned exponent = (row.exponents[i / 4u] >> ((i & 3u) * 8u)) & 255u;
    const int shift = int(exponent) - int(unit(row));
    const unsigned significand = shift >= 0 ? magnitude >> unsigned(shift)
                                          : magnitude << unsigned(-shift);
    return uint16_t((negative ? 0x8000u : 0u) | (exponent << 7u) | (significand & 127u));
}
QRT_COMPACT_DOT4_INLINE Row prepare(const uint16_t* input) {
    Row result{};
    unsigned maximum = 0u, live = 0u;
    bool valid = true;
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t value = input[i];
        const unsigned exponent = (value >> 7u) & 255u;
        if (!(value & 0x7fffu)) { valid &= value == 0u; continue; }
        valid &= exponent != 0u && exponent != 255u;
        maximum = maximum > exponent ? maximum : exponent;
        live |= 1u << i;
    }
    const unsigned scale = live && maximum > 7u ? maximum - 7u : 127u;
    valid &= !live || maximum > 7u;
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t value = input[i];
        unsigned coefficient = 0u;
        if (value & 0x7fffu) {
            const unsigned exponent = (value >> 7u) & 255u;
            const unsigned significand = 128u | (value & 127u);
            const int shift = int(exponent) - int(scale);
            if (shift < -7 || shift > 7) valid = false;
            else if (shift < 0) {
                valid &= (significand & ((1u << unsigned(-shift)) - 1u)) == 0u;
                coefficient = significand >> unsigned(-shift);
            } else coefficient = significand << unsigned(shift);
            if (value & 0x8000u) coefficient = uint16_t(0u - coefficient);
            result.exponents[i / 4u] |= exponent << ((i & 3u) * 8u);
        }
        result.pairs[i / 2u] |= coefficient << ((i & 1u) * 16u);
    }
    if (valid) result.control = scale | (maximum << 8u) | (live << 16u);
    else {
        result = Row{};
        for (unsigned i = 0u; i < 16u; ++i)
            result.pairs[i / 2u] |= uint32_t(input[i]) << ((i & 1u) * 16u);
    }
    return result;
}
QRT_COMPACT_DOT4_INLINE Value fallback(Value carry, const Row& a, const Row& b) {
    Value values[17]; values[0] = carry;
    for (unsigned i = 0u; i < 16u; ++i)
        values[i + 1u] = qrt_q1_moe_hawkeye::multiply_bf16(original(a, i), original(b, i), -133);
    return qrt_q1_moe_hawkeye::group_sum<26, -133>(values, 17u);
}
QRT_COMPACT_DOT4_INLINE int maximum(Value carry, const Row& a, const Row& b) {
    const int initial = carry.exponent > -133 ? carry.exponent : -133;
    const int upper = int((a.control >> 8u) & 255u) + int((b.control >> 8u) & 255u) - 254;
    if (initial >= upper) return initial;
    uint32_t result = unsigned(initial + 254) * 0x00010001u;
    const unsigned active = nonzero(a) & nonzero(b);
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) {
        const uint32_t x = a.exponents[i], y = b.exponents[i];
        const uint32_t even = (x & 0x00ff00ffu) + (y & 0x00ff00ffu);
        const uint32_t odd = ((x >> 8u) & 0x00ff00ffu) + ((y >> 8u) & 0x00ff00ffu);
        const unsigned mask = (active >> (4u * i)) & 15u;
        const uint32_t em = ((mask & 1u) ? 65535u : 0u) | ((mask & 4u) ? 0xffff0000u : 0u);
        const uint32_t om = ((mask & 2u) ? 65535u : 0u) | ((mask & 8u) ? 0xffff0000u : 0u);
        result = qrt_sm121_prepared_integer_pairs::maximum_pair(result, even & em);
        result = qrt_sm121_prepared_integer_pairs::maximum_pair(result, odd & om);
    }
    return int((result & 65535u) > (result >> 16u) ? result & 65535u : result >> 16u) - 254;
}
QRT_COMPACT_DOT4_INLINE int64_t product(const Row& a, const Row& b) {
    int32_t hh = 0, hl = 0, lh = 0, ll = 0;
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) {
        const uint32_t a0 = a.pairs[2u * i], a1 = a.pairs[2u * i + 1u];
        const uint32_t b0 = b.pairs[2u * i], b1 = b.pairs[2u * i + 1u];
        const uint32_t al = (a0 & 0x00ff00ffu) | ((a1 & 0x00ff00ffu) << 8u);
        const uint32_t ah = ((a0 >> 8u) & 0x00ff00ffu) | (a1 & 0xff00ff00u);
        const uint32_t bl = (b0 & 0x00ff00ffu) | ((b1 & 0x00ff00ffu) << 8u);
        const uint32_t bh = ((b0 >> 8u) & 0x00ff00ffu) | (b1 & 0xff00ff00u);
        hh = qrt_sm121_scalar_integer_core::dot4<true, true>(ah, bh, hh);
        hl = qrt_sm121_scalar_integer_core::dot4<true, false>(ah, bl, hl);
        lh = qrt_sm121_scalar_integer_core::dot4<false, true>(al, bh, lh);
        ll = qrt_sm121_scalar_integer_core::dot4<false, false>(al, bl, ll);
    }
    return int64_t(hh) * 65536 + (int64_t(hl) + lh) * 256 + ll;
}
QRT_COMPACT_DOT4_INLINE Value accumulate(Value carry, const Row& a, const Row& b, unsigned* path = nullptr) {
    if (path) *path = 0u;
    if (!unit(a) || !unit(b)) return fallback(carry, a, b);
    const int exponent = maximum(carry, a, b);
    const int shift = exponent - (int(unit(a) + unit(b)) - 254) - 11;
    if (shift < -25 || shift > 16) return fallback(carry, a, b);
    int64_t mathematical = product(a, b);
    if (shift > 0) {
        const uint32_t mask = (1u << unsigned(shift)) - 1u;
        int32_t discarded = 0;
#pragma unroll
        for (unsigned i = 0u; i < 8u; ++i) {
            const uint32_t x = a.pairs[i], y = b.pairs[i];
            const uint32_t residue = qrt_sm121_core_remainder::multiply_low16(x, y);
            const uint32_t lo = residue & mask, hi = (residue >> 16u) & mask;
            discarded += int32_t(lo) - (((x ^ y) & 0x8000u) && lo ? int32_t(mask + 1u) : 0);
            discarded += int32_t(hi) - (((x ^ y) & 0x80000000u) && hi ? int32_t(mask + 1u) : 0);
        }
        mathematical -= discarded;
        mathematical = mathematical < 0 ? -int64_t(uint64_t(-mathematical) >> unsigned(shift))
                                        : int64_t(uint64_t(mathematical) >> unsigned(shift));
    } else mathematical *= int64_t(1) << unsigned(-shift);
    const unsigned carry_shift = unsigned(exponent - carry.exponent);
    const uint32_t aligned = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    mathematical += carry.negative ? -int64_t(aligned) : int64_t(aligned);
    if (mathematical < -int64_t(UINT32_MAX) || mathematical > int64_t(UINT32_MAX))
        return fallback(carry, a, b);
    if (path) *path = shift > 0 ? 2u : 1u;
    const auto sum = qrt_sm121_group16::decode_modulo_sum(uint32_t(mathematical),
        ((a.pairs[0] ^ b.pairs[0]) & 0x8000u) != 0u);
    return qrt_sm121_canonical::normalize(sum.magnitude, sum.negative, exponent);
}
} // namespace qrt_sm121_compact_integer_dot4
#undef QRT_COMPACT_DOT4_INLINE
