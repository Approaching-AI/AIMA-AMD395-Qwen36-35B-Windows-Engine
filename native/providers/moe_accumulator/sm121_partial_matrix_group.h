#pragma once
#include "sm121_matrix_remainder_group.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PARTIAL_MATRIX_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PARTIAL_MATRIX_INLINE inline
#endif

namespace qrt_sm121_partial_matrix_group {
namespace group = qrt_sm121_compact_matrix_group;
namespace compact = group::compact;

// Each exception retains its original BF16 bits in the coefficient slot. Its
// matrix coefficient is zero. Exponents/nonzero still describe every original
// operand, including exceptions, so alignment never uses an incomplete dot.
struct Row { group::Row common; uint32_t exceptions; };
static_assert(sizeof(Row) == 64u);

QRT_PARTIAL_MATRIX_INLINE uint32_t coefficient_pair(const Row& row, unsigned pair) {
    if (!compact::unit(row.common.encoded)) return 0u;
    const unsigned mask = row.exceptions >> (2u * pair);
    return row.common.encoded.pairs[pair] &
        ((mask & 1u ? 0u : 65535u) | (mask & 2u ? 0u : 0xffff0000u));
}
QRT_PARTIAL_MATRIX_INLINE uint16_t original(const Row& row, unsigned i) {
    return row.exceptions & (1u << i) ? compact::word(row.common.encoded, i)
                                    : compact::original(row.common.encoded, i);
}
QRT_PARTIAL_MATRIX_INLINE Row prepare(const uint16_t* input) {
    Row result{};
    // Complete-domain admission belongs to the caller; invalid inputs retain
    // a lossless raw row and cannot enter this accumulator even in isolation.
    for (unsigned i = 0u; i < 16u; ++i)
        if (!qrt_sm121_narrow_f32_carry::eligible(input[i])) {
            for (unsigned j = 0u; j < 16u; ++j)
                result.common.encoded.pairs[j / 2u] |= uint32_t(input[j]) << ((j & 1u) * 16u);
            return result;
        }
    result.common = group::prepare(input);
    if (compact::unit(result.common.encoded)) return result;
    result = Row{};
    unsigned maximum = 0u, live = 0u;
    for (unsigned i = 0u; i < 16u; ++i) {
        if (!(input[i] & 0x7fffu)) continue;
        const unsigned exponent = (input[i] >> 7u) & 255u;
        maximum = maximum > exponent ? maximum : exponent;
        live |= 1u << i;
    }
    const unsigned unit = live ? maximum - 7u : 127u;
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t bits = input[i];
        const unsigned exponent = bits & 0x7fffu ? (bits >> 7u) & 255u : 0u;
        const unsigned significand = 128u | (bits & 127u);
        const int shift = int(exponent) - int(unit);
        bool exception = bits == 0x8000u;
        unsigned coefficient = 0u;
        if (bits & 0x7fffu) {
            exception = shift < -7 || shift > 7;
            if (!exception && shift < 0) {
                exception = (significand & ((1u << unsigned(-shift)) - 1u)) != 0u;
                coefficient = significand >> unsigned(-shift);
            } else if (!exception) coefficient = significand << unsigned(shift);
            if (bits & 0x8000u) coefficient = uint16_t(0u - coefficient);
        }
        if (exception) { result.exceptions |= 1u << i; coefficient = bits; }
        result.common.encoded.pairs[i / 2u] |= coefficient << ((i & 1u) * 16u);
        result.common.encoded.exponents[i / 4u] |= exponent << ((i & 3u) * 8u);
        const unsigned trailing = !exception && coefficient
            ? qrt_sm121_integer_parts::trailing_bits(uint16_t(coefficient)) : 15u;
        result.common.trailing[i / 8u] |= trailing << ((i & 7u) * 4u);
    }
    result.common.encoded.control = unit | (maximum << 8u) | (live << 16u);
    return result;
}

// Mathematical contains only products whose two coefficients are lossless.
// Add each omitted original product after its individual 26-bit alignment.
// No exceptional value is approximated, and rejection leaves output untouched.
// The caller certifies the same complete narrow domain and K<=8192 as group.
QRT_PARTIAL_MATRIX_INLINE bool accumulate(float carry, const Row& a, const Row& b,
    int64_t mathematical, float* output) {
    const uint32_t exceptions = a.exceptions | b.exceptions;
    if (!exceptions)
        return qrt_sm121_matrix_remainder_group::accumulate(
            carry, a.common, b.common, mathematical, output);
    if (!compact::unit(a.common.encoded) || !compact::unit(b.common.encoded)) return false;
    const unsigned active = compact::nonzero(a.common.encoded) & compact::nonzero(b.common.encoded);
    if (!active) { *output = carry; return true; }
    int maximum = group::product_maximum(a.common, b.common);
    const int exponent = int((group::f32::bits(carry) & 0x7fffffffu) >> 23u) - 127;
    maximum = maximum > exponent ? maximum : exponent;
    maximum = maximum > -89 ? maximum : -89;
    const int shift = maximum - (int(compact::unit(a.common.encoded) +
        compact::unit(b.common.encoded)) - 254) - 11;
    if (shift < -25 || shift > 16) return false;
    if (shift > 0) {
        const uint32_t mask = (1u << unsigned(shift)) - 1u;
        int32_t discarded = 0;
#pragma unroll
        for (unsigned pair = 0u; pair < 8u; ++pair) {
            const uint32_t x = coefficient_pair(a, pair), y = coefficient_pair(b, pair);
            const uint32_t residue = qrt_sm121_core_remainder::multiply_low16(x, y);
            const uint32_t lo = residue & mask, hi = (residue >> 16u) & mask;
            discarded += int32_t(lo) - (((x ^ y) & 0x8000u) && lo ? int32_t(mask + 1u) : 0);
            discarded += int32_t(hi) - (((x ^ y) & 0x80000000u) && hi ? int32_t(mask + 1u) : 0);
        }
        mathematical -= discarded;
        mathematical = mathematical < 0
            ? -int64_t(uint64_t(-mathematical) >> unsigned(shift))
            : int64_t(uint64_t(mathematical) >> unsigned(shift));
    } else mathematical *= int64_t(1) << unsigned(-shift);
    const float scale = group::f32::alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = uint32_t(mathematical) + uint32_t(int32_t(carry * scale));
    // Sparse exceptions are typically the small-magnitude terms. Only these
    // products expand to FP32; BF16 products and power-of-two alignment remain
    // exact in the complete narrow domain, as in the original narrow producer.
    uint32_t remaining = exceptions & active;
    while (remaining) {
        const unsigned i = unsigned(__builtin_ctz(remaining));
        remaining &= remaining - 1u;
        const float x = group::f32::alignment::from_bits(uint32_t(original(a, i)) << 16u);
        const float y = group::f32::alignment::from_bits(uint32_t(original(b, i)) << 16u);
        const float product = x * y;
        modulo += uint32_t(int32_t(product * scale));
    }
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo,
        ((original(a, 0u) ^ original(b, 0u)) & 0x8000u) != 0u);
    *output = group::normalize(sum.magnitude, sum.negative, maximum);
    return true;
}
} // namespace qrt_sm121_partial_matrix_group
#undef QRT_PARTIAL_MATRIX_INLINE
