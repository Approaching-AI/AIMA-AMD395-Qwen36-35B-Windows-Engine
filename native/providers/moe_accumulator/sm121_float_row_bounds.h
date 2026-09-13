#ifndef QRT_SM121_FLOAT_ROW_BOUNDS_H
#define QRT_SM121_FLOAT_ROW_BOUNDS_H
#include "sm121_float_alignment.h"

#if defined(__HIPCC__)
#define QRT_FLOAT_BOUNDS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_FLOAT_BOUNDS_INLINE inline
#endif

namespace qrt_sm121_float_row_bounds {
using Value = qrt_q1_moe_hawkeye::Value;
constexpr uint32_t valid_bit = 1u << 16u;

// Each original K16 row needs one word: maximum biased exponent, minimum
// biased exponent plus significand trailing zeros, and scalar eligibility.
// Zero contributes neither extremum. No tensor value is rounded or replaced.
QRT_FLOAT_BOUNDS_INLINE uint32_t prepare(const uint16_t* row) {
    unsigned maximum = 0u, minimum = 255u;
    bool valid = true;
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t word = row[i];
        if (!(word & 0x7fffu)) continue;
        const unsigned exponent = (word >> 7u) & 255u;
        valid &= exponent >= 64u && exponent <= 190u;
        const unsigned low_bit = exponent + unsigned(__builtin_ctz(128u | (word & 127u)));
        maximum = exponent > maximum ? exponent : maximum;
        minimum = low_bit < minimum ? low_bit : minimum;
    }
    return maximum | (minimum << 8u) | (valid ? valid_bit : 0u);
}

// If the carry dominates both row maxima, this is the original alignment
// exponent without a paired-product scan. Otherwise the larger row bound is
// equivalent only when every product AND the carry are exact on that grid.
// Normalization of the same exact sum then retains identical FP32 carry bits.
QRT_FLOAT_BOUNDS_INLINE bool exponent(Value carry, uint32_t left, uint32_t right, int* output) {
    if (!(left & right & valid_bit)) return false;
    const bool empty = !(left & 255u) || !(right & 255u);
    const int upper = empty ? -133 : int(left & 255u) + int(right & 255u) - 254;
    int maximum = carry.exponent > upper ? carry.exponent : upper;
    maximum = maximum > -133 ? maximum : -133;
    if (maximum < -101 || maximum > 151) return false;
    if (maximum > carry.exponent) {
        const int integer_limit = int((left >> 8u) & 255u) + int((right >> 8u) & 255u) - 243;
        if (!empty && maximum > integer_limit) return false;
        const unsigned shift = unsigned(maximum - carry.exponent);
        if (carry.significand && shift > 2u &&
            (shift >= 32u || (carry.significand & ((1u << (shift - 2u)) - 1u)))) return false;
    }
    *output = maximum;
    return true;
}

// Host/device arithmetic oracle for the reusable metadata contract. GPU
// cooperative dots distribute the same sixteen scalar terms across lanes.
QRT_FLOAT_BOUNDS_INLINE bool sum(Value carry, const uint16_t* left, const uint16_t* right,
    uint32_t left_bounds, uint32_t right_bounds, qrt_sm121_group16::AlignedSum* output) {
    int maximum;
    if (!exponent(carry, left_bounds, right_bounds, &maximum)) return false;
    const float scale = qrt_sm121_float_alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = 0u;
    for (unsigned i = 0u; i < 16u; ++i) {
        const float product = qrt_sm121_float_alignment::from_bits(uint32_t(left[i]) << 16u) *
            qrt_sm121_float_alignment::from_bits(uint32_t(right[i]) << 16u);
        modulo += uint32_t(int32_t(product * scale));
    }
    const unsigned shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
    modulo += carry.negative ? 0u - aligned : aligned;
    *output = {qrt_sm121_group16::decode_modulo_sum(modulo, ((left[0] ^ right[0]) & 0x8000u) != 0u), maximum};
    return true;
}
} // namespace qrt_sm121_float_row_bounds
#undef QRT_FLOAT_BOUNDS_INLINE
#endif
