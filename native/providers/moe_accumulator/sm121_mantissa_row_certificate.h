#ifndef QRT_SM121_MANTISSA_ROW_CERTIFICATE_H
#define QRT_SM121_MANTISSA_ROW_CERTIFICATE_H

#include "sm121_mantissa_parts.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MANTISSA_ROW_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MANTISSA_ROW_INLINE inline
#endif

namespace qrt_sm121_mantissa_row_certificate {
// Diagnostic reconstruction conditions, conditional on exact matrix partials.
// CPU certificates do not establish that gfx1151 floating WMMA produces those
// partials: both BF16 and FP16 have recorded counterexamples. This header is
// not connected to a runtime admission path; native validation is required.
struct Range {
    int minimum = 1000, maximum = -1000;
    bool valid = true;
};

// Reusable per operand row. A zero contributes no exponent; exceptional or
// non-normal split operands require the original exact calculation.
QRT_MANTISSA_ROW_INLINE Range prepare(const uint16_t* row) {
    Range result;
    for (unsigned i = 0; i < 16u; ++i) {
        const uint16_t word = row[i];
        if (!(word & 0x7fffu)) continue;
        const int exponent = (word >> 7u) & 255u;
        result.valid = result.valid && exponent > 0 && exponent < 255 &&
            (exponent > 7 || !(word & 15u));
        if (exponent < result.minimum) result.minimum = exponent;
        if (exponent > result.maximum) result.maximum = exponent;
    }
    return result;
}

// Sufficient conditions only. Unlike the original per-cell mantissa helper,
// this fast path examines two row ranges and the carry, without scanning the
// sixteen paired products. Wide ranges still require the original path.
QRT_MANTISSA_ROW_INLINE bool eligible(
    const Range& left, const Range& right, qrt_q1_moe_hawkeye::Value carry,
    int* maximum_exponent) {
    if (!left.valid || !right.valid || left.minimum == 1000 || right.minimum == 1000)
        return false;
    const int minimum = left.minimum + right.minimum - 254;
    const int maximum = left.maximum + right.maximum - 254;
    const int aligned = maximum > carry.exponent ? maximum : carry.exponent;
    if (minimum < -112 || maximum > 122 || aligned > 122 ||
        maximum - minimum > 12 || aligned - minimum > 11) return false;
    // All BF16 products lie on this conservative alignment grid. The FP32
    // carry must do so too: increasing the alignment exponent must not discard
    // carry bits that the exact cell-specific exponent would have preserved.
    const unsigned shift = unsigned(aligned - carry.exponent);
    if (carry.significand && shift > 2u &&
        (shift >= 32u || (carry.significand & ((1u << (shift - 2u)) - 1u))))
        return false;
    *maximum_exponent = aligned;
    return true;
}

QRT_MANTISSA_ROW_INLINE bool sum(
    const Range& left, const Range& right, qrt_q1_moe_hawkeye::Value carry,
    bool first_product_negative, const float (&partials)[4],
    qrt_sm121_group16::AlignedSum* output) {
    int exponent;
    if (!eligible(left, right, carry, &exponent)) return false;
    for (unsigned i = 0; i < 4u; ++i)
        if ((qrt_sm121_native_product::float_bits(partials[i]) & 0x7f800000u) == 0x7f800000u)
            return false;
    // Each four-bit partial needs at most 24 significant bits. Their combined
    // exact sum fits FP64. The certificate guarantees no product/carry bits
    // are discarded on this grid, so there is no per-product remainder pass.
    const double mathematical = (double(partials[0]) + partials[1]) +
        (double(partials[2]) + partials[3]);
    const double scaled = mathematical * qrt_sm121_mantissa_parts::power_of_two(25 - exponent);
    if (!(scaled >= -2147483648.0 && scaled <= 2147483647.0)) return false;
    const int32_t products = static_cast<int32_t>(scaled);
    const unsigned shift = unsigned(exponent - carry.exponent);
    const uint32_t aligned_carry = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
    const uint32_t modulo = uint32_t(products) + (carry.negative ? 0u - aligned_carry : aligned_carry);
    *output = {qrt_sm121_group16::decode_modulo_sum(modulo, first_product_negative), exponent};
    return true;
}
} // namespace qrt_sm121_mantissa_row_certificate

#undef QRT_MANTISSA_ROW_INLINE
#endif
