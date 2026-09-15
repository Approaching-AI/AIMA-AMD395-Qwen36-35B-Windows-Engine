#ifndef QRT_SM121_MATRIX_MAXIMUM_H
#define QRT_SM121_MATRIX_MAXIMUM_H
#include "sm121_f32_carry.h"
#if defined(__HIPCC__)
#define QRT_MATRIX_MAX_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MATRIX_MAX_INLINE inline
#endif

namespace qrt_sm121_matrix_maximum {
constexpr uint16_t invalid_operand = 0x8000u;
constexpr int invalid_maximum = -32768;

// Encode a nonzero BF16 exponent e as the positive power 2^(5e). Signs and
// significands are irrelevant to the original alignment exponent. The bounded
// domain keeps every nonzero encoded product normal and the full K16 sum
// finite. Other values retain the original per-product maximum calculation.
QRT_MATRIX_MAX_INLINE uint16_t encode(uint16_t value) {
    if (!(value & 0x7fffu)) return 0u;
    const int exponent = int((value >> 7u) & 255u) - 127;
    if (exponent < -12 || exponent > 12) return invalid_operand;
    return uint16_t((127 + 5 * exponent) << 7u);
}
QRT_MATRIX_MAX_INLINE uint32_t pack(uint16_t value) {
    return (uint32_t(value) << 16u) | encode(value);
}

// If M is the largest paired exponent, the exact positive matrix dot S lies
// in [2^(5M),16*2^(5M)]. Under the explicit relative native-error condition
// |native-S| <= S/4, multiplying native by 1.5 places it strictly between
// 2^(5M) and 32*2^(5M), even after one FP32 rounding. Its exponent divided by
// five (floor, including negatives) therefore recovers M exactly. This wide
// separation does not establish a universal hardware error guarantee.
// An exact zero dot requires the native instruction to preserve zero.
QRT_MATRIX_MAX_INLINE int recover(float native) {
    const uint32_t raw = qrt_sm121_f32_carry::bits(native);
    if (!(raw & 0x7fffffffu)) return -133;
    if ((raw & 0x80000000u) || raw >= 0x7f800000u) return invalid_maximum;
    const uint32_t scaled = qrt_sm121_f32_carry::bits(native * 1.5f);
    const int exponent = int((scaled >> 23u) & 255u) - 127;
    if (exponent < -120 || exponent > 124) return invalid_maximum;
    return exponent >= 0 ? exponent / 5 : -((-exponent + 4) / 5);
}
}
#undef QRT_MATRIX_MAX_INLINE
#endif
