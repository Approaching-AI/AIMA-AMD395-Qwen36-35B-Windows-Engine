#ifndef QRT_SM121_RANGE_NORMALIZE_H
#define QRT_SM121_RANGE_NORMALIZE_H
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_RANGE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_RANGE_INLINE inline
#endif
namespace qrt_sm121_range {
constexpr unsigned kMinimumExponent = 77u;
constexpr unsigned kMaximumExponent = 179u;
constexpr unsigned kMaximumColumns = 16384u;

QRT_RANGE_INLINE bool bounded_operand(uint16_t value) {
    const unsigned exponent = (value >> 7u) & 255u;
    return !(value & 0x7fffu) ||
        (exponent >= kMinimumExponent && exponent <= kMaximumExponent);
}
QRT_RANGE_INLINE bool bounded_shape(unsigned columns) {
    return columns && columns <= kMaximumColumns && !(columns & 15u);
}

// For admitted nonzero BF16 operands, 2^-50 <= |x| < 2^53, so every
// nonzero product is in [2^-100, 2^106). At most 2^14 terms give an
// absolute running-sum bound <2^120. Truncating each aligned magnitude and
// the resulting carry cannot increase this bound. Thus a nonzero-product
// K16 group has maximum exponent in [-100,119]. Its smallest nonzero
// aligned sum has exponent maximum-25 >= -125: no subnormal adjustment is
// possible. Zero-product groups with a smaller carry preserve that carry.
// This specialization is valid only after that row-pair certificate.
QRT_RANGE_INLINE qrt_q1_moe_hawkeye::Value normalize(
    uint32_t magnitude, bool negative, int maximum) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    const unsigned leading = __clz(magnitude | 1u);
#else
    const unsigned leading = 32u - qrt_q1_moe_hawkeye::bit_width_u64(magnitude | 1u);
#endif
    const uint32_t significand = (magnitude << leading) >> 8u;
    return {significand, int16_t(significand ? maximum + 6 - int(leading) : -133), negative};
}
} // namespace qrt_sm121_range
#undef QRT_RANGE_INLINE
#endif
