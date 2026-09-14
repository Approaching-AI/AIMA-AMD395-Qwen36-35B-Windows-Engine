#ifndef QRT_SM121_PROJECTION_INTERVAL_H
#define QRT_SM121_PROJECTION_INTERVAL_H
#include "sm121_pv_error_bound.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PROJECTION_INTERVAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PROJECTION_INTERVAL_INLINE inline
#endif

// Experimental interval propagation; no product dispatcher uses this header.
// The native zero-C product keeps the existing conservative 2^-19 coefficient.
// Canonical alignment loss is separate from directed carry additions, avoiding
// a second generic native dot-error allowance on the growing carry itself.
namespace qrt_sm121_projection_interval {
namespace bound = qrt_sm121_pv_bound;
struct Interval { float lower, upper; };
struct Row { int minimum = 255, maximum = -1; bool valid = true; };

QRT_PROJECTION_INTERVAL_INLINE void include(Row& row, uint16_t value) {
    if (!(value & 0x7fffu)) return;
    const int exponent = int((value >> 7u) & 255u);
    // This domain keeps nonzero K16 alignment units normal in FP32 and
    // products far from overflow. Other operands require original replay.
    row.valid = row.valid && exponent >= 80 && exponent <= 174;
    row.minimum = exponent < row.minimum ? exponent : row.minimum;
    row.maximum = exponent > row.maximum ? exponent : row.maximum;
}
QRT_PROJECTION_INTERVAL_INLINE Interval invalid() {
    return {-bound::infinity(), bound::infinity()};
}
QRT_PROJECTION_INTERVAL_INLINE bool valid(Interval value) {
    return bound::finite(value.lower) && bound::finite(value.upper) && value.lower <= value.upper;
}
QRT_PROJECTION_INTERVAL_INLINE float directed_add(float a, float b, bool positive) {
#if defined(__HIP_DEVICE_COMPILE__)
    return positive ? __ocml_add_rtp_f32(a, b) : __ocml_add_rtn_f32(a, b);
#elif defined(__CUDA_ARCH__)
    return positive ? __fadd_ru(a, b) : __fadd_rd(a, b);
#else
    // TwoSum's residual decides whether nearest-even needs one outward step.
    // Volatile intermediates prevent reassociation in the host diagnostic.
    volatile float sum = a + b;
    if (!bound::finite(sum)) return positive ? bound::infinity() : -bound::infinity();
    volatile float recovered = sum - a;
    volatile float first = sum - recovered;
    volatile float left = a - first;
    volatile float right = b - recovered;
    volatile float error = left + right;
    return (positive ? error > 0.0f : error < 0.0f) ? bound::next(sum, positive) : float(sum);
#endif
}
QRT_PROJECTION_INTERVAL_INLINE int exponent(float magnitude) {
    return magnitude == 0.0f ? -133 : int((bound::bits(magnitude) >> 23u) & 255u) - 127;
}

QRT_PROJECTION_INTERVAL_INLINE Interval group(Interval carry, float product,
    float absolute_dot, const Row& left, const Row& right) {
    if (!valid(carry) || !left.valid || !right.valid || !bound::finite(product) ||
        !bound::finite(absolute_dot) || absolute_dot < 0.0f) return invalid();
    if (left.maximum < 0 || right.maximum < 0) return carry;
    const float abs_lower = bound::absolute(carry.lower), abs_upper = bound::absolute(carry.upper);
    const float carry_magnitude = abs_lower > abs_upper ? abs_lower : abs_upper;
    int maximum = left.maximum + right.maximum - 254;
    const int carry_maximum = exponent(carry_magnitude);
    maximum = maximum > carry_maximum ? maximum : carry_maximum;
    // Nonzero eligible products imply maximum >= -94. A large or corrupt
    // carry fails closed before constructing a scale or adding endpoints.
    if (maximum < -94 || maximum > 120) return invalid();
    const float unit = bound::value(uint32_t(maximum + 102) << 23u); // 2^(maximum-25)

    // Each BF16 product has eleven zero alignment bits at width 26. If even
    // the row-wise minimum exponent is close enough, every alignment is exact.
    const int minimum_product = left.minimum + right.minimum - 254;
    unsigned lost_terms = maximum - minimum_product <= 11 ? 0u : 16u;
    bool exact_carry = carry.lower == 0.0f && carry.upper == 0.0f;
    if (carry.lower > 0.0f || carry.upper < 0.0f) {
        const float nearest_zero = carry.lower > 0.0f ? carry.lower : -carry.upper;
        exact_carry = maximum - exponent(nearest_zero) <= 2;
    }
    lost_terms += exact_carry ? 0u : 1u;
    // The actual group exponent cannot exceed maximum. Each inexact signed
    // alignment loses strictly less than unit, toward zero. This symmetric
    // interval covers either sign without guessing cancellation or a bias.
    const float products = bound::upper(absolute_dot * (1.0f + 0x1p-19f) + 0x1p-118f);
    const float native_error = bound::upper(products * 0x1p-19f + 0x1p-118f);
    const float loss = bound::upper(native_error + float(lost_terms) * unit);
    if (!bound::finite(loss)) return invalid();
    const Interval next{
        directed_add(directed_add(carry.lower, product, false), -loss, false),
        directed_add(directed_add(carry.upper, product, true), loss, true)};
    // Canonical normalization truncates to FP32. Outward FP32 endpoints
    // enclose that result without accumulating a generic normalization error.
    return valid(next) ? next : invalid();
}
QRT_PROJECTION_INTERVAL_INLINE bool same_bf16(Interval value) {
    return valid(value) && bound::bf16(value.lower) == bound::bf16(value.upper);
}
} // namespace qrt_sm121_projection_interval
#undef QRT_PROJECTION_INTERVAL_INLINE
#endif
