#pragma once
#include "routed_consumer_interval.h"

#if defined(__HIPCC__)
#define QRT_DOWN_HD __host__ __device__
#else
#define QRT_DOWN_HD
#endif
namespace qrt_moe_down_consumer {
namespace c = qrt_routed_consumer;
constexpr unsigned valid_bit = 256u;
struct Snapshot { float low, high, center; unsigned flags; uint16_t residual, reserved; };
static_assert(sizeof(Snapshot) == 20u);
QRT_DOWN_HD inline float add(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __fadd_rn(a, b);
#else
    volatile float value = a + b; return value;
#endif
}
QRT_DOWN_HD inline float multiply(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    return __fmul_rn(a, b);
#else
    volatile float value = a * b; return value;
#endif
}
QRT_DOWN_HD inline float rounded(float x) { return c::widen(c::rounded(x)); }

// The retained vt4 tree pairs routes0/4,1/5,2/6,3/7 and adds those pairs in
// order. Every input has already received the weighted BF16 endpoint.
QRT_DOWN_HD inline float sum(const float (&x)[8]) {
    float value = add(add(x[0], x[4]), add(x[1], x[5]));
    value = add(value, add(x[2], x[6]));
    return add(value, add(x[3], x[7]));
}
QRT_DOWN_HD inline Snapshot enclose(const float (&raw)[8], const float (&error)[8], unsigned selected) {
    float lower[8], upper[8], center[8]; bool valid = true;
    for (unsigned route = 0u; route < 8u; ++route) {
        center[route] = rounded(raw[route]); lower[route] = upper[route] = center[route];
        valid = valid && c::finite(center[route]);
        if (selected & (1u << route)) {
            const auto range = c::range(raw[route], error[route]);
            valid = valid && range.valid;
            if (range.valid) {
                lower[route] = c::widen(c::unordered(range.low));
                upper[route] = c::widen(c::unordered(range.high));
            }
        }
    }
    const float low = sum(lower), high = sum(upper), value = sum(center);
    valid = valid && c::finite(low) && c::finite(high) && c::finite(value) && low <= value && value <= high;
    return {low, high, value, (selected & 255u) | (valid ? valid_bit : 0u), 0u, 0u};
}
QRT_DOWN_HD inline float combined(float routed, float shared) {
    return rounded(add(rounded(routed), rounded(shared)));
}
QRT_DOWN_HD inline float residual(float routed, float shared, float hidden) {
    return add(rounded(hidden), combined(routed, shared));
}
QRT_DOWN_HD inline bool routed_constant(Snapshot s) {
    return (s.flags & valid_bit) && c::rounded(s.low) == c::rounded(s.high);
}
QRT_DOWN_HD inline bool combined_constant(Snapshot s, float shared) {
    if (!(s.flags & valid_bit) || !c::finite(shared)) return false;
    const float low = combined(s.low, shared), high = combined(s.high, shared);
    return c::finite(low) && c::finite(high) && c::bits(low) == c::bits(high);
}
} // namespace qrt_moe_down_consumer
#undef QRT_DOWN_HD
