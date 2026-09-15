#pragma once
#include "out_consumer_interval.h"
#include <cstring>

#if defined(__HIPCC__)
#define QRT_OUT_RESIDUAL_HD __host__ __device__
#else
#define QRT_OUT_RESIDUAL_HD
#endif
namespace qrt_out_residual_filter {
inline int mode(const char* setting) {
    if (!setting || !*setting || !std::strcmp(setting, "0")) return 0;
    return !std::strcmp(setting, "1") ? 1 : -1;
}
constexpr bool shape(unsigned rows, unsigned tokens, unsigned width) {
    return rows == 2048u && tokens == 8192u && width == 4096u;
}
// Conditional only on the original empirical projection envelope. This
// preserves the rounded residual, not the unrounded RMSNorm variance. A full
// GB10 token/logit gate is required for any inference/performance claim.
QRT_OUT_RESIDUAL_HD inline bool invariant(float raw, float residual, float cauchy,
    unsigned ppb, unsigned radius) {
    namespace interval = qrt_out_consumer_interval;
    if (!interval::finite(residual)) return false;
    const float previous = interval::rounded(residual);
    interval::Interval projected;
    if (!interval::finite(previous) || !interval::projection(raw, cauchy, ppb, radius, &projected))
        return false;
#if defined(__HIP_DEVICE_COMPILE__)
    const float lower = __fadd_rn(previous, projected.lower);
    const float upper = __fadd_rn(previous, projected.upper);
#else
    const float lower = previous + projected.lower;
    const float upper = previous + projected.upper;
#endif
    return interval::finite(lower) && interval::finite(upper) &&
        interval::bf16(lower) == interval::bf16(upper);
}
} // namespace qrt_out_residual_filter
#undef QRT_OUT_RESIDUAL_HD
