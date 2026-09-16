#pragma once
#include "sm121_pv_final_bound.h"
#include <cmath>
#include <cfloat>

#if defined(__HIPCC__)
#define QRT_PV_PREFIX_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PV_PREFIX_INLINE inline
#endif

// Component-only prefix refinement of the SAME carried native PV producer.
// This does not restart canonical arithmetic from an approximate checkpoint.
// The replay supplies a prefix computed exactly from the original zero carry.
namespace qrt_sm121_pv_prefix_bound {
namespace bound = qrt_sm121_pv_bound;
namespace deferred = qrt_sm121_pv_final_bound;
struct Checkpoint { float center, state; };
struct Weight { double lower, upper; };
struct Interval { double lower, upper; };
QRT_PV_PREFIX_INLINE bool finite(double x) { return x >= -DBL_MAX && x <= DBL_MAX; }
QRT_PV_PREFIX_INLINE double down(double x) { return ::nextafter(x, -INFINITY); }
QRT_PV_PREFIX_INLINE double up(double x) { return ::nextafter(x, INFINITY); }
QRT_PV_PREFIX_INLINE bool valid(Weight w) {
    return w.lower >= 0.0 && w.lower <= w.upper && w.upper <= 1.0;
}
QRT_PV_PREFIX_INLINE Weight multiply(Weight w, float alpha) {
    if (!valid(w) || bound::bits(alpha) > 0x3f800000u) return {-1.0, -1.0};
    if (alpha == 0.0f) return {0.0, 0.0};
    if (alpha == 1.0f) return w;
    const double lo = down(w.lower * double(alpha)), hi = up(w.upper * double(alpha));
    return {lo > 0.0 ? lo : 0.0, hi < 1.0 ? hi : 1.0};
}

// Let E be the original outward per-step envelope and S the deferred state.
// In exact nonnegative arithmetic the original group adds at least
// 2^-19*(abs(native_carry)+absolute_dot). Its rescale adds at least
// 2^-22*alpha*abs(native_carry), besides alpha*E. Thus E >= 2^-19*S_real.
// At most four rounded state operations per K16 give gamma(4G) < G*2^-16
// for G <= 512. The subtractive G*2^-112 floor dominates all state
// normal/subnormal flushing after the 2^-19 scaling. This is a LOWER bound
// on the old E, unlike deferred::finalize, which is an UPPER bound.
QRT_PV_PREFIX_INLINE double prefix_lower(float state, unsigned groups) {
    if (!groups || groups > 512u || (groups & 1u) || bound::bits(state) >= deferred::cap_bits)
        return -1.0;
    const double scaled = double(state) * 0x1p-19;
    const double result = down(down(scaled * (1.0 - double(groups) * 0x1p-16)) -
        double(groups) * 0x1p-112);
    return result > 0.0 ? result : 0.0;
}

// Unroll the old envelope using the ORIGINAL native/canonical paths. With
// A the suffix product of the original alphas, E_final - A*E_prefix bounds
// the weighted sum of all suffix local errors. Consequently
//   canonical_final = native_final + A*(exact_prefix-native_prefix) + delta.
// Replace E_final by its deferred upper bound and E_prefix by the lower
// bound above. This keeps every inherited local-error allowance; it does
// not assume that rounded K16 transitions are Lipschitz or associative.
// The conditional native WMMA assumption is inherited, not strengthened.
QRT_PV_PREFIX_INLINE Interval suffix(Checkpoint prefix, Checkpoint final,
    float exact_prefix, Weight weight, unsigned prefix_groups, unsigned final_groups) {
    const Interval invalid{-INFINITY, INFINITY};
    if (!valid(weight) || prefix_groups >= final_groups || !bound::finite(prefix.center) ||
        !bound::finite(final.center) || !bound::finite(exact_prefix)) return invalid;
    const double before = prefix_lower(prefix.state, prefix_groups);
    const float after = deferred::finalize(final.state, final_groups);
    if (before < 0.0 || !bound::finite(after)) return invalid;
    const double remaining = up(double(after) - down(weight.lower * before));
    if (!(remaining >= 0.0) || !finite(remaining)) return invalid;
    const double lo = down(double(exact_prefix) - double(prefix.center));
    const double hi = up(double(exact_prefix) - double(prefix.center));
    const double a = weight.lower * lo, b = weight.upper * lo;
    const double c = weight.lower * hi, d = weight.upper * hi;
    const double low_product = ::fmin(::fmin(a,b),::fmin(c,d));
    const double high_product = ::fmax(::fmax(a,b),::fmax(c,d));
    return {down(down(double(final.center) + down(low_product)) - remaining),
            up(up(double(final.center) + up(high_product)) + remaining)};
}
QRT_PV_PREFIX_INLINE bool certified(Interval interval, float reciprocal, float* representative) {
    if (!finite(interval.lower) || !finite(interval.upper) || interval.lower > interval.upper ||
        !bound::finite(reciprocal) || reciprocal < 0.0f) return false;
    const float lower = bound::next(float(interval.lower), false);
    const float upper = bound::next(float(interval.upper), true);
    if (!bound::finite(lower) || !bound::finite(upper)) return false;
    const float out_lower = deferred::multiply(lower, reciprocal);
    const float out_upper = deferred::multiply(upper, reciprocal);
    if (!bound::finite(out_lower) || !bound::finite(out_upper) ||
        bound::bf16(out_lower) != bound::bf16(out_upper)) return false;
    const float center = float(interval.lower * 0.5 + interval.upper * 0.5);
    const float output = deferred::multiply(center, reciprocal);
    if (!bound::finite(output) || bound::bf16(output) != bound::bf16(out_lower)) return false;
    *representative = output;
    return true;
}
} // namespace qrt_sm121_pv_prefix_bound
#undef QRT_PV_PREFIX_INLINE
