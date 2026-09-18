#pragma once
#include "sm121_pv_final_bound.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PV_LONG_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PV_LONG_INLINE inline
#endif

// Isolated extension of the deferred metadata proof. Existing short-context
// finalization, arithmetic accumulators and provider defaults are unchanged.
namespace qrt_sm121_pv_long_final_bound {
namespace deferred=qrt_sm121_pv_final_bound;
namespace scalar=qrt_sm121_pv_bound;
using deferred::group;
using deferred::rescale;
constexpr unsigned max_groups=16546u; // ceil(264736/32)*2

// The existing capped group/rescale metadata has a length-independent
// per-call comparison factor q=1+2^-17 and additive floor8*2^-118.
// There are G group calls and at most G/2 rescale calls, hence M=3G/2.
// For nonnegative x and M*x<1, the binomial expansion gives
//   (1+x)^M <= 1/(1-M*x).
// At G=16546, M=24819 and M*2^-17<0.19. The subtraction below is exact:
// M and its power-of-two-scaled complement have fewer than24 significant
// bits. Two outward reciprocal steps cover its FP32 division rounding.
// The inherited64*G*2^-118 floor exceeds M*8*2^-118 times this growth,
// including the final multiplication/underflow allowance. Invalid metadata
// stays sticky through the existing2^100 cap. No native coefficient changes.
QRT_PV_LONG_INLINE float finalize(float state,unsigned groups) {
    if(groups<=deferred::max_groups)return deferred::finalize(state,groups);
    if(groups>max_groups || (groups&1u) || scalar::bits(state)>=deferred::cap_bits)
        return scalar::infinity();
    const unsigned calls=groups+groups/2u;
    const float denominator=1.0f-float(calls)*0x1p-17f;
    const float inflation=scalar::upper(1.0f/denominator);
    const float scaled=deferred::multiply(state,0x1p-19f);
    const float inflated=deferred::multiply(scaled,inflation);
    return scalar::upper(scalar::upper(inflated)+float(groups)*0x1p-112f);
}
struct Finalizer {
    QRT_PV_LONG_INLINE static float finalize(float state,unsigned groups) {
        return qrt_sm121_pv_long_final_bound::finalize(state,groups);
    }
};
} // namespace qrt_sm121_pv_long_final_bound
#undef QRT_PV_LONG_INLINE
