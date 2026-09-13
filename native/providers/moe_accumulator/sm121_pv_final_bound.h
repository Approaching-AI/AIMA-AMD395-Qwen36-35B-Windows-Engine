#ifndef QRT_SM121_PV_FINAL_BOUND_H
#define QRT_SM121_PV_FINAL_BOUND_H

#include "sm121_pv_error_bound.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PV_FINAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PV_FINAL_INLINE inline
#endif

namespace qrt_sm121_pv_final_bound {
namespace old = qrt_sm121_pv_bound;

// Metadata only: native PV's K16 WMMA, online K32 rescaling and final RCP
// multiplication retain their original order. Store error / 2^-19 without
// propagating its small relative inflation at every step. Restrict this route
// to <= 512 K16 groups, at most one alpha in [0,1] per two groups.
//
// For normal nonnegative arithmetic, including each old upper()'s two ULPs,
// the old group is <= (1+2^-18)*(error + 2^-19*(|carry|+dot)) + 8*2^-118.
// The old rescale is <= (1+2^-18)*(alpha*error + 2^-22*alpha*|carry|)
// + 8*2^-118. Comparing these to the rounded metadata below costs less than
// 1+2^-17 per call (also allowing four downward metadata ULPs per call).
// For G <= 512, (1+2^-17)^(3G/2) <= 1+G*2^-15. The 64*G*2^-118
// additive term covers the old floors and FP32 normal/subnormal flushing.
// A 2^100 metadata cap prevents intermediate overflow from invalidating
// this comparison; exceptional values select exact replay and stay sticky.
// Thus finalization enlarges the old envelope, rather than tightening the
// admission contract. This remains subject to native and GB10 validation.
constexpr unsigned max_groups = 512u;
constexpr uint32_t cap_bits = 0x71800000u; // 2^100

// Keep individual FP32 rounding without the private-memory traffic generated
// by volatile local metadata on gfx1151. The inference accumulator is untouched.
QRT_PV_FINAL_INLINE float add(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
#else
    volatile float result = a + b; return result;
#endif
}
QRT_PV_FINAL_INLINE float multiply(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;
    asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
#else
    volatile float result = a * b; return result;
#endif
}

QRT_PV_FINAL_INLINE float cap(float x) {
    return old::bits(x) < cap_bits ? x : old::infinity();
}

QRT_PV_FINAL_INLINE float group(float state, float carry, float absolute_dot) {
    const float sum = add(add(state, old::absolute(carry)), absolute_dot);
    if ((old::bits(state) | old::bits(absolute_dot)) & 0x80000000u)
        return old::infinity();
    return cap(sum);
}

QRT_PV_FINAL_INLINE float rescale(float state, float carry, float alpha) {
    if (old::bits(state) >= cap_bits ||
        (old::bits(carry) & 0x7fffffffu) >= cap_bits ||
        old::bits(alpha) > 0x3f800000u) return old::infinity();
    if (alpha == 0.0f) return 0.0f;
    if (alpha == 1.0f) return state;
    const float scaled_state = multiply(state, alpha);
    const float scaled_carry = multiply(old::absolute(carry), alpha);
    const float rounding = multiply(scaled_carry, 0.125f);
    const float sum = add(scaled_state, rounding);
    return cap(sum);
}

QRT_PV_FINAL_INLINE float finalize(float state, unsigned groups) {
    if (!groups || groups > max_groups || (groups & 1u) || old::bits(state) >= cap_bits)
        return old::infinity();
    const float inflation = 1.0f + float(groups) * 0x1p-15f;
    const float scaled = multiply(state, 0x1p-19f);
    const float inflated = multiply(scaled, inflation);
    return old::upper(old::upper(inflated) + float(groups) * 0x1p-112f);
}
} // namespace qrt_sm121_pv_final_bound

#undef QRT_PV_FINAL_INLINE
#endif
