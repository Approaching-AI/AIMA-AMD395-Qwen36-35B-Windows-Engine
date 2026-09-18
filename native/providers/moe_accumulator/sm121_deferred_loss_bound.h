#pragma once
#include "sm121_signed_loss_bound.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DEFERRED_LOSS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DEFERRED_LOSS_INLINE inline
#endif

// Component only. Inputs to advance() are complete eligible C64 native
// partials, with the original zero-C K16 WMMA order and conditional2^-19
// premise. Counts describe the actual nonzero signed products. No reference
// carry or output participates in this recurrence.
namespace qrt_sm121_deferred_loss_bound {
namespace old=qrt_sm121_signed_loss_bound;
namespace scalar=old::scalar;
struct State { float center=0.0f,lower_sum=0.0f,upper_sum=0.0f; };
constexpr unsigned max_blocks=128u;
constexpr uint32_t cap_bits=0x7d800000u; // 2^124

// Keep the metadata's stated FP32 operations without volatile spills on GPU.
QRT_DEFERRED_LOSS_INLINE float add(float a,float b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a),"v"(b));return result;
#else
    volatile float result=a+b;return result;
#endif
}
QRT_DEFERRED_LOSS_INLINE float multiply(float a,float b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a),"v"(b));return result;
#else
    volatile float result=a*b;return result;
#endif
}
QRT_DEFERRED_LOSS_INLINE bool width_supported(unsigned width) {
    return width && width<=8192u && !(width&15u);
}

// Store error increments in units of2^-25. For A=absolute partial,
// C=incoming native carry, P=signed partial and c=product_count+20:
//   I(c) = 80*A + 4*(|C|+|P|) + c*(|C|+A).
// The80 includes the unchanged native2^-19 allowance (64 units) and
// four explicit K16 partial additions (16 units). The c term charges each
// potentially inexact product and all four carry/normalization boundaries.
// Dropping the sign barrier enlarges the old directed canonical charge.
// No outward rounding, exponent extraction or inherited error propagation
// occurs inside this metadata loop; center retains the original addition.
QRT_DEFERRED_LOSS_INLINE State advance(State before,float partial,float absolute,
    old::Counts products) {
    const float carry=scalar::absolute(before.center);
    const float magnitude=add(carry,absolute);
    const float common=add(multiply(80.0f,absolute),
        multiply(4.0f,add(carry,scalar::absolute(partial))));
    const float lower=add(common,multiply(float(products.positive+20u),magnitude));
    const float upper=add(common,multiply(float(products.negative+20u),magnitude));
    return {before.center+partial,add(before.lower_sum,lower),add(before.upper_sum,upper)};
}

// A complete eligible row has BF16 exponents80..174. At most128 C64
// blocks give |C|<2^109, A<2^102 and both scaled metadata sums<2^124.
// The old recurrence is bounded, on each side, by
//   q*(E_side + 2^-25*(I(side)+84*max(E_lower,E_upper))) + 2^-110,
// with q=1+2^-16. This covers every old two-ULP upper(), the absolute
// producer inflation and unit(x,b)<=x*2^-b; floors cover zero/subnormal
// terms. Each increment and its sum use at most8 normal FP32 operations.
// For N<=128, combining q, the84*2^-25 feedback, metadata rounding and
// their cross terms costs less than1+N*2^-15. Since all increments are
// nonnegative, each side is bounded by its stored sum plus N*2^-15 times
// the larger final sum. N*2^-108 covers the additive recurrence floors.
// Final outward steps cover finalization rounding. This enlarges the old
// directional envelope; it does not assert a universal hardware error bound.
QRT_DEFERRED_LOSS_INLINE old::State finalize(State state,unsigned blocks) {
    const float inf=scalar::infinity();
    if(!blocks || blocks>max_blocks || !scalar::finite(state.center) ||
        scalar::bits(state.lower_sum)>=cap_bits || scalar::bits(state.upper_sum)>=cap_bits)
        return {state.center,inf,inf};
    const float maximum=state.lower_sum>state.upper_sum?state.lower_sum:state.upper_sum;
    const float allowance=scalar::upper(multiply(float(blocks)*0x1p-15f,maximum));
    const float floor=float(blocks)*0x1p-108f;
    const float lower=scalar::upper(scalar::upper(multiply(scalar::upper(add(state.lower_sum,allowance)),0x1p-25f))+floor);
    const float upper=scalar::upper(scalar::upper(multiply(scalar::upper(add(state.upper_sum,allowance)),0x1p-25f))+floor);
    return {state.center,lower,upper};
}
} // namespace qrt_sm121_deferred_loss_bound
#undef QRT_DEFERRED_LOSS_INLINE
