#ifndef QRT_SM121_COARSE_PROJECTION_BOUND_H
#define QRT_SM121_COARSE_PROJECTION_BOUND_H
#include "sm121_pv_error_bound.h"
#if defined(__HIPCC__)
#define QRT_COARSE_BOUND_INLINE __host__ __device__ __forceinline__
#else
#define QRT_COARSE_BOUND_INLINE inline
#endif

// Experimental BF16 producer envelope. Production defaults to2^-19. The
// explicit2^-20 diagnostic arm has an arithmetic derivation conditional on
// the observed eight-DOT2 hardware model; finite hardware comparisons do not
// prove that premise. Canonical loss accounting is independent of the native
// assumption and includes EVERY K16 alignment and normalization.
namespace qrt_sm121_coarse_projection_bound {
namespace scalar=qrt_sm121_pv_bound;
struct State{float center=0.0f,error=0.0f;};
QRT_COARSE_BOUND_INLINE bool eligible(uint16_t x) {
    const unsigned exponent=(x>>7u)&255u;
    return !(x&0x7fffu) || (exponent>=80u && exponent<=174u);
}
QRT_COARSE_BOUND_INLINE float unit(float magnitude,unsigned fractional) {
    if(magnitude==0.0f)return 0.0f;
    if(!scalar::finite(magnitude) || magnitude<0.0f)return scalar::infinity();
    const int exponent=int((scalar::bits(magnitude)>>23u)&255u)-127-int(fractional);
    if(exponent>=128)return scalar::infinity();
    if(exponent>=-126)return scalar::value(unsigned(exponent+127)<<23u);
    // Below the normal scale, saturate unrepresentable error units to the
    // minimum positive float rather than constructing a negative exponent.
    return scalar::value(exponent>=-149?1u<<unsigned(exponent+149):1u);
}
template<unsigned Groups,unsigned NativeErrorBits=19u>
QRT_COARSE_BOUND_INLINE State advance(State before,float partial,float absolute) {
    static_assert(Groups==4u || Groups==8u || Groups==16u || Groups==32u);
    static_assert(NativeErrorBits==19u || NativeErrorBits==20u);
    State result{before.center+partial,scalar::infinity()};
    if(!scalar::finite(before.center)||!scalar::finite(before.error)||before.error<0.0f||
        !scalar::finite(partial)||!scalar::finite(absolute)||absolute<0.0f)return result;
    constexpr float epsilon=1.0f/float(1u<<NativeErrorBits),floor=float(Groups)*0x1p-118f;
    // The extra Groups*2^-23 covers the explicit FP32 sums of zero-C K16
    // outputs. Two outward steps cover constant reciprocal/product rounding.
    constexpr float inflation=1.0f/(1.0f-epsilon-float(Groups)*0x1p-23f);
    const float positive=scalar::upper(scalar::upper(absolute*inflation)+floor);
    const float native_magnitude=scalar::upper(positive*inflation);
    const float native_error=scalar::upper(scalar::upper(positive*epsilon)+
        scalar::upper(float(Groups)*unit(native_magnitude,23u))+floor);

    // Magnitude cannot grow beyond the entry carry plus absolute products:
    // every canonical signed alignment and normalization truncates magnitude.
    // This bounds every interior exponent of the complete coarse block.
    const float magnitude=scalar::upper(scalar::upper(scalar::absolute(before.center)+before.error)+positive);
    // All16 products and the carry may each lose one26-bit alignment unit.
    // FP32 normalization loses at most four such units at the same exponent.
    // Hence21 units per original K16 group, without assuming cancellation,
    // a product exponent distribution, or exact unrounded RMS consumers.
    const float canonical_error=scalar::upper(float(21u*Groups)*unit(magnitude,25u)+floor);
    const float addition_error=unit(scalar::upper(scalar::absolute(before.center)+scalar::absolute(partial)),23u);
    result.error=scalar::upper(scalar::upper(before.error+native_error)+scalar::upper(canonical_error+addition_error));
    return result;
}
QRT_COARSE_BOUND_INLINE bool certified(State value) {
    return scalar::same_bf16(value.center,value.error);
}
}
#undef QRT_COARSE_BOUND_INLINE
#endif
