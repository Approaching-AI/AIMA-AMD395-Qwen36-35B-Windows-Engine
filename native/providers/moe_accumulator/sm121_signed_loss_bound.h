#pragma once
#include "sm121_coarse_projection_bound.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SIGNED_LOSS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SIGNED_LOSS_INLINE inline
#endif

// Isolated C64 envelope. This changes canonical loss accounting, not the
// inherited conditional native 2^-19 premise. No provider dispatch uses it.
namespace qrt_sm121_signed_loss_bound {
namespace coarse=qrt_sm121_coarse_projection_bound;
namespace scalar=coarse::scalar;
struct Summary { uint64_t negative=0u,nonzero=0u; bool valid=true; };
struct Counts { unsigned positive=0u,negative=0u; };
struct State { float center=0.0f,lower_error=0.0f,upper_error=0.0f; };
QRT_SIGNED_LOSS_INLINE Summary summarize(const uint16_t* words,unsigned length=64u) {
    Summary result;
    if(!words || length>64u){result.valid=false;return result;}
    for(unsigned i=0u;i<length;++i){
        result.valid &= coarse::eligible(words[i]);
        result.negative |= uint64_t((words[i]>>15u)&1u)<<i;
        result.nonzero |= uint64_t(bool(words[i]&0x7fffu))<<i;
    }
    return result;
}
QRT_SIGNED_LOSS_INLINE unsigned population(uint64_t x) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __popcll(x);
#else
    unsigned n=0u;while(x){x&=x-1u;++n;}return n;
#endif
}
QRT_SIGNED_LOSS_INLINE Counts counts(Summary a,Summary b) {
    const uint64_t active=a.nonzero&b.nonzero;
    const unsigned negative=population((a.negative^b.negative)&active);
    return {population(active)-negative,negative};
}
QRT_SIGNED_LOSS_INLINE bool valid(State v) {
    return scalar::finite(v.center)&&scalar::finite(v.lower_error)&&
        scalar::finite(v.upper_error)&&v.lower_error>=0.0f&&v.upper_error>=0.0f;
}
QRT_SIGNED_LOSS_INLINE State advance(State before,float partial,float absolute,
    Counts products,unsigned* stable_sign=nullptr) {
    if(stable_sign)*stable_sign=0u;
    State result{before.center+partial,scalar::infinity(),scalar::infinity()};
    if(!valid(before)||!scalar::finite(partial)||!scalar::finite(absolute)||
        absolute<0.0f||products.positive>64u||products.negative>64u-products.positive)
        return result;
    constexpr float epsilon=0x1p-19f,floor=4.0f*0x1p-118f;
    constexpr float inflation=1.0f/(1.0f-epsilon-4.0f*0x1p-23f);
    const float positive=scalar::upper(scalar::upper(absolute*inflation)+floor);
    const float native_magnitude=scalar::upper(positive*inflation);
    const float native_error=scalar::upper(scalar::upper(positive*epsilon)+
        scalar::upper(4.0f*coarse::unit(native_magnitude,23u))+floor);
    const float entry_error=before.lower_error>before.upper_error?before.lower_error:before.upper_error;
    const float magnitude=scalar::upper(scalar::upper(scalar::absolute(before.center)+entry_error)+positive);
    const float unit=coarse::unit(magnitude,25u);
    const float all_canonical=scalar::upper(84.0f*unit+floor);
    // Every original prefix differs from the entry carry plus exact products
    // by at most the original 84-unit bound. This barrier establishes the
    // signs of ALL four incoming carries and pre-normalization sums; an entry
    // sign alone cannot do so. No actual/reference carry is consulted.
    const float barrier=scalar::upper(scalar::upper(entry_error+positive)+all_canonical);
    const bool positive_prefix=before.center>barrier;
    const bool negative_prefix=before.center< -barrier;
    if(stable_sign)*stable_sign=positive_prefix?1u:(negative_prefix?2u:0u);
    // A positive product's magnitude truncation contributes only downward
    // error; a negative product contributes only upward error. Four carry
    // alignments and four normalizations cost 4*(1+4) units, charged on the
    // known side when the complete-block sign barrier succeeds, both sides
    // otherwise. Zero products have no alignment loss, regardless of sign.
    const float lower_loss=scalar::upper(float(products.positive+(negative_prefix?0u:20u))*unit+floor);
    const float upper_loss=scalar::upper(float(products.negative+(positive_prefix?0u:20u))*unit+floor);
    const float addition_error=coarse::unit(scalar::upper(scalar::absolute(before.center)+scalar::absolute(partial)),23u);
    result.lower_error=scalar::upper(scalar::upper(before.lower_error+native_error)+scalar::upper(lower_loss+addition_error));
    result.upper_error=scalar::upper(scalar::upper(before.upper_error+native_error)+scalar::upper(upper_loss+addition_error));
    return result;
}
QRT_SIGNED_LOSS_INLINE State advance(State before,float partial,float absolute,Summary a,Summary b,
    unsigned* stable_sign=nullptr) {
    if(!a.valid||!b.valid)return {before.center+partial,scalar::infinity(),scalar::infinity()};
    return advance(before,partial,absolute,counts(a,b),stable_sign);
}
QRT_SIGNED_LOSS_INLINE bool certified(State v) {
    if(!valid(v))return false;
    const float lower=scalar::next(v.center-v.lower_error,false);
    const float upper=scalar::next(v.center+v.upper_error,true);
    return scalar::finite(lower)&&scalar::finite(upper)&&scalar::bf16(lower)==scalar::bf16(upper);
}
}
#undef QRT_SIGNED_LOSS_INLINE
