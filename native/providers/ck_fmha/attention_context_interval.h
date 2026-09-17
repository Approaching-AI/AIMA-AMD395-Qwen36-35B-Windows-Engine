#pragma once
#include "../gdn/sm121_attention_rcp.h"
#include "../moe_accumulator/sm121_pv_error_bound.h"

#if defined(__HIPCC__)
#define QRT_CONTEXT_INTERVAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CONTEXT_INTERVAL_INLINE inline
#endif

// Isolated final-context certificate. Both numerator and denominator may
// still be intervals. A successful result fixes the BF16 output for their
// complete Cartesian product; it does not require an exact PV numerator.
// The caller supplies independently validated arithmetic enclosures and the
// SHA-bound original reciprocal table. This helper does not establish either.
namespace qrt_attention_context_interval {
namespace bits = qrt_sm121_pv_bound;
struct Interval { float low,high; };

QRT_CONTEXT_INTERVAL_INLINE float multiply(float a,float b) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a),"v"(b));return result;
#else
    volatile float result=a*b;return result;
#endif
}
QRT_CONTEXT_INTERVAL_INLINE bool valid(Interval x) {
    return bits::finite(x.low) && bits::finite(x.high) && x.low<=x.high;
}
QRT_CONTEXT_INTERVAL_INLINE bool denominator_domain(Interval x) {
    return valid(x) && x.low>=1.0f && x.high<0x1p19f;
}
QRT_CONTEXT_INTERVAL_INLINE Interval reciprocal(Interval denominator,const unsigned char* table) {
    const float first=qrt_sm121_attention_rcp::evaluate(table,denominator.low);
    if(bits::bits(denominator.low)==bits::bits(denominator.high))return {first,first};
    const float last=qrt_sm121_attention_rcp::evaluate(table,denominator.high);
    // The pinned table has a signed correction in [-1,+1] to rounded 1/m.
    // Two output-word steps enclose its possible nonmonotone excursions.
    return {bits::value(bits::bits(last)-2u),bits::value(bits::bits(first)+2u)};
}
QRT_CONTEXT_INTERVAL_INLINE bool stable(Interval numerator,Interval denominator,
    const unsigned char* table,uint16_t* result) {
    if(!table || !result || !valid(numerator) || !denominator_domain(denominator))return false;
    const auto inverse=reciprocal(denominator,table);
    if(!valid(inverse) || !(inverse.low>0.0f))return false;
    const float a[2]={numerator.low,numerator.high},b[2]={inverse.low,inverse.high};
    uint16_t fixed=0u;
#if defined(__HIPCC__)
#pragma unroll
#endif
    for(unsigned i=0u;i<2u;++i){
#if defined(__HIPCC__)
#pragma unroll
#endif
        for(unsigned j=0u;j<2u;++j){
            const float product=multiply(a[i],b[j]);
            const uint32_t magnitude=bits::bits(product)&0x7fffffffu;
            // Exceptional/subnormal products decline, including nonzero
            // underflow to zero. Their host/device flushing need not agree.
            if(magnitude>=0x7f800000u || (magnitude && magnitude<0x00800000u) ||
                (!magnitude && (bits::bits(a[i])&0x7fffffffu)))return false;
            const uint16_t rounded=bits::bf16(product);
            if((rounded&0x7f80u)==0x7f80u)return false;
            if(i || j){if(rounded!=fixed)return false;}else fixed=rounded;
        }
    }
    *result=fixed;return true;
}
} // namespace qrt_attention_context_interval
#undef QRT_CONTEXT_INTERVAL_INLINE
