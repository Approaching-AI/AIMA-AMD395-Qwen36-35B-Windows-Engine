#pragma once
#include "sm121_byte_exponents.h"
#include "sm121_predicted_group.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_NARROW_PLAN_INLINE __host__ __device__ __forceinline__
#else
#define QRT_NARROW_PLAN_INLINE inline
#endif
namespace qrt_sm121_narrow_qk_plan {
namespace byte=qrt_sm121_byte_exponents;
namespace plan=qrt_sm121_predicted_group;
using Plan=plan::Plan;
QRT_NARROW_PLAN_INLINE int predicted_exponent(float value){
    const uint32_t bits=byte::f32::bits(value)&0x7fffffffu;
    if(!bits)return -133;
    if(bits<0x00800000u||bits>=0x7f800000u)return 512;
    return int(bits>>23u)-127;
}
QRT_NARROW_PLAN_INLINE int maximum(int product,int predicted){
    if(predicted< -133||predicted>73)return 512;
    const int combined=product>predicted?product:predicted;
    return combined> -89?combined:-89;
}
// Metadata and original operands satisfy the byte/narrow K256 admission.
// A prediction chooses work only. The actual carry is checked before use.
QRT_NARROW_PLAN_INLINE Plan prepare(const uint16_t* a,const uint16_t* b,
    const byte::Metadata& am,const byte::Metadata& bm,int predicted){
    const int product=byte::product_maximum(am,bm),aligned=maximum(product,predicted);
    if(aligned==512)return {};
    const float scale=byte::f32::alignment::from_bits(uint32_t(152-aligned)<<23u);
    uint32_t modulo=0u;
    for(unsigned i=0u;i<16u;++i){
        const float p=byte::f32::alignment::from_bits(uint32_t(a[i])<<16u)*
            byte::f32::alignment::from_bits(uint32_t(b[i])<<16u);
        modulo+=uint32_t(int32_t(p*scale));
    }
    return plan::encode(modulo,aligned,product,((a[0]^b[0])&0x8000u)!=0u);
}
QRT_NARROW_PLAN_INLINE bool apply(float carry,Plan prepared,float* output){
    if(!output||!plan::valid(prepared))return false;
    const int actual=byte::alignment(carry,plan::product_maximum(prepared));
    if(actual!=plan::maximum(prepared))return false;
    const float scale=byte::f32::alignment::from_bits(uint32_t(152-actual)<<23u);
    const uint32_t modulo=prepared.modulo+uint32_t(int32_t(carry*scale));
    *output=byte::finish(modulo,(prepared.control&(1u<<20u))!=0u,actual);
    return true;
}
} // namespace qrt_sm121_narrow_qk_plan
#undef QRT_NARROW_PLAN_INLINE
