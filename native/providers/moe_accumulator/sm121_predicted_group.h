#pragma once
#include "sm121_float_alignment.h"
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PREDICTED_GROUP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PREDICTED_GROUP_INLINE inline
#endif

// Isolated experiment: prepare product alignment before the preceding exact
// carry is available. A predicted exponent is a scheduling hint only. The
// consumer checks the actual common exponent before accepting the plan.
namespace qrt_sm121_predicted_group {
using Value=qrt_q1_moe_hawkeye::Value;
struct Plan { uint32_t modulo=0u,control=0x80000000u; };
static_assert(sizeof(Plan)==8u);
QRT_PREDICTED_GROUP_INLINE Plan encode(uint32_t modulo,int maximum,int product_maximum,bool negative) {
    return {modulo,uint32_t(maximum+254)|(uint32_t(product_maximum+254)<<10u)|(uint32_t(negative)<<20u)};
}
QRT_PREDICTED_GROUP_INLINE int maximum(Plan plan) {return int(plan.control&1023u)-254;}
QRT_PREDICTED_GROUP_INLINE int product_maximum(Plan plan) {return int((plan.control>>10u)&1023u)-254;}
QRT_PREDICTED_GROUP_INLINE bool valid(Plan plan) {return !(plan.control&0x80000000u);}

// Products are the unmodified multiply_bf16(...,-133) results encoded by
// pack_product. Their original exponent/sign and all significant bits remain.
QRT_PREDICTED_GROUP_INLINE Plan prepare_packed(const uint32_t (&products)[16],int predicted_exponent) {
    if(predicted_exponent < -133 || predicted_exponent > 511)return {};
    int product_max=-133;
    for(unsigned i=0u;i<16u;++i){const int e=qrt_sm121_group16::packed_exponent(products[i]);if(e>product_max)product_max=e;}
    const int alignment=product_max>predicted_exponent?product_max:predicted_exponent;
    uint32_t modulo=0u;
    for(unsigned i=0u;i<16u;++i){
        const auto p=products[i];const unsigned shift=unsigned(alignment-qrt_sm121_group16::packed_exponent(p));
        const uint32_t value=shift>=32u?0u:((p&0xffffu)<<11u)>>shift;
        modulo+=(p&0x80000000u)?0u-value:value;
    }
    return encode(modulo,alignment,product_max,(products[0]&0x80000000u)!=0u);
}

// The caller establishes the existing scalar-product eligibility predicate.
// This is the same power-of-two conversion used by float_alignment::sum.
QRT_PREDICTED_GROUP_INLINE Plan prepare_float(const qrt_sm121_float_alignment::Group& group,int predicted_exponent) {
    if(predicted_exponent < -133 || predicted_exponent > 511)return {};
    const int alignment=group.maximum>predicted_exponent?group.maximum:predicted_exponent;
    if(alignment==-133)return encode(0u,-133,group.maximum,group.first_negative);
    if(alignment < -101 || alignment > 127)return {};
    const float scale=qrt_sm121_float_alignment::from_bits(uint32_t(152-alignment)<<23u);
    uint32_t modulo=0u;
    for(unsigned i=0u;i<16u;++i)modulo+=uint32_t(int32_t(group.products[i]*scale));
    return encode(modulo,alignment,group.maximum,group.first_negative);
}

// The input carry is an original normalized K16 endpoint (at most24 bits),
// including zero, subnormal and extended exponents. Rejection leaves output
// untouched, and the caller recomputes this group with the original routine.
QRT_PREDICTED_GROUP_INLINE bool apply(Value carry,Plan plan,Value* output) {
    if(!output || !valid(plan) || carry.significand>0xffffffu)return false;
    const int product_max=product_maximum(plan);
    const int actual=product_max>carry.exponent?product_max:carry.exponent;
    if(actual!=maximum(plan))return false;
    const unsigned shift=unsigned(actual-carry.exponent);
    const uint32_t value=shift>=32u?0u:(carry.significand<<2u)>>shift;
    const uint32_t modulo=plan.modulo+(carry.negative?0u-value:value);
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,(plan.control&(1u<<20u))!=0u);
    *output=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,actual);
    return true;
}
} // namespace qrt_sm121_predicted_group
#undef QRT_PREDICTED_GROUP_INLINE
