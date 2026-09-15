#ifndef QRT_SM121_CARRY_TRANSFER_H
#define QRT_SM121_CARRY_TRANSFER_H
#include "sm121_group16_modulo.h"
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_CARRY_TRANSFER_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CARRY_TRANSFER_INLINE inline
#endif
namespace qrt_sm121_carry_transfer {
using Value=qrt_q1_moe_hawkeye::Value;
constexpr uint32_t product_limit=16u*qrt_sm121_group16::kMaxAlignedProduct;
static_assert(product_limit<0x7fffffffu-3u);
QRT_CARRY_TRANSFER_INLINE bool regular(Value carry) {
    return carry.significand>=0x800000u && carry.significand<0x1000000u &&
        carry.exponent>=-126 && carry.exponent<=127;
}
QRT_CARRY_TRANSFER_INLINE bool product_sum(uint32_t modulo) {
    return modulo<=product_limit || modulo>=0u-product_limit;
}
// With a fixed alignment exponent, positive carries add floor(S/4), while
// negative carries add ceil(S/4) in signed-significand coordinates. The
// product-only sum S fits signed32; neither conversion nor division overflows.
QRT_CARRY_TRANSFER_INLINE int32_t increment(uint32_t modulo,bool negative) {
    const int32_t sum=modulo>0x7fffffffu?-int32_t(0u-modulo):int32_t(modulo);
    return negative ? (sum>=0?(sum+3)/4:-(-sum/4)) :
        (sum>=0?sum/4:-((-sum+3)/4));
}
template<unsigned Groups>
QRT_CARRY_TRANSFER_INLINE bool apply(Value carry,const uint32_t (&sums)[Groups],
    bool last_product_negative,Value* result) {
    static_assert(Groups==2u || Groups==4u || Groups==8u);
    // Caller establishes that every original product exponent is at most the
    // incoming carry exponent. Check every interior carry, not just endpoints.
    if(!result || !regular(carry))return false;
    int32_t current=carry.negative?-int32_t(carry.significand):int32_t(carry.significand);
    for(unsigned group=0u;group+1u<Groups;++group) {
        if(!product_sum(sums[group]))return false;
        const int32_t step=increment(sums[group],carry.negative);
        // Two same-binade significands differ by less than2^23. This bound
        // also makes the signed addition safe before the interior range test.
        if(step<=-0x800000 || step>=0x800000)return false;
        current+=step;
        if(carry.negative ? current>-0x800000 || current<=-0x1000000 :
            current<0x800000 || current>=0x1000000)return false;
    }
    if(!product_sum(sums[Groups-1u]))return false;
    // The final group may change sign or exponent. Normalize it exactly once,
    // preserving the original modulo sign-disambiguation and underflow rules.
    const auto sum=qrt_sm121_group16::decode_modulo_sum(
        uint32_t(current)*4u+sums[Groups-1u],last_product_negative);
    *result=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,carry.exponent);
    return true;
}
}
#undef QRT_CARRY_TRANSFER_INLINE
#endif
