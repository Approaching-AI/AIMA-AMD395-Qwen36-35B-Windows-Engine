#ifndef QRT_SM121_BYTE_RESIDUE32_CORE_H
#define QRT_SM121_BYTE_RESIDUE32_CORE_H
#include "sm121_byte_residue_core.h"
#if defined(__HIPCC__)
#define QRT_RESIDUE32_INLINE __host__ __device__ __forceinline__
#else
#define QRT_RESIDUE32_INLINE inline
#endif

// Isolated headroom5 specialization. The FP16 estimate still requires the
// explicit error<128 condition; integer range proofs do not prove that
// hardware condition. No product dispatcher uses this header.
namespace qrt_sm121_byte_residue32 {
namespace original=qrt_sm121_byte_residue_core;
using Row=original::Row;
using Value=original::Value;
using AlignedSum=original::AlignedSum;
constexpr int32_t maximum_dot=16*8160*8160;
static_assert(maximum_dot==1065369600 && maximum_dot+1024<INT32_MAX);

QRT_RESIDUE32_INLINE void prepare(Row& row) { original::prepare<5u>(row); }
QRT_RESIDUE32_INLINE bool recover(float approximate,uint32_t residue,int32_t* output) {
    // Guard conversion and subsequent offset addition before narrowing.
    if(!(approximate>-float(maximum_dot+512) && approximate<float(maximum_dot+512)))return false;
    const int32_t truncated=int32_t(approximate);
    const uint32_t offset=(residue-uint32_t(truncated))&255u;
    if(offset==128u && approximate==float(truncated))return false;
    int32_t result=truncated+int32_t(offset);
    if(offset>128u || (offset==128u && approximate<float(truncated)))result-=256;
    if(result < -maximum_dot || result > maximum_dot)return false;
    *output=result;return true;
}

QRT_RESIDUE32_INLINE uint32_t shifted(int32_t value,int shift) {
    if(!value || shift>=26)return 0u;
    if(shift<=0)return uint32_t(value)<<unsigned(-shift);
    const uint32_t magnitude=uint32_t(value<0?-value:value)>>unsigned(shift);
    return value<0?0u-magnitude:magnitude;
}

QRT_RESIDUE32_INLINE AlignedSum finish(uint32_t products,Value carry,int maximum,
    const Row& left,const Row& right) {
    const unsigned shift=unsigned(maximum-carry.exponent);
    const uint32_t aligned=shift>=32u?0u:(carry.significand<<2u)>>shift;
    const uint32_t total=products+(carry.negative?0u-aligned:aligned);
    const bool first_negative=((left.original[0]^right.original[0])&0x8000u)!=0u;
    // Sixteen individually bounded products plus one carry can exceed
    // INT32_MAX. Preserve the original unsigned modulo/sign-disambiguation
    // proof instead of narrowing that complete aligned sum to signed32.
    return {qrt_sm121_group16::decode_modulo_sum(total,first_negative),maximum};
}

QRT_RESIDUE32_INLINE bool sum(Value carry,const Row& left,const Row& right,
    int32_t mathematical,AlignedSum* output) {
    if(left.unit<0 || right.unit<0 || mathematical < -maximum_dot || mathematical > maximum_dot)return false;
    const int minimum=left.unit+right.unit-254;
    const uint32_t exceptions=(left.exceptions|right.exceptions)&left.nonzero&right.nonzero;
    const int row_bound=left.maximum+right.maximum-254;
    int maximum=carry.exponent>-133?carry.exponent:-133;
    const int bound=row_bound>maximum?row_bound:maximum;
    const int bound_shift=bound-minimum-11;
    const unsigned carry_shift=unsigned(bound-carry.exponent);
    const uint32_t carry_bits=carry.significand<<2u;
    const bool carry_exact=!carry_bits || (carry_shift<32u && !(carry_bits&((1u<<carry_shift)-1u)));
    if(!exceptions && carry_exact && (!mathematical || bound_shift>=-25) &&
        (bound_shift<=0 || qrt_sm121_integer_parts::products_divisible(left.trailing,right.trailing,unsigned(bound_shift)))) {
        *output=finish(shifted(mathematical,bound_shift),carry,bound,left,right);return true;
    }
    if(row_bound>maximum)maximum=original::paired_maximum(left,right,maximum);
    const int shift=maximum-minimum-11;
    if(shift < -25 && mathematical)return false;
    // Each headroom5 core product is below2^26. Beyond that shift every
    // original core term is zero. For smaller shifts, signed discarded sums
    // and their compensated dot stay within +/-maximum_dot.
    const uint32_t losses=shift>0 && shift<26?original::remainder_mask(left,right,unsigned(shift)):0u;
    uint32_t pending=losses|exceptions,corrections=0u;int32_t discarded=0;
    while(pending) {
        const unsigned i=qrt_sm121_integer_core::first_bit(pending);pending&=pending-1u;
        const uint16_t a=left.original[i],b=right.original[i];
        const int ac=qrt_sm121_integer_core::signed_core(a,left.unit),bc=qrt_sm121_integer_core::signed_core(b,right.unit);
        const uint32_t magnitude=uint32_t(ac<0?-ac:ac)*uint32_t(bc<0?-bc:bc);
        const bool negative=((a^b)&0x8000u)!=0u;
        if(losses&(1u<<i)) {
            const int32_t remainder=int32_t(magnitude&((1u<<unsigned(shift))-1u));
            discarded+=negative?-remainder:remainder;
        }
        if(exceptions&(1u<<i)) {
            const auto product=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
            const unsigned original_shift=unsigned(maximum-product.exponent);
            const uint32_t aligned=original_shift>=32u?0u:(product.significand<<2u)>>original_shift;
            const uint32_t core_aligned=!magnitude || shift>=26?0u:shift>0?magnitude>>unsigned(shift):magnitude<<unsigned(-shift);
            const uint32_t difference=aligned-core_aligned;
            corrections+=negative?0u-difference:difference;
        }
    }
    const uint32_t products=shifted(mathematical-discarded,shift)+corrections;
    *output=finish(products,carry,maximum,left,right);return true;
}
QRT_RESIDUE32_INLINE AlignedSum fallback(Value carry,const Row& left,const Row& right) {
    uint32_t products[16];
    for(unsigned i=0u;i<16u;++i)products[i]=qrt_sm121_group16::pack_product(
        qrt_q1_moe_hawkeye::multiply_bf16(left.original[i],right.original[i],-133));
    return qrt_sm121_group16::sum_packed(carry,products);
}
#if defined(__HIPCC__)
using original::products;
#endif
}
#undef QRT_RESIDUE32_INLINE
#endif
