#pragma once
#include "sm121_packed_core_remainder.h"
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SCALAR_CORE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SCALAR_CORE_INLINE inline
#endif
namespace qrt_sm121_scalar_integer_core {
using Core = qrt_sm121_integer_core::Row;
struct Row {
    Core core;
    qrt_sm121_core_remainder::Prepared remainder;
    uint32_t padding;
};
static_assert(sizeof(Row)==156u); // Odd dword LDS stride.

QRT_SCALAR_CORE_INLINE void prepare(Row& row) {
    qrt_sm121_integer_core::prepare(row.core);
    row.remainder=qrt_sm121_core_remainder::prepare(row.core);
    row.padding=0u;
}

// The byte interpretation flags are explicit. Each partial sum over K16 is
// within signed32: HH is signed/signed, HL and LH have one unsigned byte,
// and LL is unsigned/unsigned. The four partials are recombined in signed64.
template<bool SignedLeft,bool SignedRight>
QRT_SCALAR_CORE_INLINE int32_t dot4(uint32_t left,uint32_t right,int32_t carry) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    if constexpr(SignedLeft && SignedRight)
        asm("v_dot4_i32_iu8 %0, %1, %2, %0 neg_lo:[1,1,0]" : "+v"(carry) : "v"(left),"v"(right));
    else if constexpr(SignedLeft)
        asm("v_dot4_i32_iu8 %0, %1, %2, %0 neg_lo:[1,0,0]" : "+v"(carry) : "v"(left),"v"(right));
    else if constexpr(SignedRight)
        asm("v_dot4_i32_iu8 %0, %1, %2, %0 neg_lo:[0,1,0]" : "+v"(carry) : "v"(left),"v"(right));
    else
        asm("v_dot4_u32_u8 %0, %1, %2, %0" : "+v"(carry) : "v"(left),"v"(right));
    return carry;
#else
    int64_t sum=carry;
    for(unsigned i=0u;i<4u;++i) {
        const unsigned a=(left>>(i*8u))&255u,b=(right>>(i*8u))&255u;
        const int av=SignedLeft && (a&128u)?int(a)-256:int(a);
        const int bv=SignedRight && (b&128u)?int(b)-256:int(b);
        sum+=av*bv;
    }
    return int32_t(sum);
#endif
}

QRT_SCALAR_CORE_INLINE int64_t product(const Core& left,const Core& right) {
    int32_t hh=0,hl=0,lh=0,ll=0;
#pragma unroll
    for(unsigned word=0u;word<4u;++word) {
        hh=dot4<true,true>(uint32_t(left.high[word]),uint32_t(right.high[word]),hh);
        hl=dot4<true,false>(uint32_t(left.high[word]),uint32_t(right.low[word]),hl);
        lh=dot4<false,true>(uint32_t(left.low[word]),uint32_t(right.high[word]),lh);
        ll=dot4<false,false>(uint32_t(left.low[word]),uint32_t(right.low[word]),ll);
    }
    return int64_t(hh)*65536+(int64_t(hl)+lh)*256+ll;
}

template<bool FastFirst>
QRT_SCALAR_CORE_INLINE qrt_q1_moe_hawkeye::Value accumulate(
    qrt_q1_moe_hawkeye::Value carry,const Row& left,const Row& right) {
    qrt_sm121_group16::AlignedSum sum;
    if(qrt_sm121_core_remainder::sum<true,FastFirst>(carry,left.core,right.core,
        left.remainder,right.remainder,product(left.core,right.core),&sum))
        return qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
    // Exceptional encodings retain the independent original wide primitive.
    qrt_q1_moe_hawkeye::Value values[17];values[0]=carry;
    for(unsigned i=0u;i<16u;++i)
        values[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(left.core.original[i],right.core.original[i],-133);
    return qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
}
} // namespace qrt_sm121_scalar_integer_core
#undef QRT_SCALAR_CORE_INLINE
