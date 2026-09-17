#pragma once
#include "sm121_group16_modulo.h"
#include "sm121_canonical_normalize.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SPARSE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SPARSE_INLINE inline
#endif

// Exact sparsity, derived solely from original BF16 words. There is no
// magnitude threshold: both signs of zero are absent, all other encodings
// (including subnormals and exceptional words) remain present.
namespace qrt_sm121_sparse_group {
using Value=qrt_q1_moe_hawkeye::Value;
template<unsigned Stride>
QRT_SPARSE_INLINE uint32_t prepare(const uint32_t* pairs) {
    uint32_t mask=0u;
#pragma unroll
    for(unsigned i=0u;i<8u;++i) {
        const uint32_t pair=pairs[i*Stride];
        mask|=unsigned((pair&0x7fffu)!=0u)<<(2u*i);
        mask|=unsigned((pair&0x7fff0000u)!=0u)<<(2u*i+1u);
    }
    return mask;
}
QRT_SPARSE_INLINE unsigned first(uint32_t mask) {return unsigned(__builtin_ctz(mask));}
QRT_SPARSE_INLINE unsigned population(uint32_t mask) {return unsigned(__builtin_popcount(mask));}
// Precondition: carry is the initial zero or an original normalized K16
// endpoint. A zero-product group preserves every nonzero endpoint, including
// subnormal and extended exponents. Its zero sum clears a prior negative zero.
QRT_SPARSE_INLINE Value empty(Value carry) {
    return carry.significand?carry:Value{0u,-133,false};
}
// Caller supplies the intersection of the two exact masks, with 1..4 bits.
// No excluded zero can affect the original maximum (initialized to -133).
// Four products plus one carry cannot reach the modulo sign-overlap region.
// Keep the original sign decoder and canonical normalizer nevertheless.
template<unsigned Stride>
QRT_SPARSE_INLINE Value small(Value carry,const uint32_t* left,
    const uint32_t* right,uint32_t mask) {
    uint32_t products[4]{};unsigned count=0u;
    int maximum=carry.exponent>-133?carry.exponent:-133;
#pragma unroll
    for(unsigned slot=0u;slot<4u;++slot) if(mask) {
        const unsigned i=first(mask);mask&=mask-1u;
        const uint16_t a=uint16_t(left[i/2u]>>((i&1u)*16u));
        const uint16_t b=uint16_t(right[(i/2u)*Stride]>>((i&1u)*16u));
        const auto product=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
        products[slot]=qrt_sm121_group16::pack_product(product);++count;
        maximum=product.exponent>maximum?product.exponent:maximum;
    }
    const unsigned distance=unsigned(maximum-carry.exponent);
    const uint32_t aligned=distance>=32u?0u:(carry.significand<<2u)>>distance;
    uint32_t modulo=carry.negative?0u-aligned:aligned;
#pragma unroll
    for(unsigned slot=0u;slot<4u;++slot) if(slot<count) {
        const uint32_t product=products[slot];
        const unsigned shift=unsigned(maximum-qrt_sm121_group16::packed_exponent(product));
        const uint32_t magnitude=shift>=32u?0u:((product&0xffffu)<<11u)>>shift;
        modulo+=(product&0x80000000u)?0u-magnitude:magnitude;
    }
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,((left[0]^right[0])&0x8000u)!=0u);
    return qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,maximum);
}
} // namespace qrt_sm121_sparse_group
#undef QRT_SPARSE_INLINE
