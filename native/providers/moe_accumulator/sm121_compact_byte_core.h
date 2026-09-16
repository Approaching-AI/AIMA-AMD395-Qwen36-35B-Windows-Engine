#pragma once
#include "sm121_byte_residue32_core.h"
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__)
#define QRT_COMPACT_BYTE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_COMPACT_BYTE_INLINE inline
#endif

// Isolated representation and arithmetic experiment. The original BF16 tensor
// remains available with a caller-supplied stride. The conditional native
// FP16 matrix error<128 requirement is inherited, not proved by this layout.
namespace qrt_sm121_compact_byte_core {
namespace wide=qrt_sm121_byte_residue_core;
namespace integer=qrt_sm121_integer_core;
namespace small=qrt_sm121_byte_residue32;
using Value=qrt_q1_moe_hawkeye::Value;
struct Row {
    uint16_t half[16];
    int low[4];
    uint32_t trailing[4];
    uint32_t control,masks,padding;
};
static_assert(sizeof(Row)==76u); // Odd dword stride in LDS, with explicit padding.
QRT_COMPACT_BYTE_INLINE int unit(const Row& row){return int(int16_t(row.control));}
QRT_COMPACT_BYTE_INLINE unsigned maximum(const Row& row){return (row.control>>16u)&255u;}
QRT_COMPACT_BYTE_INLINE unsigned exceptions(const Row& row){return row.masks&65535u;}
QRT_COMPACT_BYTE_INLINE unsigned nonzero(const Row& row){return row.masks>>16u;}
QRT_COMPACT_BYTE_INLINE bool eligible(const Row& row){return (row.control&0x01000000u)!=0u;}
QRT_COMPACT_BYTE_INLINE Row prepare(const uint16_t* original){
    wide::Row source{};
    for(unsigned i=0u;i<16u;++i)source.original[i]=original[i];
    wide::prepare<5u>(source);
    Row result{};
    for(unsigned i=0u;i<16u;++i) {
        // Preserve signed zeros, including a negative original rounded to an
        // integer-core zero. This does not change the integer matrix value.
        result.half[i]=uint16_t((source.half[i]&0x7fffu)|(original[i]&0x8000u));
    }
    for(unsigned i=0u;i<4u;++i){result.low[i]=source.low[i];result.trailing[i]=source.trailing[i];}
    result.control=uint16_t(source.unit)|(uint32_t(source.maximum)<<16u)|(uint32_t(source.original[17])<<24u);
    result.masks=source.exceptions|(source.nonzero<<16u);
    return result;
}

// A non-exception core is exactly the original value times 2^(134-unit).
// Its FP16 exponent therefore maps to BF16 exponent half_exp+unit-22.
// All seven original fraction bits survive that exact power-of-two relation.
QRT_COMPACT_BYTE_INLINE uint16_t original(const Row& row,const uint16_t* words,
    size_t stride,unsigned index){
    if(unit(row)<0 || (exceptions(row)&(1u<<index)))return words[size_t(index)*stride];
    const uint16_t value=row.half[index];
    if(!(value&0x7fffu))return value&0x8000u;
    return uint16_t((value&0x8000u)|(unsigned(int((value>>10u)&31u)+unit(row)-22)<<7u)|((value&1023u)>>3u));
}
QRT_COMPACT_BYTE_INLINE unsigned remainder_mask(const Row& a,const Row& b,unsigned shift){
    if(!shift)return 0u;
    const uint32_t threshold=shift*0x01010101u;uint32_t result=0u;
    for(unsigned word=0u;word<4u;++word){
        const uint32_t high=~(a.trailing[word]+b.trailing[word]+0x80808080u-threshold)&0x80808080u;
        result|=((((high>>7u)*0x01020408u)>>24u)&15u)<<(word*4u);
    }
    return result;
}

// The actual carry fixes alignment on this path. Rejection leaves output and
// statistics untouched; the caller uses original scalar arithmetic for that
// one group. No 64-bit general integer-core path is needed here.
QRT_COMPACT_BYTE_INLINE bool dominant(Value carry,const Row& a,const Row& b,
    const uint16_t* original_a,size_t stride_a,const uint16_t* original_b,size_t stride_b,
    int32_t mathematical,Value* output,unsigned* replayed=nullptr){
    if(!output || unit(a)<1 || unit(b)<1 || carry.significand>0xffffffu ||
        carry.exponent < -101 || carry.exponent > 127 ||
        carry.exponent < int(maximum(a)+maximum(b))-254 ||
        mathematical < -small::maximum_dot || mathematical > small::maximum_dot)return false;
    const int shift=carry.exponent-(unit(a)+unit(b)-254)-11;
    if(shift < -1 || shift > 16)return false;
    const unsigned losses=shift>0?remainder_mask(a,b,unsigned(shift)):0u;
    const unsigned exceptional=(exceptions(a)|exceptions(b))&nonzero(a)&nonzero(b);
    unsigned pending=losses|exceptional,count=0u;int32_t discarded=0;uint32_t correction=0u;
    while(pending){
        const unsigned index=integer::first_bit(pending);pending&=pending-1u;++count;
        const uint16_t left=original(a,original_a,stride_a,index),right=original(b,original_b,stride_b,index);
        const int ac=integer::signed_core(left,unit(a)),bc=integer::signed_core(right,unit(b));
        const uint32_t magnitude=uint32_t(ac<0?-ac:ac)*uint32_t(bc<0?-bc:bc);
        const bool negative=((left^right)&0x8000u)!=0u;
        if(losses&(1u<<index)){
            const int32_t remainder=int32_t(magnitude&((1u<<unsigned(shift))-1u));
            discarded+=negative?-remainder:remainder;
        }
        if(exceptional&(1u<<index)){
            const auto product=qrt_q1_moe_hawkeye::multiply_bf16(left,right,-133);
            const unsigned distance=unsigned(carry.exponent-product.exponent);
            const uint32_t aligned=distance>=32u?0u:(product.significand<<2u)>>distance;
            const uint32_t core_aligned=shift>0?magnitude>>unsigned(shift):magnitude<<unsigned(-shift);
            const uint32_t difference=aligned-core_aligned;
            correction+=negative?0u-difference:difference;
        }
    }
    const uint32_t products=small::shifted(mathematical-discarded,shift)+correction;
    const uint32_t aligned=carry.significand<<2u;
    const uint32_t total=products+(carry.negative?0u-aligned:aligned);
    const auto sum=qrt_sm121_group16::decode_modulo_sum(total,((a.half[0]^b.half[0])&0x8000u)!=0u);
    *output=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,carry.exponent);
    if(replayed)*replayed=count;
    return true;
}
QRT_COMPACT_BYTE_INLINE Value fallback(Value carry,const Row& a,const Row& b,
    const uint16_t* original_a,size_t stride_a,const uint16_t* original_b,size_t stride_b){
    qrt_sm121_group16::AlignedSum sum;
    if(eligible(a)&&eligible(b)&&(!carry.significand||carry.exponent>=-126)){
        qrt_sm121_float_alignment::Group group;
        for(unsigned i=0u;i<16u;++i)group.set(i,original(a,original_a,stride_a,i),original(b,original_b,stride_b,i));
        if(qrt_sm121_float_alignment::sum(carry,group,&sum))
            return qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
    }
    uint32_t products[16];
    for(unsigned i=0u;i<16u;++i)products[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
        original(a,original_a,stride_a,i),original(b,original_b,stride_b,i),-133));
    sum=qrt_sm121_group16::sum_packed(carry,products);
    return qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
}
#if defined(__HIPCC__)
__device__ __forceinline__ wide::Products products(const Row& a,const Row& b){
    wide::F16x16 left{},right{};wide::I32x4 low_a{},low_b{};
#pragma unroll
    for(unsigned i=0u;i<16u;++i){left[i]=__builtin_bit_cast(_Float16,a.half[i]);right[i]=__builtin_bit_cast(_Float16,b.half[i]);}
#pragma unroll
    for(unsigned i=0u;i<4u;++i){low_a[i]=a.low[i];low_b[i]=b.low[i];}
    return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(left,right,wide::F32x8{}),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,low_a,false,low_b,wide::I32x8{},false)};
}
#endif
} // namespace qrt_sm121_compact_byte_core
#undef QRT_COMPACT_BYTE_INLINE
