#ifndef QRT_SM121_BYTE_RESIDUE_CORE_H
#define QRT_SM121_BYTE_RESIDUE_CORE_H
#include "sm121_modular_integer_core.h"
#if defined(__HIPCC__)
#define QRT_BYTE_RESIDUE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_BYTE_RESIDUE_INLINE inline
#endif
// Isolated two-matrix hypothesis. One IU8 product supplies the exact low byte
// of each signed integer dot. FP16 estimates locate its congruence class ONLY
// under the explicit error<128 condition. Neither this lemma nor finite GPU
// comparisons prove that hardware condition for arbitrary inputs.
namespace qrt_sm121_byte_residue_core {
using Value=qrt_q1_moe_hawkeye::Value;
using AlignedSum=qrt_sm121_group16::AlignedSum;
namespace integer=qrt_sm121_integer_core;
struct Row {
    uint16_t original[18],half[16];
    int low[4]; uint32_t trailing[4];
    int unit; uint32_t exceptions,nonzero; int maximum;
};
static_assert(sizeof(Row)==116u);
template<unsigned Headroom>
QRT_BYTE_RESIDUE_INLINE void prepare(Row& row) {
    static_assert(Headroom==4u || Headroom==5u || Headroom==6u);
    integer::Row original{};
    for(unsigned i=0u;i<16u;++i)original.original[i]=row.original[i];
    integer::prepare(original);
    row.maximum=original.maximum;row.nonzero=original.nonzero;
    row.unit=original.unit<0?-1:!row.nonzero?127:row.maximum>int(Headroom+1u)?row.maximum-int(Headroom):1;
    row.exceptions=0u;row.original[16]=0u;bool eligible=true;
    for(unsigned word=0u;word<4u;++word) {
        uint32_t low=0u,trailing=0u;
        for(unsigned byte=0u;byte<4u;++byte) {
            const unsigned i=word*4u+byte;const uint16_t x=row.original[i];
            const int core=integer::signed_core(x,row.unit),shift=int((x>>7u)&255u)-row.unit;
            const unsigned magnitude=unsigned(core<0?-core:core);
            row.half[i]=qrt_sm121_modular_core::half_bits(magnitude,core<0);
            low|=(uint32_t(core)&255u)<<(byte*8u);
            trailing|=qrt_sm121_integer_parts::trailing_bits(uint16_t(core))<<(byte*8u);
            if((x&0x7fffu) && row.unit>=0 && shift<0 &&
                (-shift>=8 || ((128u|(x&127u))&((1u<<(-shift))-1u))))row.exceptions|=1u<<i;
            row.original[16]|=uint16_t(unsigned((x&0x8000u)!=0u)<<i);
            eligible=eligible && qrt_sm121_float_alignment::eligible(x);
        }
        row.low[word]=int(low);row.trailing[word]=trailing;
    }
    row.original[17]=uint16_t(eligible);
}
template<unsigned Headroom>
QRT_BYTE_RESIDUE_INLINE bool recover(float approximate,uint32_t residue,int64_t* result) {
    static_assert(Headroom==4u || Headroom==5u || Headroom==6u);
    constexpr int64_t max_core=int64_t(255)<<Headroom;
    constexpr int64_t maximum_dot=16*max_core*max_core;
    if(!(approximate>-float(maximum_dot+512) && approximate<float(maximum_dot+512)))return false;
    const int64_t truncated=int64_t(approximate);
    const uint32_t offset=(residue-uint32_t(truncated))&255u;
    if(offset==128u && approximate==float(truncated))return false;
    int64_t value=truncated+offset;
    if(offset>128u || (offset==128u && approximate<float(truncated)))value-=256;
    if(value < -maximum_dot || value > maximum_dot)return false;
    *result=value;return true;
}
QRT_BYTE_RESIDUE_INLINE uint32_t remainder_mask(const Row& left,const Row& right,unsigned shift) {
    if(!shift || shift>=30u)return 0u;
    const uint32_t threshold=shift*0x01010101u;uint32_t mask=0u;
    for(unsigned word=0u;word<4u;++word) {
        const uint32_t high=~(left.trailing[word]+right.trailing[word]+0x80808080u-threshold)&0x80808080u;
        mask|=((((high>>7u)*0x01020408u)>>24u)&15u)<<(word*4u);
    }
    return mask;
}
QRT_BYTE_RESIDUE_INLINE int paired_maximum(const Row& left,const Row& right,int maximum) {
    const uint32_t nonzero=left.nonzero&right.nonzero;
    for(unsigned pair=0u;pair<8u;++pair) {
        const uint32_t a=uint32_t(left.original[2u*pair])|(uint32_t(left.original[2u*pair+1u])<<16u);
        const uint32_t b=uint32_t(right.original[2u*pair])|(uint32_t(right.original[2u*pair+1u])<<16u);
        const uint32_t sum=((a>>7u)&0x00ff00ffu)+((b>>7u)&0x00ff00ffu);
        const int first=(nonzero&(1u<<(2u*pair)))?int(sum&65535u)-254:-133;
        const int second=(nonzero&(2u<<(2u*pair)))?int(sum>>16u)-254:-133;
        maximum=first>maximum?first:maximum;maximum=second>maximum?second:maximum;
    }
    return maximum;
}
// Original integer-core compensation, with the narrower decomposition and
// original BF16 exception products. Exact-lattice rows skip all pair replay;
// otherwise only the original exception/remainder mask is visited.
QRT_BYTE_RESIDUE_INLINE bool sum(Value carry,const Row& left,const Row& right,int64_t mathematical,AlignedSum* output) {
    if(left.unit<0 || right.unit<0)return false;
    const uint32_t exceptions=(left.exceptions|right.exceptions)&left.nonzero&right.nonzero;
    if(!exceptions && qrt_sm121_integer_parts::sum_exact_integer_product(carry,mathematical,
        left.unit,left.maximum,right.unit,right.maximum,output,left.trailing,right.trailing))return true;
    int maximum=carry.exponent>-133?carry.exponent:-133;
    if(left.maximum+right.maximum-254>maximum)maximum=paired_maximum(left,right,maximum);
    const int shift=maximum-(left.unit+right.unit-254)-11;
    if(shift < -25 && mathematical)return false;
    const uint32_t loss=shift>0?remainder_mask(left,right,unsigned(shift)):0u;
    uint32_t pending=loss|exceptions;int64_t discarded=0,corrections=0;
    while(pending) {
        const unsigned i=integer::first_bit(pending);pending&=pending-1u;
        const uint16_t a=left.original[i],b=right.original[i];
        const int ac=integer::signed_core(a,left.unit),bc=integer::signed_core(b,right.unit);
        const uint32_t magnitude=uint32_t(ac<0?-ac:ac)*uint32_t(bc<0?-bc:bc);
        const bool negative=((a^b)&0x8000u)!=0u;
        if(loss&(1u<<i)) {
            const uint32_t remainder=magnitude&((1u<<shift)-1u);
            discarded+=negative?-int64_t(remainder):int64_t(remainder);
        }
        if(exceptions&(1u<<i)) {
            const auto original=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
            const unsigned original_shift=unsigned(maximum-original.exponent);
            const uint32_t aligned=original_shift>=32u?0u:(original.significand<<2u)>>original_shift;
            const uint32_t core_aligned=!magnitude?0u:shift>=30?0u:shift>0?magnitude>>shift:uint32_t(uint64_t(magnitude)<<(-shift));
            const int64_t difference=int64_t(aligned)-core_aligned;
            corrections+=negative?-difference:difference;
        }
    }
    const int64_t compensated=mathematical-discarded;
    const int64_t products=!compensated || shift>=30?0:shift<=0?compensated*(int64_t(1)<<(-shift)):
        compensated<0?-int64_t(uint64_t(-compensated)>>shift):int64_t(uint64_t(compensated)>>shift);
    const unsigned carry_shift=unsigned(maximum-carry.exponent);
    const uint32_t aligned=carry_shift>=32u?0u:(carry.significand<<2u)>>carry_shift;
    const int64_t total=products+corrections+(carry.negative?-int64_t(aligned):int64_t(aligned));
    *output={{uint32_t(total<0?-total:total),total<0},maximum};return true;
}
#if defined(__HIPCC__)
using F16x16=_Float16 __attribute__((ext_vector_type(16)));
using F32x8=float __attribute__((ext_vector_type(8)));
using I32x4=int __attribute__((ext_vector_type(4)));
using I32x8=int __attribute__((ext_vector_type(8)));
struct Products { F32x8 approximate; I32x8 residue; };
__device__ __forceinline__ Products products(const Row& left,const Row& right) {
    F16x16 a{},b{};I32x4 low_a{},low_b{};
#pragma unroll
    for(unsigned i=0u;i<16u;++i){a[i]=__builtin_bit_cast(_Float16,left.half[i]);b[i]=__builtin_bit_cast(_Float16,right.half[i]);}
#pragma unroll
    for(unsigned i=0u;i<4u;++i){low_a[i]=left.low[i];low_b[i]=right.low[i];}
    return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a,b,F32x8{}),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,low_a,false,low_b,I32x8{},false)};
}
#endif
}
#undef QRT_BYTE_RESIDUE_INLINE
#endif
