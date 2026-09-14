#ifndef QRT_SM121_SCALED_HALF_PRODUCTS_H
#define QRT_SM121_SCALED_HALF_PRODUCTS_H
#include "sm121_float_alignment.h"
#include "sm121_prepared_integer_pairs.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SCALED_HALF_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SCALED_HALF_INLINE inline
#endif
namespace qrt_sm121_scaled_half_products {
using Value=qrt_q1_moe_hawkeye::Value;
using AlignedSum=qrt_sm121_group16::AlignedSum;
struct Row { uint32_t pairs[8],control; };
static_assert(sizeof(Row)==36u);
QRT_SCALED_HALF_INLINE int unit(const Row& row){return int(int16_t(row.control));}

// A supported row is scaled by one exact power of two. Every nonzero BF16
// maps to a normal FP16 with its seven fraction bits intact. Unsupported
// rows retain original BF16 words in the same storage, marked by unit-32768.
QRT_SCALED_HALF_INLINE Row prepare(const uint16_t* input) {
    Row row{};unsigned maximum=0u,minimum=255u,nonzero=0u;bool valid=true;
    for(unsigned i=0u;i<16u;++i)if(input[i]&0x7fffu) {
        const unsigned exponent=(input[i]>>7u)&255u;
        maximum=exponent>maximum?exponent:maximum;minimum=exponent<minimum?exponent:minimum;
        valid=valid && exponent && exponent<255u;nonzero|=1u<<i;
    }
    valid=valid && (!nonzero || maximum-minimum<=29u);
    const int scale=valid?(nonzero?int(maximum)-142:-15):-32768;
    row.control=(nonzero<<16u)|uint16_t(scale);
    for(unsigned i=0u;i<16u;++i) {
        const uint16_t x=input[i];uint16_t encoded=x;
        if(valid)encoded=uint16_t((x&0x8000u)|((x&0x7fffu)?
            (unsigned(int((x>>7u)&255u)-112-scale)<<10u)|((x&127u)<<3u):0u));
        row.pairs[i/2u]|=uint32_t(encoded)<<(i%2u*16u);
    }
    return row;
}
QRT_SCALED_HALF_INLINE uint16_t original(const Row& row,unsigned i) {
    const uint16_t x=uint16_t(row.pairs[i/2u]>>(i%2u*16u));
    if(unit(row)==-32768)return x;
    if(!(x&0x7fffu))return uint16_t(x&0x8000u);
    return uint16_t((x&0x8000u)|(unsigned(int((x>>10u)&31u)+112+unit(row))<<7u)|((x&1023u)>>3u));
}
QRT_SCALED_HALF_INLINE float half_value(uint16_t x) {
    if(!(x&0x7fffu))return 0.0f;
    return qrt_sm121_float_alignment::from_bits((uint32_t(x&0x8000u)<<16u)|
        ((uint32_t((x>>10u)&31u)+112u)<<23u)|(uint32_t(x&1023u)<<13u));
}
template<bool High>
QRT_SCALED_HALF_INLINE float product(uint32_t left,uint32_t right) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    float result;
    if constexpr(High)
        asm("v_fma_mix_f32 %0, %1, %2, 0 op_sel:[1,1,0] op_sel_hi:[1,1,0]" : "=v"(result) : "v"(left),"v"(right));
    else
        asm("v_fma_mix_f32 %0, %1, %2, 0 op_sel:[0,0,0] op_sel_hi:[1,1,0]" : "=v"(result) : "v"(left),"v"(right));
    return result;
#else
    return half_value(uint16_t(left>>(High?16u:0u)))*half_value(uint16_t(right>>(High?16u:0u)));
#endif
}
template<bool AllNonzero>
QRT_SCALED_HALF_INLINE bool sum_products(Value carry,const Row& left,const Row& right,
    unsigned active,AlignedSum* output) {
    float products[16];uint32_t paired_maximum=0u;
#pragma unroll
    for(unsigned i=0u;i<8u;++i) {
        const uint32_t a=left.pairs[i],b=right.pairs[i];
        products[2u*i]=product<false>(a,b);products[2u*i+1u]=product<true>(a,b);
        uint32_t exponents=((a>>10u)&0x001f001fu)+((b>>10u)&0x001f001fu);
        if constexpr(!AllNonzero) {
            const unsigned bits=(active>>(2u*i))&3u;
            const uint32_t mask=((bits&1u)|((bits&2u)<<15u))*65535u;
            exponents&=mask;
        }
        paired_maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum,exponents);
    }
    const int half_maximum=int((paired_maximum&65535u)>(paired_maximum>>16u)?paired_maximum&65535u:paired_maximum>>16u);
    const int combined_unit=unit(left)+unit(right);
    int maximum=half_maximum-30+combined_unit;
    maximum=maximum>carry.exponent?maximum:carry.exponent;maximum=maximum>-133?maximum:-133;
    const int product_power=25-maximum+combined_unit;
    if(maximum<-101 || maximum>127 || carry.exponent>127 || product_power<-126 || product_power>127)return false;
    const float carry_scale=qrt_sm121_float_alignment::from_bits(unsigned(152-maximum)<<23u);
    const float product_scale=qrt_sm121_float_alignment::from_bits(unsigned(127+product_power)<<23u);
    uint32_t total=uint32_t(int32_t(qrt_q1_moe_hawkeye::value_to_float(carry)*carry_scale));
#pragma unroll
    for(unsigned i=0u;i<16u;++i)total+=uint32_t(int32_t(products[i]*product_scale));
    *output={qrt_sm121_group16::decode_modulo_sum(total,((left.pairs[0]^right.pairs[0])&0x8000u)!=0u),maximum};
    return true;
}
QRT_SCALED_HALF_INLINE Value accumulate(Value carry,const Row& left,const Row& right,unsigned* path=nullptr) {
    if(path)*path=0u;
    if(unit(left)!=-32768 && unit(right)!=-32768) {
        const unsigned active=(left.control&right.control)>>16u;
        if(!active) {
            const int maximum=carry.exponent>-133?carry.exponent:-133;
            const unsigned shift=unsigned(maximum-carry.exponent);
            const uint32_t magnitude=shift>=32u?0u:(carry.significand<<2u)>>shift;
            if(path)*path=1u;
            return qrt_sm121_canonical::normalize(magnitude,magnitude && carry.negative,maximum);
        }
        AlignedSum sum;
        const bool accepted=active==65535u?sum_products<true>(carry,left,right,active,&sum):sum_products<false>(carry,left,right,active,&sum);
        if(accepted) {
            if(path)*path=active==65535u?2u:3u;
            return qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
        }
    }
    Value values[17];values[0]=carry;
    for(unsigned i=0u;i<16u;++i)values[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(original(left,i),original(right,i),-133);
    return qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
}
} // namespace qrt_sm121_scaled_half_products
#undef QRT_SCALED_HALF_INLINE
#endif
