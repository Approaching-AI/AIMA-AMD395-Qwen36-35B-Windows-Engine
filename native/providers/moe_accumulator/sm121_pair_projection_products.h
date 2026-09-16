#pragma once
#include "sm121_pair_residue.h"
#include "sm121_group16_modulo.h"
#include "sm121_canonical_normalize.h"
#if defined(__HIPCC__)
#define QRT_PAIR_PRODUCTS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PAIR_PRODUCTS_INLINE inline
#endif

// Lossless per-pair units permit a core integer representation even when
// the complete K16 operand has a wider exponent range. Unsupported pairs
// retain their two original BF16 words. No runtime dispatcher uses this view.
namespace qrt_sm121_pair_projection_products {
namespace core=qrt_sm121_pair_residue;
using Pair=core::Pair;
using Value=qrt_q1_moe_hawkeye::Value;
struct Row { Pair pairs[8]; };
static_assert(sizeof(Row)==64u);
QRT_PAIR_PRODUCTS_INLINE bool valid(Pair p) {return (p.bytes&0x01000000u)!=0u;}
QRT_PAIR_PRODUCTS_INLINE unsigned unit(Pair p) {return (p.bytes>>16u)&255u;}
QRT_PAIR_PRODUCTS_INLINE Pair prepare_pair(uint16_t a,uint16_t b) {
    const unsigned ea=(a>>7u)&255u,eb=(b>>7u)&255u;
    const unsigned lo=ea<eb?ea:eb,hi=ea>eb?ea:eb;
    if(!lo || hi==255u || hi-lo>7u) return {uint32_t(a)|(uint32_t(b)<<16u),0u};
    Pair p=core::prepare((a&127u)|128u,ea-lo,(a&32768u)!=0u,
        (b&127u)|128u,eb-lo,(b&32768u)!=0u);
    p.bytes|=(lo<<16u)|0x01000000u;return p;
}
QRT_PAIR_PRODUCTS_INLINE Row prepare(const uint16_t* input) {
    Row row{};for(unsigned i=0;i<8;++i)row.pairs[i]=prepare_pair(input[2u*i],input[2u*i+1u]);return row;
}
QRT_PAIR_PRODUCTS_INLINE uint16_t original_pair(Pair p,unsigned i) {
    const uint16_t h=uint16_t(p.half>>(i*16u));
    if(!valid(p))return h;
    const unsigned exponent=unit(p)+((h>>10u)&31u)-22u;
    return uint16_t((h&0x8000u)|(exponent<<7u)|((h&1023u)>>3u));
}
QRT_PAIR_PRODUCTS_INLINE uint16_t original(const Row& row,unsigned i) {return original_pair(row.pairs[i/2u],i&1u);}
QRT_PAIR_PRODUCTS_INLINE int maximum(Pair a,Pair b) {
    if(valid(a)&&valid(b)) {
        const uint32_t e=((a.half>>10u)&0x001f001fu)+((b.half>>10u)&0x001f001fu);
        return int(unit(a)+unit(b))+int((e&65535u)>(e>>16u)?e&65535u:e>>16u)-298;
    }
    const auto x=qrt_q1_moe_hawkeye::multiply_bf16(original_pair(a,0u),original_pair(b,0u),-133);
    const auto y=qrt_q1_moe_hawkeye::multiply_bf16(original_pair(a,1u),original_pair(b,1u),-133);
    return x.exponent>y.exponent?x.exponent:y.exponent;
}
QRT_PAIR_PRODUCTS_INLINE int32_t integer(Pair p,unsigned i) {
    const uint16_t h=uint16_t(p.half>>(i*16u));
    const int32_t magnitude=int32_t(((h&1023u)>>3u)|128u)<<(((h>>10u)&31u)-22u);
    return h&32768u?-magnitude:magnitude;
}
QRT_PAIR_PRODUCTS_INLINE uint32_t aligned_original(Value value,int exponent) {
    const unsigned shift=unsigned(exponent-value.exponent);
    const uint32_t magnitude=shift>=32u?0u:(value.significand<<2u)>>shift;
    return value.negative?0u-magnitude:magnitude;
}
QRT_PAIR_PRODUCTS_INLINE uint32_t aligned(Pair a,Pair b,int exponent,bool* recovered=nullptr) {
    if(recovered)*recovered=false;
    if(valid(a)&&valid(b)) {
        const int shift=exponent-int(unit(a)+unit(b))+243;
        if(shift>=-11 && shift<=8) {
            // Metadata occupies the upper two bytes; the residue opcode
            // receives only the two stored operand bytes.
            const Pair ca{a.half,a.bytes&65535u},cb{b.half,b.bytes&65535u};
            float estimate;uint32_t residue;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
            estimate=core::estimate(ca,cb);residue=core::residue(ca,cb);
#else
            const int32_t exact=integer(a,0u)*integer(b,0u)+integer(a,1u)*integer(b,1u);
            estimate=float(exact);residue=uint32_t(exact)&255u;
#endif
            int32_t mathematical=0;
            if(core::recover(estimate,residue,&mathematical)) {
                if(recovered)*recovered=true;
                return shift<=0?uint32_t(mathematical)<<unsigned(-shift):core::aligned(mathematical,ca,cb,unsigned(shift));
            }
        }
    }
    return aligned_original(qrt_q1_moe_hawkeye::multiply_bf16(original_pair(a,0u),original_pair(b,0u),-133),exponent)+
        aligned_original(qrt_q1_moe_hawkeye::multiply_bf16(original_pair(a,1u),original_pair(b,1u),-133),exponent);
}
QRT_PAIR_PRODUCTS_INLINE Value accumulate(Value carry,const Row& a,const Row& b,unsigned* recovered=nullptr) {
    int exponent=carry.exponent>-133?carry.exponent:-133;
    for(unsigned i=0;i<8;++i){const int e=maximum(a.pairs[i],b.pairs[i]);exponent=e>exponent?e:exponent;}
    unsigned count=0;uint32_t total=aligned_original(carry,exponent);
    for(unsigned i=0;i<8;++i){bool used=false;total+=aligned(a.pairs[i],b.pairs[i],exponent,&used);count+=used;}
    const bool negative=((original(a,0u)^original(b,0u))&32768u)!=0u;
    const auto sum=qrt_sm121_group16::decode_modulo_sum(total,negative);
    if(recovered)*recovered=count;
    return qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,exponent);
}
}
#undef QRT_PAIR_PRODUCTS_INLINE
