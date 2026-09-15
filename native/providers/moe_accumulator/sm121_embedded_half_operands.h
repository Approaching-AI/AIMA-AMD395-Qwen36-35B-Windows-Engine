#pragma once
#include "sm121_scaled_half_products.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_EMBEDDED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_EMBEDDED_INLINE inline
#endif
namespace qrt_sm121_embedded_half {
namespace half=qrt_sm121_scaled_half_products;
// Every supported FP16 operand has three zero low fraction bits. Four
// operands leave twelve bits: store the same eight-bit biased unit and
// all-nonzero flag in each lane's two dwords. No operand bit is discarded.
// Unsupported groups keep the original BF16 payload and require the separate
// support bitmap; their metadata bits must never be interpreted as FP16.
struct Row { uint32_t pairs[8]; };
static_assert(sizeof(Row)==32u);
constexpr uint32_t payload_mask=0xfff8fff8u;
QRT_EMBEDDED_INLINE unsigned metadata(uint32_t first,uint32_t second) {
    return (first&7u)|((first>>13u)&56u)|((second&7u)<<6u);
}
QRT_EMBEDDED_INLINE bool prepare(const uint16_t* original,Row* output) {
    const auto source=half::prepare(original);
    const int unit=half::unit(source);
    if(unit==-32768) {
        for(unsigned i=0u;i<8u;++i)output->pairs[i]=source.pairs[i];
        return false;
    }
    const unsigned meta=unsigned(unit+142)|((source.control>>16u)==65535u?256u:0u);
    for(unsigned lane=0u;lane<4u;++lane) {
        output->pairs[2u*lane]=source.pairs[2u*lane]|(meta&7u)|((meta&56u)<<13u);
        output->pairs[2u*lane+1u]=source.pairs[2u*lane+1u]|((meta>>6u)&7u);
    }
    return true;
}
QRT_EMBEDDED_INLINE uint16_t original(const Row& row,unsigned i,bool supported) {
    const uint16_t stored=uint16_t(row.pairs[i/2u]>>(i%2u*16u));
    if(!supported)return stored;
    const uint16_t value=stored&0xfff8u;
    if(!(value&0x7fffu))return value&0x8000u;
    const unsigned lane=i/4u;
    const int unit=int(metadata(row.pairs[2u*lane],row.pairs[2u*lane+1u])&255u)-142;
    return uint16_t((value&0x8000u)|(unsigned(int((value>>10u)&31u)+112+unit)<<7u)|((value&1023u)>>3u));
}
QRT_EMBEDDED_INLINE unsigned flag_words(unsigned width) { return 1u+(width/16u+31u)/32u; }
} // namespace qrt_sm121_embedded_half
#undef QRT_EMBEDDED_INLINE
