#pragma once
#include "sm121_exp2_native_delta.h"

#if defined(__HIPCC__)
#define QRT_EXP_REDUCED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_EXP_REDUCED_INLINE inline
#endif

// Isolated derived representation. A fractional index shares one correction
// over negative inputs with magnitudes [1,126). Derive it at -(1+fraction),
// then audit EVERY input in the admitted domain. Any unequal native/source
// correction marks that fraction as an original-table escape for all inputs.
// No SFU periodicity, error tolerance or portable native table is assumed.
namespace qrt_sm121_exp2_reduced_delta {
namespace full=qrt_sm121_exp2_native_delta;
namespace source=qrt_sm121_exp2_interpolated;
constexpr uint32_t magnitude_begin=0x3f800000u,magnitude_end=0x42fc0000u;
constexpr uint32_t fractions=0x800000u;
constexpr size_t packed_bytes=fractions/4u,escape_words=fractions/32u;
QRT_EXP_REDUCED_INLINE bool admitted(uint32_t input){
    const uint32_t m=input&0x7fffffffu;
    return (input>>31u) && m>=magnitude_begin && m<magnitude_end;
}
// Caller establishes the magnitude domain before this shift. The exponent
// is in [0,6]; the fraction is an exact integer in units of 2^-23.
QRT_EXP_REDUCED_INLINE uint32_t fractional_index(uint32_t magnitude){
    const unsigned exponent=(magnitude>>23u)-127u;
    return ((0x800000u|(magnitude&0x7fffffu))<<exponent)&0x7fffffu;
}
#if defined(__HIPCC__)
__device__ __forceinline__ float evaluate(const unsigned char* original,
    const unsigned char* packed,float argument){
    const uint32_t input=qrt_sm121_exp2::bits(argument);
    if(!admitted(input))return source::evaluate(original,argument);
    const unsigned correction=full::code(packed,fractional_index(input&0x7fffffffu));
    if(correction==3u)return source::evaluate(original,argument);
    return qrt_sm121_exp2::value(full::apply(qrt_sm121_exp2::bits(full::native_exp(argument)),correction));
}
__global__ void build(const unsigned char* original,unsigned char* packed){
    for(size_t byte=size_t(blockIdx.x)*blockDim.x+threadIdx.x;byte<packed_bytes;byte+=size_t(gridDim.x)*blockDim.x){
        unsigned word=0u;
#pragma unroll
        for(unsigned part=0u;part<4u;++part){
            const uint32_t magnitude=magnitude_begin+uint32_t(byte*4u+part);
            const float argument=qrt_sm121_exp2::value(0x80000000u|magnitude);
            word|=full::encode(qrt_sm121_exp2::bits(full::native_exp(argument)),
                source::decode(original,magnitude-source::begin))<<(part*2u);
        }
        packed[byte]=static_cast<unsigned char>(word);
    }
}
// Packed corrections are immutable during discovery. A separate monotone
// atomic bitmap avoids read/write races between different integer classes.
__global__ void find_escapes(const unsigned char* original,const unsigned char* packed,
    unsigned* escapes){
    for(uint32_t offset=blockIdx.x*blockDim.x+threadIdx.x;offset<magnitude_end-magnitude_begin;offset+=gridDim.x*blockDim.x){
        const uint32_t magnitude=magnitude_begin+offset;
        const float argument=qrt_sm121_exp2::value(0x80000000u|magnitude);
        if(qrt_sm121_exp2::bits(evaluate(original,packed,argument))!=source::decode(original,magnitude-source::begin)){
            const uint32_t fraction=fractional_index(magnitude);
            atomicOr(escapes+fraction/32u,1u<<(fraction%32u));
        }
    }
}
__global__ void apply_escapes(unsigned char* packed,const unsigned* escapes){
    for(size_t byte=size_t(blockIdx.x)*blockDim.x+threadIdx.x;byte<packed_bytes;byte+=size_t(gridDim.x)*blockDim.x){
        unsigned value=packed[byte];
#pragma unroll
        for(unsigned part=0u;part<4u;++part){
            const uint32_t fraction=uint32_t(byte*4u+part);
            if(escapes[fraction/32u]&(1u<<(fraction%32u)))value|=3u<<(part*2u);
        }
        packed[byte]=static_cast<unsigned char>(value);
    }
}
#endif
} // namespace qrt_sm121_exp2_reduced_delta
#undef QRT_EXP_REDUCED_INLINE
