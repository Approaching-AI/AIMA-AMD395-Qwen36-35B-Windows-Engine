#pragma once
#include "sm121_compact_integer_dot4.h"
#include "sm121_narrow_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_COMPACT_MATRIX_INLINE __host__ __device__ __forceinline__
#else
#define QRT_COMPACT_MATRIX_INLINE inline
#endif

namespace qrt_sm121_compact_matrix_group {
namespace compact=qrt_sm121_compact_integer_dot4;
namespace f32=qrt_sm121_f32_carry;
struct Row { compact::Row encoded;uint32_t trailing[2]; };
static_assert(sizeof(Row)==60u);
QRT_COMPACT_MATRIX_INLINE Row prepare(const uint16_t* input){
    Row result{};result.encoded=compact::prepare(input);
    if(compact::unit(result.encoded))for(unsigned i=0u;i<16u;++i){
        const uint16_t value=compact::word(result.encoded,i);
        const unsigned trailing=value?qrt_sm121_integer_parts::trailing_bits(value):15u;
        result.trailing[i/8u]|=trailing<<(4u*(i%8u));
    }
    return result;
}
QRT_COMPACT_MATRIX_INLINE bool divisible(const Row& a,const Row& b,unsigned shift){
    if(shift>=30u)return false;
    const uint32_t threshold=shift*0x01010101u;
    for(unsigned i=0u;i<2u;++i){
        const uint32_t even=(a.trailing[i]&0x0f0f0f0fu)+(b.trailing[i]&0x0f0f0f0fu);
        const uint32_t odd=((a.trailing[i]>>4u)&0x0f0f0f0fu)+((b.trailing[i]>>4u)&0x0f0f0f0fu);
        if(((even+0x80808080u-threshold)&0x80808080u)!=0x80808080u ||
           ((odd+0x80808080u-threshold)&0x80808080u)!=0x80808080u)return false;
    }
    return true;
}
QRT_COMPACT_MATRIX_INLINE int product_maximum(const Row& a,const Row& b){
    const unsigned active=compact::nonzero(a.encoded)&compact::nonzero(b.encoded);
    uint32_t maximum=0u;
    for(unsigned i=0u;i<4u;++i){
        const uint32_t x=a.encoded.exponents[i],y=b.encoded.exponents[i];
        uint32_t even=(x&0x00ff00ffu)+(y&0x00ff00ffu);
        uint32_t odd=((x>>8u)&0x00ff00ffu)+((y>>8u)&0x00ff00ffu);
        const unsigned live=(active>>(4u*i))&15u;
        even&=((live&1u?65535u:0u)|(live&4u?0xffff0000u:0u));
        odd&=((live&2u?65535u:0u)|(live&8u?0xffff0000u:0u));
        maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(maximum,even);
        maximum=qrt_sm121_prepared_integer_pairs::maximum_pair(maximum,odd);
    }
    return active?int((maximum&65535u)>(maximum>>16u)?maximum&65535u:maximum>>16u)-254:-133;
}
QRT_COMPACT_MATRIX_INLINE float normalize(uint32_t magnitude,bool negative,int maximum){
    const unsigned shift=f32::leading(magnitude);
    const uint32_t mantissa=((magnitude<<shift)>>8u)&0x7fffffu;
    const int exponent=maximum+6-int(shift);
    return f32::alignment::from_bits(magnitude?((negative?0x80000000u:0u)|
        (uint32_t(exponent+127)<<23u)|mantissa):0u);
}

// The caller certifies complete signed-zero/exponent95..159 rows and K<=8192.
// Mathematical is the exact signed16 dot reconstructed from four IU8 matrices.
// A rejected group leaves output untouched and uses the original narrow group.
QRT_COMPACT_MATRIX_INLINE bool accumulate(float carry,const Row& a,const Row& b,
    int64_t mathematical,float* output){
    if(!compact::unit(a.encoded)||!compact::unit(b.encoded))return false;
    if(!(compact::nonzero(a.encoded)&compact::nonzero(b.encoded))){*output=carry;return true;}
    int maximum=product_maximum(a,b);
    const int exponent=int((f32::bits(carry)&0x7fffffffu)>>23u)-127;
    maximum=maximum>exponent?maximum:exponent;maximum=maximum>-89?maximum:-89;
    const int shift=maximum-(int(compact::unit(a.encoded)+compact::unit(b.encoded))-254)-11;
    if(shift < -25 || shift>=30 || (shift>0&&!divisible(a,b,unsigned(shift))))return false;
    const int64_t aligned=shift<=0?mathematical*(int64_t(1)<<unsigned(-shift)):
        mathematical<0?-int64_t(uint64_t(-mathematical)>>unsigned(shift)):
        int64_t(uint64_t(mathematical)>>unsigned(shift));
    const float scale=f32::alignment::from_bits(uint32_t(152-maximum)<<23u);
    const uint32_t modulo=uint32_t(aligned)+uint32_t(int32_t(carry*scale));
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,
        ((a.encoded.pairs[0]^b.encoded.pairs[0])&0x8000u)!=0u);
    *output=normalize(sum.magnitude,sum.negative,maximum);return true;
}

// The caller passes a set-bit rank. Five population-count partitions select
// its source lane without an LDS queue or a loop proportional to the rank.
QRT_COMPACT_MATRIX_INLINE unsigned select_bit(uint32_t mask,unsigned rank){
    unsigned index=0u;
    for(unsigned width=16u;width;width>>=1u){
        const uint32_t low=mask&((uint32_t(1u)<<width)-1u);
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        const unsigned count=__popc(low);
#else
        const unsigned count=unsigned(__builtin_popcount(low));
#endif
        if(rank>=count){rank-=count;mask>>=width;index+=width;}
        else mask=low;
    }
    return index;
}
} // namespace qrt_sm121_compact_matrix_group
#undef QRT_COMPACT_MATRIX_INLINE
