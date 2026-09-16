#pragma once
#include "sm121_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_TWO_PASS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_TWO_PASS_INLINE inline
#endif
namespace qrt_sm121_two_pass_packed_group {
// Preconditions match prepared decoded QK: eligible BF16 words packed above
// their signed exponent, zero exponent -512, and zero/normal finite carry.
// Volatile operand reads keep the first pass's words out of the product pass.
// Products are consumed in small batches once the common exponent is known.
template<unsigned Chunk, unsigned LeftStride, unsigned RightStride>
QRT_TWO_PASS_INLINE bool accumulate(float carry,const volatile uint32_t* left,
    const volatile uint32_t* right,float* output) {
    static_assert(Chunk==1u || Chunk==2u || Chunk==4u || Chunk==8u);
    static_assert(LeftStride && RightStride);
    namespace f32=qrt_sm121_f32_carry;
    const uint32_t absolute=f32::bits(carry)&0x7fffffffu;
    int maximum=absolute?int(absolute>>23u)-127:-133;
    maximum=maximum>-133?maximum:-133;
#pragma unroll
    for(unsigned i=0u;i<16u;++i) {
        const int exponent=int(int16_t(left[i*LeftStride]))+int(int16_t(right[i*RightStride]));
        maximum=exponent>maximum?exponent:maximum;
    }
    if(maximum==-133 && !absolute){*output=0.0f;return true;}
    if(maximum<-101 || maximum>127)return false;
    const float scale=f32::alignment::from_bits(uint32_t(152-maximum)<<23u);
    uint32_t modulo=uint32_t(int32_t(carry*scale));
    bool first_negative=false;
#pragma unroll 1
    for(unsigned base=0u;base<16u;base+=Chunk) {
#pragma unroll
        for(unsigned item=0u;item<Chunk;++item) {
            const unsigned i=base+item;
            const uint32_t a=left[i*LeftStride],b=right[i*RightStride];
            const float product=f32::alignment::from_bits(a&0xffff0000u)*
                f32::alignment::from_bits(b&0xffff0000u);
            modulo+=uint32_t(int32_t(product*scale));
            if(!i)first_negative=((a^b)&0x80000000u)!=0u;
        }
    }
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,first_negative);
    return f32::normalize<0u>(sum.magnitude,sum.negative,maximum,output);
}
} // namespace qrt_sm121_two_pass_packed_group
#undef QRT_TWO_PASS_INLINE
