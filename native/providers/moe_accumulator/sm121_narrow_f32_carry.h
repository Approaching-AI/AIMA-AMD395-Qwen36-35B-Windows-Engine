#pragma once
#include "sm121_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_NARROW_INLINE __host__ __device__ __forceinline__
#else
#define QRT_NARROW_INLINE inline
#endif

// Exact K256 specialization, entered only when every operand is signed zero
// or normal BF16 with unbiased exponent in [-32,32]. A nonzero K16 product
// has exponent at least -64. Its smallest nonzero aligned sum is 2^(-64-25),
// so all nonzero carried endpoints have exponent >= -89. An all-zero group
// preserves that endpoint. Absolute growth over 256 products stays < 2^74.
// Thus all scale factors, conversions and normalized carries are in the
// original admitted domain. The -89 floor changes only an all-zero sum.
namespace qrt_sm121_narrow_f32_carry {
namespace original=qrt_sm121_f32_carry;
namespace alignment=qrt_sm121_float_alignment;
QRT_NARROW_INLINE bool eligible(uint16_t x){
    const unsigned exponent=(x>>7u)&255u;
    return !(x&0x7fffu) || (exponent>=95u && exponent<=159u);
}
QRT_NARROW_INLINE float accumulate(float carry,const alignment::Group& group){
    const int carry_exponent=int((original::bits(carry)&0x7fffffffu)>>23u)-127;
    int maximum=group.maximum>carry_exponent?group.maximum:carry_exponent;
    maximum=maximum>-89?maximum:-89;
    const float scale=alignment::from_bits(uint32_t(152-maximum)<<23u);
    uint32_t modulo=uint32_t(int32_t(carry*scale));
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for(unsigned i=0u;i<16u;++i)modulo+=uint32_t(int32_t(group.products[i]*scale));
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,group.first_negative);
    const unsigned shift=original::leading(sum.magnitude);
    const uint32_t mantissa=((sum.magnitude<<shift)>>8u)&0x7fffffu;
    const int exponent=maximum+6-int(shift);
    const uint32_t encoded=(sum.negative?0x80000000u:0u)|(uint32_t(exponent+127)<<23u)|mantissa;
    return alignment::from_bits(sum.magnitude?encoded:0u);
}
} // namespace qrt_sm121_narrow_f32_carry
#undef QRT_NARROW_INLINE
