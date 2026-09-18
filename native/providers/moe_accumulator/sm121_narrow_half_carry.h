#pragma once
#include "sm121_scaled_half_products.h"
#include "sm121_narrow_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_NARROW_HALF_INLINE __host__ __device__ __forceinline__
#else
#define QRT_NARROW_HALF_INLINE inline
#endif

// Whole-dot admission: K<=8192, signed zeros or BF16 exponents95..159,
// and every K16 operand has a lossless normal-half encoding. Products have
// exponent>=-64; a nonzero ordered carry is at least2^-89. Absolute growth
// through8192 products is below2^79. Thus every nonzero carry is normal and
// finite, and 2^(25-maximum) is normal. For nonempty groups the half-product
// scale exponent is <=53. A scale below2^-126 contributes only integer zero.
// These are admission preconditions, not permission to change K16 ordering.
namespace qrt_sm121_narrow_half_carry {
namespace half=qrt_sm121_scaled_half_products;
namespace f32=qrt_sm121_f32_carry;
namespace narrow=qrt_sm121_narrow_f32_carry;
using Row=half::Row;

QRT_NARROW_HALF_INLINE bool eligible(const Row& row) {
    if(half::unit(row)==-32768)return false;
    for(unsigned i=0u;i<16u;++i)if(!narrow::eligible(half::original(row,i)))return false;
    return true;
}
QRT_NARROW_HALF_INLINE float normalize(uint32_t magnitude,bool negative,int maximum) {
    const unsigned shift=f32::leading(magnitude);
    const uint32_t mantissa=((magnitude<<shift)>>8u)&0x7fffffu;
    const int exponent=maximum+6-int(shift);
    const uint32_t encoded=(negative?0x80000000u:0u)|(uint32_t(exponent+127)<<23u)|mantissa;
    return f32::alignment::from_bits(magnitude?encoded:0u);
}

// Host/device scalar oracle for the same admitted half representation. The
// device projection below distributes these sixteen products across four lanes.
QRT_NARROW_HALF_INLINE float accumulate(float carry,const Row& left,const Row& right) {
    const unsigned active=(left.control&right.control)>>16u;
    if(!active)return carry;
    float products[16];int half_maximum=0;
    for(unsigned i=0u;i<8u;++i){
        const uint32_t a=left.pairs[i],b=right.pairs[i];
        products[2u*i]=half::product<false>(a,b);products[2u*i+1u]=half::product<true>(a,b);
        for(unsigned j=0u;j<2u;++j)if(active&(1u<<(2u*i+j))){
            const int e=int((a>>(16u*j+10u))&31u)+int((b>>(16u*j+10u))&31u);
            half_maximum=e>half_maximum?e:half_maximum;
        }
    }
    const int unit=half::unit(left)+half::unit(right);
    int maximum=half_maximum-30+unit;
    const int carry_exponent=int((f32::bits(carry)&0x7fffffffu)>>23u)-127;
    maximum=maximum>carry_exponent?maximum:carry_exponent;
    maximum=maximum>-89?maximum:-89;
    const float carry_scale=f32::alignment::from_bits(uint32_t(152-maximum)<<23u);
    uint32_t modulo=uint32_t(int32_t(carry*carry_scale));
    const int power=25-maximum+unit;
    if(power>=-126){
        const float scale=f32::alignment::from_bits(uint32_t(127+power)<<23u);
        for(unsigned i=0u;i<16u;++i)modulo+=uint32_t(int32_t(products[i]*scale));
    }
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,((left.pairs[0]^right.pairs[0])&0x8000u)!=0u);
    return normalize(sum.magnitude,sum.negative,maximum);
}
} // namespace qrt_sm121_narrow_half_carry
#undef QRT_NARROW_HALF_INLINE
