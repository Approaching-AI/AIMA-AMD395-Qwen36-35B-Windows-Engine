#pragma once
#include "sm121_narrow_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_BYTE_EXP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_BYTE_EXP_INLINE inline
#endif
namespace qrt_sm121_byte_exponents {
namespace narrow=qrt_sm121_narrow_f32_carry;
namespace f32=qrt_sm121_f32_carry;
struct Metadata { uint32_t deficits[4]; int maximum; };
static_assert(sizeof(Metadata)==20u);

// Complete K16 rows use the proved narrow BF16 domain and a nonzero
// exponent span <=31. Signed zeros have deficit63. Invalid rows are never
// admitted to the fast tile. An all-zero row is valid with maximum0.
QRT_BYTE_EXP_INLINE Metadata prepare(const uint16_t* words) {
    Metadata result{};bool valid=true;unsigned minimum=255u;
    for(unsigned i=0u;i<16u;++i){
        const auto x=words[i];valid&=narrow::eligible(x);
        if(x&0x7fffu){
            const unsigned exponent=(x>>7u)&255u;
            result.maximum=int(exponent)>result.maximum?int(exponent):result.maximum;
            minimum=exponent<minimum?exponent:minimum;
        }
    }
    if(result.maximum && unsigned(result.maximum)-minimum>31u)valid=false;
    for(unsigned i=0u;i<16u;++i){
        const auto x=words[i];
        const unsigned deficit=(x&0x7fffu)?unsigned(result.maximum)-((x>>7u)&255u):63u;
        result.deficits[i/4u]|=deficit<<(8u*(i%4u));
    }
    if(!valid)result.maximum=-1;
    return result;
}

// Each byte is <=126. Adding the local high bit before subtraction prevents
// inter-byte borrow. The resulting mask chooses the smaller low seven bits.
QRT_BYTE_EXP_INLINE uint32_t minimum_bytes(uint32_t a,uint32_t b) {
    const uint32_t high=((b|0x80808080u)-a)&0x80808080u;
    const uint32_t mask=high-(high>>7u);
    return (a&mask)|(b&~mask);
}

// Four independent byte additions cannot carry: each deficit is <=63.
// Nonzero pairs have sum <=62; any zero pair has sum >=63. Thus the minimum
// either gives the exact original paired exponent or establishes all zeros.
QRT_BYTE_EXP_INLINE int product_maximum(const Metadata& a,const Metadata& b) {
    const uint32_t x=minimum_bytes(a.deficits[0]+b.deficits[0],a.deficits[1]+b.deficits[1]);
    const uint32_t y=minimum_bytes(a.deficits[2]+b.deficits[2],a.deficits[3]+b.deficits[3]);
    const uint32_t z=minimum_bytes(x,y);
    const unsigned p=(z&255u)<((z>>8u)&255u)?(z&255u):((z>>8u)&255u);
    const unsigned q=((z>>16u)&255u)<(z>>24u)?((z>>16u)&255u):(z>>24u);
    const unsigned m=p<q?p:q;
    return m<63u?a.maximum+b.maximum-int(m)-254:-133;
}

// The unchanged K256 narrow-domain proof supplies the normal-carry range.
// These helpers let a GPU consume each exact product immediately, without
// keeping sixteen products live until their maximum is known.
QRT_BYTE_EXP_INLINE int alignment(float carry,int product_maximum) {
    const int carried=int((f32::bits(carry)&0x7fffffffu)>>23u)-127;
    const int maximum=product_maximum>carried?product_maximum:carried;
    return maximum>-89?maximum:-89;
}
QRT_BYTE_EXP_INLINE float finish(uint32_t modulo,bool first_negative,int maximum) {
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,first_negative);
    const unsigned shift=f32::leading(sum.magnitude);
    const uint32_t mantissa=((sum.magnitude<<shift)>>8u)&0x7fffffu;
    const int exponent=maximum+6-int(shift);
    const uint32_t encoded=(sum.negative?0x80000000u:0u)|(uint32_t(exponent+127)<<23u)|mantissa;
    return f32::alignment::from_bits(sum.magnitude?encoded:0u);
}
QRT_BYTE_EXP_INLINE float accumulate(float carry,const uint16_t* a,const uint16_t* b,
    const Metadata& am,const Metadata& bm) {
    const int maximum=alignment(carry,product_maximum(am,bm));
    const float scale=f32::alignment::from_bits(uint32_t(152-maximum)<<23u);
    uint32_t modulo=uint32_t(int32_t(carry*scale));
    for(unsigned i=0u;i<16u;++i){
        const float product=f32::alignment::from_bits(uint32_t(a[i])<<16u)*
            f32::alignment::from_bits(uint32_t(b[i])<<16u);
        modulo+=uint32_t(int32_t(product*scale));
    }
    return finish(modulo,((a[0]^b[0])&0x8000u)!=0u,maximum);
}
} // namespace qrt_sm121_byte_exponents
#undef QRT_BYTE_EXP_INLINE
