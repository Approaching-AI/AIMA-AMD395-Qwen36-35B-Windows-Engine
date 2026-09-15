#pragma once
#include "sm121_integer_core.h"
#include "sm121_canonical_normalize.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DOMINANT_CORE_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DOMINANT_CORE_INLINE inline
#endif

// Exact K16 integer matrix products, kept as two signed 32-bit components.
// Their mathematical sum is high*65536+lower; lower need not be a radix digit.
// Inputs come from core::prepare and four exact signed/unsigned IU8 dots.
namespace qrt_sm121_dominant_integer_core {
namespace core=qrt_sm121_integer_core;
using Value=qrt_q1_moe_hawkeye::Value;
struct Product { int32_t high,lower; };
static_assert(sizeof(Product)==8u);
constexpr int32_t lower_bound=16*128*255*2*256+16*255*255;
constexpr int32_t discarded_bound=16*65535;
static_assert(lower_bound+discarded_bound<INT32_MAX);
static_assert(uint64_t(16)*qrt_sm121_group16::kMaxAlignedProduct<INT32_MAX);

QRT_DOMINANT_CORE_INLINE Product combine(int32_t hh,int32_t hl,int32_t lh,int32_t ll) {
    // Each cross dot is bounded by16*128*255; this multiply/add fits int32.
    return {hh,(hl+lh)*256+ll};
}
QRT_DOMINANT_CORE_INLINE int64_t mathematical(Product product) {
    return int64_t(product.high)*65536+product.lower;
}

// If the actual carry dominates both row maxima, it fixes the original
// alignment without scanning the sixteen paired exponents. Normalized row
// units then put the alignment shift at least3. Up to16, the high component
// is divisible by2^shift. Remove each discarded signed remainder from lower,
// divide that exact multiple, then add all terms modulo2^32. Original BF16
// exception corrections remain; the established bounded decoder handles the
// positive/negative overlap above INT32_MAX. No prediction is involved.
// Rejection leaves both outputs unchanged. A caller retains the full original
// path for unsupported rows, non-dominant carries and other shift ranges.
QRT_DOMINANT_CORE_INLINE bool accumulate(Value carry,const core::Row& left,
    const core::Row& right,Product product,Value* output,unsigned* replayed_pairs=nullptr) {
    if(!output || carry.significand>0xffffffu || carry.exponent < -101 || carry.exponent > 127 ||
        left.unit<1 || right.unit<1 || left.maximum!=left.unit+7 || right.maximum!=right.unit+7 ||
        carry.exponent < left.maximum+right.maximum-254) return false;
    const int shift=carry.exponent-(left.unit+right.unit-254)-11;
    if(shift<3 || shift>16) return false;
    const unsigned distance=unsigned(shift),mask=(1u<<distance)-1u;
    const unsigned losses=core::remainder_mask(left,right,distance);
    const unsigned exceptions=(left.exceptions|right.exceptions)&left.nonzero&right.nonzero;
    unsigned pending=losses|exceptions,pairs=0u;
    int32_t discarded=0,corrections=0;
    while(pending) {
        const unsigned i=core::first_bit(pending);pending&=pending-1u;++pairs;
        const uint16_t a=left.original[i],b=right.original[i];
        const int ac=core::signed_core(a,left.unit),bc=core::signed_core(b,right.unit);
        const uint32_t magnitude=uint32_t(ac<0?-ac:ac)*uint32_t(bc<0?-bc:bc);
        const bool negative=((a^b)&0x8000u)!=0u;
        if(losses&(1u<<i)) {
            const int32_t remainder=int32_t(magnitude&mask);
            discarded+=negative?-remainder:remainder;
        }
        if(exceptions&(1u<<i)) {
            const auto original=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
            const unsigned delta=unsigned(carry.exponent-original.exponent);
            const uint32_t aligned=delta>=32u?0u:(original.significand<<2u)>>delta;
            const int32_t difference=int32_t(aligned)-int32_t(magnitude>>distance);
            corrections+=negative?-difference:difference;
        }
    }
    // Divisibility follows from the exact matrix product minus its individual
    // signed remainders, because high*65536 is divisible throughout this range.
    const int32_t numerator=product.lower-discarded;
    const int32_t quotient=numerator<0?-int32_t(uint32_t(-numerator)>>distance):
        int32_t(uint32_t(numerator)>>distance);
    uint32_t modulo=(uint32_t(product.high)<<(16u-distance))+uint32_t(quotient)+uint32_t(corrections);
    const uint32_t aligned_carry=carry.significand<<2u;
    modulo+=carry.negative?0u-aligned_carry:aligned_carry;
    const auto sum=qrt_sm121_group16::decode_modulo_sum(modulo,((left.original[0]^right.original[0])&0x8000u)!=0u);
    *output=qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,carry.exponent);
    if(replayed_pairs)*replayed_pairs=pairs;
    return true;
}
} // namespace qrt_sm121_dominant_integer_core
#undef QRT_DOMINANT_CORE_INLINE
