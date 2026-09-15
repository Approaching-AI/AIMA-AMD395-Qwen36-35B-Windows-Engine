#pragma once
#include "sm121_scaled_half_products.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DOMINANT_HALF_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DOMINANT_HALF_INLINE inline
#endif
namespace qrt_sm121_dominant_half_products {
namespace half=qrt_sm121_scaled_half_products;
using Value=half::Value;

// Each supported nonzero operand has a normal FP16 exponent at most30.
// The original BF16 product exponent is therefore at most30+left_unit+
// right_unit. When the incoming carry dominates this bound and the -133
// floor, it is exactly the original K16 group's alignment exponent.
QRT_DOMINANT_HALF_INLINE bool eligible(Value carry,uint32_t left_control,uint32_t right_control) {
    const int left=int(int16_t(left_control)),right=int(int16_t(right_control));
    return left!=-32768 && right!=-32768 && ((left_control&right_control)>>16u) &&
        carry.exponent>=-133 && carry.exponent>=30+left+right;
}

// Caller has supported rows and the original group's alignment exponent.
// Normal FP16 products are exact FP32 values here: each operand retains
// only the seven BF16 fraction bits. Scaling reproduces the original
// bounded product alignment, with truncation toward zero.
template<unsigned Pairs>
QRT_DOMINANT_HALF_INLINE uint32_t products(const uint32_t* left,const uint32_t* right,int power) {
    uint32_t total=0u;
    // If the scale is below2^-126, every product aligns to integer zero.
    if(power>=-126) {
        const float scale=qrt_sm121_float_alignment::from_bits(unsigned(127+power)<<23u);
#pragma unroll
        for(unsigned i=0u;i<Pairs;++i) {
            total+=uint32_t(int32_t(half::product<false>(left[i],right[i])*scale));
            total+=uint32_t(int32_t(half::product<true>(left[i],right[i])*scale));
        }
    }
    return total;
}
QRT_DOMINANT_HALF_INLINE Value finish(Value carry,uint32_t total,bool first_negative,int maximum) {
    const unsigned shift=unsigned(maximum-carry.exponent);
    const uint32_t aligned=shift>=32u?0u:(carry.significand<<2u)>>shift;
    total+=carry.negative?0u-aligned:aligned;
    const auto sum=qrt_sm121_group16::decode_modulo_sum(total,first_negative);
    return qrt_sm121_canonical::normalize(sum.magnitude,sum.negative,maximum);
}
} // namespace qrt_sm121_dominant_half_products
#undef QRT_DOMINANT_HALF_INLINE
