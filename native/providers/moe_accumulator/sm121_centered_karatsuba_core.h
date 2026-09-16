#ifndef QRT_SM121_CENTERED_KARATSUBA_CORE_H
#define QRT_SM121_CENTERED_KARATSUBA_CORE_H
#include "sm121_biased_karatsuba_core.h"
#if defined(__HIPCC__)
#define QRT_CENTERED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CENTERED_INLINE inline
#endif

// Isolated signed16 integer core decomposition. HH and LL are exact IU8
// matrices. S=H+L-128 lies in[-256,254] and is exactly representable in FP16.
// Under the observed zero-C natural-DOT2 model its K16 forward error is at
// most85/256<1/2, permitting nearest-integer recovery of SS. The arithmetic
// implication is conditional; it does not prove the hardware premise.
namespace qrt_sm121_centered_karatsuba {
struct Prepared{uint16_t combined[16];int32_t sum;};
static_assert(sizeof(Prepared)==36u);
QRT_CENTERED_INLINE int digit(uint16_t value){
    const int high=int(value>>8u)-((value&0x8000u)?256:0);
    return high+int(value&255u)-128;
}
template<class Row>
QRT_CENTERED_INLINE Prepared prepare(const Row& row){
    Prepared result{};
    for(unsigned i=0u;i<16u;++i){
        const int value=digit(qrt_sm121_biased_karatsuba::core(row,i));
        result.combined[i]=qrt_sm121_biased_karatsuba::signed_half(value);
        result.sum+=value;
    }
    return result;
}
QRT_CENTERED_INLINE int64_t reconstruct(int32_t high,int32_t low,int32_t combined,int32_t left_sum,int32_t right_sum){
    const int64_t cross=int64_t(combined)-high-low+int64_t(left_sum+right_sum)*128+262144;
    return int64_t(high)*65536+cross*256+low;
}
#if defined(__HIPCC__)
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
using H16=_Float16 __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
struct Parts{I8 high,low;F8 combined;};
template<class Row>
__device__ __forceinline__ Parts products(const Row& left,const Row& right,const Prepared& a,const Prepared& b){
    I4 ah{},al{},bh{},bl{};H16 as{},bs{};
#pragma unroll
    for(unsigned i=0u;i<4u;++i){ah[i]=left.high[i];al[i]=left.low[i];bh[i]=right.high[i];bl[i]=right.low[i];}
#pragma unroll
    for(unsigned i=0u;i<16u;++i){as[i]=__builtin_bit_cast(_Float16,a.combined[i]);bs[i]=__builtin_bit_cast(_Float16,b.combined[i]);}
    return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,true,bh,I8{},false),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,false,bl,I8{},false),
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(as,bs,F8{})};
}
// Every supported combined dot is in[-2^20,2^20], so the native conversion
// cannot overflow. The caller still needs the conditional error<1/2 premise.
__device__ __forceinline__ int32_t nearest(float value){return __float2int_rn(value);}
#endif
}
#undef QRT_CENTERED_INLINE
#endif
