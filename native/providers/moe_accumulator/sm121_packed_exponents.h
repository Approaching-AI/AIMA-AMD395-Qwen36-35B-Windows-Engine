#ifndef QRT_SM121_PACKED_EXPONENTS_H
#define QRT_SM121_PACKED_EXPONENTS_H
#include "sm121_float_alignment.h"
#if defined(__HIPCC__)
#define QRT_PACKED_EXP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PACKED_EXP_INLINE inline
#endif

namespace qrt_sm121_packed_exponents {
// Each five-bit field holds an exponent deficit clamped to15. Two such
// fields sum to at most30, so ordinary dword addition cannot carry into the
// next field. Three dwords hold all sixteen deficits. The original BF16
// values remain available whenever this summary cannot determine a maximum.
struct Metadata{uint32_t deficits[3];int maximum;};
static_assert(sizeof(Metadata)==16u);
struct Row{uint16_t original[16];Metadata metadata;};
static_assert(sizeof(Row)==48u);
QRT_PACKED_EXP_INLINE Metadata prepare(const uint16_t* words){
    Metadata result{};bool eligible=true;
    for(unsigned i=0u;i<16u;++i){
        const uint16_t value=words[i];
        eligible=eligible && qrt_sm121_float_alignment::eligible(value);
        const int exponent=(value&32767u)?int((value>>7u)&255u):0;
        result.maximum=exponent>result.maximum?exponent:result.maximum;
    }
    for(unsigned i=0u;i<16u;++i){
        const int exponent=int((words[i]>>7u)&255u);
        const unsigned gap=(words[i]&32767u)?unsigned(result.maximum-exponent):15u;
        const unsigned deficit=gap<15u?gap:15u;
        result.deficits[i/6u]|=deficit<<(i%6u*5u);
    }
    if(!eligible)result.maximum=-1;
    return result;
}
// If the minimum sum is below15, neither member of a minimizing pair was
// saturated or zero. Its original exponent sum is therefore exact. Every
// other pair has an equal or larger true deficit. A larger minimum declines
// without changing output; the caller computes the original paired maximum.
QRT_PACKED_EXP_INLINE bool maximum(const Metadata& left,const Metadata& right,int* output){
    if(left.maximum<0 || right.maximum<0)return false;
    if(!left.maximum || !right.maximum){*output=-133;return true;}
    const uint32_t sums[3]={left.deficits[0]+right.deficits[0],left.deficits[1]+right.deficits[1],left.deficits[2]+right.deficits[2]};
    unsigned minimum=30u;
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for(unsigned i=0u;i<16u;++i){
        const unsigned sum=(sums[i/6u]>>(i%6u*5u))&31u;
        minimum=sum<minimum?sum:minimum;
    }
    if(minimum>=15u)return false;
    *output=left.maximum+right.maximum-int(minimum)-254;return true;
}
}
#undef QRT_PACKED_EXP_INLINE
#endif
