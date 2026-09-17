#pragma once
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_GROUP_WINDOW_INLINE __host__ __device__ __forceinline__
#else
#define QRT_GROUP_WINDOW_INLINE inline
#endif
namespace qrt_sm121_group_window {
// One bit for each K16 group containing any nonzero BF16 operand. This keeps
// subnormal and exceptional encodings active; positive/negative zero are absent.
template<unsigned Groups,unsigned Stride=1u>
QRT_GROUP_WINDOW_INLINE uint32_t prepare(const uint32_t* pairs) {
    static_assert(Groups>=1u&&Groups<=8u&&Stride>=1u);
    uint32_t mask=0u;
    for(unsigned g=0u;g<Groups;++g){
        uint32_t any=0u;
        for(unsigned p=0u;p<8u;++p)any|=pairs[(g*8u+p)*Stride];
        mask|=uint32_t(bool(any&0x7fff7fffu))<<g;
    }
    return mask;
}
QRT_GROUP_WINDOW_INLINE unsigned first(uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return mask?unsigned(__ffs(mask)-1):0u;
#else
    unsigned n=0u;if(mask)while(!(mask&1u)){mask>>=1u;++n;}return n;
#endif
}
QRT_GROUP_WINDOW_INLINE unsigned end(uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return mask?32u-unsigned(__clz(mask)):0u;
#else
    unsigned n=0u;while(mask){mask>>=1u;++n;}return n;
#endif
}
QRT_GROUP_WINDOW_INLINE unsigned population(uint32_t mask) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return unsigned(__popc(mask));
#else
    unsigned n=0u;while(mask){mask&=mask-1u;++n;}return n;
#endif
}
}
#undef QRT_GROUP_WINDOW_INLINE
