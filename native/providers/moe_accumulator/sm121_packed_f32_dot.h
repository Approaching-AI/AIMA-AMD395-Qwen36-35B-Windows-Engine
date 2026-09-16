#pragma once
#include "sm121_f32_carry.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PACKED_F32_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PACKED_F32_INLINE inline
#endif

namespace qrt_sm121_packed_f32_dot {
// The caller validates every original BF16 operand. A rejected call leaves
// output untouched; its caller restarts the complete original dot. Keeping
// FP32 endpoints is valid only while each original K16 endpoint is normal
// or zero. No shared operand representation or arithmetic order changes.
template<unsigned Width,unsigned Columns>
QRT_PACKED_F32_INLINE bool try_dot(const uint32_t* left,const uint32_t* right,
    unsigned column,bool valid,float* output) {
    static_assert(Width && Width%16u==0u && Columns);
    if(!valid)return false;
    float carry=0.0f;
    for(unsigned base=0u;base<Width;base+=16u) {
        qrt_sm121_float_alignment::Group group;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for(unsigned i=0u;i<16u;i+=2u) {
            const uint32_t a=left[(base+i)/2u],b=right[((base+i)/2u)*Columns+column];
            group.set(i,uint16_t(a),uint16_t(b));
            group.set(i+1u,uint16_t(a>>16u),uint16_t(b>>16u));
        }
        float next;
        if(!qrt_sm121_f32_carry::accumulate<0u>(carry,group,&next))return false;
        carry=next;
    }
    *output=carry;
    return true;
}
}
#undef QRT_PACKED_F32_INLINE
