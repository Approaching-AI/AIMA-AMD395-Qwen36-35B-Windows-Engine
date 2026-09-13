#ifndef QRT_SM121_TILED_PROJECTION_H
#define QRT_SM121_TILED_PROJECTION_H
#include "sm121_scalar_projection.h"

namespace qrt_sm121_tiled_projection {
constexpr unsigned threads=256u,lanes=4u,groups=threads/lanes,k_tile=64u;
using Value=qrt_q1_moe_hawkeye::Value;

__device__ __forceinline__ float rounded(float value) {
    const uint32_t bits=__float_as_uint(value);
    return qrt_sm121_float_alignment::from_bits((bits+0x7fffu+((bits>>16u)&1u))&0xffff0000u);
}
__global__ void mark_kernel(const unsigned* indices,unsigned count,unsigned cells,unsigned* mask) {
    const unsigned slot=blockIdx.x*blockDim.x+threadIdx.x;
    if (slot<count) {
        const unsigned cell=indices[slot];
        if (cell<cells) atomicOr(mask+cell/32u,1u<<(cell&31u));
    }
}
__device__ __forceinline__ bool selected(const unsigned* mask,unsigned rows,unsigned tokens,
    unsigned row,unsigned token) {
    if (row>=rows || token>=tokens) return false;
    const size_t cell=size_t(token)*rows+row;
    return (mask[cell/32u]>>(cell&31u))&1u;
}

// Sparse tiles keep several independent carries per subgroup and reuse the
// same original BF16 input/weight slabs across those selected cells. Dense
// tiles use the original four-lane dot without reserving unbounded carries.
template<unsigned Rows,unsigned Tokens,unsigned Capacity>
__global__ void replay_kernel(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_flags,const unsigned* input_flags,const unsigned* mask,
    float* output,unsigned rows,unsigned tokens,unsigned width) {
    static_assert(Rows==64u || Rows==128u);
    static_assert(Tokens==32u && Capacity==Rows*4u);
    constexpr unsigned carries=Capacity/groups;
    __shared__ unsigned count;
    __shared__ uint16_t candidates[Capacity];
    __shared__ uint32_t left[Tokens][33],right[Rows][33];
    const unsigned tid=threadIdx.x,lane=tid&(lanes-1u),group=tid/lanes;
    const unsigned first_row=blockIdx.x*Rows,first_token=blockIdx.y*Tokens;
    if (!tid) count=0u;
    __syncthreads();
    for (unsigned cell=tid;cell<Rows*Tokens;cell+=threads) {
        const unsigned row=first_row+cell%Rows,token=first_token+cell/Rows;
        if (selected(mask,rows,tokens,row,token)) {
            const unsigned slot=atomicAdd(&count,1u);
            if (slot<Capacity) candidates[slot]=uint16_t(cell);
        }
    }
    __syncthreads();
    if (!count) return;
    if (count>Capacity) {
        for (unsigned cell=group;cell<Rows*Tokens;cell+=groups) {
            const unsigned row=first_row+cell%Rows,token=first_token+cell/Rows;
            if (!selected(mask,rows,tokens,row,token)) continue;
            const float value=qrt_sm121_scalar_projection::validated_dot<lanes>(
                inputs+size_t(token)*width,weights+size_t(row)*width,width,
                input_flags[token] && weight_flags[row]);
            if (!lane) output[size_t(token)*rows+row]=rounded(value);
        }
        return;
    }
    Value carry[carries];
    unsigned local_row[carries],local_token[carries];
    bool valid[carries];
#pragma unroll
    for (unsigned slot=0u;slot<carries;++slot) {
        const unsigned index=slot*groups+group;
        const unsigned cell=index<count ? unsigned(candidates[index]) : 0u;
        local_row[slot]=cell%Rows;local_token[slot]=cell/Rows;
        carry[slot]={0u,-133,false};
        valid[slot]=index<count && weight_flags[first_row+local_row[slot]] && input_flags[first_token+local_token[slot]];
    }
    for (unsigned base=0u;base<width;base+=k_tile) {
        for (unsigned cell=tid;cell<Tokens*32u;cell+=threads) {
            const unsigned token=first_token+cell/32u,k=base+(cell%32u)*2u;
            uint16_t a=0u,b=0u;
            if (token<tokens && k<width) {
                a=inputs[size_t(token)*width+k];b=inputs[size_t(token)*width+k+1u];
            }
            left[cell/32u][cell%32u]=uint32_t(a)|(uint32_t(b)<<16u);
        }
        for (unsigned cell=tid;cell<Rows*32u;cell+=threads) {
            const unsigned row=first_row+cell/32u,k=base+(cell%32u)*2u;
            uint16_t a=0u,b=0u;
            if (row<rows && k<width) {
                a=weights[size_t(row)*width+k];b=weights[size_t(row)*width+k+1u];
            }
            right[cell/32u][cell%32u]=uint32_t(a)|(uint32_t(b)<<16u);
        }
        __syncthreads();
#pragma unroll
        for (unsigned begin=0u;begin<k_tile;begin+=16u) if (base+begin<width) {
#pragma unroll
            for (unsigned slot=0u;slot<carries;++slot) if (slot*groups+group<count) {
                qrt_sm121_float_subgroup::Product products[4];
#pragma unroll
                for (unsigned pair=0u;pair<2u;++pair) {
                    const unsigned k=(begin+lane*4u)/2u+pair;
                    const uint32_t a=left[local_token[slot]][k],b=right[local_row[slot]][k];
#pragma unroll
                    for (unsigned half=0u;half<2u;++half) {
                        const uint16_t x=uint16_t(a>>(half*16u)),y=uint16_t(b>>(half*16u));
                        const uint32_t original=uint32_t(x)|(uint32_t(y)<<16u);
                        if (!valid[slot]) products[pair*2u+half]={0.0f,original,512};
                        else {
                            const bool zero=!(x&0x7fffu) || !(y&0x7fffu);
                            products[pair*2u+half]={
                                qrt_sm121_float_alignment::from_bits(uint32_t(x)<<16u)*
                                qrt_sm121_float_alignment::from_bits(uint32_t(y)<<16u),original,
                                zero ? -133 : int((x>>7u)&255u)+int((y>>7u)&255u)-254};
                        }
                    }
                }
                carry[slot]=qrt_sm121_float_subgroup::accumulate<lanes>(carry[slot],products);
            }
        }
        __syncthreads();
    }
    if (!lane) {
#pragma unroll
        for (unsigned slot=0u;slot<carries;++slot) if (slot*groups+group<count) {
            const float value=qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry[slot]));
            output[size_t(first_token+local_token[slot])*rows+first_row+local_row[slot]]=rounded(value);
        }
    }
}

inline hipError_t mark(const unsigned* indices,unsigned count,unsigned cells,
    unsigned* mask,size_t mask_words,hipStream_t stream) {
    if (!indices || !mask || !cells || cells>16384u*8192u || count>cells || mask_words<(size_t(cells)+31u)/32u)
        return hipErrorInvalidValue;
    const auto status=hipMemsetAsync(mask,0,((size_t(cells)+31u)/32u)*sizeof(unsigned),stream);
    if (status!=hipSuccess || !count) return status;
    hipLaunchKernelGGL(mark_kernel,dim3((count+threads-1u)/threads),dim3(threads),0u,stream,indices,count,cells,mask);
    return hipGetLastError();
}

inline hipError_t launch(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_flags,const unsigned* input_flags,const unsigned* mask,
    size_t mask_words,float* output,unsigned rows,unsigned tokens,unsigned width,
    unsigned tile_rows,hipStream_t stream) {
    const size_t cells=size_t(rows)*tokens;
    if (!weights || !inputs || !weight_flags || !input_flags || !mask || !output ||
        !rows || rows>16384u || !tokens || tokens>8192u || !width || width>4096u || width%16u ||
        mask_words<(cells+31u)/32u || (tile_rows!=64u && tile_rows!=128u))
        return hipErrorInvalidValue;
    if (tile_rows==64u) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<64u,32u,256u>),
            dim3((rows+63u)/64u,(tokens+31u)/32u),dim3(threads),0u,stream,
            weights,inputs,weight_flags,input_flags,mask,output,rows,tokens,width);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<128u,32u,512u>),
            dim3((rows+127u)/128u,(tokens+31u)/32u),dim3(threads),0u,stream,
            weights,inputs,weight_flags,input_flags,mask,output,rows,tokens,width);
    }
    return hipGetLastError();
}
} // namespace qrt_sm121_tiled_projection
#endif
