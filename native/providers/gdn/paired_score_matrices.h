#pragma once
#include "blackwell_scalar_matrices.h"

// Isolated complete score producer. Adjacent value heads use the same Q/K
// head; compute its original raw dot once and apply both original gates.
// Callers own exactly count*32*64 score words; no padded query is stored.
namespace qrt_fla_paired_score {
namespace scalar=qrt_fla_blackwell_scalar;
constexpr unsigned rows=8u,columns=32u,pairs=64u,threads=256u;
__global__ void kernel(const uint16_t* q,const uint16_t* k,const float* g,
    uint16_t* scores,unsigned count,const unsigned char* table) {
    __shared__ uint32_t queries[rows][65],keys[pairs][columns];
    __shared__ unsigned unsupported;
    const unsigned chunk=blockIdx.z/(64u/rows),first_row=(blockIdx.z%(64u/rows))*rows;
    const unsigned offset=chunk*64u,valid=min(64u,count-offset);
    const unsigned first_column=blockIdx.x*columns,qk_head=blockIdx.y;
    const unsigned row=first_row+threadIdx.x/columns,column=first_column+threadIdx.x%columns;
    if(first_column>first_row+rows-1u||first_row>=valid){
        if(row>=valid)return;
#pragma unroll
        for(unsigned half=0u;half<2u;++half)
            scores[(size_t(offset+row)*32u+qk_head*2u+half)*64u+column]=0u;
        return;
    }
    if(!threadIdx.x)unsupported=0u;
    __syncthreads();
    bool invalid=false;
    for(unsigned cell=threadIdx.x;cell<rows*pairs;cell+=threads){
        const unsigned r=cell/pairs,pair=cell%pairs;uint16_t a=0u,b=0u;
        if(first_row+r<valid){
            const size_t index=(size_t(offset+first_row+r)*16u+qk_head)*128u+pair*2u;
            a=q[index];b=q[index+1u];
        }
        queries[r][pair]=scalar::pack(a,b);invalid |= !scalar::eligible(a,b);
    }
    for(unsigned cell=threadIdx.x;cell<pairs*columns;cell+=threads){
        const unsigned pair=cell/columns,c=cell%columns;uint16_t a=0u,b=0u;
        if(first_column+c<valid){
            const size_t index=(size_t(offset+first_column+c)*16u+qk_head)*128u+pair*2u;
            a=k[index];b=k[index+1u];
        }
        keys[pair][c]=scalar::pack(a,b);invalid |= !scalar::eligible(a,b);
    }
    if(invalid)atomicOr(&unsupported,1u);
    __syncthreads();
    if(row>=valid)return;
    if(column>row){
#pragma unroll
        for(unsigned half=0u;half<2u;++half)
            scores[(size_t(offset+row)*32u+qk_head*2u+half)*64u+column]=0u;
        return;
    }
    const float sum=scalar::dot<128u,columns>(queries[threadIdx.x/columns],keys,
        threadIdx.x%columns,!unsupported);
#pragma unroll
    for(unsigned half=0u;half<2u;++half){
        const unsigned head=qk_head*2u+half;
        const float difference=g[size_t(offset+row)*32u+head]-g[size_t(offset+column)*32u+head];
        scores[(size_t(offset+row)*32u+head)*64u+column]=scalar::to_bf16(sum*scalar::exponential(difference,table));
    }
}
}
