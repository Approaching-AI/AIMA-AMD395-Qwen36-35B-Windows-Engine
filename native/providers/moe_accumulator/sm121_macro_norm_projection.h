#pragma once
#include "sm121_coarse_projection_matrix.h"
#include "sm121_macro_norm_bound.h"

// Isolated single-matrix producer. Every K16 signed product uses zero-C WMMA
// followed by explicit scalar FP32 addition. Conservative operand metadata
// replaces the absolute-product matrix; no reference output is an input.
// Internal launch contracts: nonempty dimensions, width divisible by16, full
// operand/flag spans and rows*ceil(width/Chunk) metadata entries for each side.
namespace qrt_sm121_macro_norm_projection {
namespace bound=qrt_sm121_macro_norm_bound;
using B16=qrt_sm121_coarse_projection_matrix::B16;
using F8=qrt_sm121_coarse_projection_matrix::F8;
using Summary=bound::Summary;
constexpr unsigned threads=256u,row_tile=128u;

template<unsigned Groups>
__global__ void prepare(const uint16_t* input,Summary* output,unsigned rows,unsigned width){
    static_assert(Groups==8u || Groups==16u || Groups==32u || Groups==64u);
    constexpr unsigned Chunk=Groups*16u;
    const unsigned chunks=(width+Chunk-1u)/Chunk;
    const size_t entry=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(entry>=size_t(rows)*chunks)return;
    const unsigned row=unsigned(entry/chunks),base=unsigned(entry%chunks)*Chunk;
    output[entry]=bound::prepare(input+size_t(row)*width+base,min(Chunk,width-base));
}

template<unsigned Groups>
__global__ __launch_bounds__(threads) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,const Summary* weight_norms,const Summary* input_norms,
    float* centers,float* errors,unsigned rows,unsigned tokens,unsigned width){
    static_assert(Groups==8u || Groups==16u || Groups==32u || Groups==64u);
    constexpr unsigned Chunk=Groups*16u;
    const unsigned chunks=(width+Chunk-1u)/Chunk;
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    const unsigned row=blockIdx.x*row_tile+wave*16u+source,first_token=blockIdx.y*16u;
    const bool valid_weight=row<rows && weight_ok[row];
    F8 center{},error{};
    for(unsigned coarse=0u;coarse<width;coarse+=Chunk){
        F8 partial{},partial_max{},prefix_max{};
#pragma unroll
        for(unsigned item=0u;item<8u;++item)prefix_max[item]=bound::scalar::absolute(center[item]);
#pragma unroll 1
        for(unsigned base=coarse;base<coarse+Chunk && base<width;base+=16u){
            B16 w{},x{};const unsigned token=first_token+source;
            const bool valid_input=token<tokens && input_ok[token];
#pragma unroll
            for(unsigned k=0u;k<16u;++k){
                w[k]=valid_weight?weights[size_t(row)*width+base+k]:0u;
                x[k]=valid_input?inputs[size_t(token)*width+base+k]:0u;
            }
            partial+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,F8{});
#pragma unroll
            for(unsigned item=0u;item<8u;++item){
                const float local=bound::scalar::absolute(partial[item]);
                const float prefix=bound::scalar::absolute(center[item]+partial[item]);
                partial_max[item]=local>partial_max[item]?local:partial_max[item];
                prefix_max[item]=prefix>prefix_max[item]?prefix:prefix_max[item];
            }
        }
        const Summary left=row<rows?weight_norms[size_t(row)*chunks+coarse/Chunk]:Summary{};
#pragma unroll
        for(unsigned item=0u;item<8u;++item){
            const unsigned token=first_token+2u*item+lane/16u;
            const Summary right=token<tokens?input_norms[size_t(token)*chunks+coarse/Chunk]:Summary{};
            const auto next=bound::advance<Groups>({center[item],error[item]},partial[item],
                bound::absolute_bound(left,right),prefix_max[item],partial_max[item],bound::product_bound(left,right));
            center[item]=next.center;error[item]=next.error;
        }
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned item=0u;item<8u;++item){
        const unsigned token=first_token+2u*item+lane/16u;
        if(token<tokens){const size_t cell=size_t(token)*rows+row;centers[cell]=center[item];
            errors[cell]=valid_weight&&input_ok[token]?error[item]:bound::scalar::infinity();}
    }
}
} // namespace qrt_sm121_macro_norm_projection
