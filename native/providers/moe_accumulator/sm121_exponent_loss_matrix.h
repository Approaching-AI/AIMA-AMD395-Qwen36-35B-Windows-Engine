#pragma once
#include "sm121_coarse_projection_matrix.h"
#include "sm121_exponent_loss_bound.h"

namespace qrt_sm121_exponent_loss_matrix {
using namespace qrt_sm121_coarse_projection_matrix;
namespace loss=qrt_sm121_exponent_loss_bound;
__global__ void prepare(const uint16_t* input,loss::Summary* metadata,unsigned rows,unsigned width){
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const unsigned chunks=(width+63u)/64u;
    if(index>=size_t(rows)*chunks)return;
    const unsigned row=unsigned(index/chunks),base=unsigned(index%chunks)*64u;
    metadata[index]=loss::summarize(input+size_t(row)*width+base,min(64u,width-base));
}

// Original C64/F1 zero-C WMMA products and explicit FP32 additions. Only the
// canonical product-loss part of the envelope consults exponent metadata.
__global__ __launch_bounds__(threads) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,const loss::Summary* weight_metadata,
    const loss::Summary* input_metadata,float* centers,float* errors,unsigned rows,unsigned tokens,unsigned width){
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    const unsigned row=blockIdx.x*row_tile+wave*16u+source;
    const unsigned first_token=blockIdx.y*16u,chunks=(width+63u)/64u;
    const bool valid_weight=row<rows&&weight_ok[row];
    F8 center{},error{};
    for(unsigned coarse=0u;coarse<width;coarse+=64u){
        F8 partial{},positive{};
#pragma unroll 1
        for(unsigned base=coarse;base<coarse+64u&&base<width;base+=16u){
            B16 w{},wa{},x{},xa{};
            const unsigned token=first_token+source;
            const bool valid_input=token<tokens&&input_ok[token];
#pragma unroll
            for(unsigned i=0u;i<16u;++i){
                w[i]=valid_weight?weights[size_t(row)*width+base+i]:0u;wa[i]=w[i]&0x7fffu;
                x[i]=valid_input?inputs[size_t(token)*width+base+i]:0u;xa[i]=x[i]&0x7fffu;
            }
            const F8 zero{};
            partial+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,zero);
            positive+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xa,wa,zero);
        }
        const auto weight=row<rows?weight_metadata[size_t(row)*chunks+coarse/64u]:loss::Summary{};
#pragma unroll
        for(unsigned item=0u;item<8u;++item){
            const unsigned token=first_token+2u*item+lane/16u;
            const auto input=token<tokens?input_metadata[size_t(token)*chunks+coarse/64u]:loss::Summary{};
            const auto next=loss::advance({center[item],error[item]},partial[item],positive[item],input,weight);
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
} // namespace qrt_sm121_exponent_loss_matrix
