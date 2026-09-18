#pragma once
#include "sm121_signed_loss_projection.h"
#include "sm121_deferred_loss_bound.h"

// Isolated producer. Sign metadata is prepared once per operand/C64 group
// and shared by all output tiles. No provider dispatch uses this component.
namespace qrt_sm121_deferred_loss_projection {
namespace layout=qrt_sm121_signed_loss_projection;
namespace bound=qrt_sm121_deferred_loss_bound;
namespace loss=qrt_sm121_signed_loss_bound;
using B16=layout::B16;using F8=layout::F8;
struct Mask { uint64_t negative,nonzero; };
static_assert(sizeof(Mask)==16u);

// One wave owns one complete or zero-padded C64 row. Each ballot corresponds
// directly to32 original BF16 words; signed zero is retained in the sign mask
// and excluded from the nonzero mask. No scalar64-element mask loop remains
// in the repeatedly executed projection producer.
__global__ void prepare(const uint16_t* input,Mask* masks,unsigned rows,unsigned width) {
    const unsigned lane=threadIdx.x%32u,blocks=(width+63u)/64u;
    const size_t entry=size_t(blockIdx.x)*8u+threadIdx.x/32u;
    if(entry>=size_t(rows)*blocks)return;
    const unsigned row=unsigned(entry/blocks),base=unsigned(entry%blocks)*64u;
    const uint16_t lo=base+lane<width?input[size_t(row)*width+base+lane]:0u;
    const uint16_t hi=base+lane+32u<width?input[size_t(row)*width+base+lane+32u]:0u;
    const uint64_t sign=uint64_t(__ballot(bool(lo&0x8000u))) |
        (uint64_t(__ballot(bool(hi&0x8000u)))<<32u);
    const uint64_t active=uint64_t(__ballot(bool(lo&0x7fffu))) |
        (uint64_t(__ballot(bool(hi&0x7fffu)))<<32u);
    if(!lane)masks[entry]={sign,active};
}

template<unsigned Fragments>
__global__ __launch_bounds__(256) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,const Mask* weight_masks,const Mask* input_masks,
    float* centers,float* lower_errors,float* upper_errors,unsigned rows,unsigned tokens,unsigned width) {
    using L=layout::Layout<64u,Fragments>;
    __shared__ uint32_t tile[L::total][33];
    __shared__ unsigned valid[L::total];
    __shared__ Mask masks[L::total];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u,source=lane%16u;
    const unsigned first_row=blockIdx.x*64u,first_token=blockIdx.y*L::tokens,blocks=(width+63u)/64u;
    if(tid<L::total)valid[tid]=tid<64u?
        unsigned(first_row+tid<rows&&weight_ok[first_row+tid]):
        unsigned(first_token+tid-64u<tokens&&input_ok[first_token+tid-64u]);
    __syncthreads();
    uint32_t words[L::words_per_thread];
    layout::load<64u,Fragments>(weights,inputs,valid,first_row,first_token,width,0u,words);
    layout::publish<64u,Fragments>(tile,words);__syncthreads();
    const unsigned wr=wave%4u*16u+source,query_base=wave/4u*16u,row=first_row+wr;
    F8 center[Fragments]{},lower[Fragments]{},upper[Fragments]{};
    for(unsigned coarse=0u;coarse<width;coarse+=64u) {
        const bool another=coarse+64u<width;
        if(another)layout::load<64u,Fragments>(weights,inputs,valid,first_row,first_token,width,coarse+64u,words);
        if(tid<L::total) {
            Mask value{};
            if(valid[tid])value=tid<64u?weight_masks[size_t(first_row+tid)*blocks+coarse/64u]:
                input_masks[size_t(first_token+tid-64u)*blocks+coarse/64u];
            masks[tid]=value;
        }
        __syncthreads();
        F8 partial[Fragments]{},positive[Fragments]{};
#pragma unroll
        for(unsigned group=0u;group<4u;++group)if(coarse+group*16u<width) {
            B16 w{},wa{};__builtin_memcpy(&w,&tile[wr][group*8u],32u);
#pragma unroll
            for(unsigned i=0u;i<16u;++i)wa[i]=w[i]&0x7fffu;
#pragma unroll
            for(unsigned fragment=0u;fragment<Fragments;++fragment) {
                B16 x{},xa{};
                __builtin_memcpy(&x,&tile[64u+query_base+fragment*L::query_step+source][group*8u],32u);
#pragma unroll
                for(unsigned i=0u;i<16u;++i)xa[i]=x[i]&0x7fffu;
                const F8 zero{};
                partial[fragment]+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,zero);
                positive[fragment]+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xa,wa,zero);
            }
        }
#pragma unroll
        for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
            for(unsigned i=0u;i<8u;++i) {
                const unsigned xr=64u+query_base+fragment*L::query_step+2u*i+lane/16u;
                const auto a=masks[wr],b=masks[xr];
                const auto products=loss::counts({a.negative,a.nonzero,true},{b.negative,b.nonzero,true});
                const auto next=bound::advance({center[fragment][i],lower[fragment][i],upper[fragment][i]},
                    partial[fragment][i],positive[fragment][i],products);
                center[fragment][i]=next.center;lower[fragment][i]=next.lower_sum;upper[fragment][i]=next.upper_sum;
            }
        }
        if(another){__syncthreads();layout::publish<64u,Fragments>(tile,words);__syncthreads();}
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
        for(unsigned i=0u;i<8u;++i) {
            const unsigned local=query_base+fragment*L::query_step+2u*i+lane/16u,token=first_token+local;
            if(token<tokens) {
                const size_t cell=size_t(token)*rows+row;
                const auto envelope=bound::finalize({center[fragment][i],lower[fragment][i],upper[fragment][i]},blocks);
                const bool supported=valid[wr]&&valid[64u+local]&&bound::width_supported(width);
                centers[cell]=envelope.center;
                lower_errors[cell]=supported?envelope.lower_error:loss::scalar::infinity();
                upper_errors[cell]=supported?envelope.upper_error:loss::scalar::infinity();
            }
        }
    }
}
} // namespace qrt_sm121_deferred_loss_projection
