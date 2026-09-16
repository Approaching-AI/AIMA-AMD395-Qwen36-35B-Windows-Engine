#pragma once
#include "sm121_coarse_projection_matrix.h"

// Isolated cooperative C64 producer. Only operand transport and matrix tile
// ownership change. Zero-C signed/absolute K16 WMMA, ascending FP32 partial
// sums and the original2^-19 envelope retain their exact instruction order.
// Internal preconditions are the original producer's complete immutable spans,
// positive rows/tokens, width divisible by16 and rows*tokens<=UINT32_MAX.
namespace qrt_sm121_shared_coarse_projection {
namespace original=qrt_sm121_coarse_projection_matrix;
namespace bound=original::bound;
using B16=original::B16;
using F8=original::F8;
template<unsigned Rows>struct Layout {
    static_assert(Rows==64u||Rows==128u);
    static constexpr unsigned tokens=4096u/Rows,total=Rows+tokens;
    static constexpr unsigned words_per_thread=total*32u/256u;
    static constexpr unsigned query_step=8u/(Rows/16u)*16u;
};
template<unsigned Rows>
__device__ __forceinline__ void load(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* valid,unsigned first_row,unsigned first_token,unsigned width,unsigned base,
    uint32_t (&words)[Layout<Rows>::words_per_thread]) {
#pragma unroll
    for(unsigned i=0;i<Layout<Rows>::words_per_thread;++i) {
        const unsigned cell=threadIdx.x+i*256u,row=cell/32u,k=(cell%32u)*2u;
        uint32_t value=0u;
        if(valid[row]&&base+k<width) {
            const auto* source=row<Rows?weights+size_t(first_row+row)*width+base+k:
                inputs+size_t(first_token+row-Rows)*width+base+k;
            // Also valid for independently two-byte-skewed operand buffers.
            __builtin_memcpy(&value,source,4u);
        }
        words[i]=value;
    }
}
template<unsigned Rows>
__device__ __forceinline__ void publish(uint32_t (&shared)[Layout<Rows>::total][33],
    const uint32_t (&words)[Layout<Rows>::words_per_thread]) {
#pragma unroll
    for(unsigned i=0;i<Layout<Rows>::words_per_thread;++i) {
        const unsigned cell=threadIdx.x+i*256u;
        shared[cell/32u][cell%32u]=words[i];
    }
}
template<unsigned Rows,bool Prefetch>
__global__ __launch_bounds__(256) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,float* centers,float* errors,
    unsigned rows,unsigned tokens,unsigned width) {
    using L=Layout<Rows>;
    __shared__ uint32_t tile[L::total][33];
    __shared__ unsigned valid[L::total];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u,source=lane%16u;
    const unsigned first_row=blockIdx.x*Rows,first_token=blockIdx.y*L::tokens;
    if(tid<L::total)valid[tid]=tid<Rows?
        unsigned(first_row+tid<rows&&weight_ok[first_row+tid]):
        unsigned(first_token+tid-Rows<tokens&&input_ok[first_token+tid-Rows]);
    __syncthreads();
    uint32_t words[L::words_per_thread];
    load<Rows>(weights,inputs,valid,first_row,first_token,width,0u,words);
    publish<Rows>(tile,words);__syncthreads();
    const unsigned wr=wave%(Rows/16u)*16u+source,query_base=wave/(Rows/16u)*16u;
    const unsigned row=first_row+wr;
    F8 center[2]{},error[2]{};
    for(unsigned coarse=0;coarse<width;coarse+=64u) {
        const bool another=coarse+64u<width;
        if constexpr(Prefetch)if(another)
            load<Rows>(weights,inputs,valid,first_row,first_token,width,coarse+64u,words);
        F8 partial[2]{},positive[2]{};
#pragma unroll
        for(unsigned group=0;group<4u;++group)if(coarse+group*16u<width) {
            B16 w{},wa{};__builtin_memcpy(&w,&tile[wr][group*8u],32u);
#pragma unroll
            for(unsigned i=0;i<16u;++i)wa[i]=w[i]&0x7fffu;
#pragma unroll
            for(unsigned fragment=0;fragment<2u;++fragment) {
                B16 x{},xa{};
                __builtin_memcpy(&x,&tile[Rows+query_base+fragment*L::query_step+source][group*8u],32u);
#pragma unroll
                for(unsigned i=0;i<16u;++i)xa[i]=x[i]&0x7fffu;
                const F8 zero{};
                partial[fragment]+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,zero);
                positive[fragment]+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xa,wa,zero);
            }
        }
#pragma unroll
        for(unsigned fragment=0;fragment<2u;++fragment) {
#pragma unroll
            for(unsigned i=0;i<8u;++i) {
                const auto next=bound::advance<4u,19u>({center[fragment][i],error[fragment][i]},partial[fragment][i],positive[fragment][i]);
                center[fragment][i]=next.center;error[fragment][i]=next.error;
            }
        }
        if(another) {
            // All readers retire before publication; no early-return lane
            // can skip either CTA barrier, including unsupported/tail rows.
            __syncthreads();
            if constexpr(!Prefetch)load<Rows>(weights,inputs,valid,first_row,first_token,width,coarse+64u,words);
            publish<Rows>(tile,words);__syncthreads();
        }
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned fragment=0;fragment<2u;++fragment) {
#pragma unroll
        for(unsigned i=0;i<8u;++i) {
            const unsigned local=query_base+fragment*L::query_step+2u*i+lane/16u,token=first_token+local;
            if(token<tokens) {
                const size_t cell=size_t(token)*rows+row;
                centers[cell]=center[fragment][i];
                errors[cell]=valid[wr]&&valid[Rows+local]?error[fragment][i]:bound::scalar::infinity();
            }
        }
    }
}
} // namespace qrt_sm121_shared_coarse_projection
