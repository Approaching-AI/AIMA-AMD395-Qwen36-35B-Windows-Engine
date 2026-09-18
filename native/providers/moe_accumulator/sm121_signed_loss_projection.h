#pragma once
#include "sm121_shared_coarse_projection.h"
#include "sm121_signed_loss_bound.h"

// Isolated native directional C64 producer. Original zero-C WMMA order,
// coefficient2^-19, complete eligibility and selected ordered K16 replay are
// unchanged. No golden result enters this kernel or its selection predicate.
// Internal launch preconditions match the shared C64 producer: immutable
// complete operands/flags, positive extents, width divisible by16, unique
// output cells and rows*tokens<=UINT32_MAX.
namespace qrt_sm121_signed_loss_projection {
namespace shared=qrt_sm121_shared_coarse_projection;
namespace loss=qrt_sm121_signed_loss_bound;
namespace bound=loss::coarse;
using B16=shared::B16;using F8=shared::F8;
// One fragment halves per-thread live centers and directional envelopes.
// Both schedules use the same C64 arithmetic and ordered full replay.
template<unsigned Rows,unsigned Fragments>struct Layout {
    static_assert((Rows==64u||Rows==128u)&&(Fragments==1u||Fragments==2u));
    static constexpr unsigned tokens=2048u*Fragments/Rows,total=Rows+tokens;
    static constexpr unsigned words_per_thread=total*32u/256u;
    static constexpr unsigned query_step=8u/(Rows/16u)*16u;
};
template<unsigned Rows,unsigned Fragments>
__device__ __forceinline__ void load(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* valid,unsigned first_row,unsigned first_token,unsigned width,unsigned base,
    uint32_t (&words)[(Layout<Rows,Fragments>::words_per_thread)]) {
#pragma unroll
    for(unsigned i=0u;i<Layout<Rows,Fragments>::words_per_thread;++i) {
        const unsigned cell=threadIdx.x+i*256u,row=cell/32u,k=(cell%32u)*2u;
        uint32_t value=0u;
        if(valid[row]&&base+k<width) {
            const auto* source=row<Rows?weights+size_t(first_row+row)*width+base+k:
                inputs+size_t(first_token+row-Rows)*width+base+k;
            __builtin_memcpy(&value,source,4u);
        }
        words[i]=value;
    }
}
template<unsigned Rows,unsigned Fragments>
__device__ __forceinline__ void publish(uint32_t (&tile)[(Layout<Rows,Fragments>::total)][33],
    const uint32_t (&words)[(Layout<Rows,Fragments>::words_per_thread)]) {
#pragma unroll
    for(unsigned i=0u;i<Layout<Rows,Fragments>::words_per_thread;++i) {
        const unsigned cell=threadIdx.x+i*256u;tile[cell/32u][cell%32u]=words[i];
    }
}
template<unsigned Rows,bool Prefetch,unsigned Fragments=2u>
__global__ __launch_bounds__(256) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,float* centers,float* lower_errors,float* upper_errors,
    unsigned rows,unsigned tokens,unsigned width) {
    using L=Layout<Rows,Fragments>;
    __shared__ uint32_t tile[L::total][33];
    __shared__ unsigned valid[L::total];
    __shared__ uint64_t negative[L::total],nonzero[L::total];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u,source=lane%16u;
    const unsigned first_row=blockIdx.x*Rows,first_token=blockIdx.y*L::tokens;
    if(tid<L::total)valid[tid]=tid<Rows?
        unsigned(first_row+tid<rows&&weight_ok[first_row+tid]):
        unsigned(first_token+tid-Rows<tokens&&input_ok[first_token+tid-Rows]);
    __syncthreads();
    uint32_t words[L::words_per_thread];
    load<Rows,Fragments>(weights,inputs,valid,first_row,first_token,width,0u,words);
    publish<Rows,Fragments>(tile,words);__syncthreads();
    const unsigned wr=wave%(Rows/16u)*16u+source,query_base=wave/(Rows/16u)*16u;
    const unsigned row=first_row+wr;
    F8 center[Fragments]{},lower[Fragments]{},upper[Fragments]{};
    for(unsigned coarse=0;coarse<width;coarse+=64u) {
        const bool another=coarse+64u<width;
        if constexpr(Prefetch)if(another)
            load<Rows,Fragments>(weights,inputs,valid,first_row,first_token,width,coarse+64u,words);
        // The entire current C64 tile is resident and zero padded. These
        // masks describe actual operand bits, including signed zero. Each
        // row has one writer; all readers retire before the next publication.
        if(tid<L::total) {
            uint64_t sign=0u,active=0u;
#pragma unroll
            for(unsigned pair=0u;pair<32u;++pair) {
                const uint32_t word=tile[tid][pair];
                sign |= uint64_t((word>>15u)&1u)<<(2u*pair);
                sign |= uint64_t(word>>31u)<<(2u*pair+1u);
                active |= uint64_t(bool(word&0x7fffu))<<(2u*pair);
                active |= uint64_t(bool(word&0x7fff0000u))<<(2u*pair+1u);
            }
            negative[tid]=sign;nonzero[tid]=active;
        }
        __syncthreads();
        F8 partial[Fragments]{},positive[Fragments]{};
#pragma unroll
        for(unsigned group=0;group<4u;++group)if(coarse+group*16u<width) {
            B16 w{},wa{};__builtin_memcpy(&w,&tile[wr][group*8u],32u);
#pragma unroll
            for(unsigned i=0;i<16u;++i)wa[i]=w[i]&0x7fffu;
#pragma unroll
            for(unsigned fragment=0;fragment<Fragments;++fragment) {
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
        for(unsigned fragment=0;fragment<Fragments;++fragment) {
#pragma unroll
            for(unsigned i=0;i<8u;++i) {
                const unsigned xr=Rows+query_base+fragment*L::query_step+2u*i+lane/16u;
                const loss::Summary a{negative[wr],nonzero[wr],true};
                const loss::Summary b{negative[xr],nonzero[xr],true};
                const auto next=loss::advance({center[fragment][i],lower[fragment][i],upper[fragment][i]},
                    partial[fragment][i],positive[fragment][i],a,b);
                center[fragment][i]=next.center;
                lower[fragment][i]=next.lower_error;upper[fragment][i]=next.upper_error;
            }
        }
        if(another) {
            // All readers retire before publication; no early-return lane
            // can skip either CTA barrier, including unsupported/tail rows.
            __syncthreads();
            if constexpr(!Prefetch)load<Rows,Fragments>(weights,inputs,valid,first_row,first_token,width,coarse+64u,words);
            publish<Rows,Fragments>(tile,words);__syncthreads();
        }
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned fragment=0;fragment<Fragments;++fragment) {
#pragma unroll
        for(unsigned i=0;i<8u;++i) {
            const unsigned local=query_base+fragment*L::query_step+2u*i+lane/16u,token=first_token+local;
            if(token<tokens) {
                const size_t cell=size_t(token)*rows+row;
                centers[cell]=center[fragment][i];
                lower_errors[cell]=valid[wr]&&valid[Rows+local]?lower[fragment][i]:bound::scalar::infinity();
                upper_errors[cell]=valid[wr]&&valid[Rows+local]?upper[fragment][i]:bound::scalar::infinity();
            }
        }
    }
}

// The producer's center is preserved. A cell is skipped only when both
// directional endpoints round to the same BF16; unsupported rows are infinite
// and necessarily enter complete original replay. One allocation per CTA.
__global__ void compact(const float* centers,const float* lower,const float* upper,
    float* output,unsigned* indices,unsigned* count,unsigned cells) {
    const unsigned cell=blockIdx.x*256u+threadIdx.x,lane=threadIdx.x%32u,wave=threadIdx.x/32u;
    const bool selected=cell<cells && !loss::certified({centers[cell],lower[cell],upper[cell]});
    const unsigned mask=__ballot(selected);__shared__ unsigned offsets[8],first;
    if(!lane)offsets[wave]=__popc(mask);__syncthreads();
    if(!threadIdx.x){unsigned total=0u;for(unsigned i=0u;i<8u;++i){const unsigned n=offsets[i];offsets[i]=total;total+=n;}first=atomicAdd(count,total);}
    __syncthreads();
    if(cell<cells)output[cell]=centers[cell];
    if(selected)indices[first+offsets[wave]+__popc(mask&((1u<<lane)-1u))]=cell;
}
} // namespace qrt_sm121_signed_loss_projection
