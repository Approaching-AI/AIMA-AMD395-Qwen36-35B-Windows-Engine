#pragma once
#include "sm121_coarse_projection_matrix.h"
#include "sm121_prefix_replay_bound.h"
#include "sm121_staged_half_projection.h"

// Isolated phased replay. Internal launch preconditions: rows*tokens fits
// uint32, width is divisible by 64*Parts, complete prepared rows and snapshots,
// unique candidate IDs, and next-work capacity at least the input count.
namespace qrt_sm121_prefix_replay_projection {
namespace bound=qrt_sm121_coarse_projection_bound;
namespace prefix=qrt_sm121_prefix_replay_bound;
namespace staged=qrt_sm121_staged_half_projection;
using Row=staged::Row;
using Value=staged::Value;
using B16=qrt_sm121_coarse_projection_matrix::B16;
using F8=qrt_sm121_coarse_projection_matrix::F8;
constexpr unsigned threads=256u,row_tile=128u;
template<unsigned Parts>
__global__ __launch_bounds__(threads) void produce(const uint16_t* weights,const uint16_t* inputs,
    const unsigned* weight_ok,const unsigned* input_ok,float* centers,float* errors,
    unsigned rows,unsigned tokens,unsigned width,bound::State* snapshots) {
    constexpr unsigned Chunk=64u,Fragments=1u;
    static_assert(Parts==2u || Parts==4u);
    const unsigned split=width/Parts;
    const size_t cells=size_t(rows)*tokens;
    static_assert(Chunk==64u || Chunk==128u || Chunk==256u);
    static_assert(Fragments==1u || Fragments==2u);
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    const unsigned row=blockIdx.x*row_tile+wave*16u+source;
    const unsigned first_token=blockIdx.y*(16u*Fragments);
    const bool valid_weight=row<rows && weight_ok[row];
    F8 centers_local[Fragments]{},errors_local[Fragments]{};
    for(unsigned coarse=0u;coarse<width;coarse+=Chunk) {
        F8 partial[Fragments]{},positive[Fragments]{};
#pragma unroll 1
        for(unsigned base=coarse;base<coarse+Chunk && base<width;base+=16u) {
            B16 w{},wa{};
#pragma unroll
            for(unsigned k=0u;k<16u;++k){w[k]=valid_weight?weights[size_t(row)*width+base+k]:0u;wa[k]=w[k]&0x7fffu;}
#pragma unroll
            for(unsigned fragment=0u;fragment<Fragments;++fragment) {
                const unsigned token=first_token+fragment*16u+source;
                const bool valid_input=token<tokens && input_ok[token];B16 x{},xa{};
#pragma unroll
                for(unsigned k=0u;k<16u;++k){x[k]=valid_input?inputs[size_t(token)*width+base+k]:0u;xa[k]=x[k]&0x7fffu;}
                const F8 zero{};
                // Explicit zero-C products followed by FP32 additions. Native
                // C accumulation is not substituted for this error model.
                const F8 signed_product=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,zero);
                const F8 absolute_product=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(xa,wa,zero);
                partial[fragment]+=signed_product;positive[fragment]+=absolute_product;
            }
        }
#pragma unroll
        for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
            for(unsigned item=0u;item<8u;++item) {
                const auto next=bound::advance<Chunk/16u>({centers_local[fragment][item],errors_local[fragment][item]},partial[fragment][item],positive[fragment][item]);
                centers_local[fragment][item]=next.center;errors_local[fragment][item]=next.error;
                if((coarse+Chunk)%split==0u && coarse+Chunk<width && row<rows) {
                    const unsigned token=first_token+fragment*16u+2u*item+lane/16u;
                    if(token<tokens) snapshots[size_t((coarse+Chunk)/split-1u)*cells+size_t(token)*rows+row]=
                        {next.center,valid_weight && input_ok[token]?next.error:bound::scalar::infinity()};
                }
            }
        }
    }
    if(row>=rows)return;
#pragma unroll
    for(unsigned fragment=0u;fragment<Fragments;++fragment) {
#pragma unroll
        for(unsigned item=0u;item<8u;++item) {
            const unsigned token=first_token+fragment*16u+2u*item+lane/16u;
            if(token<tokens){const size_t cell=size_t(token)*rows+row;centers[cell]=centers_local[fragment][item];
                errors[cell]=valid_weight && input_ok[token]?errors_local[fragment][item]:bound::scalar::infinity();}
        }
    }
}

// Explicit words avoid Value padding in the compacted continuation record.
struct Work { uint32_t cell,significand,state; };
static_assert(sizeof(Work)==12u);
__device__ __forceinline__ Value carry(const Work& work) {
    return {work.significand,int16_t(work.state&65535u),bool(work.state&65536u)};
}
__device__ __forceinline__ Work work(unsigned cell,Value value) {
    return {cell,value.significand,uint32_t(uint16_t(value.exponent))|(unsigned(value.negative)<<16u)};
}
template<bool Indexed>
__global__ void full_replay(const Row* weights,const Row* inputs,const unsigned* indices,
    float* output,unsigned rows,unsigned width,unsigned offset,unsigned count) {
    const unsigned rank=blockIdx.x*64u+threadIdx.x/4u;
    if(rank>=count)return;
    const unsigned cell=Indexed?indices[offset+rank]:offset+rank;
    const float value=staged::dot<2u>(weights+size_t(cell%rows)*(width/16u),inputs+size_t(cell/rows)*(width/16u),width);
    if(!(threadIdx.x&3u))output[cell]=value;
}

template<unsigned Parts,bool Audit=false>
__global__ void phase(const Row* weights,const Row* inputs,const unsigned* indices,
    const Work* previous,const float* centers,const float* errors,const bound::State* snapshots,
    float* output,Work* next,unsigned* next_count,unsigned rows,unsigned tokens,unsigned width,
    unsigned stage,unsigned offset,unsigned count,unsigned* finished_stage=nullptr) {
    static_assert(Parts==2u || Parts==4u);
    const unsigned rank=offset+blockIdx.x*64u+threadIdx.x/4u,lane=threadIdx.x&3u;
    const bool live=rank<offset+count;
    const unsigned groups=width/16u,first=stage*(groups/Parts),end=(stage+1u)*(groups/Parts);
    const size_t cells=size_t(rows)*tokens;
    unsigned cell=0u;Value value{0u,-133,false};bool selected=false;
    if(live) {
        if(!stage)cell=indices[rank];else{const auto item=previous[rank];cell=item.cell;value=carry(item);}
        const auto* w=weights+size_t(cell%rows)*groups;
        const auto* x=inputs+size_t(cell/rows)*groups;
#pragma unroll 1
        for(unsigned base=first;base<end;base+=2u) {
            const auto a=staged::load(w[base],x[base]);
            const auto b=staged::load(w[base+1u],x[base+1u]);
            value=staged::accumulate(value,a);value=staged::accumulate(value,b);
        }
        if(!lane) {
            const float exact=qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(value));
            bool done=stage+1u==Parts;float representative=exact;
            if(!done && (!value.significand || (value.significand>=0x800000u && value.exponent>=-126 && value.exponent<=127)))
                done=prefix::certificate(snapshots[size_t(stage)*cells+cell],{centers[cell],errors[cell]},exact,&representative);
            if(done) {
                output[cell]=representative;
                if constexpr(Audit)finished_stage[cell]=stage+1u;
            }else selected=true;
        }
    }
    // Compact unfinished cells between kernels, so completed dots do not leave
    // idle subgroups in subsequent long suffix loops. All threads reach these
    // barriers, including the partial final CTA and unsupported operand rows.
    if(stage+1u==Parts)return;
    const unsigned wave=threadIdx.x/32u,wlane=threadIdx.x%32u,mask=__ballot(selected);
    __shared__ unsigned offsets[8],begin;
    if(!wlane)offsets[wave]=__popc(mask);
    __syncthreads();
    if(!threadIdx.x){unsigned n=0u;for(unsigned i=0u;i<8u;++i){const unsigned old=offsets[i];offsets[i]=n;n+=old;}begin=atomicAdd(next_count,n);}
    __syncthreads();
    if(selected)next[begin+offsets[wave]+__popc(mask&((1u<<wlane)-1u))]=work(cell,value);
}
} // namespace qrt_sm121_prefix_replay_projection
