#pragma once
#include "predicted_group_qk.h"

// Component experiment only. Keep exact alignment plans within the producing
// thread instead of writing and rereading a full score-by-K16 global arena.
namespace qrt_fused_predicted_group_qk {
using namespace qrt_predicted_group_qk;

template<unsigned PlanGroups,bool Audit=false>
__global__ void scores(const uint16_t* query,const uint16_t* key,
    const unsigned* query_flags,const unsigned* key_flags,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    unsigned long long* counters=nullptr) {
    static_assert(PlanGroups==4u||PlanGroups==16u);
    __shared__ uint16_t q[16][kHeadDim],k[kHeadDim][16];
    __shared__ float partials[groups][256];
    const unsigned head=blockIdx.y,kv=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned qr=threadIdx.x/16u,kc=threadIdx.x%16u;
    const unsigned row=query_tile+qr,column=key_tile+kc;
    const bool live=row<count&&column<stride,active=live&&column<=start+row;
    const size_t cell=(size_t(row)*kQueryHeads+head)*stride+column;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    if(key_tile>last){if(live)output[cell]=-INFINITY;return;}
    for(unsigned item=threadIdx.x;item<16u*kHeadDim;item+=kThreads){
        const unsigned r=item/kHeadDim,feature=item%kHeadDim;
        q[r][feature]=query_tile+r<count&&query_flags[(start+query_tile+r)*kQueryHeads+head]
            ?query[(size_t(start+query_tile+r)*kQueryHeads+head)*kHeadDim+feature]:0u;
        const unsigned key_feature=item/16u,c=item%16u;
        k[key_feature][c]=key_tile+c<stride&&key_flags[(key_tile+c)*kKvHeads+kv]
            ?key[(size_t(kv)*kHeadDim+key_feature)*key_stride+key_tile+c]:0u;
    }
    __syncthreads();
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    for(unsigned group=wave;group<groups;group+=8u){
        B16 left{},right{};
#pragma unroll
        for(unsigned i=0u;i<16u;++i){left[i]=q[source][group*16u+i];right[i]=k[group*16u+i][source];}
        const F8 zero{};
        const F8 partial=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(left,right,zero);
#pragma unroll
        for(unsigned i=0u;i<8u;++i)partials[group][(2u*i+lane/16u)*16u+source]=partial[i];
    }
    __syncthreads();
    qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
    bool fallback=active&&(!query_flags[(start+row)*kQueryHeads+head]||!key_flags[column*kKvHeads+kv]);
    unsigned accepted=0u;float running=0.0f;
    if(active&&!fallback){
#pragma unroll 1
        for(unsigned base=0u;base<groups&&!fallback;base+=PlanGroups){
            planned::Plan plans[PlanGroups];
            // Constant indices allow the compiler to retain independent plans
            // in registers. Resource metadata must verify the actual allocation.
#pragma unroll
            for(unsigned index=0u;index<PlanGroups;++index){
                const unsigned group=base+index;
                const unsigned bits=qrt_sm121_f32_carry::bits(running)&0x7fffffffu;
                const int predicted=bits>=0x7f800000u?512:bits?int(bits>>23u)-127:-133;
                running+=partials[group][threadIdx.x];
                qrt_sm121_float_alignment::Group products;
#pragma unroll
                for(unsigned i=0u;i<16u;++i)products.set(i,q[qr][group*16u+i],k[group*16u+i][kc]);
                plans[index]=planned::prepare_float(products,predicted);
            }
#pragma unroll
            for(unsigned index=0u;index<PlanGroups;++index){
                if(!fallback){
                    qrt_q1_moe_hawkeye::Value next;
                    if(!planned::apply(carry,plans[index],&next))fallback=true;
                    else{carry=next;++accepted;}
                }
            }
        }
    }
    if constexpr(Audit){
        __shared__ unsigned hits[kThreads],misses[kThreads];
        hits[threadIdx.x]=accepted;misses[threadIdx.x]=unsigned(fallback);__syncthreads();
        for(unsigned step=kThreads/2u;step;step/=2u){
            if(threadIdx.x<step){hits[threadIdx.x]+=hits[threadIdx.x+step];misses[threadIdx.x]+=misses[threadIdx.x+step];}
            __syncthreads();
        }
        if(!threadIdx.x){atomicAdd(counters,static_cast<unsigned long long>(hits[0]));atomicAdd(counters+1u,static_cast<unsigned long long>(misses[0]));}
    }
    if(live)output[cell]=!active?-INFINITY:fallback?qrt_decoded_window_qk::raw_dot(
        query+(size_t(start+row)*kQueryHeads+head)*kHeadDim,
        key+size_t(kv)*kHeadDim*key_stride+column,key_stride):
        qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry))*kExactScale;
}
} // namespace qrt_fused_predicted_group_qk
