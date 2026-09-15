#pragma once
#include "prepared_decoded_qk.h"
#include "../moe_accumulator/sm121_predicted_group.h"

// Component experiment only. Native zero-C WMMA supplies exponent guesses,
// never accepted arithmetic. Scalar BF16 products form exact integer plans;
// the consumer checks each actual alignment exponent before using a plan.
namespace qrt_predicted_group_qk {
using namespace qrt_blackwell_attention;
namespace planned=qrt_sm121_predicted_group;
using B16=uint16_t __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
constexpr unsigned groups=kHeadDim/16u;
static_assert(groups==16u&&kThreads==256u);

// Coalesced layout makes the score cell contiguous within each K16 plane,
// in both LDS prefixes and global plans. Each warp then prepares different
// scores in one group, avoiding the original same-bank per-group accesses.
template<bool Coalesced>
__device__ __forceinline__ unsigned prefix_index(unsigned cell,unsigned group){return Coalesced?group*256u+cell:cell*groups+group;}
template<bool Coalesced>
__device__ __forceinline__ size_t plan_index(size_t cell,unsigned group,size_t cells){return Coalesced?size_t(group)*cells+cell:cell*groups+group;}

template<bool Coalesced=false>
__global__ void prepare(const uint16_t* query,const uint16_t* key,
    const unsigned* query_flags,const unsigned* key_flags,planned::Plan* plans,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride) {
    __shared__ uint16_t q[16][kHeadDim],k[kHeadDim][16];
    __shared__ float prefix[256u*groups];
    const size_t cells=size_t(count)*kQueryHeads*stride;
    const unsigned head=blockIdx.y,kv=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    if(key_tile>last){
        for(unsigned job=threadIdx.x;job<256u*groups;job+=kThreads){
            const unsigned cell=Coalesced?job%256u:job/groups,group=Coalesced?job/256u:job%groups;
            const unsigned row=query_tile+cell/16u,column=key_tile+cell%16u;
            if(row<count&&column<stride)plans[plan_index<Coalesced>((size_t(row)*kQueryHeads+head)*stride+column,group,cells)]={};
        }
        return;
    }
    for(unsigned item=threadIdx.x;item<16u*kHeadDim;item+=kThreads){
        const unsigned row=item/kHeadDim,feature=item%kHeadDim;
        q[row][feature]=query_tile+row<count&&query_flags[(start+query_tile+row)*kQueryHeads+head]
            ?query[(size_t(start+query_tile+row)*kQueryHeads+head)*kHeadDim+feature]:0u;
        const unsigned key_feature=item/16u,column=item%16u;
        k[key_feature][column]=key_tile+column<stride&&key_flags[(key_tile+column)*kKvHeads+kv]
            ?key[(size_t(kv)*kHeadDim+key_feature)*key_stride+key_tile+column]:0u;
    }
    __syncthreads();
    for(unsigned group=wave;group<groups;group+=8u){
        B16 left{},right{};
#pragma unroll
        for(unsigned i=0u;i<16u;++i){left[i]=q[source][group*16u+i];right[i]=k[group*16u+i][source];}
        const F8 zero{};
        const F8 partial=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(left,right,zero);
#pragma unroll
        for(unsigned i=0u;i<8u;++i)prefix[prefix_index<Coalesced>((2u*i+lane/16u)*16u+source,group)]=partial[i];
    }
    __syncthreads();
    float running=0.0f;
#pragma unroll 1
    for(unsigned group=0u;group<groups;++group){const float partial=prefix[prefix_index<Coalesced>(threadIdx.x,group)];prefix[prefix_index<Coalesced>(threadIdx.x,group)]=running;running+=partial;}
    __syncthreads();
    for(unsigned job=threadIdx.x;job<256u*groups;job+=kThreads){
        const unsigned cell=Coalesced?job%256u:job/groups,group=Coalesced?job/256u:job%groups;
        const unsigned qr=cell/16u,kc=cell%16u;
        const unsigned row=query_tile+qr,column=key_tile+kc;
        if(row>=count||column>=stride)continue;
        planned::Plan result{};
        if(column<=start+row&&query_flags[(start+row)*kQueryHeads+head]&&key_flags[column*kKvHeads+kv]){
            const unsigned bits=qrt_sm121_f32_carry::bits(prefix[prefix_index<Coalesced>(cell,group)])&0x7fffffffu;
            const int predicted=bits>=0x7f800000u?512:bits?int(bits>>23u)-127:-133;
            qrt_sm121_float_alignment::Group products;
#pragma unroll
            for(unsigned i=0u;i<16u;++i)products.set(i,q[qr][group*16u+i],k[group*16u+i][kc]);
            result=planned::prepare_float(products,predicted);
        }
        plans[plan_index<Coalesced>((size_t(row)*kQueryHeads+head)*stride+column,group,cells)]=result;
    }
}

template<bool Audit=false,bool Coalesced=false>
__global__ void finish(const uint16_t* query,const uint16_t* key,const planned::Plan* plans,
    float* output,unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    unsigned long long* counters=nullptr) {
    const size_t cell=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const bool live=cell<size_t(count)*kQueryHeads*stride;
    const unsigned column=unsigned(cell%stride),row=unsigned(cell/stride)/kQueryHeads;
    const bool active=live&&column<=start+row;
    const unsigned head=unsigned(cell/stride)%kQueryHeads,kv=head/(kQueryHeads/kKvHeads);
    qrt_q1_moe_hawkeye::Value carry{0u,-133,false};bool fallback=false;unsigned accepted=0u;
#pragma unroll 1
    for(unsigned group=0u;active&&group<groups;++group){
        qrt_q1_moe_hawkeye::Value next;
        if(!planned::apply(carry,plans[plan_index<Coalesced>(cell,group,size_t(count)*kQueryHeads*stride)],&next)){fallback=true;break;}
        carry=next;++accepted;
    }
    if constexpr(Audit){
        __shared__ unsigned hits[kThreads],misses[kThreads];
        hits[threadIdx.x]=accepted;misses[threadIdx.x]=unsigned(fallback);__syncthreads();
        for(unsigned step=kThreads/2u;step;step/=2u){if(threadIdx.x<step){hits[threadIdx.x]+=hits[threadIdx.x+step];misses[threadIdx.x]+=misses[threadIdx.x+step];}__syncthreads();}
        if(!threadIdx.x){atomicAdd(counters,static_cast<unsigned long long>(hits[0]));atomicAdd(counters+1u,static_cast<unsigned long long>(misses[0]));}
    }
    if(live)output[cell]=!active?-INFINITY:fallback?qrt_decoded_window_qk::raw_dot(
        query+(size_t(start+row)*kQueryHeads+head)*kHeadDim,
        key+size_t(kv)*kHeadDim*key_stride+column,key_stride):
        qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry))*kExactScale;
}
} // namespace qrt_predicted_group_qk
