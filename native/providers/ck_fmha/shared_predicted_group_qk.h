#pragma once
#include "predicted_group_qk.h"

// Isolated component schedule: different waves prepare different K16 groups
// concurrently. Bounded LDS plans hand those independent jobs to one ordered
// carry owner per score. No global plan table or approximate output is used.
namespace qrt_shared_predicted_group_qk {
using namespace qrt_predicted_group_qk;

template<unsigned WindowGroups,bool Audit=false>
__global__ void scores(const uint16_t* query,const uint16_t* key,
    const unsigned* query_flags,const unsigned* key_flags,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    unsigned long long* counters=nullptr) {
    static_assert(WindowGroups==4u||WindowGroups==8u);
    constexpr unsigned width=WindowGroups*16u,waves_per_group=8u/WindowGroups;
    __shared__ uint16_t q[16][width],k[width][16];
    // Plain, separate words avoid device-side object initialization and give
    // consecutive score owners consecutive LDS banks in both plan fields.
    __shared__ uint32_t plan_modulo[WindowGroups][256],plan_control[WindowGroups][256];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u,source=lane%16u;
    const unsigned head=blockIdx.y,kv=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned qr=tid/16u,kc=tid%16u,row=query_tile+qr,column=key_tile+kc;
    const bool live=row<count&&column<stride,active=live&&column<=start+row;
    const size_t cell=(size_t(row)*kQueryHeads+head)*stride+column;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    if(key_tile>last){if(live)output[cell]=-INFINITY;return;}
    qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
    bool fallback=active&&(!query_flags[(start+row)*kQueryHeads+head]||!key_flags[column*kKvHeads+kv]);
    unsigned accepted=0u;float running=0.0f;
    for(unsigned base=0u;base<kHeadDim;base+=width){
        for(unsigned item=tid;item<16u*width;item+=kThreads){
            const unsigned r=item/width,feature=item%width;
            q[r][feature]=query_tile+r<count&&query_flags[(start+query_tile+r)*kQueryHeads+head]
                ?query[(size_t(start+query_tile+r)*kQueryHeads+head)*kHeadDim+base+feature]:0u;
            const unsigned key_feature=item/16u,c=item%16u;
            k[key_feature][c]=key_tile+c<stride&&key_flags[(key_tile+c)*kKvHeads+kv]
                ?key[(size_t(kv)*kHeadDim+base+key_feature)*key_stride+key_tile+c]:0u;
        }
        __syncthreads();
        if(wave<WindowGroups){
            B16 left{},right{};
#pragma unroll
            for(unsigned i=0u;i<16u;++i){left[i]=q[source][wave*16u+i];right[i]=k[wave*16u+i][source];}
            const F8 zero{};
            const F8 partial=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(left,right,zero);
#pragma unroll
            for(unsigned i=0u;i<8u;++i)
                plan_modulo[wave][(2u*i+lane/16u)*16u+source]=qrt_sm121_f32_carry::bits(partial[i]);
        }
        __syncthreads();
        // Each score owns its ascending FP32 predictor across every window.
        // The temporary control word is overwritten by a complete exact plan
        // only after all predictions are visible to the producer waves.
#pragma unroll 1
        for(unsigned group=0u;group<WindowGroups;++group){
            const unsigned bits=qrt_sm121_f32_carry::bits(running)&0x7fffffffu;
            const int predicted=bits>=0x7f800000u?512:bits?int(bits>>23u)-127:-133;
            running+=qrt_sm121_float_alignment::from_bits(plan_modulo[group][tid]);
            plan_control[group][tid]=uint32_t(predicted);
        }
        __syncthreads();
        const unsigned group=wave%WindowGroups;
        for(unsigned job=(wave/WindowGroups)*32u+lane;job<256u;job+=waves_per_group*32u){
            const unsigned r=job/16u,c=job%16u;
            const unsigned output_row=query_tile+r,output_column=key_tile+c;
            planned::Plan result{};
            if(output_row<count&&output_column<stride&&output_column<=start+output_row&&
                query_flags[(start+output_row)*kQueryHeads+head]&&key_flags[output_column*kKvHeads+kv]){
                const int predicted=int32_t(plan_control[group][job]);
                qrt_sm121_float_alignment::Group products;
#pragma unroll
                for(unsigned i=0u;i<16u;++i)products.set(i,q[r][group*16u+i],k[group*16u+i][c]);
                result=planned::prepare_float(products,predicted);
            }
            plan_modulo[group][job]=result.modulo;plan_control[group][job]=result.control;
        }
        __syncthreads();
        if(active&&!fallback){
#pragma unroll 1
            for(unsigned index=0u;index<WindowGroups;++index){
                qrt_q1_moe_hawkeye::Value next;
                if(!planned::apply(carry,{plan_modulo[index][tid],plan_control[index][tid]},&next)){fallback=true;break;}
                carry=next;++accepted;
            }
        }
        __syncthreads();
    }
    if constexpr(Audit){
        __shared__ unsigned hits[kThreads],misses[kThreads];
        hits[tid]=accepted;misses[tid]=unsigned(fallback);__syncthreads();
        for(unsigned step=kThreads/2u;step;step/=2u){
            if(tid<step){hits[tid]+=hits[tid+step];misses[tid]+=misses[tid+step];}
            __syncthreads();
        }
        if(!tid){atomicAdd(counters,static_cast<unsigned long long>(hits[0]));atomicAdd(counters+1u,static_cast<unsigned long long>(misses[0]));}
    }
    if(live)output[cell]=!active?-INFINITY:fallback?qrt_decoded_window_qk::raw_dot(
        query+(size_t(start+row)*kQueryHeads+head)*kHeadDim,
        key+size_t(kv)*kHeadDim*key_stride+column,key_stride):
        qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry))*kExactScale;
}
} // namespace qrt_shared_predicted_group_qk
