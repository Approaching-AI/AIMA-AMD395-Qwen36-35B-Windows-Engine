#pragma once
#include "predicted_group_qk.h"
#include "../moe_accumulator/sm121_matrix_predicted_group.h"

// Independent integer WMMA products feed exact sparse-compensation plans.
// Only the exponent predictor is floating; every accepted plan is exact.
namespace qrt_matrix_predicted_group_qk {
using namespace qrt_predicted_group_qk;
namespace matrix_plan=qrt_sm121_matrix_predicted_group;
using Row=matrix_plan::Row;
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
struct Parts{I8 high,cross_left,cross_right,low;};
__device__ __forceinline__ Parts products(const Row& left,const Row& right){
    I4 ah{},al{},bh{},bl{};
#pragma unroll
    for(unsigned i=0u;i<4u;++i){ah[i]=left.high[i];al[i]=left.low[i];bh[i]=right.high[i];bl[i]=right.low[i];}
    const I8 zero{};
    return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,true,bh,zero,false),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,false,bl,zero,false),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,true,bh,zero,false),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,false,bl,zero,false)};
}
template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned tokens){
    static_assert(Kind==IntegerRowKind::Query||Kind==IntegerRowKind::Key);
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=integer_row_count(Kind,tokens,tokens))return;
    Row value{};
    for(unsigned i=0u;i<16u;++i)value.original[i]=input[integer_row_input_index(Kind,row,i,tokens,0u,tokens)];
    qrt_sm121_integer_core::prepare(value);output[row]=value;
}

template<unsigned WindowGroups,bool Audit=false>
__global__ void scores(const uint16_t* query,const uint16_t* key,
    const unsigned* query_flags,const unsigned* key_flags,const Row* core_query,const Row* core_key,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    unsigned long long* counters=nullptr) {
    static_assert(WindowGroups==4u||WindowGroups==8u);
    constexpr unsigned words=sizeof(Row)/4u,waves_per_group=8u/WindowGroups;
    __shared__ Row left_rows[WindowGroups][16],right_rows[WindowGroups][16];
    __shared__ uint32_t predictor[WindowGroups][256];
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
    for(unsigned base=0u;base<groups;base+=WindowGroups){
        for(unsigned item=tid;item<WindowGroups*32u*words;item+=kThreads){
            const unsigned row_index=item/words,word=item%words,group=row_index/32u,r=row_index%32u;
            const bool is_query=r<16u;
            const unsigned position=is_query?query_tile+r:key_tile+r-16u;
            uint32_t value=0u;
            if(is_query?(position<count&&query_flags[(start+position)*kQueryHeads+head]):
                (position<stride&&key_flags[position*kKvHeads+kv])){
                const Row* from=is_query?core_query+(size_t(start+position)*kQueryHeads+head)*groups+base+group:
                    core_key+(size_t(kv)*groups+base+group)*key_stride+position;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(from)+word*4u,4u);
            }
            Row* to=is_query?&left_rows[group][r]:&right_rows[group][r-16u];
            __builtin_memcpy(reinterpret_cast<unsigned char*>(to)+word*4u,&value,4u);
        }
        __syncthreads();
        if(wave<WindowGroups){
            const Row& a=left_rows[wave][source];const Row& b=right_rows[wave][source];
            const Parts integer=products(a,b);
            B16 left{},right{};
#pragma unroll
            for(unsigned i=0u;i<16u;++i){left[i]=a.original[i];right[i]=b.original[i];}
            const F8 zero{};
            const F8 partial=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(left,right,zero);
#pragma unroll
            for(unsigned i=0u;i<8u;++i){
                const unsigned destination=(2u*i+lane/16u)*16u+source;
                const int64_t mathematical=int64_t(integer.high[i])*65536+
                    (int64_t(integer.cross_left[i])+integer.cross_right[i])*256+integer.low[i];
                plan_modulo[wave][destination]=uint32_t(mathematical);
                plan_control[wave][destination]=uint32_t(uint64_t(mathematical)>>32u);
                predictor[wave][destination]=qrt_sm121_f32_carry::bits(partial[i]);
            }
        }
        __syncthreads();
        // Each score owns its ascending FP32 predictor across every window.
        // Predictor slots keep the guessed exponents while the separate integer
        // product words are replaced by complete plans after a CTA barrier.
#pragma unroll 1
        for(unsigned group=0u;group<WindowGroups;++group){
            const unsigned bits=qrt_sm121_f32_carry::bits(running)&0x7fffffffu;
            const int predicted=bits>=0x7f800000u?512:bits?int(bits>>23u)-127:-133;
            running+=qrt_sm121_float_alignment::from_bits(predictor[group][tid]);
            predictor[group][tid]=uint32_t(predicted);
        }
        __syncthreads();
        const unsigned group=wave%WindowGroups;
        for(unsigned job=(wave/WindowGroups)*32u+lane;job<256u;job+=waves_per_group*32u){
            const unsigned r=job/16u,c=job%16u;
            const unsigned output_row=query_tile+r,output_column=key_tile+c;
            planned::Plan result{};
            if(output_row<count&&output_column<stride&&output_column<=start+output_row&&
                query_flags[(start+output_row)*kQueryHeads+head]&&key_flags[output_column*kKvHeads+kv]){
                const int predicted=int32_t(predictor[group][job]);
                const uint64_t bits=uint64_t(plan_modulo[group][job])|(uint64_t(plan_control[group][job])<<32u);
                int64_t mathematical;__builtin_memcpy(&mathematical,&bits,8u);
                result=matrix_plan::prepare(left_rows[group][r],right_rows[group][c],mathematical,predicted);
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
} // namespace qrt_matrix_predicted_group_qk
