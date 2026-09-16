#pragma once
#include "prepared_decoded_qk.h"
#include "../moe_accumulator/sm121_compact_byte_core.h"

// Isolated two-matrix experiment with compact76-byte rows. Original BF16
// operands are reconstructed from exact FP16 integer cores where possible;
// only encoding exceptions fetch the retained original tensor. Every actual
// carry is consumed in order, using original scalar fallback for rejected
// groups. The native matrix error<128 condition remains conditional.
namespace qrt_compact_byte_qk {
using namespace qrt_blackwell_attention;
namespace compact=qrt_sm121_compact_byte_core;
using Row=compact::Row;
using Value=compact::Value;

template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned tokens,unsigned start,unsigned count,
    uint16_t* transposed=nullptr) {
    static_assert(Kind==IntegerRowKind::Query || Kind==IntegerRowKind::Key);
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=integer_row_count(Kind,tokens,count))return;
    uint16_t original[16];
    for(unsigned i=0u;i<16u;++i){
        original[i]=input[integer_row_input_index(Kind,row,i,tokens,start,count)];
        if constexpr(Kind==IntegerRowKind::Key)
            if(transposed)transposed[(row/tokens*16u+i)*tokens+row%tokens]=original[i];
    }
    output[row]=compact::prepare(original);
}
template<unsigned WindowGroups,bool Audit=false>
__global__ void scores(const Row* query,const Row* key,const uint16_t* raw_query,const uint16_t* raw_key,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    unsigned long long* counters=nullptr) {
    static_assert(WindowGroups==4u || WindowGroups==8u);
    constexpr unsigned groups=kHeadDim/16u,words=sizeof(Row)/4u;
    __shared__ Row rows[WindowGroups][32];
    __shared__ float approximate[WindowGroups][256];
    __shared__ int32_t residue[WindowGroups][256];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u,source=lane%16u;
    const unsigned head=blockIdx.y,kv=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned qr=tid/16u,kc=tid%16u,row=query_tile+qr,column=key_tile+kc;
    const bool live=row<count && column<stride,active=live && column<=start+row;
    const size_t cell=(size_t(row)*kQueryHeads+head)*stride+column;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    if(key_tile>last){if(live)output[cell]=-INFINITY;return;}
    Value carry{0u,-133,false};unsigned accepted=0u,rejected=0u,replayed=0u,recovery_rejected=0u;
    for(unsigned base=0u;base<groups;base+=WindowGroups) {
        for(unsigned item=tid;item<WindowGroups*32u*words;item+=kThreads) {
            const unsigned record=item/words,word=item%words,group=record/32u,r=record%32u;
            const bool left=r<16u;const unsigned position=left?query_tile+r:key_tile+r-16u;
            uint32_t value=0u;
            if(left?position<count:position<stride) {
                const Row* from=left?query+(size_t(position)*kQueryHeads+head)*groups+base+group:
                    key+(size_t(kv)*groups+base+group)*key_stride+position;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(from)+word*4u,4u);
            }
            __builtin_memcpy(reinterpret_cast<unsigned char*>(&rows[group][r])+word*4u,&value,4u);
        }
        __syncthreads();
        if(wave<WindowGroups) {
            const auto parts=compact::products(rows[wave][source],rows[wave][16u+source]);
#pragma unroll
            for(unsigned i=0u;i<8u;++i) {
                const unsigned destination=(2u*i+lane/16u)*16u+source;
                approximate[wave][destination]=parts.approximate[i];residue[wave][destination]=parts.residue[i];
            }
        }
        __syncthreads();
        if(active) {
#pragma unroll 1
            for(unsigned group=0u;group<WindowGroups;++group) {
                const auto& a=rows[group][qr];const auto& b=rows[group][16u+kc];
                const auto* raw_a=raw_query+(size_t(start+row)*kQueryHeads+head)*kHeadDim+(base+group)*16u;
                const auto* raw_b=raw_key+(size_t(kv)*kHeadDim+(base+group)*16u)*key_stride+column;
                Value next;unsigned pairs=0u;int32_t mathematical=0;
                const bool recovered=compact::small::recover(approximate[group][tid],uint32_t(residue[group][tid]),&mathematical);
                if(recovered && compact::dominant(carry,a,b,raw_a,1u,raw_b,key_stride,mathematical,&next,Audit?&pairs:nullptr)) {
                    carry=next;
                    if constexpr(Audit){++accepted;replayed+=pairs;}
                } else {
                    carry=compact::fallback(carry,a,b,raw_a,1u,raw_b,key_stride);
                    if constexpr(Audit){++rejected;recovery_rejected+=unsigned(!recovered);}
                }
            }
        }
        if(base+WindowGroups<groups)__syncthreads();
    }
    if constexpr(Audit) {
        __shared__ unsigned totals[4][kThreads];
        totals[0][tid]=accepted;totals[1][tid]=rejected;totals[2][tid]=replayed;totals[3][tid]=recovery_rejected;__syncthreads();
        for(unsigned step=kThreads/2u;step;step/=2u) {
            if(tid<step)for(unsigned i=0u;i<4u;++i)totals[i][tid]+=totals[i][tid+step];
            __syncthreads();
        }
        if(!tid)for(unsigned i=0u;i<4u;++i)atomicAdd(counters+i,static_cast<unsigned long long>(totals[i][0]));
    }
    if(live)output[cell]=active?qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(carry))*kExactScale:-INFINITY;
}
} // namespace qrt_compact_byte_qk
