#pragma once
#include "prepared_decoded_qk.h"
#include "../moe_accumulator/sm121_dominant_integer_core.h"

// Reuse staged integer products but keep high/lower in separate LDS planes.
// One lane owns one actual carry. Dominant groups consume the 32-bit shortcut;
// every rejected group retains original exact integer/exception arithmetic.
// There are no native exponent guesses or intermediate acceptance plans.
namespace qrt_dominant_integer_qk {
using namespace qrt_blackwell_attention;
namespace compact=qrt_sm121_dominant_integer_core;
namespace core=qrt_sm121_integer_core;
using Row=core::Row;
using Value=compact::Value;

template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned tokens,unsigned start,unsigned count) {
    static_assert(Kind==IntegerRowKind::Query || Kind==IntegerRowKind::Key);
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=integer_row_count(Kind,tokens,count))return;
    Row value{};
    for(unsigned i=0u;i<16u;++i)value.original[i]=input[integer_row_input_index(Kind,row,i,tokens,start,count)];
    core::prepare(value);output[row]=value;
}
__device__ __forceinline__ Value original(Value carry,const Row& left,const Row& right,
    compact::Product product) {
    qrt_sm121_group16::AlignedSum sum;
    if(!core::sum_integer_product(carry,left,right,compact::mathematical(product),&sum)) {
        uint32_t products[16];
#pragma unroll
        for(unsigned i=0u;i<16u;++i)products[i]=qrt_sm121_group16::pack_product(
            qrt_q1_moe_hawkeye::multiply_bf16(left.original[i],right.original[i],-133));
        sum=qrt_sm121_group16::sum_packed(carry,products);
    }
    return qrt_sm121_canonical::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
}

template<unsigned WindowGroups,bool Audit=false>
__global__ void scores(const Row* query,const Row* key,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride,
    unsigned long long* counters=nullptr) {
    static_assert(WindowGroups==4u || WindowGroups==8u);
    constexpr unsigned groups=kHeadDim/16u,words=sizeof(Row)/4u;
    __shared__ Row rows[WindowGroups][32];
    __shared__ int32_t high[WindowGroups][256],lower[WindowGroups][256];
    const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u,source=lane%16u;
    const unsigned head=blockIdx.y,kv=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned qr=tid/16u,kc=tid%16u,row=query_tile+qr,column=key_tile+kc;
    const bool live=row<count && column<stride,active=live && column<=start+row;
    const size_t cell=(size_t(row)*kQueryHeads+head)*stride+column;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    if(key_tile>last){if(live)output[cell]=-INFINITY;return;}
    Value carry{0u,-133,false};unsigned accepted=0u,rejected=0u,replayed=0u;
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
            const auto parts=blackwell_integer_prepared_products(rows[wave][source],rows[wave][16u+source]);
#pragma unroll
            for(unsigned i=0u;i<8u;++i) {
                const auto product=compact::combine(parts.value[0][i],parts.value[1][i],parts.value[2][i],parts.value[3][i]);
                const unsigned destination=(2u*i+lane/16u)*16u+source;
                high[wave][destination]=product.high;lower[wave][destination]=product.lower;
            }
        }
        __syncthreads();
        if(active) {
#pragma unroll 1
            for(unsigned group=0u;group<WindowGroups;++group) {
                const auto& a=rows[group][qr];const auto& b=rows[group][16u+kc];
                const compact::Product product{high[group][tid],lower[group][tid]};
                Value next;unsigned pairs=0u;
                if(compact::accumulate(carry,a,b,product,&next,Audit?&pairs:nullptr)) {
                    carry=next;
                    if constexpr(Audit){++accepted;replayed+=pairs;}
                } else {
                    carry=original(carry,a,b,product);
                    if constexpr(Audit)++rejected;
                }
            }
        }
        if(base+WindowGroups<groups)__syncthreads();
    }
    if constexpr(Audit) {
        __shared__ unsigned totals[3][kThreads];
        totals[0][tid]=accepted;totals[1][tid]=rejected;totals[2][tid]=replayed;__syncthreads();
        for(unsigned step=kThreads/2u;step;step/=2u) {
            if(tid<step)for(unsigned i=0u;i<3u;++i)totals[i][tid]+=totals[i][tid+step];
            __syncthreads();
        }
        if(!tid)for(unsigned i=0u;i<3u;++i)atomicAdd(counters+i,static_cast<unsigned long long>(totals[i][0]));
    }
    if(live)output[cell]=active?qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(carry))*kExactScale:-INFINITY;
}
} // namespace qrt_dominant_integer_qk
