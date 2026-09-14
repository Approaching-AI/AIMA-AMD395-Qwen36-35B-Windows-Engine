#ifndef QRT_SCALAR_INTEGER_QK_H
#define QRT_SCALAR_INTEGER_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_scalar_integer_core.h"

// Component experiment only: exact byte dot4 products with one scalar owner
// per score. Cooperative K64/K128 windows reuse both prepared operand rows.
namespace qrt_scalar_integer_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_sm121_scalar_integer_core::Row;
using qrt_sm121_scalar_integer_core::prepare;
template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input,Row* output,
    unsigned tokens,unsigned start,unsigned count) {
    static_assert(Kind==IntegerRowKind::Query || Kind==IntegerRowKind::Key);
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=integer_row_count(Kind,tokens,count))return;
    Row result{};
    for(unsigned i=0u;i<16u;++i)
        result.core.original[i]=input[integer_row_input_index(Kind,row,i,tokens,start,count)];
    prepare(result);output[row]=result;
}

template<unsigned Window,bool FastFirst>
__global__ void scores(const Row* query,const Row* key,float* output,
    unsigned query_start,unsigned query_count,unsigned stride,unsigned key_stride) {
    static_assert(Window==64u || Window==128u);
    constexpr unsigned groups=Window/16u,words=sizeof(Row)/4u;
    __shared__ Row left[groups][16],right[groups][16];
    const unsigned head=blockIdx.y,kv_head=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned row=threadIdx.x/16u,column=threadIdx.x%16u;
    const unsigned output_row=query_tile+row,key_row=key_tile+column;
    const unsigned last_query=query_start+min(query_tile+16u,query_count)-1u;
    const bool active=output_row<query_count && key_row<stride && key_row<=query_start+output_row;
    qrt_q1_moe_hawkeye::Value carry{0u,kBlackwellZeroExponent,false};
    if(key_tile<=last_query) for(unsigned start=0u;start<16u;start+=groups) {
        for(unsigned item=threadIdx.x;item<groups*32u*words;item+=kThreads) {
            const unsigned index=item/words,word=item%words,group=index/32u,local=index%32u;
            const bool is_query=local<16u;
            const unsigned input_row=is_query?query_tile+local:key_tile+local-16u;
            uint32_t value=0u;
            if(input_row<(is_query?query_count:stride)) {
                const Row* source=is_query?query+(size_t(input_row)*kQueryHeads+head)*16u+start+group
                    :key+(size_t(kv_head)*16u+start+group)*key_stride+input_row;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+word*4u,4u);
            }
            Row* target=is_query?&left[group][local]:&right[group][local-16u];
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target)+word*4u,&value,4u);
        }
        __syncthreads();
        if(active) {
#pragma unroll 1
            for(unsigned group=0u;group<groups;++group)
                carry=qrt_sm121_scalar_integer_core::accumulate<FastFirst>(carry,left[group][row],right[group][column]);
        }
        __syncthreads();
    }
    if(output_row<query_count && key_row<stride)
        output[(size_t(output_row)*kQueryHeads+head)*stride+key_row]=active?
            qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry))*kExactScale:-INFINITY;
}
} // namespace qrt_scalar_integer_qk
#endif
