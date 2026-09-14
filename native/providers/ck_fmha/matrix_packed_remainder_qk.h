#ifndef QRT_MATRIX_PACKED_REMAINDER_QK_H
#define QRT_MATRIX_PACKED_REMAINDER_QK_H
#include "positive_integer_qk.h"
#include "../moe_accumulator/sm121_packed_core_remainder.h"

// Component experiment only. Separate matrix-lane ownership from scalar
// correction ownership while retaining all original integer/core arithmetic.
namespace qrt_matrix_packed_remainder_qk {
using namespace qrt_blackwell_attention;
struct Row {
    qrt_positive_integer_qk::Row matrix;
    qrt_sm121_core_remainder::Prepared packed;
    uint32_t padding;
};
static_assert(sizeof(Row)==188u); // Odd dword LDS stride.
__host__ __device__ __forceinline__ void prepare(Row& row) {
    qrt_positive_integer_qk::prepare(row.matrix);
    row.matrix.core.original[17]=uint16_t(qrt_sm121_matrix_float_fallback::eligible(row.matrix.core.original));
    row.packed=qrt_sm121_core_remainder::prepare(row.matrix.core);row.padding=0u;
}
template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned tokens,unsigned start,unsigned count) {
    static_assert(Kind==IntegerRowKind::Query || Kind==IntegerRowKind::Key);
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=integer_row_count(Kind,tokens,count))return;
    Row result{};
    for(unsigned i=0u;i<16u;++i)result.matrix.core.original[i]=input[integer_row_input_index(Kind,row,i,tokens,start,count)];
    prepare(result);output[row]=result;
}

template<unsigned Columns, unsigned Policy>
__global__ void scores(const Row* query,const Row* key,float* output,
    unsigned query_start,unsigned query_count,unsigned stride,unsigned key_stride) {
    static_assert(Columns==16u || Columns==32u || Columns==64u);
    static_assert(Policy<=4u);
    constexpr unsigned cells=16u*Columns,per_thread=cells/kThreads,words=sizeof(Row)/4u;
    __shared__ Row left[16],right[Columns];
    __shared__ int32_t partials[4][cells];
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u;
    const unsigned head=blockIdx.y,kv_head=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*Columns;
    const unsigned last_query=query_start+min(query_tile+16u,query_count)-1u;
    qrt_q1_moe_hawkeye::Value accumulators[per_thread];
    for(unsigned i=0u;i<per_thread;++i)accumulators[i]={0u,kBlackwellZeroExponent,false};
    if (key_tile<=last_query) for (unsigned group=0u;group<16u;++group) {
        for (unsigned item=threadIdx.x;item<(16u+Columns)*words;item+=kThreads) {
            const unsigned row=item/words,word=item%words;
            const bool is_query=row<16u;
            const unsigned input_row=is_query?query_tile+row:key_tile+row-16u;
            uint32_t value=0u;
            if (input_row<(is_query?query_count:stride)) {
                const Row* source=is_query?query+(size_t(input_row)*kQueryHeads+head)*16u+group
                    :key+(size_t(kv_head)*16u+group)*key_stride+input_row;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+word*4u,4u);
            }
            Row* target=is_query?left+row:right+row-16u;
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target)+word*4u,&value,4u);
        }
        __syncthreads();
        // Only complete waves enter WMMA. Publish each exact partial once;
        // all256 scalar lanes then consume one, two or four independent cells.
        if (wave<Columns/16u) {
            IntegerMatrixParts matrix{};
            {
                const auto result=qrt_positive_integer_qk::products(left[lane%16u].matrix,right[wave*16u+lane%16u].matrix);
#pragma unroll
                for (unsigned element=0u;element<8u;++element) {
                    const unsigned row=2u*element+lane/16u,column=wave*16u+lane%16u;
                    const int total=int(left[row].matrix.core.original[16])+int(right[column].matrix.core.original[16]);
                    matrix.value[0][element]=result.high[element];
                    matrix.value[1][element]=int(result.combined[element])-result.high[element]-result.low[element]-total*128+262144;
                    matrix.value[3][element]=result.low[element];
                }
            }
#pragma unroll
            for (unsigned element=0u;element<8u;++element) {
                const unsigned row=2u*element+lane/16u,column=wave*16u+lane%16u;
#pragma unroll
                for (unsigned part=0u;part<4u;++part)
                    partials[part][row*Columns+column]=matrix.value[part][element];
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned item=0u;item<per_thread;++item) {
            const unsigned cell=threadIdx.x+item*kThreads,row=cell/Columns,column=cell%Columns;
            const unsigned output_row=query_tile+row,key_row=key_tile+column;
            if (output_row<query_count && key_row<stride && key_row<=query_start+output_row) {
                const int32_t values[4]={partials[0][cell],partials[1][cell],partials[2][cell],partials[3][cell]};
                const auto carry=accumulators[item];
                const int64_t mathematical=int64_t(values[0])*65536+(int64_t(values[1])+values[2])*256+values[3];
                qrt_sm121_group16::AlignedSum sum;
                bool accepted;
                if constexpr(Policy==4u)
                    accepted=qrt_sm121_integer_core::sum_integer_product(carry,left[row].matrix.core,right[column].matrix.core,mathematical,&sum);
                else
                    accepted=qrt_sm121_core_remainder::sum<(Policy&1u)!=0u,(Policy&2u)!=0u>(carry,left[row].matrix.core,right[column].matrix.core,left[row].packed,right[column].packed,mathematical,&sum);
                if(!accepted)
                    qrt_sm121_matrix_float_fallback::sum(carry,left[row].matrix.core,right[column].matrix.core,values,
                        left[row].matrix.core.original[17] && right[column].matrix.core.original[17],&sum);
                accumulators[item]=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned item=0u;item<per_thread;++item) {
        const unsigned cell=threadIdx.x+item*kThreads,row=query_tile+cell/Columns,key_row=key_tile+cell%Columns;
        if (row<query_count && key_row<stride)
            output[(size_t(row)*kQueryHeads+head)*stride+key_row]=key_row<=query_start+row?qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(accumulators[item]))*kExactScale:-INFINITY;
    }
}
} // namespace qrt_matrix_packed_remainder_qk
#endif
