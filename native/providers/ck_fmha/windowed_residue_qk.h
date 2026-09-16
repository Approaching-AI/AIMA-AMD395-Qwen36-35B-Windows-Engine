#ifndef QRT_WINDOWED_RESIDUE_QK_H
#define QRT_WINDOWED_RESIDUE_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_byte_residue4_core.h"

// Batch independent K16 matrix work before serial canonical carries. All
// intermediate integer dots stay in LDS, without a global partial tensor.
// One wave owns each16x16 group tile; every live consumer preserves K order.
namespace qrt_windowed_residue_qk {
using namespace qrt_blackwell_attention;
namespace core=qrt_sm121_byte_residue4;
using Row=core::Row;
using core::prepare;
template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input,Row* output,unsigned tokens,unsigned start,unsigned count){
    static_assert(Kind==IntegerRowKind::Query || Kind==IntegerRowKind::Key);
    const size_t row=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=integer_row_count(Kind,tokens,count))return;
    Row result{};
    for(unsigned i=0u;i<16u;++i)result.original[i]=input[integer_row_input_index(Kind,row,i,tokens,start,count)];
    prepare(result);output[row]=result;
}
template<unsigned WindowGroups,unsigned Columns>
__global__ void scores(const Row* query,const Row* key,float* output,
    unsigned start,unsigned count,unsigned stride,unsigned key_stride){
    static_assert(WindowGroups==1u || WindowGroups==2u || WindowGroups==4u);
    static_assert(Columns==16u || Columns==32u);
    static_assert(WindowGroups*(Columns/16u)<=8u);
    constexpr unsigned cells=16u*Columns,per_thread=cells/kThreads,words=sizeof(Row)/4u;
    __shared__ Row left[WindowGroups][16],right[WindowGroups][Columns];
    __shared__ int32_t mathematical[WindowGroups][cells];
    const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u;
    const unsigned head=blockIdx.y,kv=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*Columns;
    const unsigned last=start+min(query_tile+16u,count)-1u;
    qrt_q1_moe_hawkeye::Value carry[per_thread];
    for(unsigned i=0u;i<per_thread;++i)carry[i]={0u,kBlackwellZeroExponent,false};
    if(key_tile<=last)for(unsigned base=0u;base<16u;base+=WindowGroups){
        for(unsigned item=threadIdx.x;item<WindowGroups*(16u+Columns)*words;item+=kThreads){
            const unsigned row_index=item/words,word=item%words,group=row_index/(16u+Columns),row=row_index%(16u+Columns);
            const bool is_query=row<16u;
            const unsigned position=is_query?query_tile+row:key_tile+row-16u;
            uint32_t value=0u;
            if(position<(is_query?count:stride)){
                const Row* source=is_query?query+(size_t(position)*kQueryHeads+head)*16u+base+group:
                    key+(size_t(kv)*16u+base+group)*key_stride+position;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(source)+word*4u,4u);
            }
            Row* target=is_query?&left[group][row]:&right[group][row-16u];
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target)+word*4u,&value,4u);
        }
        __syncthreads();
        if(wave<WindowGroups*(Columns/16u)){
            const unsigned group=wave/(Columns/16u),column_tile=(wave%(Columns/16u))*16u;
            const auto products=core::products(left[group][lane%16u],right[group][column_tile+lane%16u]);
#pragma unroll
            for(unsigned item=0u;item<8u;++item){
                int32_t value=INT32_MIN;
                core::recover(products.approximate[item],uint32_t(products.residue[item]),&value);
                mathematical[group][(2u*item+lane/16u)*Columns+column_tile+lane%16u]=value;
            }
        }
        __syncthreads();
#pragma unroll
        for(unsigned item=0u;item<per_thread;++item){
            const unsigned cell=threadIdx.x+item*kThreads,row=cell/Columns,column=cell%Columns;
            const unsigned output_row=query_tile+row,key_row=key_tile+column;
            if(output_row<count && key_row<stride && key_row<=start+output_row){
#pragma unroll 1
                for(unsigned group=0u;group<WindowGroups;++group){
                    qrt_sm121_group16::AlignedSum sum;
                    if(!core::sum(carry[item],left[group][row],right[group][column],mathematical[group][cell],&sum))
                        sum=core::fallback(carry[item],left[group][row],right[group][column]);
                    carry[item]=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(unsigned item=0u;item<per_thread;++item){
        const unsigned cell=threadIdx.x+item*kThreads,row=query_tile+cell/Columns,column=key_tile+cell%Columns;
        if(row<count && column<stride)output[(size_t(row)*kQueryHeads+head)*stride+column]=column<=start+row?
            qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry[item]))*kExactScale:-INFINITY;
    }
}
}
#endif
