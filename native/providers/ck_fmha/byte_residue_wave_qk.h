#ifndef QRT_BYTE_RESIDUE_WAVE_QK_H
#define QRT_BYTE_RESIDUE_WAVE_QK_H
#include "byte_residue32_qk.h"

// Isolated layout experiment: one wave owns the complete16x16 score tile.
// Matrix results stay in their original lanes through canonical carry update;
// there is no shared partial/residue array or redistribution to other waves.
// Arithmetic,116-byte lossless rows and the conditional native error<128
// requirement are inherited unchanged. No production dispatcher uses it.
namespace qrt_byte_residue_wave_qk {
using namespace qrt_blackwell_attention;
using Row=qrt_sm121_byte_residue32::Row;
template<bool PrivateKey>
__global__ __launch_bounds__(32) void scores(const Row* query,const Row* key,float* output,
    unsigned query_start,unsigned query_count,unsigned stride,unsigned key_stride) {
    constexpr unsigned words=sizeof(Row)/4u;
    __shared__ Row left[16],shared_right[PrivateKey?1u:16u];
    const unsigned lane=threadIdx.x,source=lane%16u,half=lane/16u;
    const unsigned head=blockIdx.y,kv_head=head/(kQueryHeads/kKvHeads);
    const unsigned query_tile=blockIdx.z*16u,key_tile=blockIdx.x*16u;
    const unsigned last_query=query_start+min(query_tile+16u,query_count)-1u;
    qrt_q1_moe_hawkeye::Value accumulators[8];
#pragma unroll
    for(unsigned i=0u;i<8u;++i)accumulators[i]={0u,kBlackwellZeroExponent,false};
    if(key_tile<=last_query)for(unsigned group=0u;group<16u;++group) {
        // Coalesced word copies supply query rows shared by all16 key columns.
        // The alternative shares keys too, trading LDS reads for fewer loads
        // and a smaller private live set during the eight carry updates.
        for(unsigned item=lane;item<(PrivateKey?16u:32u)*words;item+=32u) {
            const unsigned row=item/words,word=item%words;const bool is_query=row<16u;
            const unsigned input_row=is_query?query_tile+row:key_tile+row-16u;
            uint32_t value=0u;
            if(input_row<(is_query?query_count:stride)) {
                const Row* original=is_query?query+(size_t(input_row)*kQueryHeads+head)*16u+group:
                    key+(size_t(kv_head)*16u+group)*key_stride+input_row;
                __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(original)+word*4u,4u);
            }
            Row* target=is_query?left+row:shared_right+row-16u;
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target)+word*4u,&value,4u);
        }
        Row private_right{};
        if constexpr(PrivateKey) {
            if(key_tile+source<stride)
                private_right=key[(size_t(kv_head)*16u+group)*key_stride+key_tile+source];
        }
        __syncthreads();
        const Row& right=PrivateKey?private_right:shared_right[source];
        const auto products=qrt_sm121_byte_residue32::products(left[source],right);
#pragma unroll
        for(unsigned item=0u;item<8u;++item) {
            const unsigned row=2u*item+half,output_row=query_tile+row,key_row=key_tile+source;
            if(output_row<query_count&&key_row<stride&&key_row<=query_start+output_row) {
                qrt_sm121_group16::AlignedSum sum;int32_t mathematical=0;
                if(!qrt_sm121_byte_residue32::recover(products.approximate[item],uint32_t(products.residue[item]),&mathematical)||
                    !qrt_sm121_byte_residue32::sum(accumulators[item],left[row],right,mathematical,&sum))
                    sum=qrt_sm121_byte_residue32::fallback(accumulators[item],left[row],right);
                accumulators[item]=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for(unsigned item=0u;item<8u;++item) {
        const unsigned row=query_tile+2u*item+half,key_row=key_tile+source;
        if(row<query_count&&key_row<stride)
            output[(size_t(row)*kQueryHeads+head)*stride+key_row]=key_row<=query_start+row?
                qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(accumulators[item]))*kExactScale:-INFINITY;
    }
}
}
#endif
