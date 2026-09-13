#ifndef QRT_FLOAT_PV_REPLAY_H
#define QRT_FLOAT_PV_REPLAY_H
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include "../moe_accumulator/sm121_float_subgroup.h"
#include "../gdn/sm121_attention_rcp.h"

namespace qrt_sm121_float_pv {
constexpr unsigned threads=256u, heads=16u, kv_heads=2u, dimensions=256u;

// One flag per causal probability row and per complete V column. The caller
// supplies scratch that remains private until replay completes on this stream.
template<bool Transposed>
__global__ void eligibility_kernel(const uint16_t* value,const uint16_t* probability,
    const uint16_t* transposed,unsigned value_stride,unsigned query_start,
    unsigned query_count,unsigned stride,unsigned* flags) {
    __shared__ unsigned all_valid;
    if (!threadIdx.x) all_valid=1u;
    __syncthreads();
    const unsigned rows=query_count*heads,row=blockIdx.x;
    const bool is_probability=row<rows;
    const unsigned feature=row-rows;
    const unsigned count=is_probability ? query_start+row/heads+1u : stride;
    unsigned valid=1u;
    for (unsigned key=threadIdx.x;key<count;key+=threads) {
        const uint16_t word=is_probability ? probability[size_t(row)*stride+key] :
            Transposed ? transposed[size_t(feature)*value_stride+key] :
                value[size_t(key)*kv_heads*dimensions+feature];
        valid &= unsigned(qrt_sm121_float_alignment::eligible(word));
    }
    for (unsigned delta=16u;delta;delta>>=1u) valid &= __shfl_xor(valid,delta,32u);
    if (!(threadIdx.x&31u) && !valid) atomicAnd(&all_valid,0u);
    __syncthreads();
    if (!threadIdx.x) flags[row]=all_valid;
}

template<unsigned Lanes,bool Transposed>
__global__ void replay_kernel(const uint16_t* value,const uint16_t* probability,
    const float* scales,float* output,unsigned query_start,unsigned query_count,
    unsigned output_start,unsigned stride,const unsigned char* rcp_table,
    float* raw_accumulator,float* raw_denominator,const unsigned* indices,
    const unsigned* count,const uint16_t* transposed,unsigned value_stride,
    const unsigned* flags) {
    static_assert(Lanes==1u || Lanes==4u);
    constexpr unsigned items=16u/Lanes;
    const unsigned lane=threadIdx.x&(Lanes-1u),step=gridDim.x*blockDim.x/Lanes;
    for (unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/Lanes;slot<*count;slot+=step) {
        const unsigned cell=indices[slot],column=cell%dimensions,row=cell/dimensions;
        const unsigned head=row%heads,query=row/heads,kv=head/(heads/kv_heads);
        const unsigned tokens=query_start+query+1u,tiles=(stride+31u)/32u;
        const bool valid=flags[row] && flags[query_count*heads+kv*dimensions+column];
        float accumulator=0.0f;
        for (unsigned tile=0u;tile<(tokens+31u)/32u;++tile) {
            const float alpha=scales[size_t(row)*(tiles+1u)+tile];
            volatile float rounded=accumulator*alpha;
            auto partial=qrt_q1_moe_hawkeye::value_from_float(rounded,-133);
            for (unsigned begin=0u;begin<32u;begin+=16u) {
                uint16_t left[items],right[items];
#pragma unroll
                for (unsigned item=0u;item<items;++item) {
                    const unsigned key=tile*32u+begin+lane*items+item;
                    left[item]=key<tokens ? probability[size_t(row)*stride+key] : 0u;
                    right[item]=key<tokens ? (Transposed ?
                        transposed[(size_t(kv)*dimensions+column)*value_stride+key] :
                        value[(size_t(key)*kv_heads+kv)*dimensions+column]) : 0u;
                }
                if constexpr (Lanes==1u) {
                    qrt_sm121_group16::AlignedSum sum;
                    bool accepted=false;
                    if (valid) {
                        qrt_sm121_float_alignment::Group group;
#pragma unroll
                        for (unsigned item=0u;item<items;++item) group.set(item,left[item],right[item]);
                        accepted=qrt_sm121_float_alignment::sum(partial,group,&sum);
                    }
                    if (!accepted) {
                        uint32_t products[items];
#pragma unroll
                        for (unsigned item=0u;item<items;++item)
                            products[item]=qrt_sm121_group16::pack_product(
                                qrt_q1_moe_hawkeye::multiply_bf16(left[item],right[item],-133));
                        sum=qrt_sm121_group16::sum_packed(partial,products);
                    }
                    partial=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
                } else {
                    qrt_sm121_float_subgroup::Product products[items];
#pragma unroll
                    for (unsigned item=0u;item<items;++item) {
                        const uint16_t a=left[item],b=right[item];
                        const uint32_t original=uint32_t(a)|(uint32_t(b)<<16u);
                        if (!valid) products[item]={0.0f,original,512};
                        else if (!(a&0x7fffu) || !(b&0x7fffu)) products[item]={0.0f,original,-133};
                        else products[item]={qrt_sm121_float_alignment::from_bits(uint32_t(a)<<16u)*
                            qrt_sm121_float_alignment::from_bits(uint32_t(b)<<16u),original,
                            int((a>>7u)&255u)+int((b>>7u)&255u)-254};
                    }
                    partial=qrt_sm121_float_subgroup::accumulate<Lanes>(partial,products);
                }
                // PV rounds after each K16 group, and rescales after each K32
                // tile. These endpoints differ from the QK dot contract.
                partial=qrt_sm121_group16::finish_accumulator(partial);
                partial=qrt_q1_moe_hawkeye::value_from_float(
                    qrt_q1_moe_hawkeye::value_to_float(partial),-133);
            }
            accumulator=qrt_q1_moe_hawkeye::value_to_float(partial);
        }
        if (!lane) {
            const float denominator=scales[size_t(row)*(tiles+1u)+tiles];
            const size_t index=size_t(output_start)*heads*dimensions+cell;
            output[index]=rcp_table ? accumulator*qrt_sm121_attention_rcp::evaluate(rcp_table,denominator) : accumulator/denominator;
            if (raw_accumulator) raw_accumulator[index]=accumulator;
            if (raw_denominator && !column) raw_denominator[size_t(output_start)*heads+row]=denominator;
        }
    }
}

inline int prepare(const uint16_t* value,const uint16_t* probability,
    unsigned query_start,unsigned query_count,unsigned stride,
    const uint16_t* transposed,unsigned value_stride,unsigned* flags,
    size_t flag_words,hipStream_t stream) {
    const size_t rows=size_t(query_count)*heads;
    if (!value || !probability || !flags || !query_count || query_count>128u ||
        query_start>=stride || query_count>stride-query_start || stride>8192u ||
        flag_words<rows+kv_heads*dimensions ||
        (transposed ? value_stride<stride || value_stride>8192u : value_stride!=0u))
        return int(hipErrorInvalidValue);
    if (transposed) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(eligibility_kernel<true>),dim3(rows+kv_heads*dimensions),dim3(threads),0u,stream,
            value,probability,transposed,value_stride,query_start,query_count,stride,flags);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(eligibility_kernel<false>),dim3(rows+kv_heads*dimensions),dim3(threads),0u,stream,
            value,probability,nullptr,0u,query_start,query_count,stride,flags);
    }
    return int(hipGetLastError());
}

inline int launch(const uint16_t* value,const uint16_t* probability,const float* scales,
    float* output,unsigned query_start,unsigned query_count,unsigned output_start,
    unsigned stride,const unsigned char* rcp_table,float* raw_accumulator,
    float* raw_denominator,const unsigned* indices,const unsigned* count,
    const uint16_t* transposed,unsigned value_stride,const unsigned* flags,
    unsigned lanes,hipStream_t stream) {
    if (!value || !probability || !scales || !output || !indices || !count || !flags ||
        (lanes!=1u && lanes!=4u) || !query_count || query_count>128u ||
        query_start>=stride || query_count>stride-query_start || stride>8192u ||
        output_start>=262144u || query_count>262144u-output_start ||
        (transposed ? value_stride<stride || value_stride>8192u : value_stride!=0u))
        return int(hipErrorInvalidValue);
    const unsigned cells=query_count*heads*dimensions;
    const unsigned maximum_blocks=(cells+threads/lanes-1u)/(threads/lanes);
    const unsigned blocks=maximum_blocks<1024u ? maximum_blocks : 1024u;
#define QRT_FLOAT_PV_LAUNCH(L,T) \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<L,T>),dim3(blocks),dim3(threads),0u,stream, \
        value,probability,scales,output,query_start,query_count,output_start,stride,rcp_table, \
        raw_accumulator,raw_denominator,indices,count,transposed,value_stride,flags)
    if (lanes==1u) {
        if (transposed) { QRT_FLOAT_PV_LAUNCH(1u,true); }
        else { QRT_FLOAT_PV_LAUNCH(1u,false); }
    } else {
        if (transposed) { QRT_FLOAT_PV_LAUNCH(4u,true); }
        else { QRT_FLOAT_PV_LAUNCH(4u,false); }
    }
#undef QRT_FLOAT_PV_LAUNCH
    return int(hipGetLastError());
}
} // namespace qrt_sm121_float_pv
#endif
