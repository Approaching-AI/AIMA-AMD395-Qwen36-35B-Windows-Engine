#pragma once
#include "blackwell_attention.h"

// Component-only scheduling change. Every selected cell retains the original
// REGISTER K32 rescale, two ordered K16 endpoints and final reciprocal.
namespace qrt_row_shared_pv {
constexpr unsigned heads=16u, dimensions=256u, lanes=4u;
struct Workspace {
    uint16_t* columns;
    unsigned* counts;
    size_t column_words, count_words;
};

__global__ void scatter(const unsigned* indices,const unsigned* count,
    uint16_t* columns,unsigned* counts) {
    const unsigned slot=blockIdx.x*blockDim.x+threadIdx.x;
    if(slot>=*count)return;
    const unsigned cell=indices[slot],row=cell/dimensions;
    columns[size_t(row)*dimensions+atomicAdd(counts+row,1u)]=uint16_t(cell%dimensions);
}

template<bool Shared, unsigned Threads=128u>
__global__ void replay(const uint16_t* probability,const uint16_t* value,
    const float* scales,float* output,float* raw_accumulator,float* raw_denominator,
    unsigned start,unsigned output_start,unsigned stride,unsigned value_stride,
    const unsigned char* rcp,Workspace w) {
    static_assert(Threads%32u==0u);
    constexpr unsigned stage=128u, cells=Threads/lanes;
    __shared__ uint16_t staged_probability[Shared?stage:1u];
    const unsigned row=blockIdx.x,selected=w.counts[row];
    if(!selected)return;
    const unsigned member=threadIdx.x%lanes,wave_lane=threadIdx.x%32u;
    const unsigned tokens=start+row/heads+1u,tiles=(stride+31u)/32u;
    const unsigned kv=row%heads/8u;
    for(unsigned base=0u;base<selected;base+=cells){
        const unsigned slot=base+threadIdx.x/lanes;
        const bool active=slot<selected;
        const unsigned column=active?w.columns[size_t(row)*dimensions+slot]:0u;
        float accumulator=0.0f;
        for(unsigned begin=0u;begin<tokens;begin+=stage){
            if constexpr(Shared){
                for(unsigned i=threadIdx.x;i<stage;i+=Threads)
                    staged_probability[i]=begin+i<tokens?probability[size_t(row)*stride+begin+i]:0u;
                __syncthreads();
            }
#pragma unroll
            for(unsigned local_tile=0u;local_tile<stage/32u;++local_tile){
                const unsigned tile=(begin+local_tile*32u)/32u;
                if(tile<(tokens+31u)/32u){
                    const float scaled=qrt_sm121_pv_final_bound::multiply(
                        accumulator,scales[size_t(row)*(tiles+1u)+tile]);
                    auto partial=qrt_q1_moe_hawkeye::value_from_float(scaled,-133);
#pragma unroll
                    for(unsigned group=0u;group<2u;++group){
                        uint32_t products[4];
#pragma unroll
                        for(unsigned item=0u;item<4u;++item){
                            const unsigned local=local_tile*32u+group*16u+member*4u+item;
                            const unsigned key=begin+local;
                            uint32_t p;
                            if constexpr(Shared)p=staged_probability[local];
                            else{
                                p=wave_lane<lanes&&key<tokens?probability[size_t(row)*stride+key]:0u;
                                // All lanes participate, including unused candidate slots.
                                p=__shfl(p,member,32u);
                            }
                            const uint16_t v=active&&key<tokens
                                ?value[(size_t(kv)*dimensions+column)*value_stride+key]:0u;
                            products[item]=qrt_sm121_group16::pack_product(
                                qrt_q1_moe_hawkeye::multiply_bf16(uint16_t(p),v,-133));
                        }
                        partial=qrt_sm121_subgroup::accumulate_products<lanes>(partial,products);
                        partial=qrt_sm121_group16::finish_accumulator(partial);
                        partial=qrt_q1_moe_hawkeye::value_from_float(
                            qrt_q1_moe_hawkeye::value_to_float(partial),-133);
                    }
                    accumulator=qrt_q1_moe_hawkeye::value_to_float(partial);
                }
            }
            if constexpr(Shared)__syncthreads();
        }
        if(active&&!member){
            const float denominator=scales[size_t(row)*(tiles+1u)+tiles];
            const size_t cell=size_t(row)*dimensions+column;
            const size_t index=size_t(output_start)*heads*dimensions+cell;
            output[index]=rcp?accumulator*qrt_sm121_attention_rcp::evaluate(rcp,denominator)
                :accumulator/denominator;
            if(raw_accumulator)raw_accumulator[index]=accumulator;
            if(raw_denominator&&!column)raw_denominator[size_t(output_start)*heads+row]=denominator;
        }
    }
}

template<bool Shared>
inline int launch(const uint16_t* probability,const uint16_t* value,
    const float* scales,float* output,unsigned start,unsigned queries,
    unsigned output_start,unsigned stride,unsigned value_stride,
    const unsigned char* rcp,float* raw_accumulator,float* raw_denominator,
    const unsigned* indices,const unsigned* count,const Workspace& w,hipStream_t stream) {
    if(!probability||!value||!scales||!output||!indices||!count||!w.columns||!w.counts||
        !queries||queries>128u||start>=stride||queries>stride-start||stride>8192u||
        value_stride<stride||value_stride>8192u||
        output_start>=qrt_sm121_attention_capacity::kTokens||
        queries>qrt_sm121_attention_capacity::kTokens-output_start||
        w.column_words<size_t(queries)*heads*dimensions||w.count_words<size_t(queries)*heads)
        return int(hipErrorInvalidValue);
    auto status=hipMemsetAsync(w.counts,0,size_t(queries)*heads*sizeof(unsigned),stream);
    if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(scatter,dim3((queries*heads*dimensions+255u)/256u),dim3(256u),0u,stream,
        indices,count,w.columns,w.counts);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((replay<Shared>),dim3(queries*heads),dim3(128u),0u,stream,
        probability,value,scales,output,raw_accumulator,raw_denominator,
        start,output_start,stride,value_stride,rcp,w);
    return int(hipGetLastError());
}
} // namespace qrt_row_shared_pv
