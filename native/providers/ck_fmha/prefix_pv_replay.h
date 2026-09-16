#pragma once
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_pv_prefix_bound.h"

// Isolated experiment. Original ordered WMMA, alpha rescaling and admission
// remain intact. Selected cells replay from zero in compacted K-chunk rounds;
// only a complete suffix interval can end a replay before its final K16.
namespace qrt_prefix_pv {
namespace original = qrt_blackwell_attention;
namespace certificate = qrt_sm121_pv_prefix_bound;
namespace deferred = qrt_sm121_pv_final_bound;
namespace bound = qrt_sm121_pv_bound;
using Checkpoint = certificate::Checkpoint;
using Weight = certificate::Weight;
using Interval = certificate::Interval;
constexpr unsigned heads=16u, dimensions=256u, threads=256u;
struct Entry { unsigned cell; float carry; };
struct Workspace {
    Checkpoint* checkpoints;
    Weight* weights;
    Entry* first;
    Entry* second;
    unsigned* counts;
    unsigned* decisions;
    Interval* intervals;
    size_t checkpoint_slots, weight_slots, queue_slots, count_slots, cell_slots;
};

template<unsigned Chunk>
__global__ void produce(const uint16_t* value, const uint16_t* probabilities,
    const float* scales, float* output, unsigned start, unsigned queries,
    unsigned output_start, unsigned stride, const unsigned char* rcp,
    float* raw_accumulator, float* raw_denominator, float* error_bounds,
    Checkpoint* checkpoints) {
    static_assert(Chunk==64u || Chunk==512u || Chunk==1024u);
    const unsigned lane=threadIdx.x%32u, wave=threadIdx.x/32u, head=blockIdx.y;
    const unsigned column_tile=blockIdx.x*original::kIntegerMatrixColumns;
    const unsigned query_tile=blockIdx.z*16u, kv=head/8u;
    const unsigned tile_stride=(stride+31u)/32u, cells=queries*heads*dimensions;
    const unsigned chunks=(stride+Chunk-1u)/Chunk;
    const unsigned last_tokens=start+min(query_tile+16u,queries), end=(last_tokens+31u)/32u*32u;
    original::MantissaF32x8 accumulator{}, state{};
    for (unsigned base=0u; base<end; base+=16u) {
        original::NativeOperandRow left{},right{};
        const unsigned operand_row=query_tile+lane%16u, tokens=start+operand_row+1u;
        const unsigned column=column_tile+wave*16u+lane%16u;
#pragma unroll
        for (unsigned i=0; i<16u; ++i) {
            const unsigned key=base+i;
            left.original[i]=operand_row<queries && key<tokens ?
                probabilities[(size_t(operand_row)*heads+head)*stride+key]:0u;
            right.original[i]=key<last_tokens ? value[(size_t(key)*2u+kv)*dimensions+column]:0u;
        }
        if (!(base%32u)) {
#pragma unroll
            for (unsigned element=0; element<8u; ++element) {
                const unsigned row=query_tile+2u*element+lane/16u;
                if (row<queries && base/32u<(start+row+32u)/32u) {
                    const float alpha=scales[(size_t(row)*heads+head)*(tile_stride+1u)+base/32u];
                    state[element]=deferred::rescale(state[element],accumulator[element],alpha);
                    volatile float rounded=accumulator[element]*alpha;accumulator[element]=rounded;
                }
            }
        }
        const auto next=original::blackwell_native_mma(left,right,accumulator);
        const auto absolute=original::blackwell_native_mma<true>(left,right,original::MantissaF32x8{});
#pragma unroll
        for (unsigned element=0; element<8u; ++element) {
            const unsigned row=query_tile+2u*element+lane/16u, row_end=(start+row+32u)/32u*32u;
            if (row<queries && base<row_end) {
                state[element]=deferred::group(state[element],accumulator[element],absolute[element]);
                accumulator[element]=next[element];
                if ((base+16u)%Chunk==0u && base+16u<row_end) {
                    const unsigned cell=(row*heads+head)*dimensions+column;
                    checkpoints[size_t((base+16u)/Chunk-1u)*cells+cell]={accumulator[element],state[element]};
                }
            }
        }
    }
#pragma unroll
    for (unsigned element=0; element<8u; ++element) {
        const unsigned row=query_tile+2u*element+lane/16u;
        const unsigned column=column_tile+wave*16u+lane%16u;
        if (row<queries) {
            const unsigned cell=(row*heads+head)*dimensions+column;
            const size_t index=size_t(output_start)*heads*dimensions+cell;
            const float denominator=scales[(size_t(row)*heads+head)*(tile_stride+1u)+tile_stride];
            const float reciprocal=qrt_sm121_attention_rcp::evaluate(rcp,denominator);
            output[index]=accumulator[element]*reciprocal;
            error_bounds[cell]=bound::finish(deferred::finalize(state[element],
                (start+row+32u)/32u*2u),accumulator[element],reciprocal);
            checkpoints[size_t(chunks-1u)*cells+cell]={accumulator[element],state[element]};
            if (raw_accumulator)raw_accumulator[index]=accumulator[element];
            if (raw_denominator && !column)raw_denominator[size_t(output_start+row)*heads+head]=denominator;
        }
    }
}

template<unsigned Chunk>
__global__ void suffix_weights(const float* scales,Weight* weights,
    unsigned start,unsigned queries,unsigned stride) {
    const unsigned row=blockIdx.x*blockDim.x+threadIdx.x, rows=queries*heads;
    if (row>=rows)return;
    const unsigned tiles=(stride+31u)/32u, end=(start+row/heads+32u)/32u;
    const unsigned chunks=(stride+Chunk-1u)/Chunk;
    Weight weight{1.0,1.0};
    weights[size_t(chunks-1u)*rows+row]=weight;
    for (unsigned tile=end;tile-->0u;) {
        weight=certificate::multiply(weight,scales[size_t(row)*(tiles+1u)+tile]);
        if (tile && tile*32u%Chunk==0u)
            weights[size_t(tile*32u/Chunk-1u)*rows+row]=weight;
    }
}
__global__ void collect(const float* output,const float* error,unsigned output_start,
    unsigned cells,Entry* entries,unsigned* count,unsigned* decisions) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=cells)return;
    decisions[cell]=0u;
    if (!bound::same_bf16(output[size_t(output_start)*heads*dimensions+cell],error[cell]))
        entries[atomicAdd(count,1u)]={cell,0.0f};
}

template<unsigned Chunk>
__global__ void replay(const uint16_t* probability,const uint16_t* transposed_value,
    const float* scales,float* output,unsigned start,unsigned queries,unsigned output_start,
    unsigned stride,unsigned value_stride,const unsigned char* rcp,float* raw_accumulator,
    const Checkpoint* checkpoints,const Weight* weights,const Entry* input,Entry* next,
    const unsigned* input_count,unsigned* next_count,unsigned* decisions,Interval* intervals,
    unsigned phase) {
    constexpr unsigned lanes=4u,items=4u;
    const unsigned member=threadIdx.x&3u,step=gridDim.x*blockDim.x/lanes;
    const unsigned cells=queries*heads*dimensions,rows=queries*heads;
    const unsigned chunks=(stride+Chunk-1u)/Chunk,tiles=(stride+31u)/32u;
    for (unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/lanes;slot<*input_count;slot+=step) {
        const Entry entry=input[slot];const unsigned cell=entry.cell,column=cell%dimensions,row=cell/dimensions;
        const unsigned query=row/heads,head=row%heads,kv=head/8u,tokens=start+query+1u;
        const unsigned end=(tokens+31u)/32u*32u,begin=phase*Chunk,limit=min(begin+Chunk,end);
        float accumulator=entry.carry;
        for (unsigned tile=begin/32u;tile<limit/32u;++tile) {
            const float alpha=scales[size_t(row)*(tiles+1u)+tile];
            volatile float rounded=accumulator*alpha;
            auto partial=qrt_q1_moe_hawkeye::value_from_float(rounded,-133);
            for (unsigned group=0;group<32u;group+=16u) {
                uint32_t products[items];
#pragma unroll
                for (unsigned item=0;item<items;++item) {
                    const unsigned key=tile*32u+group+member*items+item;
                    const uint16_t p=key<tokens?probability[size_t(row)*stride+key]:0u;
                    const uint16_t v=key<tokens?transposed_value[(size_t(kv)*dimensions+column)*value_stride+key]:0u;
                    products[item]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(p,v,-133));
                }
                partial=qrt_sm121_subgroup::accumulate_products<lanes>(partial,products);
                partial=qrt_sm121_group16::finish_accumulator(partial);
                partial=qrt_q1_moe_hawkeye::value_from_float(qrt_q1_moe_hawkeye::value_to_float(partial),-133);
            }
            accumulator=qrt_q1_moe_hawkeye::value_to_float(partial);
        }
        if (!member) {
            const float reciprocal=qrt_sm121_attention_rcp::evaluate(rcp,scales[size_t(row)*(tiles+1u)+tiles]);
            const size_t index=size_t(output_start)*heads*dimensions+cell;
            if (limit==end) {
                output[index]=accumulator*reciprocal;
                if(raw_accumulator)raw_accumulator[index]=accumulator;
                decisions[cell]=0x80000000u|limit/16u;
                intervals[cell]={double(accumulator),double(accumulator)};
            } else {
                const auto interval=certificate::suffix(checkpoints[size_t(phase)*cells+cell],
                    checkpoints[size_t(chunks-1u)*cells+cell],accumulator,
                    weights[size_t(phase)*rows+row],limit/16u,end/16u);
                float representative=0.0f;
                if(certificate::certified(interval,reciprocal,&representative)) {
                    output[index]=representative;
                    // This is a certified representative, not an exact raw endpoint.
                    if(raw_accumulator)raw_accumulator[index]=float(interval.lower*0.5+interval.upper*0.5);
                    decisions[cell]=limit/16u;intervals[cell]=interval;
                } else next[atomicAdd(next_count,1u)]={cell,accumulator};
            }
        }
    }
}

template<unsigned Chunk>
inline int launch(const uint16_t* value,const uint16_t* probability,const float* scales,
    float* output,float* error,unsigned start,unsigned queries,unsigned output_start,
    unsigned stride,const unsigned char* rcp,const uint16_t* transposed_value,
    unsigned value_stride,Workspace workspace,hipStream_t stream,
    float* raw_accumulator=nullptr,float* raw_denominator=nullptr) {
    static_assert(Chunk==64u || Chunk==512u || Chunk==1024u);
    const unsigned cells=queries*heads*dimensions,rows=queries*heads,chunks=(stride+Chunk-1u)/Chunk;
    if(!value||!probability||!scales||!output||!error||!rcp||!transposed_value||
        !queries||queries>128u||!stride||stride>8192u||start>=stride||queries!=stride-start||
        value_stride<stride||value_stride>8192u||output_start>=qrt_sm121_attention_capacity::kTokens||
        queries>qrt_sm121_attention_capacity::kTokens-output_start||
        !workspace.checkpoints||!workspace.weights||!workspace.first||!workspace.second||
        !workspace.counts||!workspace.decisions||!workspace.intervals||
        workspace.checkpoint_slots<size_t(cells)*chunks||workspace.weight_slots<size_t(rows)*chunks||
        workspace.queue_slots<cells||workspace.count_slots<chunks+1u||workspace.cell_slots<cells)
        return int(hipErrorInvalidValue);
    auto status=hipMemsetAsync(workspace.counts,0,(chunks+1u)*sizeof(unsigned),stream);
    if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((produce<Chunk>),dim3(dimensions/original::kIntegerMatrixColumns,heads,(queries+15u)/16u),
        dim3(threads),0u,stream,value,probability,scales,output,start,queries,output_start,stride,rcp,
        raw_accumulator,raw_denominator,error,workspace.checkpoints);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL((suffix_weights<Chunk>),dim3((rows+255u)/256u),dim3(256u),0u,stream,
        scales,workspace.weights,start,queries,stride);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(collect,dim3((cells+255u)/256u),dim3(256u),0u,stream,
        output,error,output_start,cells,workspace.first,workspace.counts,workspace.decisions);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    const unsigned blocks=(cells+63u)/64u<1024u?(cells+63u)/64u:1024u;
    for(unsigned phase=0;phase<chunks;++phase) {
        hipLaunchKernelGGL((replay<Chunk>),dim3(blocks),dim3(threads),0u,stream,
            probability,transposed_value,scales,output,start,queries,output_start,stride,value_stride,rcp,
            raw_accumulator,workspace.checkpoints,workspace.weights,phase&1u?workspace.second:workspace.first,
            phase&1u?workspace.first:workspace.second,workspace.counts+phase,workspace.counts+phase+1u,
            workspace.decisions,workspace.intervals,phase);
        status=hipGetLastError();if(status!=hipSuccess)return int(status);
    }
    return int(hipSuccess);
}
} // namespace qrt_prefix_pv
