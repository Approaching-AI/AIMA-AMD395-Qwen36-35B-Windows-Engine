#pragma once
#include "sm121_staged_half_projection.h"

// Component only. A CTA owns one token's candidates and stages its complete
// lossless input row once. Weight rows stay in global memory. There is one
// entry barrier, no K-window barrier, no dense staging of unselected weights,
// and the original staged2 four-lane recurrence owns each selected output.
namespace qrt_sm121_resident_input_projection {
namespace staged=qrt_sm121_staged_half_projection;
using Row=staged::Row;
constexpr unsigned threads=256u,outputs=threads/4u;

template<unsigned Capacity,unsigned Parts,bool Audit>
__global__ __launch_bounds__(threads) void replay(const Row* weights,const Row* inputs,
    const unsigned* indices,unsigned count,const unsigned* starts,unsigned* status,
    float* output,unsigned rows,unsigned tokens,unsigned width,uint32_t* trace,
    staged::Stats* statistics) {
    static_assert(Capacity==16u || Capacity==128u || Capacity==256u || Capacity==512u);
    static_assert(Parts==1u || Parts==2u);
    __shared__ Row resident[Capacity];
    const unsigned token=blockIdx.x/Parts,part=blockIdx.x%Parts,tid=threadIdx.x;
    const unsigned slot=tid/4u,lane=tid%4u,groups=width/16u;
    if(token>=tokens || *status)return;
    const unsigned begin=starts[token],end=starts[token+1u];
    if(begin>end || end>count) { if(!tid)atomicOr(status,16u);return; }
    if(begin+part*outputs>=end)return;
    for(unsigned word=tid;word<groups*9u;word+=threads) {
        uint32_t value;
        __builtin_memcpy(&value,reinterpret_cast<const unsigned char*>(inputs+size_t(token)*groups)+word*4u,4u);
        __builtin_memcpy(reinterpret_cast<unsigned char*>(resident)+word*4u,&value,4u);
    }
    __syncthreads();
    for(unsigned base=begin+part*outputs;base<end;base+=Parts*outputs) {
        const unsigned index=base+slot;
        if(index<end) {
            const unsigned cell=indices[index];
            if(cell/rows!=token) { if(!lane)atomicOr(status,32u);continue; }
            staged::Stats stats;
            const float value=staged::dot<2u,Audit>(resident,weights+size_t(cell%rows)*groups,
                width,Audit&&trace?trace+size_t(cell)*groups*3u:nullptr,Audit?&stats:nullptr);
            if(!lane) {
                output[cell]=value;
                if constexpr(Audit)if(statistics)statistics[cell]=stats;
            }
        }
    }
}
template<unsigned Capacity,unsigned Parts>
inline hipError_t dispatch(const Row* weights,const Row* inputs,const unsigned* indices,
    unsigned count,const unsigned* starts,unsigned* status,float* output,
    unsigned rows,unsigned tokens,unsigned width,hipStream_t stream,
    uint32_t* trace,staged::Stats* statistics) {
    if(trace || statistics) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(replay<Capacity,Parts,true>),dim3(tokens*Parts),dim3(threads),0u,stream,
            weights,inputs,indices,count,starts,status,output,rows,tokens,width,trace,statistics);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(replay<Capacity,Parts,false>),dim3(tokens*Parts),dim3(threads),0u,stream,
            weights,inputs,indices,count,starts,status,output,rows,tokens,width,nullptr,nullptr);
    }
    return hipGetLastError();
}
// starts has tokens+1 entries describing an exact permutation of unique
// candidate identities. The preceding bucketer supplies that contract. This
// launcher also rejects malformed spans and wrong-token identities on device;
// callers retain every buffer until stream completion and check status.
inline hipError_t launch(const Row* weights,size_t weight_records,const Row* inputs,size_t input_records,
    const unsigned* indices,size_t index_words,unsigned count,const unsigned* starts,size_t start_words,
    unsigned* status,float* output,size_t output_cells,unsigned rows,unsigned tokens,unsigned width,
    unsigned parts,hipStream_t stream,uint32_t* trace=nullptr,size_t trace_words=0u,
    staged::Stats* statistics=nullptr,size_t statistics_cells=0u) {
    const size_t cells=size_t(rows)*tokens;
    if(!weights || !inputs || !indices || !starts || !status || !output ||
        !rows || rows>16384u || !tokens || tokens>8192u || !width || width>8192u || width%16u ||
        (parts!=1u && parts!=2u) || count>cells || index_words<count || start_words<size_t(tokens)+1u ||
        weight_records<size_t(rows)*(width/16u) || input_records<size_t(tokens)*(width/16u) || output_cells<cells ||
        (trace && trace_words<cells*(width/16u)*3u) || (!trace && trace_words) ||
        (statistics && statistics_cells<cells) || (!statistics && statistics_cells))return hipErrorInvalidValue;
#define QRT_RESIDENT_INPUT_DISPATCH(capacity,partitions) \
    return dispatch<capacity,partitions>(weights,inputs,indices,count,starts,status,output,rows,tokens,width,stream,trace,statistics)
    if(width<=256u){if(parts==1u){QRT_RESIDENT_INPUT_DISPATCH(16u,1u);}QRT_RESIDENT_INPUT_DISPATCH(16u,2u);}
    if(width<=2048u){if(parts==1u){QRT_RESIDENT_INPUT_DISPATCH(128u,1u);}QRT_RESIDENT_INPUT_DISPATCH(128u,2u);}
    if(width<=4096u){if(parts==1u){QRT_RESIDENT_INPUT_DISPATCH(256u,1u);}QRT_RESIDENT_INPUT_DISPATCH(256u,2u);}
    if(parts==1u){QRT_RESIDENT_INPUT_DISPATCH(512u,1u);}QRT_RESIDENT_INPUT_DISPATCH(512u,2u);
#undef QRT_RESIDENT_INPUT_DISPATCH
}
} // namespace qrt_sm121_resident_input_projection
