#pragma once
#include "streamed_exact_attention.h"
#include "native_exp2_workspace.h"

namespace qrt_fused_probability_pv {
// Consume only a validated, immutable native EXP owner. No allocation or
// host synchronization is introduced; every launch uses the caller's stream.
inline int launch(const void* state,const float* scores,const uint16_t* value,
    uint16_t* probability,float* scales,float* output,float* errors,
    float* raw_accumulator,float* raw_denominator,unsigned start,unsigned count,
    unsigned output_start,unsigned stride,const unsigned char* original,
    const unsigned char* rcp,bool vllm_sum,hipStream_t stream){
    if(!state||!scores||!value||!probability||!scales||!output||!errors||!original||!rcp||
        !vllm_sum||!count||count>128u||start>=8192u||count>8192u-start||stride!=start+count||
        output_start>=qrt_blackwell_attention::kSplitMaxTokens||
        count>qrt_blackwell_attention::kSplitMaxTokens-output_start)return int(hipErrorInvalidValue);
    const auto& workspace=*static_cast<const qrt_native_exp2_workspace::Workspace*>(state);
    if(!workspace.packed||workspace.original!=original)return int(hipErrorInvalidValue);
    const size_t offset=size_t(output_start)*4096u;
    hipLaunchKernelGGL((qrt_streamed_exact_attention::produce<false>),
        dim3(16u,(count+31u)/32u),dim3(256u),0u,stream,
        nullptr,nullptr,value,nullptr,nullptr,nullptr,nullptr,
        probability,scales,output+offset,errors,
        raw_accumulator?raw_accumulator+offset:nullptr,
        raw_denominator?raw_denominator+size_t(output_start)*16u:nullptr,nullptr,
        start,count,stride,stride,original,workspace.packed,rcp,true,scores);
    return int(hipGetLastError());
}
} // namespace qrt_fused_probability_pv
