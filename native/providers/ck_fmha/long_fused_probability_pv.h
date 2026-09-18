#pragma once
#include "streamed_exact_attention.h"
#include "native_exp2_workspace.h"

// Component-only long-history producer. Preserve every original per-group
// error bound rather than extending the short-context final-envelope proof.
namespace qrt_long_fused_probability_pv {
inline int launch(const void* state,const float* scores,const uint16_t* value,
    uint16_t* probability,float* scales,float* output,float* errors,
    float* raw_accumulator,float* raw_denominator,unsigned start,unsigned count,
    unsigned output_start,unsigned stride,const unsigned char* original,
    const unsigned char* rcp,bool vllm_sum,hipStream_t stream) {
    constexpr unsigned maximum=qrt_blackwell_attention::kSplitMaxTokens;
    if(!state||!scores||!value||!probability||!scales||!output||!errors||!original||!rcp||
        !vllm_sum||!count||count>128u||start>=maximum||count>maximum-start||
        stride!=start+count||output_start>=maximum||count>maximum-output_start)
        return int(hipErrorInvalidValue);
    const auto& w=*static_cast<const qrt_native_exp2_workspace::Workspace*>(state);
    if(!w.packed||w.original!=original)return int(hipErrorInvalidValue);
    const size_t offset=size_t(output_start)*4096u;
    hipLaunchKernelGGL((qrt_streamed_exact_attention::produce<false,
        qrt_streamed_exact_attention::NativeExp,false>),
        dim3(16u,(count+31u)/32u),dim3(256u),0u,stream,
        nullptr,nullptr,value,nullptr,nullptr,nullptr,nullptr,
        probability,scales,output+offset,errors,
        raw_accumulator?raw_accumulator+offset:nullptr,
        raw_denominator?raw_denominator+size_t(output_start)*16u:nullptr,nullptr,
        start,count,stride,stride,original,w.packed,rcp,true,scores);
    return int(hipGetLastError());
}
} // namespace qrt_long_fused_probability_pv
