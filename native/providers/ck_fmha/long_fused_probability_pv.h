#pragma once
#include "streamed_exact_attention.h"
#include "native_exp2_workspace.h"

// Component-only long-history producer. Preserve every original per-group
// error bound rather than extending the short-context final-envelope proof.
namespace qrt_long_fused_probability_pv {
template<bool InplaceProbability = false, bool PackedProbability = false>
inline int replay(const uint16_t* value,const uint16_t* transposed_value,
    const uint16_t* probability,const float* scales,float* output,const float* errors,
    float* raw_accumulator,float* raw_denominator,unsigned* indices,unsigned* selected,
    unsigned start,unsigned count,unsigned output_start,unsigned stride,unsigned value_stride,
    const unsigned char* rcp,bool register_rescale,hipStream_t stream,
    qrt_blackwell_attention::SplitCompletionObserver* observer=nullptr) {
    namespace original=qrt_blackwell_attention;
    constexpr unsigned maximum=original::kSplitMaxTokens;
    if(!value||!transposed_value||!probability||!scales||!output||!errors||!indices||!selected||!rcp||
        !count||count>128u||start>=maximum||count>maximum-start||stride!=start+count||
        value_stride<stride||value_stride>maximum||output_start>=maximum||count>maximum-output_start)
        return int(hipErrorInvalidValue);
    const unsigned cells=count*4096u,blocks=std::min((cells+63u)/64u,1024u);
    auto status=hipMemsetAsync(selected,0,sizeof(unsigned),stream);
    if(status!=hipSuccess)return int(status);
    hipLaunchKernelGGL(original::blackwell_collect_pv_replay_kernel,
        dim3((cells+255u)/256u),dim3(256u),0u,stream,
        output,errors,output_start,cells,indices,selected);
    status=hipGetLastError();if(status!=hipSuccess)return int(status);
    const int collected=original::observe_split_stage(observer,3u,stream);
    if(collected!=int(hipSuccess))return collected;
    if(register_rescale) {
        hipLaunchKernelGGL((original::blackwell_compacted_pv_replay_kernel<true,false,true,InplaceProbability,PackedProbability>),
            dim3(blocks),dim3(256u),0u,stream,value,probability,scales,output,start,output_start,
            stride,rcp,raw_accumulator,raw_denominator,indices,selected,transposed_value,value_stride,0u);
    }else {
        hipLaunchKernelGGL((original::blackwell_compacted_pv_replay_kernel<true,false,false,InplaceProbability,PackedProbability>),
            dim3(blocks),dim3(256u),0u,stream,value,probability,scales,output,start,output_start,
            stride,rcp,raw_accumulator,raw_denominator,indices,selected,transposed_value,value_stride,0u);
    }
    status=hipGetLastError();
    return status==hipSuccess?original::observe_split_stage(observer,4u,stream):int(status);
}
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
