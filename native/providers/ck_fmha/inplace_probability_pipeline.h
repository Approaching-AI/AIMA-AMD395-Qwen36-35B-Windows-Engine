#pragma once
#include "long_attention_pipeline.h"

// Isolated full long-attention experiment. QK, softmax, native PV, original
// bounds, candidate collection and original K16 replay keep their arithmetic.
// Only score/P ownership differs. No runtime environment option selects it.
namespace qrt_inplace_probability_pipeline {
inline int probability(const qrt_native_exp2_workspace::Workspace& owner,
    float* scores,const uint16_t* value,float* scales,float* output,float* errors,
    float* raw_accumulator,float* raw_denominator,unsigned start,unsigned count,
    unsigned output_start,unsigned stride,const unsigned char* exp,
    const unsigned char* rcp,hipStream_t stream,bool final_bound) {
    constexpr unsigned maximum=qrt_long_attention_layout::maximum_tokens;
    if(!scores||!value||!scales||!output||!errors||!exp||!rcp||!owner.packed||owner.original!=exp||
        !count||count>128u||start>=maximum||count>maximum-start||stride!=start+count||
        output_start>=maximum||count>maximum-output_start)
        return int(hipErrorInvalidValue);
    const size_t offset=size_t(output_start)*4096u;
    auto* p=reinterpret_cast<uint16_t*>(scores);
    if(final_bound) {
        hipLaunchKernelGGL((qrt_streamed_exact_attention::produce<false,
            qrt_streamed_exact_attention::NativeExp,true,qrt_sm121_pv_long_final_bound::Finalizer,true>),
            dim3(16u,(count+31u)/32u),dim3(256u),0u,stream,
            nullptr,nullptr,value,nullptr,nullptr,nullptr,nullptr,p,scales,output+offset,errors,
            raw_accumulator?raw_accumulator+offset:nullptr,
            raw_denominator?raw_denominator+size_t(output_start)*16u:nullptr,nullptr,
            start,count,stride,stride,exp,owner.packed,rcp,true,scores);
    } else {
        hipLaunchKernelGGL((qrt_streamed_exact_attention::produce<false,
            qrt_streamed_exact_attention::NativeExp,false,qrt_streamed_exact_attention::ShortFinalizer,true>),
            dim3(16u,(count+31u)/32u),dim3(256u),0u,stream,
            nullptr,nullptr,value,nullptr,nullptr,nullptr,nullptr,p,scales,output+offset,errors,
            raw_accumulator?raw_accumulator+offset:nullptr,
            raw_denominator?raw_denominator+size_t(output_start)*16u:nullptr,nullptr,
            start,count,stride,stride,exp,owner.packed,rcp,true,scores);
    }
    return int(hipGetLastError());
}
inline int launch(const qrt_long_narrow_qk::Workspace& qk,
    const qrt_native_exp2_workspace::Workspace& exp_owner,
    const uint16_t* query,const uint16_t* transposed_key,const uint16_t* value,
    const uint16_t* transposed_value,float* output,unsigned start,unsigned count,
    unsigned output_start,unsigned key_stride,const unsigned char* exp,
    const unsigned char* rcp,float* scratch,size_t scratch_elements,hipStream_t stream,
    qrt_blackwell_attention::SplitCompletionObserver* observer=nullptr,bool final_bound=false) {
    namespace original=qrt_blackwell_attention;
    constexpr unsigned maximum=qrt_long_attention_layout::maximum_tokens;
    if(!query||!transposed_key||!value||!transposed_value||!output||!scratch||!exp||!rcp||
        !count||start>=maximum||count>maximum-start||output_start>=maximum||count>maximum-output_start||
        key_stride<start+count||key_stride>maximum||!exp_owner.packed||exp_owner.original!=exp)
        return int(hipErrorInvalidValue);
    const unsigned stride=start+count;
    const auto offsets=qrt_inplace_probability_storage::layout(count,stride);
    if(!offsets.elements||scratch_elements<offsets.elements)return int(hipErrorInvalidValue);
    auto* p=reinterpret_cast<uint16_t*>(scratch);
    auto* scales=scratch+offsets.scales;
    auto* errors=scratch+offsets.errors;
    auto* indices=reinterpret_cast<unsigned*>(scratch+offsets.indices);
    auto* selected=reinterpret_cast<unsigned*>(scratch+offsets.count);
    int status=qrt_long_narrow_qk::launch_workspace(&qk,query,transposed_key,scratch,
        stream,start,count,stride,key_stride);
    if(status!=int(hipSuccess))return status;
    status=original::observe_split_stage(observer,0u,stream);
    if(status!=int(hipSuccess))return status;
    status=probability(exp_owner,scratch,value,scales,output,errors,nullptr,nullptr,
        start,count,output_start,stride,exp,rcp,stream,final_bound);
    if(status!=int(hipSuccess))return status;
    for(unsigned stage=1u;stage<=2u;++stage) {
        status=original::observe_split_stage(observer,stage,stream);
        if(status!=int(hipSuccess))return status;
    }
    return qrt_long_fused_probability_pv::replay<true>(value,transposed_value,p,scales,output,
        errors,nullptr,nullptr,indices,selected,start,count,output_start,stride,key_stride,
        rcp,true,stream,observer);
}
} // namespace qrt_inplace_probability_pipeline
