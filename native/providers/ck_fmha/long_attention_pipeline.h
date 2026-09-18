#pragma once
#include "long_attention_layout.h"
#include "long_narrow_qk.h"
#include "long_fused_probability_pv.h"

namespace qrt_long_attention_pipeline {
inline int launch(const qrt_long_narrow_qk::Workspace& qk,
    const qrt_native_exp2_workspace::Workspace& exp_owner,
    const uint16_t* query,const uint16_t* transposed_key,const uint16_t* value,
    const uint16_t* transposed_value,float* output,unsigned start,unsigned count,
    unsigned output_start,unsigned key_stride,const unsigned char* exp,
    const unsigned char* rcp,float* scratch,size_t scratch_elements,hipStream_t stream,
    qrt_blackwell_attention::SplitCompletionObserver* observer=nullptr) {
    namespace original=qrt_blackwell_attention;
    constexpr unsigned maximum=qrt_long_attention_layout::maximum_tokens;
    if(!query||!transposed_key||!value||!transposed_value||!output||!scratch||!exp||!rcp||
        !count||start>=maximum||count>maximum-start||output_start>=maximum||count>maximum-output_start||
        key_stride<start+count||key_stride>maximum||!exp_owner.packed||exp_owner.original!=exp)
        return int(hipErrorInvalidValue);
    const unsigned stride=start+count;
    const auto offsets=qrt_long_attention_layout::layout(count,stride);
    if(!offsets.elements||scratch_elements<offsets.elements)return int(hipErrorInvalidValue);
    auto* p=reinterpret_cast<uint16_t*>(scratch+offsets.probability);
    auto* scales=scratch+offsets.scales;
    auto* errors=scratch+offsets.errors;
    auto* indices=reinterpret_cast<unsigned*>(scratch+offsets.indices);
    auto* selected=reinterpret_cast<unsigned*>(scratch+offsets.count);
    int status=qrt_long_narrow_qk::launch_workspace(&qk,query,transposed_key,scratch,
        stream,start,count,stride,key_stride);
    if(status!=int(hipSuccess))return status;
    status=original::observe_split_stage(observer,0u,stream);
    if(status!=int(hipSuccess))return status;
    status=qrt_long_fused_probability_pv::launch(&exp_owner,scratch,value,p,scales,output,
        errors,nullptr,nullptr,start,count,output_start,stride,exp,rcp,true,stream);
    if(status!=int(hipSuccess))return status;
    for(unsigned stage=1u;stage<=2u;++stage) {
        status=original::observe_split_stage(observer,stage,stream);
        if(status!=int(hipSuccess))return status;
    }
    return qrt_long_fused_probability_pv::replay(value,transposed_value,p,scales,output,
        errors,nullptr,nullptr,indices,selected,start,count,output_start,stride,key_stride,
        rcp,true,stream,observer);
}
} // namespace qrt_long_attention_pipeline
