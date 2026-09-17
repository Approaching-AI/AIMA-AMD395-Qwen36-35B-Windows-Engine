#pragma once
#include "native_delta_probability.h"

namespace qrt_native_exp2_workspace {
namespace delta=qrt_sm121_exp2_native_delta;
struct Workspace {
    unsigned char* packed=nullptr;
    const unsigned char* original=nullptr;
};
// The caller owns the same lock as the immutable source table. Publish only
// after a separate complete-domain check succeeds on this execution device.
inline int prepare(Workspace& workspace,const unsigned char* original,hipStream_t stream){
    if(!original)return int(hipErrorInvalidValue);
    if(workspace.packed)return workspace.original==original?int(hipSuccess):int(hipErrorInvalidValue);
    unsigned char* next=nullptr;unsigned* bad=nullptr;unsigned errors=0u;
    auto status=hipMalloc(reinterpret_cast<void**>(&next),delta::packed_bytes);
    if(status==hipSuccess)status=hipMalloc(reinterpret_cast<void**>(&bad),sizeof(unsigned));
    if(status==hipSuccess)status=hipMemsetAsync(bad,0,sizeof(unsigned),stream);
    if(status==hipSuccess){
        hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,stream,original,next);
        status=hipGetLastError();
    }
    if(status==hipSuccess){
        hipLaunchKernelGGL(delta::verify,dim3(4096u),dim3(256u),0u,stream,original,next,bad);
        status=hipGetLastError();
    }
    if(status==hipSuccess)status=hipMemcpyAsync(&errors,bad,sizeof(unsigned),hipMemcpyDeviceToHost,stream);
    // Drain submitted work before freeing buffers or publishing the owner,
    // including on a failed intermediate submission.
    const auto completed=hipStreamSynchronize(stream);
    if(status==hipSuccess)status=completed;
    (void)hipFree(bad);
    if(status==hipSuccess&&errors)status=hipErrorInvalidValue;
    if(status!=hipSuccess){(void)hipFree(next);return int(status);}
    workspace={next,original};return int(hipSuccess);
}
inline void release(Workspace& workspace){
    (void)hipFree(workspace.packed);workspace={};
}
inline int launch(const void* state,const float* scores,uint16_t* probability,float* scales,
    unsigned start,unsigned count,unsigned stride,const unsigned char* original,
    bool vllm_sum,hipStream_t stream){
    if(!state||!scores||!probability||!scales||!original||!count||count>128u||
        start>=8192u||count>8192u-start||stride!=start+count)return int(hipErrorInvalidValue);
    const auto& workspace=*static_cast<const Workspace*>(state);
    if(!workspace.packed||workspace.original!=original)return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(qrt_native_delta_probability::probabilities,
        dim3(qrt_blackwell_attention::kQueryHeads,count),dim3(32u),0u,stream,
        scores,probability,scales,start,stride,original,vllm_sum,workspace.packed);
    return int(hipGetLastError());
}
} // namespace qrt_native_exp2_workspace
