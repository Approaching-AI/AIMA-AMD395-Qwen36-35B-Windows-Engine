#pragma once
#include "q8192_out_l1_policy.h"
#include "q8192_matrix_producer_policy.h"

namespace qrt_out_l1_replay {
constexpr unsigned rows=2048u,tokens=8192u,width=4096u;
__global__ void finish_bounds(float* magnitude,const unsigned* wf,const unsigned* xf,
    const float* input_norm,const float* weight_norm,unsigned factor) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell>=rows*tokens)return;
    const unsigned row=cell%rows,token=cell/rows;
    const float upper=wf[row] && xf[token]
        ? qrt_bf16_positive_sum_bound::finish(magnitude[cell],width)
        : qrt_bf16_positive_sum_bound::value(0x7f800000u);
    magnitude[cell]=qrt_out_l1_policy::capped(upper,input_norm[token]*weight_norm[row],factor);
}

// Experimental empirical envelope, default off. A matching product GB10
// token/logit boundary is required before retaining any performance result.
// The callback uses the original replay and consumes only the new bound.
// Excluded magnitude rows retain their original Cauchy bound.
template<class Replay>
inline hipError_t run(const uint16_t* weights,const uint16_t* inputs,
    const float* input_norm,const float* weight_norm,unsigned factor,
    unsigned ppb,unsigned radius,hipStream_t stream,Replay replay) {
    if(!weights || !inputs || !input_norm || !weight_norm || !ppb ||
        !factor || factor>32u || (factor&(factor-1u)))return hipErrorInvalidValue;
    unsigned algorithm=99u;
    if(!qrt_q8192_matrix_producer::resolve(
            std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM"),rows,width,tokens,true,&algorithm,
            std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE")) || algorithm!=0u)
        return hipErrorInvalidValue;
#if !defined(QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER)
    return hipErrorInvalidConfiguration;
#else
    const auto start=std::chrono::steady_clock::now();
    auto complete=[&]() -> hipError_t {
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(250);
        for(;;) {
            const hipError_t status=hipStreamQuery(stream);
            if(status!=hipErrorNotReady)return status;
            if(std::chrono::steady_clock::now()>=deadline)return hipErrorLaunchTimeOut;
            std::this_thread::yield();
        }
    };
    // Complete the original producer before reading its operands or starting
    // preparation. Its wait remains inside the product request's timing.
    hipError_t status=complete();if(status!=hipSuccess)return status;
    const size_t operand_words=size_t(rows+tokens)*width;
    const std::array<size_t,3> bytes={operand_words*sizeof(uint16_t),
        size_t(rows)*tokens*sizeof(float),size_t(rows+tokens)*sizeof(unsigned)};
    std::array<void*,3> owned{};
    double preparation_ms=0.0,replay_ms=0.0;
    status=[&]() -> hipError_t {
        hipError_t result;
        for(unsigned i=0u;i<owned.size();++i)
            if((result=hipMalloc(&owned[i],bytes[i]))!=hipSuccess)return result;
        auto* mw=static_cast<uint16_t*>(owned[0]);auto* mx=mw+size_t(rows)*width;
        auto* bound=static_cast<float*>(owned[1]);
        auto* wf=static_cast<unsigned*>(owned[2]);auto* xf=wf+rows;
        hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,
            dim3(rows),dim3(256u),0u,stream,weights,wf,rows,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,
            dim3(tokens),dim3(256u),0u,stream,inputs,xf,tokens,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel,
            dim3(rows),dim3(256u),0u,stream,weights,wf,mw,rows,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        hipLaunchKernelGGL(qrt_bf16_absolute_product_views::prepare_rows_kernel,
            dim3(tokens),dim3(256u),0u,stream,inputs,xf,mx,tokens,width);
        if((result=hipGetLastError())!=hipSuccess || (result=complete())!=hipSuccess)return result;
        std::string failed_stage,failure;
        if(!resident_bf16_matrix_matmul_f32_output_with_heuristic_index(mw,mx,bound,
                rows,width,tokens,4u,stream,"out_l1_replay_magnitude4",&failed_stage,&failure)) {
            std::fprintf(stderr,"BATCH_MARK out_l1_replay_failed stage=%s detail=%s\n",failed_stage.c_str(),failure.c_str());
            return hipErrorInvalidConfiguration;
        }
        hipLaunchKernelGGL(finish_bounds,dim3((rows*tokens+255u)/256u),dim3(256u),0u,stream,
            bound,wf,xf,input_norm,weight_norm,factor);
        if((result=hipGetLastError())!=hipSuccess || (result=complete())!=hipSuccess)return result;
        preparation_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        const auto replay_start=std::chrono::steady_clock::now();
        result=replay(bound);
        replay_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-replay_start).count();
        return result;
    }();
    // The callback and every intermediate launch may leave work pending on
    // failure. Drain before releasing any of the three owning allocations.
    const hipError_t drain=hipStreamSynchronize(stream);
    if(status==hipSuccess)status=drain;
    for(unsigned i=0u;i<owned.size();++i)if(owned[i]) {
        const hipError_t freed=hipFree(owned[i]);if(status==hipSuccess)status=freed;
    }
    std::fprintf(stderr,"BATCH_MARK out_l1_replay rows=%u tokens=%u k=%u factor=%u ppb=%u midpoint_radius=%u workspace_bytes=%zu preparation_ms=%.6f replay_ms=%.6f magnitude_algorithm=4 cauchy_cap=1 empirical_envelope=1 original_k16=1 completed=%u\n",
        rows,tokens,width,factor,ppb,radius,bytes[0]+bytes[1]+bytes[2],preparation_ms,replay_ms,status==hipSuccess?1u:0u);
    std::fflush(stderr);
    return status;
#endif
}
} // namespace qrt_out_l1_replay
