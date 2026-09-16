#pragma once

// Included after the matrix/replay kernels, BF16 conversion and descriptor
// memory owner. hipMalloc and free_device must use the same descriptor pool.
namespace qrt_coarse_out {
constexpr unsigned rows=2048u,tokens=8192u,width=4096u,cells=rows*tokens;
using Row=qrt_sm121_staged_half_projection::Row;
constexpr size_t prepared_groups=size_t(rows+tokens)*(width/16u);
constexpr size_t workspace_bytes=2u*size_t(cells)*sizeof(float)+
    (size_t(cells)+1u+rows+tokens)*sizeof(unsigned)+prepared_groups*sizeof(Row);
struct Stats {
    unsigned candidates=0u,dispatches=0u;
    double completed_ms=0.0;
    const char* operation="arguments";
};
inline int setting(const char* value) {
    if(!value || !*value || !std::strcmp(value,"0"))return 0;
    return !std::strcmp(value,"1")?1:-1;
}
inline bool applicable(unsigned r,unsigned t,unsigned k,unsigned radius,unsigned ppb) {
    return r==rows && t==tokens && k==width && radius==512u && ppb==10000u;
}
inline bool linear_applicable(unsigned r,unsigned t,unsigned k,unsigned radius,unsigned ppb,
    bool corrected,bool bf16_consumer,bool host_diagnostics) {
    return r==rows && t==tokens && k==width && radius==512u && ppb==1000u &&
        corrected && bf16_consumer && !host_diagnostics;
}
inline bool linear_options_compatible() {
    // These options inspect or change the original matrix/correction route.
    // Never claim to execute them after replacing that route with this owner.
    for(const char* name:{
        "QRT_QWEN36_Q8192_OUT_MATRIX_SHADOW_AUDIT",
        "QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT",
        "QRT_QWEN36_Q8192_OUT_L1_BOUND",
        "QRT_QWEN36_Q8192_OUT_RESIDUAL_FILTER",
        "QRT_QWEN36_Q8192_LINEAR_OUT_VARIANCE_REPLAY",
        "QRT_QWEN36_Q8192_OUT_VARIANCE_BUDGET_AUDIT",
        "QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT"})
        if(setting(std::getenv(name))!=0)return false;
    return true;
}
inline hipError_t run(const uint16_t* weights,const uint16_t* inputs,uint16_t* output,
    unsigned maximum_blocks,hipStream_t stream,Stats* stats,float* rounded_output=nullptr) {
    if(!weights || !inputs || !output || !stats || !maximum_blocks || maximum_blocks>4096u)
        return hipErrorInvalidValue;
    *stats=Stats{};const auto start=std::chrono::steady_clock::now();
    void* storage=nullptr;
    hipError_t status=[&]() -> hipError_t {
        hipError_t result;
        stats->operation="allocate";
        if((result=hipMalloc(&storage,workspace_bytes))!=hipSuccess)return result;
        auto* raw=static_cast<float*>(storage);auto* error=raw+cells;
        auto* ids=reinterpret_cast<unsigned*>(error+cells);auto* count=ids+cells;
        auto* wf=count+1u;auto* xf=wf+rows;
        auto* pw=reinterpret_cast<Row*>(xf+tokens);auto* px=pw+size_t(rows)*(width/16u);
        static_assert(alignof(Row)<=alignof(unsigned) && sizeof(Row)==36u);
        stats->operation="clear_counter";
        if((result=hipMemsetAsync(count,0,sizeof(unsigned),stream))!=hipSuccess)return result;
        stats->operation="prepare_weights";
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,
            dim3((size_t(rows)*(width/16u)+255u)/256u),dim3(256u),0u,stream,weights,pw,rows,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        stats->operation="prepare_inputs";
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,
            dim3((size_t(tokens)*(width/16u)+255u)/256u),dim3(256u),0u,stream,inputs,px,tokens,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        stats->operation="weight_domain";
        hipLaunchKernelGGL(qrt_sm121_coarse_projection_matrix::eligibility,dim3(rows),dim3(256u),0u,stream,weights,wf,rows,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        stats->operation="input_domain";
        hipLaunchKernelGGL(qrt_sm121_coarse_projection_matrix::eligibility,dim3(tokens),dim3(256u),0u,stream,inputs,xf,tokens,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        stats->operation="producer";
        hipLaunchKernelGGL((qrt_sm121_coarse_projection_matrix::produce<64u,1u>),
            dim3(rows/128u,tokens/16u),dim3(256u),0u,stream,weights,inputs,wf,xf,raw,error,rows,tokens,width);
        if((result=hipGetLastError())!=hipSuccess)return result;
        stats->operation="compaction";
        // Each thread reads/writes only its own center; the alias needs no
        // second full output allocation. Replay starts after the count read.
        hipLaunchKernelGGL(qrt_sm121_coarse_projection_matrix::compact,dim3(cells/256u),dim3(256u),0u,stream,
            raw,error,raw,ids,count,cells);
        if((result=hipGetLastError())!=hipSuccess)return result;
        stats->operation="selection_complete";
        if((result=hipStreamSynchronize(stream))!=hipSuccess)return result;
        stats->operation="candidate_count";
        if((result=hipMemcpy(&stats->candidates,count,sizeof(unsigned),hipMemcpyDeviceToHost))!=hipSuccess)return result;
        if(stats->candidates>cells)return hipErrorInvalidValue;
        stats->operation="original_replay";
        const unsigned capacity=maximum_blocks*64u;
        for(unsigned offset=0u;offset<stats->candidates;offset+=capacity) {
            const unsigned n=(std::min)(capacity,stats->candidates-offset);
            hipLaunchKernelGGL(selected_bf16_projection_hawkeye_staged_half_kernel,
                dim3((n+63u)/64u),dim3(256u),0u,stream,pw,px,raw,rows,width,ids,offset,offset+n);
            if((result=hipGetLastError())!=hipSuccess)return result;
            ++stats->dispatches;
        }
        stats->operation="bf16";
        hipLaunchKernelGGL(f32_to_bf16_kernel,dim3(cells/256u),dim3(256u),0u,stream,raw,output,size_t(cells));
        if((result=hipGetLastError())!=hipSuccess)return result;
        if(rounded_output) {
            // Linear tracing can still read the F32 carrier. Populate it from
            // the actual BF16 endpoint; residual/RMSNorm consumes output.
            stats->operation="rounded_f32_carrier";
            hipLaunchKernelGGL(bf16_to_f32_kernel,dim3(cells/256u),dim3(256u),0u,stream,
                output,rounded_output,size_t(cells));
            return hipGetLastError();
        }
        return hipSuccess;
    }();
    // A failed launch/count read can still leave earlier kernels pending.
    // Drain before returning the one allocation to its descriptor owner.
    const hipError_t drained=hipStreamSynchronize(stream);
    if(status==hipSuccess){stats->operation="completion";status=drained;}
    if(storage)free_device(storage);
    stats->completed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return status;
}
}
