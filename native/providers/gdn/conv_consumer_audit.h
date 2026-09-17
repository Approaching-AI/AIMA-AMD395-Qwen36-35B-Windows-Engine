#pragma once
// Included after the unchanged production selector and convolution kernels.
// Read-only observation: the callback always performs every original replay.
namespace qrt_conv_consumer_audit {
namespace c = qrt_conv_consumer;
constexpr unsigned rows = 8192u, tokens = 8192u;
constexpr size_t cells = size_t(rows)*tokens;
constexpr unsigned constant_bit = 1u<<16u, selected_bit = 1u<<17u;
constexpr unsigned guard = 128u, metrics = 16u;
enum Metric { Elements, Selected, Valid, Wide, Tiny, Omittable, WideOmittable,
    Constant, Halo, Changed, OmittedChanged, RangeFailures, CertificateFailures,
    UnselectedChanges, Nonfinite, TinyNonzero };
struct View {
    const float* native;
    const float* input_l2;
    const float* weight_l2;
    const uint16_t* weights;
    const unsigned char* table;
};

__global__ void certificates(View view,uint32_t* result) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=cells)return;
    const unsigned token=unsigned(i/rows),feature=unsigned(i%rows);
    c::c::Range ranges[4];uint16_t taps[4];unsigned present=0u;bool own_selected=false;
    for(unsigned tap=0u;tap<4u;++tap){
        taps[tap]=view.weights[feature*4u+tap];
        if(token+tap<3u)continue;
        const unsigned source=token+tap-3u;const size_t index=size_t(source)*rows+feature;
        const bool selected=selected_bf16_projection_hawkeye_candidate(view.native[index],index,rows,
            512u,0u,1000u,nullptr,view.input_l2,view.weight_l2);
        const float error=(view.input_l2[source]*view.weight_l2[feature])*(1000.0f*1.0e-9f);
        ranges[tap]=c::endpoint(view.native[index],error,selected);present|=1u<<tap;
        if(tap==3u)own_selected=selected;
    }
    const auto cert=c::certify(ranges,taps,present,view.table);
    result[i]=(cert.constant?constant_bit|cert.output:0u)|(own_selected?selected_bit:0u);
}

__global__ __launch_bounds__(256) void inspect(View view,const float* corrected,
    const float* convolution,const uint32_t* certificate,unsigned long long* result) {
    unsigned local[metrics]{};
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<cells;i+=size_t(gridDim.x)*blockDim.x){
        const unsigned token=unsigned(i/rows),feature=unsigned(i%rows),word=certificate[i];
        const bool selected=(word&selected_bit)!=0u,constant=(word&constant_bit)!=0u;
        const float native=view.native[i],exact=corrected[i];
        const bool tiny=((c::c::bits(native)>>23u)&255u)<32u;
        const bool changed=c::c::rounded(native)!=c::c::rounded(exact);
        const float error=(view.input_l2[token]*view.weight_l2[feature])*(1000.0f*1.0e-9f);
        const auto range=c::endpoint(native,error,selected);
        const bool wide=range.valid&&unsigned(range.high-range.low)>8u;
        bool following[4]{};
        for(unsigned j=0u;j<4u&&token+j<tokens;++j)following[j]=(certificate[i+size_t(j)*rows]&constant_bit)!=0u;
        const bool omit=selected&&c::can_omit(token,tokens,following);
        ++local[Elements];local[Selected]+=selected;local[Valid]+=selected&&range.valid;
        local[Wide]+=selected&&wide;local[Tiny]+=selected&&tiny;
        local[Omittable]+=omit;local[WideOmittable]+=omit&&wide;
        local[Constant]+=constant;local[Halo]+=selected&&tokens-token<=3u;
        local[Changed]+=changed;local[OmittedChanged]+=omit&&changed;
        local[RangeFailures]+=selected&&range.valid&&!c::c::contains(range,exact);
        local[CertificateFailures]+=constant&&uint16_t(word)!=c::c::rounded(convolution[i]);
        local[UnselectedChanges]+=!selected&&changed;
        local[Nonfinite]+=!c::c::finite(native)||!c::c::finite(exact)||!c::c::finite(convolution[i]);
        local[TinyNonzero]+=selected&&tiny&&(c::c::bits(native)&0x7fffffffu)!=0u;
    }
    __shared__ unsigned partial[metrics][8];
    for(unsigned m=0u;m<metrics;++m){
        unsigned value=local[m];
        for(unsigned step=16u;step;step>>=1u)value+=__shfl_down(value,step,32u);
        if((threadIdx.x&31u)==0u)partial[m][threadIdx.x/32u]=value;
    }
    __syncthreads();
    if(threadIdx.x<metrics){unsigned value=0u;for(unsigned w=0u;w<8u;++w)value+=partial[threadIdx.x][w];
        atomicAdd(result+threadIdx.x,static_cast<unsigned long long>(value));}
}
inline hipError_t complete(hipStream_t stream) {
    hipEvent_t event=nullptr;auto status=hipEventCreateWithFlags(&event,hipEventDisableTiming);
    if(status!=hipSuccess)return status;
    status=hipEventRecord(event,stream);
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    while(status==hipSuccess){
        status=hipEventQuery(event);
        if(status!=hipErrorNotReady)break;
        if(std::chrono::steady_clock::now()>=deadline){status=hipErrorNotReady;break;}
        std::this_thread::yield();status=hipSuccess;
    }
    const auto cleanup=hipEventDestroy(event);return status==hipSuccess?cleanup:status;
}
struct Storage {
    uint32_t* base=nullptr;hipStream_t stream=nullptr;
    ~Storage(){if(base){(void)hipStreamSynchronize(stream);(void)hipFree(base);}}
};

template<class Replay> hipError_t run(float* output,const float* input_l2,const float* weight_l2,
    const uint16_t* weights,const unsigned char* table,unsigned layer,hipStream_t stream,
    Replay replay,unsigned long long* copied_metrics=nullptr) {
    if(!output||!input_l2||!weight_l2||!weights||!table)return hipErrorInvalidValue;
    auto status=complete(stream);if(status!=hipSuccess)return status;
    const auto start=std::chrono::steady_clock::now();
    constexpr size_t span=cells+2u*guard,words=3u*span+metrics*2u+2u*guard;
    Storage storage;storage.stream=stream;
    status=hipMalloc(reinterpret_cast<void**>(&storage.base),words*sizeof(uint32_t));
    if(status!=hipSuccess)return status;
    uint32_t* buffers[4]={storage.base,storage.base+span,storage.base+2u*span,storage.base+3u*span};
    const size_t lengths[4]={cells,cells,cells,metrics*2u};
    for(unsigned b=0u;b<4u;++b){
        status=hipMemsetAsync(buffers[b],0xa5,guard*sizeof(uint32_t),stream);if(status!=hipSuccess)return status;
        status=hipMemsetAsync(buffers[b]+guard+lengths[b],0xa5,guard*sizeof(uint32_t),stream);if(status!=hipSuccess)return status;
    }
    auto* native=reinterpret_cast<float*>(buffers[0]+guard);
    auto* convolution=reinterpret_cast<float*>(buffers[1]+guard);
    auto* certificate=buffers[2]+guard;
    auto* counters=reinterpret_cast<unsigned long long*>(buffers[3]+guard);
    status=hipMemsetAsync(counters,0,metrics*sizeof(*counters),stream);if(status!=hipSuccess)return status;
    status=hipMemcpyAsync(native,output,cells*sizeof(float),hipMemcpyDeviceToDevice,stream);if(status!=hipSuccess)return status;
    const View view{native,input_l2,weight_l2,weights,table};
    hipLaunchKernelGGL(certificates,dim3((cells+255u)/256u),dim3(256u),0u,stream,view,certificate);
    status=hipGetLastError();if(status!=hipSuccess)return status;
    // This observer never removes a candidate or changes an output value.
    status=replay();if(status!=hipSuccess)return status;
    hipLaunchKernelGGL(selected_conv_qkv_window_kernel,dim3(rows/256u,tokens),dim3(256u),0u,stream,
        output,weights,nullptr,convolution,tokens,3u,nullptr,nullptr,0u,table);
    status=hipGetLastError();if(status!=hipSuccess)return status;
    hipLaunchKernelGGL(inspect,dim3(1024u),dim3(256u),0u,stream,view,output,convolution,certificate,counters);
    status=hipGetLastError();if(status!=hipSuccess)return status;
    unsigned long long counts[metrics]{};uint32_t redzones[8][guard]{};
    status=hipMemcpyAsync(counts,counters,sizeof(counts),hipMemcpyDeviceToHost,stream);if(status!=hipSuccess)return status;
    for(unsigned b=0u;b<4u;++b){
        status=hipMemcpyAsync(redzones[2u*b],buffers[b],sizeof(redzones[0]),hipMemcpyDeviceToHost,stream);if(status!=hipSuccess)return status;
        status=hipMemcpyAsync(redzones[2u*b+1u],buffers[b]+guard+lengths[b],sizeof(redzones[0]),hipMemcpyDeviceToHost,stream);if(status!=hipSuccess)return status;
    }
    status=complete(stream);if(status!=hipSuccess)return status;
    for(const auto& edge:redzones)for(uint32_t word:edge)if(word!=0xa5a5a5a5u)return hipErrorUnknown;
    if(counts[Elements]!=cells||counts[Omittable]>counts[Valid]||counts[Valid]>counts[Selected]||
        counts[Selected]>cells||counts[WideOmittable]>counts[Omittable]||counts[Constant]>cells)return hipErrorUnknown;
    if(copied_metrics)std::copy_n(counts,metrics,copied_metrics);
    status=hipFree(storage.base);storage.base=nullptr;if(status!=hipSuccess)return status;
    const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::fprintf(stderr,"BATCH_MARK conv_consumer_audit layer=%u elements=%llu selected=%llu valid_ranges=%llu wide_ranges=%llu tiny_selected=%llu omittable=%llu wide_omittable=%llu constant_outputs=%llu halo_selected_protected=%llu projection_bf16_changes=%llu omittable_bf16_changes=%llu range_failures=%llu certificate_failures=%llu unselected_endpoint_changes=%llu nonfinite=%llu tiny_nonzero=%llu workspace_bytes=%zu completed_owner_ms=%.6f all_original_candidates_replayed=1 guards_pass=1 diagnostic_only=1\n",
        layer,counts[Elements],counts[Selected],counts[Valid],counts[Wide],counts[Tiny],counts[Omittable],counts[WideOmittable],counts[Constant],counts[Halo],counts[Changed],counts[OmittedChanged],counts[RangeFailures],counts[CertificateFailures],counts[UnselectedChanges],counts[Nonfinite],counts[TinyNonzero],words*sizeof(uint32_t),elapsed);
    std::fflush(stderr);
    // Numerical certificate failures are observations. Original inference
    // continues so its GB10 verdict can be distinguished from this proposal.
    return hipSuccess;
}
} // namespace qrt_conv_consumer_audit
