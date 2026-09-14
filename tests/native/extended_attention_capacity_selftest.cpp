#include <hip/hip_runtime.h>
#include "../../native/providers/ck_fmha/blackwell_attention.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <stdexcept>
#include <thread>

// Address and memory-ownership evidence only. Zero queries/keys and constant
// values have an analytic mean of 0.5; real long-context GB10 runs are separate.
namespace test {
namespace attention=qrt_blackwell_attention;
constexpr size_t guard=128u;
constexpr unsigned threads=256u,blocks=1024u;
void check(hipError_t status) {
    if(status!=hipSuccess)throw std::runtime_error(hipGetErrorString(status));
}
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
template<class T> struct Device {
    T* base=nullptr;size_t elements;
    explicit Device(size_t count):elements(count) {
        check(hipMalloc(reinterpret_cast<void**>(&base),(elements+2u*guard)*sizeof(T)));
        check(hipMemset(base,0xa5,(elements+2u*guard)*sizeof(T)));
    }
    ~Device(){if(base)(void)hipFree(base);}
    Device(const Device&)=delete;Device& operator=(const Device&)=delete;
    T* data(){return base+guard;}
};
__global__ void fill_values(uint16_t* values,size_t elements) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=size_t(blockDim.x)*gridDim.x)
        values[i]=0x3f00u;
}
__global__ void verify_input(const uint16_t* values,size_t elements,uint16_t expected,unsigned* errors) {
    unsigned bad=0u;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements+2u*guard;i+=size_t(blockDim.x)*gridDim.x)
        bad+=values[i]!=(i<guard || i>=guard+elements?uint16_t(0xa5a5u):expected);
    if(bad)atomicAdd(errors,bad);
}
__global__ void verify_output(const float* values,size_t elements,size_t first,unsigned* errors) {
    unsigned bad=0u,selected=0u;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements+2u*guard;i+=size_t(blockDim.x)*gridDim.x) {
        const uint32_t value=__float_as_uint(values[i]);
        if(i>=guard+first && i<guard+first+4096u) {
            ++selected;
            bad+=!isfinite(values[i]) || attention::f32_to_bf16(values[i])!=0x3f00u;
        } else bad+=value!=0xa5a5a5a5u;
    }
    if(bad)atomicAdd(errors,bad);
    if(selected)atomicAdd(errors+1u,selected);
}
__global__ void verify_guards(const uint32_t* values,size_t elements,unsigned* errors) {
    const unsigned i=threadIdx.x;
    if(i<guard && (values[i]!=0xa5a5a5a5u || values[guard+elements+i]!=0xa5a5a5a5u))atomicAdd(errors,1u);
}
void finish() {
    hipEvent_t event=nullptr;check(hipEventCreate(&event));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;) {
        const auto status=hipEventQuery(event);if(status==hipSuccess)break;
        if(status!=hipErrorNotReady)check(status);
        require(std::chrono::steady_clock::now()<deadline,"extended attention completion deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
void run() {
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    require(!std::strncmp(properties.gcnArchName,"gfx1151",7u),"requires gfx1151");
    constexpr unsigned maximum=qrt_sm121_attention_capacity::kTokens;
    constexpr size_t query_words=size_t(maximum)*4096u,kv_words=size_t(maximum)*512u;
    static_assert(query_words*4u>UINT32_MAX);
    Device<uint16_t> q(query_words),k(kv_words),v(kv_words);
    Device<float> output(query_words),scores(attention::split_scratch_elements(1u,maximum,2u));
    Device<unsigned> counts(2u);
    check(hipMemset(q.data(),0,query_words*2u));check(hipMemset(k.data(),0,kv_words*2u));
    hipLaunchKernelGGL(fill_values,dim3(blocks),dim3(threads),0u,nullptr,v.data(),kv_words);
    check(hipGetLastError());finish();
    for(unsigned tokens:{131073u,132096u,263168u,maximum}) {
        check(hipMemset(output.base,0xa5,(query_words+2u*guard)*4u));
        check(hipMemset(counts.data(),0,8u));
        const size_t first=size_t(tokens-1u)*4096u;
        const auto begin=std::chrono::steady_clock::now();
        check(hipError_t(attention::launch_queries(q.data(),k.data(),v.data(),output.data(),nullptr,
            tokens-1u,1u,tokens-1u,nullptr,nullptr,nullptr,true,nullptr,2u,scores.data(),scores.elements)));
        finish();
        const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
        hipLaunchKernelGGL(verify_output,dim3(blocks),dim3(threads),0u,nullptr,
            output.base,query_words,first,counts.data());check(hipGetLastError());
        for(unsigned side=0u;side<3u;++side) {
            auto& input=side==0u?q:side==1u?k:v;
            hipLaunchKernelGGL(verify_input,dim3(blocks),dim3(threads),0u,nullptr,
                input.base,input.elements,uint16_t(side==2u?0x3f00u:0u),counts.data());check(hipGetLastError());
        }
        hipLaunchKernelGGL(verify_guards,dim3(1u),dim3(threads),0u,nullptr,
            reinterpret_cast<const uint32_t*>(scores.base),scores.elements,counts.data());check(hipGetLastError());
        hipLaunchKernelGGL(verify_guards,dim3(1u),dim3(threads),0u,nullptr,
            counts.base,counts.elements,counts.data());check(hipGetLastError());
        finish();unsigned actual[2]{};check(hipMemcpy(actual,counts.data(),sizeof(actual),hipMemcpyDeviceToHost));
        require(!actual[0] && actual[1]==4096u,"extended attention mean,unselected output or input guard differs");
        std::printf("{\"kind\":\"extended_attention_address_boundary\",\"tokens\":%u,\"maximum_tokens\":%u,"
            "\"query_start\":%u,\"output_start\":%u,\"output_byte_offset\":%zu,\"output_allocation_bytes\":%zu,"
            "\"analytic_bf16_mean\":16128,\"output_cells\":4096,\"bf16_mismatches\":0,\"kernel_ms\":%.6f,"
            "\"all_unselected_outputs_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,"
            "\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens,maximum,tokens-1u,tokens-1u,first*4u,(query_words+2u*guard)*4u,elapsed);
        std::fflush(stdout);
    }
    require(attention::launch_queries(q.data(),k.data(),v.data(),output.data(),nullptr,
        maximum-1u,2u,0u)==int(hipErrorInvalidValue),"out-of-capacity query accepted");
    require(attention::launch_queries(q.data(),k.data(),v.data(),output.data(),nullptr,
        0u,2u,maximum-1u)==int(hipErrorInvalidValue),"out-of-capacity output accepted");
    require(attention::launch_queries(q.data(),k.data(),v.data(),output.data(),nullptr,
        0xffffffffu,1u,0u)==int(hipErrorInvalidValue),"wrapped query accepted");
}
}
int main()try{test::run();return 0;}
catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
