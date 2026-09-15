#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <string>
#include <thread>
#include <vector>
#include "q8192_out_l1_policy.h"
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorInvalidConfiguration,
    hipErrorUnknown,hipErrorNotReady,hipErrorLaunchTimeOut};
using hipStream_t=void*;
struct dim3 {unsigned x;explicit dim3(unsigned v):x(v){}};
struct Index {unsigned x=0u;} blockIdx,blockDim,threadIdx;
constexpr unsigned red=128u;
struct Allocation {unsigned char* base;size_t bytes;};
std::vector<Allocation> owned;
std::deque<std::function<void()>> work;
unsigned step=0u,fail_step=0u,callbacks=0u,preparations=0u,matrix_calls=0u;
bool validate_all=false;
uint16_t original_weight=123u,original_input=456u;
std::vector<float> input_norm(8192u),weight_norm(2048u);
hipError_t touch(){return ++step==fail_step?hipErrorUnknown:hipSuccess;}
void span(const void* pointer,size_t bytes) {
    const auto begin=reinterpret_cast<uintptr_t>(pointer);
    assert(std::any_of(owned.begin(),owned.end(),[&](const Allocation& a){
        const auto p=reinterpret_cast<uintptr_t>(a.base+red);
        return begin>=p && begin-p<=a.bytes && bytes<=a.bytes-(begin-p);
    }));
}
hipError_t hipMalloc(void** pointer,size_t bytes) {
    const auto status=touch();*pointer=nullptr;if(status!=hipSuccess)return status;
    auto* base=static_cast<unsigned char*>(std::malloc(bytes+2u*red));assert(base);
    std::memset(base,0xa5,red);std::memset(base+red+bytes,0xa5,red);
    owned.push_back({base,bytes});*pointer=base+red;return hipSuccess;
}
void drain(){while(!work.empty()){auto action=std::move(work.front());work.pop_front();action();}}
hipError_t hipFree(void* pointer) {
    assert(work.empty());const auto status=touch();
    const auto it=std::find_if(owned.begin(),owned.end(),[&](const Allocation& a){return a.base+red==pointer;});assert(it!=owned.end());
    for(unsigned i=0u;i<red;++i)assert(it->base[i]==0xa5u && it->base[red+it->bytes+i]==0xa5u);
    std::free(it->base);owned.erase(it);return status;
}
hipError_t hipStreamQuery(hipStream_t) {
    const auto status=touch();if(status!=hipSuccess)return status;
    if(work.empty())return hipSuccess;
    auto action=std::move(work.front());work.pop_front();action();return hipErrorNotReady;
}
hipError_t hipStreamSynchronize(hipStream_t){const auto status=touch();drain();return status;}
hipError_t hipGetLastError(){return touch();}
#define __global__
#define QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER 1
#define hipLaunchKernelGGL(kernel,blocks,threads,shared,stream,...) \
    do {const auto g=blocks,b=threads;assert(!(shared));work.push_back([=]{blockDim.x=b.x;for(blockIdx.x=0u;blockIdx.x<g.x;++blockIdx.x)for(threadIdx.x=0u;threadIdx.x<b.x;++threadIdx.x)kernel(__VA_ARGS__);});}while(0)
namespace qrt_sm121_scalar_projection {
void eligible_rows_kernel(const uint16_t* input,unsigned* flags,unsigned rows,unsigned width) {
    if(threadIdx.x)return;
    assert(width==4096u && ((rows==2048u&&input==&original_weight)||(rows==8192u&&input==&original_input)));
    span(flags,size_t(rows)*4u);flags[blockIdx.x]=(blockIdx.x%7u)!=0u;
}
}
namespace qrt_bf16_absolute_product_views {
void prepare_rows_kernel(const uint16_t* input,const unsigned* flags,uint16_t* output,unsigned rows,unsigned width) {
    if(threadIdx.x || blockIdx.x)return;
    assert(width==4096u && ((rows==2048u&&input==&original_weight)||(rows==8192u&&input==&original_input)));
    span(flags,size_t(rows)*4u);span(output,size_t(rows)*width*2u);
    for(unsigned i=0u;i<rows;++i)assert(flags[i]==unsigned(i%7u!=0u));
    ++preparations;
}
}
float raw_magnitude(size_t i) {
    if(i%31u==0u)return std::numeric_limits<float>::infinity();
    if(i%31u==1u)return std::numeric_limits<float>::quiet_NaN();
    if(i%31u==2u)return -1.0f;
    return float(i%4093u)/32.0f;
}
bool resident_bf16_matrix_matmul_f32_output_with_heuristic_index(
    const uint16_t* w,const uint16_t* x,float* out,unsigned rows,unsigned width,unsigned tokens,
    unsigned algorithm,hipStream_t,const std::string&,std::string*,std::string*) {
    if(touch()!=hipSuccess)return false;
    assert(work.empty() && preparations==2u && algorithm==4u && rows==2048u && width==4096u && tokens==8192u);
    assert(x==w+size_t(rows)*width);span(w,size_t(rows+tokens)*width*2u);span(out,size_t(rows)*tokens*4u);
    ++matrix_calls;work.push_back([=]{for(size_t i=0u;i<size_t(rows)*tokens;++i)out[i]=raw_magnitude(i);});return true;
}
#include "q8192_out_l1_replay.h"
float reference(size_t cell,unsigned factor) {
    const unsigned row=unsigned(cell%2048u),token=unsigned(cell/2048u);
    const float cauchy=input_norm[token]*weight_norm[row],raw=raw_magnitude(cell);
    if(row%7u==0u || token%7u==0u || !std::isfinite(raw) || raw<0.0f)return cauchy;
    const float inf=std::numeric_limits<float>::infinity();
    const float inflated=std::nextafter(std::nextafter(raw*1.00390625f,inf)+4096.0f*0x1p-126f,inf);
    return std::min(std::nextafter(inflated*float(factor),inf),cauchy);
}
hipError_t invoke(unsigned factor=1u) {
    return qrt_out_l1_replay::run(&original_weight,&original_input,input_norm.data(),weight_norm.data(),factor,10000u,512u,nullptr,[&](const float* bounds){
        ++callbacks;assert(work.empty() && matrix_calls==1u);span(bounds,size_t(2048u)*8192u*4u);
        const size_t cells=size_t(2048u)*8192u;
        for(size_t i=0u;i<cells;i+=validate_all?1u:7919u){const float expected=reference(i,factor);assert(!std::memcmp(&expected,bounds+i,4u));}
        // Deliberately leave a consumer pending even on a callback error.
        work.push_back([=]{span(bounds,cells*4u);assert(bounds[cells-1u]>=0.0f);});
        return touch();
    });
}
void reset(unsigned failure=0u) {assert(owned.empty() && work.empty());step=callbacks=preparations=matrix_calls=0u;fail_step=failure;}
void setting(const char* name,const char* value) {
#ifdef _WIN32
    _putenv_s(name,value);
#else
    setenv(name,value,1);
#endif
}
int main() {
    for(unsigned i=0u;i<input_norm.size();++i)input_norm[i]=1.0f+float(i%13u);
    for(unsigned i=0u;i<weight_norm.size();++i)weight_norm[i]=1.0f+float(i%17u);
    for(const char* text:std::array<const char*,3>{nullptr,"","0"})assert(qrt_out_l1_policy::selected_factor(text)==0);
    for(const char* text:{"-1","3","01","1x"," 1","64"})assert(qrt_out_l1_policy::selected_factor(text)==-1);
    for(unsigned f:{1u,2u,4u,8u,16u,32u})assert(qrt_out_l1_policy::selected_factor(std::to_string(f).c_str())==int(f));
    assert(qrt_out_l1_policy::replay_shape(2048u,8192u,4096u));
    assert(!qrt_out_l1_policy::replay_shape(2048u,7169u,4096u));
    assert(!qrt_out_l1_policy::replay_shape(8192u,8192u,2048u));
    for(unsigned f:{0u,3u,64u,UINT32_MAX}){reset();assert(invoke(f)==hipErrorInvalidValue && !step);}
    for(unsigned missing=0u;missing<5u;++missing) {
        reset();const auto status=qrt_out_l1_replay::run(missing==0u?nullptr:&original_weight,
            missing==1u?nullptr:&original_input,missing==2u?nullptr:input_norm.data(),
            missing==3u?nullptr:weight_norm.data(),1u,missing==4u?0u:10000u,512u,nullptr,
            [](const float*){assert(false);return hipSuccess;});
        assert(status==hipErrorInvalidValue && !step);
    }
    setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM","4");
    for(const char* scope:{"all","out","invalid"}) {
        setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE",scope);reset();
        assert(invoke()==hipErrorInvalidValue && !step);
    }
    setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","qkv");
    reset();validate_all=true;assert(invoke()==hipSuccess);const unsigned steps=step;
    assert(callbacks==1u && owned.empty() && work.empty());validate_all=false;
    for(unsigned f:{2u,4u,8u,16u,32u}){reset();assert(invoke(f)==hipSuccess && callbacks==1u && owned.empty() && work.empty());}
    for(unsigned failure=1u;failure<=steps;++failure){reset(failure);assert(invoke()!=hipSuccess);assert(owned.empty() && work.empty());}
    assert(original_weight==123u && original_input==456u);
    std::printf("out_l1_replay_host_pass full_bound_cells=%u factors=6 injected_failures=%u\n",2048u*8192u,steps);
}
