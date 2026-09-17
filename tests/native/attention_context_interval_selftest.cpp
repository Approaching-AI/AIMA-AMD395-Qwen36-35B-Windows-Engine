#include <hip/hip_runtime.h>
#include "../../native/providers/ck_fmha/attention_context_interval.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace context=qrt_attention_context_interval;
namespace rcp=qrt_sm121_attention_rcp;
constexpr size_t redzone=128u;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
void check(hipError_t code){if(code!=hipSuccess)throw std::runtime_error(hipGetErrorString(code));}
struct Device {
    unsigned char* base=nullptr;size_t size;
    explicit Device(size_t n):size(n){check(hipMalloc(reinterpret_cast<void**>(&base),n+2u*redzone));check(hipMemset(base,0xa5,n+2u*redzone));}
    ~Device(){if(base && hipFree(base)!=hipSuccess)std::abort();}
    template<class T=unsigned char>T* data(){return reinterpret_cast<T*>(base+redzone);}
    std::vector<unsigned char> snapshot(){
        std::vector<unsigned char> out(size+2u*redzone);check(hipMemcpy(out.data(),base,out.size(),hipMemcpyDeviceToHost));
        for(size_t i=0u;i<redzone;++i)require(out[i]==0xa5u && out[redzone+size+i]==0xa5u,"device redzone changed");return out;
    }
};
void finish(){
    check(hipGetLastError());hipEvent_t event;check(hipEventCreateWithFlags(&event,hipEventDisableTiming));check(hipEventRecord(event));
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);
    for(;;){const auto result=hipEventQuery(event);if(result==hipSuccess)break;check(result==hipErrorNotReady?hipSuccess:result);require(std::chrono::steady_clock::now()<deadline,"context interval deadline");std::this_thread::yield();}
    check(hipEventDestroy(event));
}
struct Case {context::Interval numerator,denominator;};
struct Stats {unsigned admitted=0u,declined=0u,cartesian_points=0u,reciprocal_points=0u,false_admissions=0u,reciprocal_escapes=0u;};
__device__ unsigned independent_round(float x){
    const unsigned word=__float_as_uint(x),tail=word&0xffffu;
    unsigned kept=word>>16u;kept+=tail>0x8000u || (tail==0x8000u && (kept&1u));return kept&0xffffu;
}
__global__ void audit(const unsigned char* table,const Case* cases,unsigned count,unsigned* results,Stats* totals){
    const unsigned index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=count)return;
    const auto c=cases[index];uint16_t result=0xa5a5u;
    const bool admitted=context::stable(c.numerator,c.denominator,table,&result);
    results[index]=(admitted?0x10000u:0u)|result;
    unsigned bad=0u,escaped=0u,points=0u,reciprocals=0u;
    if(index<65536u){
        const auto inverse=context::reciprocal(c.denominator,table);
        const unsigned first=rcp::bits(c.denominator.low),last=rcp::bits(c.denominator.high);
        for(unsigned d=first;d<=last;++d){
            const float value=rcp::evaluate(table,rcp::value(d));++reciprocals;
            escaped+=!(value>=inverse.low && value<=inverse.high);
            if(admitted){
                uint32_t n=rcp::bits(c.numerator.low);
                for(unsigned i=0u;i<9u;++i){
                    volatile float product=rcp::value(n)*value;
                    bad+=!isfinite(product) || result!=independent_round(product);++points;
                    n+=(n&0x80000000u)?uint32_t(-1):1u;
                }
            }
        }
    }
    atomicAdd(admitted?&totals->admitted:&totals->declined,1u);
    atomicAdd(&totals->cartesian_points,points);atomicAdd(&totals->reciprocal_points,reciprocals);
    atomicAdd(&totals->false_admissions,bad);atomicAdd(&totals->reciprocal_escapes,escaped);
}
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
int main(int argc,char** argv)try{
    require(argc==2,"requires SHA-verified original reciprocal table");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));require(!std::strncmp(properties.gcnArchName,"gfx1151",7u),"requires gfx1151");
    std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(file && file.tellg()==std::streamoff(rcp::table_bytes),"reciprocal span");
    std::vector<unsigned char> table(rcp::table_bytes);file.seekg(0);
    require(bool(file.read(reinterpret_cast<char*>(table.data()),table.size())) && rcp::valid_layout(table.data(),table.size()),"reciprocal layout");
    std::vector<Case> cases;
    for(unsigned trial=0u;trial<65536u;++trial){
        const uint32_t den=((127u+trial%19u)<<23u)|((random_word()&0x7fffffu)&~63u);
        const float inverse=rcp::evaluate(table.data(),rcp::value(den+16u));
        const uint32_t target=(random_word()&0x80000000u)|((100u+random_word()%50u)<<23u)|
            ((random_word()&127u)<<16u)|(trial&1u?0x8000u:0x1800u);
        volatile float center=rcp::value(target)/inverse;float low=center,high=center;
        for(unsigned i=0u;i<4u;++i){low=std::nextafter(low,-INFINITY);high=std::nextafter(high,INFINITY);}
        cases.push_back({{low,high},{rcp::value(den),rcp::value(den+32u)}});
    }
    for(auto n:{context::Interval{1,-1},context::Interval{NAN,0},context::Interval{-INFINITY,0},
        context::Interval{0,INFINITY},context::Interval{0x1p-126f,0x1p-126f},
        context::Interval{rcp::value(0x7f7fffffu),rcp::value(0x7f7fffffu)}})
        cases.push_back({n,{1,2}});
    for(auto d:{context::Interval{0,1},context::Interval{1,0},context::Interval{1,0x1p19f},
        context::Interval{NAN,2},context::Interval{1,INFINITY}})cases.push_back({{1,1},d});
    cases.push_back({{0,0},{1,8192}});cases.push_back({{-0.0f,-0.0f},{1,8192}});cases.push_back({{-0.0f,0.0f},{1,2}});
    std::vector<unsigned> expected;
    for(const auto& c:cases){uint16_t out=0xa5a5u;const bool accepted=context::stable(c.numerator,c.denominator,table.data(),&out);expected.push_back((accepted?0x10000u:0u)|out);}
    Device dt(table.size()),dc(cases.size()*sizeof(Case)),output(cases.size()*4u),stats(sizeof(Stats));
    check(hipMemcpy(dt.data(),table.data(),table.size(),hipMemcpyHostToDevice));
    check(hipMemcpy(dc.data(),cases.data(),dc.size,hipMemcpyHostToDevice));check(hipMemset(stats.data(),0,stats.size));
    const auto table_before=dt.snapshot(),cases_before=dc.snapshot();
    hipLaunchKernelGGL(audit,dim3((cases.size()+255u)/256u),dim3(256u),0u,nullptr,dt.data(),dc.data<Case>(),unsigned(cases.size()),output.data<unsigned>(),stats.data<Stats>());finish();
    const auto actual=output.snapshot();require(!std::memcmp(actual.data()+redzone,expected.data(),output.size),"native and host admission/output differ");
    const auto counted=stats.snapshot();Stats totals;std::memcpy(&totals,counted.data()+redzone,sizeof(totals));
    require(totals.admitted==32770u && totals.declined==32780u && totals.cartesian_points==9732096u && totals.reciprocal_points==2162688u && !totals.false_admissions && !totals.reciprocal_escapes,"native context interval audit failed");
    require(dt.snapshot()==table_before && dc.snapshot()==cases_before,"immutable table or cases changed");
    std::printf("{\"kind\":\"joint_context_interval_native\",\"cases\":%zu,\"admitted\":%u,\"declined\":%u,\"cartesian_points\":%u,\"reciprocal_points\":%u,\"false_admissions\":0,\"reciprocal_escapes\":0,\"host_decisions_and_outputs_match\":true,\"immutable_inputs\":true,\"redzones_pass\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",cases.size(),totals.admitted,totals.declined,totals.cartesian_points,totals.reciprocal_points);
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
