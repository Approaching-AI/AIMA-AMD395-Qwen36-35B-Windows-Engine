// Compile the actual host owner with a HIP transport mock. This checks memory
// ownership and failure cleanup; numerical acceptance belongs to native tests.
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "q8192_matrix_producer_policy.h"
#include "q8192_out_l1_policy.h"
#include "moe_accumulator/bf16_midpoint_selector.h"
using hipStream_t = void*;
using hipError_t = int;
constexpr int hipSuccess=0, hipErrorInvalidValue=1, hipErrorInvalidConfiguration=2,
 hipErrorNotReady=3, hipErrorLaunchTimeOut=4, hipMemcpyHostToDevice=5, hipMemcpyDeviceToHost=6;
constexpr int transport_error=99;
std::map<uintptr_t,size_t> allocations;
uintptr_t next_address=0x100000000ull, stats_address=0;
std::array<unsigned,19> stats{};
unsigned operation=0, fail_operation=0, mallocs=0, frees=0, launches=0, matrices=0, drains=0;
bool pending=false;
bool fail() { return ++operation == fail_operation; }
void range(const void* p,size_t n) {
 const uintptr_t address=reinterpret_cast<uintptr_t>(p);
 auto it=allocations.upper_bound(address); assert(it!=allocations.begin()); --it;
 assert(address>=it->first && n<=it->second && address-it->first<=it->second-n);
}
int hipMalloc(void** p,size_t n) {
 if(fail()) return transport_error;
 *p=reinterpret_cast<void*>(next_address); allocations.emplace(next_address,n);
 next_address+=n+4096; ++mallocs; return hipSuccess;
}
int hipMemsetAsync(void* p,int value,size_t n,hipStream_t) {
 range(p,n);assert(value==0xa5); if(fail()) return transport_error;
 pending=true;return hipSuccess;
}
int hipMemcpyAsync(void* dst,const void* src,size_t n,int kind,hipStream_t) {
 range(dst,n);assert(kind==hipMemcpyHostToDevice && n==sizeof(stats));
 if(fail()) return transport_error;
 stats_address=reinterpret_cast<uintptr_t>(dst);std::memcpy(stats.data(),src,n);
 pending=true;return hipSuccess;
}
int hipMemcpy(void* dst,const void* src,size_t n,int kind) {
 assert(!pending && kind==hipMemcpyDeviceToHost);range(src,n);
 if(fail()) return transport_error;
 if(reinterpret_cast<uintptr_t>(src)==stats_address) {assert(n==sizeof(stats));std::memcpy(dst,stats.data(),n);}
 else {assert(n==256u || n==512u);std::memset(dst,0xa5,n);}
 return hipSuccess;
}
void launch() {++launches;pending=true;}
#define hipLaunchKernelGGL(...) launch()
int hipGetLastError() {return fail()?transport_error:hipSuccess;}
int hipStreamQuery(hipStream_t) {if(fail())return transport_error;pending=false;return hipSuccess;}
int hipStreamSynchronize(hipStream_t) {++drains;pending=false;return hipSuccess;}
int hipFree(void* p) {assert(!pending);assert(allocations.erase(reinterpret_cast<uintptr_t>(p))==1u);++frees;return fail()?transport_error:hipSuccess;}
namespace qrt_q1_moe_hawkeye {float accumulate_bf16_hopper_blackwell(float,const uint16_t*,const uint16_t*,unsigned){assert(false);return 0;}}
bool resident_bf16_matrix_matmul_f32_output_with_heuristic_index(
 const uint16_t* w,const uint16_t* x,float* y,unsigned rows,unsigned k,unsigned tokens,
 unsigned choice,hipStream_t,const std::string&,std::string*,std::string*) {
 assert(rows==2048 && k==4096 && tokens==8192 && choice==(matrices?4u:0u));
 range(y,size_t(rows)*tokens*sizeof(float));
 if(matrices){range(w,size_t(rows)*k*sizeof(uint16_t));range(x,size_t(tokens)*k*sizeof(uint16_t));}
 ++matrices;if(fail())return false;pending=true;return true;
}
#define QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER 1
#include "out_l1_actual_run.h"
#include "out_l1_actual_selector.h"
void setting(const char* name,const char* value){
#ifdef _WIN32
 _putenv_s(name,value?value:"");
#else
 if(value)setenv(name,value,1);else unsetenv(name);
#endif
}
void reset(unsigned failure=0){
 assert(allocations.empty() && !pending);operation=mallocs=frees=launches=matrices=drains=0;fail_operation=failure;
 setting("QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT","1");
 setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM","4");
 setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","qkv");
}
int main(){
 using namespace qrt_bf16_positive_sum_bound;
 unsigned cases=0;
 const std::array<unsigned,5> factors={1,4,8,16,32};
 for(unsigned v=0;v<factors.size();++v)assert(qrt_out_l1_policy::multiplier(v)==factors[v]);
 for(unsigned f:factors){
  assert(qrt_out_l1_policy::capped(2.0f,3.0f,f)==(f==1u?next_up(2.0f):3.0f));
  for(float excluded:{INFINITY,NAN,-1.0f})assert(qrt_out_l1_policy::capped(excluded,3.0f,f)==3.0f);
  assert(qrt_out_l1_policy::capped(value(0x7f7fffffu),3.0f,f)==3.0f);
 }
 // Preserve the first captured L1 admission counterexample. Multiplying its
 // coefficient by four restores admission, without claiming other cells pass.
 const float raw=value(0xbd207cecu),in=value(0x420a89c5u),wn=value(0x3f6cee1du),l1=value(0x3f8f2355u);
 assert(selected_bf16_projection_hawkeye_candidate(raw,0,2048,512,0,1000,nullptr,&in,&wn));
 for(unsigned f:factors){
  const float bound=qrt_out_l1_policy::capped(l1,in*wn,f);
  assert(selected_bf16_projection_hawkeye_candidate(raw,0,2048,512,0,1000,&bound,nullptr,nullptr)==(f!=1u));
 }
 const uint16_t w=1,x=1;const float y=1,n=1;
 auto run=[&](unsigned rows=2048,unsigned tokens=8192,unsigned k=4096,unsigned ppb=1000,bool missing=false){
  return qrt_out_l1_shadow::run(&w,&x,missing?nullptr:&y,&n,&n,rows,tokens,k,512,ppb,nullptr,"mock",true);
 };
 for(const char* disabled:{static_cast<const char*>(nullptr),"","0"}){
  reset();setting("QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT",disabled);assert(run()==hipSuccess && !operation);++cases;
 }
 for(const char* invalid:{"2","-1"," 1","01"}){
  reset();setting("QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT",invalid);assert(run()==hipErrorInvalidValue && !operation);++cases;
 }
 for(unsigned tokens:{1u,7169u,8191u,8193u,65536u}){reset();assert(run(2048,tokens)==hipSuccess && !operation);++cases;}
 reset();assert(run(2047)==hipSuccess && !operation);++cases;
 reset();assert(run(2048,8192,4095)==hipSuccess && !operation);++cases;
 reset();assert(run(2048,8192,4096,0)==hipErrorInvalidValue && !operation);++cases;
 reset();assert(run(2048,8192,4096,1000,true)==hipErrorInvalidValue && !operation);++cases;
 reset();setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","all");assert(run()==hipErrorInvalidValue && !operation);++cases;
 reset();assert(run()==hipSuccess && mallocs==4 && frees==4 && matrices==2 && launches==5 && allocations.empty() && !pending);++cases;
 const unsigned operations=operation;
 for(unsigned failure=1;failure<=operations;++failure){
  reset(failure);assert(run()!=hipSuccess && frees==mallocs && allocations.empty() && !pending);++cases;
 }
 std::printf("out_l1_host_pass cases=%u injected_transport_failures=%u numerical_acceptance=0\n",cases,operations);
}
