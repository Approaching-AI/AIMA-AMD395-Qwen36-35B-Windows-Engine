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
#include <string>
#include <thread>
#include <vector>
#include "out_consumer_interval.h"
#include "out_residual_filter.h"
#include "q8192_matrix_producer_policy.h"
#include "q8192_out_l1_policy.h"
using hipStream_t=void*;using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorInvalidConfiguration=2,hipErrorNotReady=3,hipErrorLaunchTimeOut=4,hipMemcpyDeviceToHost=5,error=99;
std::map<uintptr_t,size_t> allocations;
uintptr_t next_address=0x100000000ull;
unsigned operation=0,fail_operation=0,allocations_seen=0,frees=0,launches=0,matrices=0,drains=0;
bool pending=false;
bool fail(){return ++operation==fail_operation;}
void range(const void* p,size_t size){const auto address=reinterpret_cast<uintptr_t>(p);auto it=allocations.upper_bound(address);assert(it!=allocations.begin());--it;assert(address>=it->first&&size<=it->second&&address-it->first<=it->second-size);}
int hipMalloc(void** p,size_t size){if(fail())return error;*p=reinterpret_cast<void*>(next_address);allocations.emplace(next_address,size);next_address+=size+4096u;++allocations_seen;return hipSuccess;}
int hipMemsetAsync(void* p,int v,size_t size,hipStream_t){range(p,size);assert(v==0xa5);if(fail())return error;pending=true;return hipSuccess;}
int hipMemcpy(void* dst,const void* src,size_t size,int kind){assert(!pending&&kind==hipMemcpyDeviceToHost);range(src,size);if(fail())return error;
 if(size==8192u*15u*4u){auto* p=static_cast<unsigned*>(dst);std::fill_n(p,8192u*15u,0u);for(unsigned t=0;t<8192u;++t){auto* r=p+t*15u;r[0]=2;r[1]=1;r[3]=1;r[9]=1;r[13]=7;r[14]=1;}}
 else {assert(size==512u);std::memset(dst,0xa5,size);}return hipSuccess;}
void launch(){++launches;pending=true;}
#define hipLaunchKernelGGL(...) launch()
int hipGetLastError(){return fail()?error:hipSuccess;}
int hipStreamQuery(hipStream_t){if(fail())return error;pending=false;return hipSuccess;}
int hipStreamSynchronize(hipStream_t){++drains;pending=false;return fail()?error:hipSuccess;}
int hipFree(void* p){assert(!pending);assert(allocations.erase(reinterpret_cast<uintptr_t>(p))==1u);++frees;return fail()?error:hipSuccess;}
bool resident_bf16_matrix_matmul_f32_output_with_heuristic_index(const uint16_t*,const uint16_t*,float* output,
 unsigned rows,unsigned width,unsigned tokens,unsigned choice,hipStream_t,const std::string&,std::string*,std::string*){
 assert(rows==2048u&&width==4096u&&tokens==8192u&&choice==0u);range(output,size_t(rows)*tokens*4u);++matrices;if(fail())return false;pending=true;return true;}
#define QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER 1
#include "out_consumer_actual_owner.h"
void setting(const char* key,const char* value){
#ifdef _WIN32
 _putenv_s(key,value?value:"");
#else
 if(value)setenv(key,value,1);else unsetenv(key);
#endif
}
void reset(unsigned failed=0){assert(allocations.empty()&&!pending);operation=allocations_seen=frees=launches=matrices=drains=0;fail_operation=failed;
 setting("QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT","1");setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM","4");setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","qkv");setting("QRT_QWEN36_Q8192_OUT_L1_BOUND","0");}
int run(unsigned tokens=8192u,bool compatible=true,bool null_source=false){const auto* bf=reinterpret_cast<const uint16_t*>(1);const auto* fp=reinterpret_cast<const float*>(1);const auto* table=reinterpret_cast<const uint8_t*>(1);
 return qrt_out_consumer_audit::run(null_source?nullptr:bf,bf,bf,fp,bf,table,tokens,512u,1000u,nullptr,"host-audit",compatible);}
int main(){
 namespace v=qrt_out_consumer_interval;unsigned intervals=0,enumerated=0,residual_certified=0;
 for(unsigned bits=0u;bits<65536u;++bits){const float residual=v::value(bits<<16u);if(!v::finite(residual))continue;
  for(unsigned sign:{0u,0x80000000u}){
   const uint32_t raw=sign|((80u+(bits%80u))<<23u)|((bits*2654435761u)&0x7fffffu);const float center=v::value(raw);
   v::Interval limits;assert(v::projection(center,std::fabs(center)*32.0f,1000u,512u,&limits));
   assert(limits.lower<=v::rounded(center)&&limits.upper>=v::rounded(center));
   const v::Interval sum{residual+limits.lower,residual+limits.upper};const auto absolute=v::absolute_range(sum);
   const float mid=residual+v::rounded(center);assert(std::fabs(mid)>=absolute.lower&&std::fabs(mid)<=absolute.upper);
   if(v::finite(sum.lower)&&v::finite(sum.upper)&&v::bf16(sum.lower)==v::bf16(sum.upper))assert(v::bf16(mid)==v::bf16(sum.lower));
   const bool certified=qrt_out_residual_filter::invariant(center,residual,std::fabs(center)*32.0f,1000u,512u);
   assert(certified==(v::finite(sum.lower)&&v::finite(sum.upper)&&v::bf16(sum.lower)==v::bf16(sum.upper)));
   residual_certified+=certified;
   ++intervals;
  }
 }
 for(unsigned base=1u;base<0x7f70u;base+=7u)for(unsigned sign:{0u,0x8000u}) {
  const float a=v::value((base|sign)<<16u),b=v::value(((base+7u)|sign)<<16u);const v::Interval limits{std::min(a,b),std::max(a,b)};const auto absolute=v::absolute_range(limits);
  for(unsigned i=0;i<=7u;++i){const float x=v::value(((base+i)|sign)<<16u);assert(std::fabs(x)>=absolute.lower&&std::fabs(x)<=absolute.upper);++enumerated;}
 }
 v::Interval result;for(float excluded:{0.0f,INFINITY,-INFINITY,NAN,v::value(1u)})assert(!v::projection(excluded,1,1000,512,&result));
 assert(!v::projection(1,1,0,512,&result)&&!v::projection(1,1,1000,32769,&result)&&!v::projection(1,1,1000,512,nullptr));
 for(float excluded:{INFINITY,-INFINITY,NAN})assert(!qrt_out_residual_filter::invariant(1,excluded,1,1000,512));
 assert(qrt_out_residual_filter::invariant(1.00390625f,128.0f,1.0f,1000u,512u));
 assert(!qrt_out_residual_filter::invariant(1.00390625f,0.0f,1.0f,1000u,512u));
 assert(qrt_out_residual_filter::mode(nullptr)==0&&qrt_out_residual_filter::mode("")==0&&qrt_out_residual_filter::mode("0")==0&&qrt_out_residual_filter::mode("1")==1);
 for(const char* bad:{"2"," 1","1x","-1"})assert(qrt_out_residual_filter::mode(bad)==-1);
 reset();assert(run()==hipSuccess&&allocations_seen==3u&&frees==3u&&launches==3u&&matrices==1u&&drains==1u);const unsigned operations=operation;
 for(unsigned failure=1u;failure<=operations;++failure){reset(failure);assert(run()!=hipSuccess);assert(allocations.empty()&&!pending&&allocations_seen==frees);}
 for(const char* off:{static_cast<const char*>(nullptr),"","0"}){reset();setting("QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT",off);assert(run(8192u,false,true)==hipSuccess&&!operation);}
 for(const char* bad:{"2"," 1","1x"}){reset();setting("QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT",bad);assert(run()==hipErrorInvalidValue&&!operation);}
 reset();assert(run(7169u,false,true)==hipSuccess&&!operation);assert(run(8192u,false)==hipErrorInvalidValue&&!operation);assert(run(8192u,true,true)==hipErrorInvalidValue&&!operation);
 reset();setting("QRT_QWEN36_Q8192_OUT_L1_BOUND","1");assert(run()==hipErrorInvalidValue&&!operation);
 reset();setting("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","all");assert(run()==hipErrorInvalidValue&&!operation);
 std::printf("out_consumer_host_pass intervals=%u enumerated_bf16_values=%u injected_failures=%u residual_filter_certified=%u\n",intervals,enumerated,operations,residual_certified);
}
