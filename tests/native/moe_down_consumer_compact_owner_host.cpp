#include "triton_moe/down_consumer_interval.h"
#include <array>
#include <cmath>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <tuple>
#include <vector>
using hipError_t=int;using hipStream_t=void*;using hipEvent_t=void*;
constexpr int hipSuccess=0,hipErrorNotReady=1,hipErrorLaunchTimeOut=2,hipErrorInvalidValue=3,hipMemcpyDeviceToHost=4,failure=99;
struct dim3 {unsigned x;explicit dim3(unsigned value):x(value){}};
struct Allocation {void* pointer;size_t bytes;};
std::vector<Allocation> live;
std::vector<std::function<void()>> pending;
unsigned operations=0,fail_at=0,allocations=0,frees=0,drains=0,launches=0,corrupt=0;
int last_error=0;
bool fail(){return ++operations==fail_at;}
void bounds(const void* p,size_t n){
 for(auto a:live){const auto begin=reinterpret_cast<uintptr_t>(a.pointer),v=reinterpret_cast<uintptr_t>(p);
  if(v>=begin&&v-begin<=a.bytes&&n<=a.bytes-(v-begin))return;}
 assert(false&&"out-of-allocation access");
}
void drain(){auto work=std::move(pending);pending.clear();for(auto& action:work)action();}
int hipMalloc(void** p,size_t n){if(fail())return failure;*p=std::malloc(n);assert(*p);live.push_back({*p,n});++allocations;return 0;}
int hipMemsetAsync(void* p,int byte,size_t n,hipStream_t){bounds(p,n);if(fail())return failure;pending.push_back([=]{bounds(p,n);std::memset(p,byte,n);});return 0;}
int hipGetLastError(){if(fail())return failure;const int result=last_error;last_error=0;return result;}
int hipStreamQuery(hipStream_t){if(fail())return failure;if(!pending.empty()){drain();return hipErrorNotReady;}return 0;}
int hipStreamSynchronize(hipStream_t){++drains;drain();return 0;}
int hipMemcpy(void* d,const void* s,size_t n,int kind){assert(pending.empty()&&kind==hipMemcpyDeviceToHost);bounds(s,n);if(fail())return failure;std::memcpy(d,s,n);return 0;}
int hipFree(void* p){assert(pending.empty());for(size_t i=0;i<live.size();++i)if(live[i].pointer==p){std::free(p);live.erase(live.begin()+i);++frees;return 0;}assert(false);return failure;}
bool shared_ready=false;
hipEvent_t expected_event=reinterpret_cast<hipEvent_t>(0x1234);
int hipStreamWaitEvent(hipStream_t stream,hipEvent_t event,unsigned flags){
 assert(!stream&&event==expected_event&&!flags);if(fail())return failure;
 pending.push_back([]{shared_ready=true;});return 0;
}
namespace qrt_moe_down_consumer_compact {
// Enqueue simulated certificate and original replay completion. The real GPU
// test covers arithmetic and queues; this fixture injects ownership failures.
template<class View>int launch(View v,unsigned first,unsigned tokens,unsigned* indices,size_t capacity,unsigned* count,unsigned* stats,hipStream_t stream){
 assert(!stream&&indices&&count&&capacity>=size_t(tokens)*16384u&&first+tokens<=v.tokens);++launches;
 if(fail())return failure;
 pending.push_back([=]{assert(shared_ready);bounds(stats,8u*sizeof(unsigned));
  stats[0]+=tokens*2048u;stats[1]+=tokens*100u;stats[2]+=tokens*70u;stats[3]+=tokens*50u;stats[4]+=tokens*100u;stats[5]+=tokens*70u;stats[6]+=tokens*30u;
  if(first+tokens==v.tokens){
   if(corrupt==1)stats[-1]=0;if(corrupt==2)stats[8]=0;
   if(corrupt==3)--stats[4];if(corrupt==4)--stats[5];if(corrupt==5)--stats[6];
   if(corrupt==6)--stats[0];if(corrupt==7)stats[7]=1;if(corrupt==8)stats[3]=v.tokens*2048u+1u;
  }
 });return 0;
}
}
#include "moe_down_compact_actual_owner.h"
int main(){
 namespace a=qrt_moe_down_consumer_compact;
 const float value=1.0f;const int32_t expert=0;const uint16_t down=0x3f80u;
 auto reset=[&](unsigned failure_point=0,unsigned corruption=0){assert(live.empty()&&pending.empty());operations=allocations=frees=drains=launches=0;fail_at=failure_point;corrupt=corruption;last_error=0;shared_ready=false;};
 auto prepare=[&](a::Owner& owner,hipEvent_t event){return owner.prepare({&value,&value,&expert,&value,&value,&down,&value,512e-9f,512u,0u,8192u},event);};
 auto run=[&](bool enabled=true,bool compatible=true){a::Owner owner(nullptr);
  auto status=owner.initialize(enabled,compatible);if(status)return status;
  status=prepare(owner,expected_event);if(status)return status;
  if(enabled){unsigned queue=0,count=0;for(unsigned first=0;first<8192u;first+=256u){
   status=owner.collect_blocks(first*64u,256u*64u,&queue,256u*16384u,&count);if(status)return status;
  }}
  return owner.finish();
 };
 reset();assert(run(false,false)==0&&!operations&&!launches);
 reset();assert(run(true,false)==hipErrorInvalidValue&&!operations);
 reset();assert(run()==0&&allocations==1&&frees==1&&launches==32&&drains==1);const unsigned successful_operations=operations;
 for(unsigned step=1;step<=successful_operations;++step){reset(step);assert(run()==failure&&live.empty()&&pending.empty()&&allocations==frees);}
 for(unsigned corruption=1;corruption<=8;++corruption){reset(0,corruption);assert(run()==hipErrorInvalidValue&&live.empty()&&pending.empty()&&allocations==frees);}
 reset();{a::Owner owner(nullptr);assert(owner.initialize(true,true)==0);assert(owner.finish()==hipErrorInvalidValue);}
 reset();{a::Owner owner(nullptr);assert(owner.initialize(true,true)==0);assert(prepare(owner,nullptr)==hipErrorInvalidValue);}
 reset();{a::Owner owner(nullptr);assert(owner.initialize(true,true)==0);assert(prepare(owner,expected_event)==0);assert(prepare(owner,expected_event)==hipErrorInvalidValue);}
 reset();{a::Owner owner(nullptr);unsigned queue=0,count=0;assert(owner.initialize(true,true)==0);assert(owner.collect_blocks(0,64,&queue,16384,&count)==hipErrorInvalidValue);assert(prepare(owner,expected_event)==0);assert(owner.collect_blocks(1,64,&queue,16384,&count)==hipErrorInvalidValue);assert(owner.collect_blocks(0,65,&queue,16384,&count)==hipErrorInvalidValue);}
 reset();
 std::printf("moe_down_compact_owner_host injected_failures=%u invalid_reports=8 actual_async_owner=1 shared_event_ordered=1 drain_before_free=1 pass=1\n",successful_operations);
}
