#include "triton_moe/down_consumer_interval.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <thread>
#include <tuple>
#include <vector>
using hipError_t=int;using hipStream_t=void*;
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
namespace qrt_moe_down_consumer_audit {
constexpr int snapshot=1,inspect=2;
template<class... Args>void launch(int kind,dim3 grid,dim3 block,unsigned shared,hipStream_t,Args... args){
 assert(grid.x==4096&&block.x==256&&!shared);++launches;
 if(fail()){last_error=failure;return;}
 const auto tuple=std::make_tuple(args...);
 if constexpr(sizeof...(Args)==12){
  assert(kind==snapshot);auto* snapshots=std::get<8>(tuple);auto* stats=std::get<9>(tuple);assert(std::get<10>(tuple)==8192u);
  bounds(snapshots,size_t(8192)*2048*sizeof(qrt_moe_down_consumer::Snapshot));
  pending.push_back([=]{bounds(stats,18u*sizeof(unsigned));stats[0]=8192u*2048u;stats[1]=100u;stats[2]=70u;stats[3]=60u;});
 }else{
  static_assert(sizeof...(Args)==8);assert(kind==inspect);auto* snapshots=std::get<0>(tuple);auto* stats=std::get<6>(tuple);assert(std::get<7>(tuple)==8192u);
  pending.push_back([=]{bounds(stats,18u*sizeof(unsigned));stats[4]=100u;stats[5]=90u;stats[8]=8192u*2048u;
   stats[10]=30u;stats[11]=40u;stats[12]=50u;stats[13]=70u;
   if(corrupt==1)stats[-1]=0;
   if(corrupt==2)stats[18]=0;
   if(corrupt==3)reinterpret_cast<unsigned*>(snapshots)[-1]=0;
   if(corrupt==4)reinterpret_cast<unsigned*>(snapshots+size_t(8192)*2048)[0]=0;
   if(corrupt==5)--stats[4];
   if(corrupt==6)--stats[0];
   if(corrupt==7)--stats[8];
   if(corrupt==8)stats[7]=1;
  });
 }
}
}
#define hipLaunchKernelGGL(...) launch(__VA_ARGS__)
#include "moe_down_actual_owner.h"
int main(){
 namespace a=qrt_moe_down_consumer_audit;
 const float value=1.0f;const int32_t expert=0;const uint16_t down=0x3f80u;
 auto reset=[&](unsigned failure_point=0,unsigned corruption=0){assert(live.empty()&&pending.empty());operations=allocations=frees=drains=launches=0;fail_at=failure_point;corrupt=corruption;last_error=0;};
 auto run=[&](bool enabled=true,bool compatible=true){a::Owner owner(nullptr,&value);
  auto status=owner.initialize(enabled,compatible);if(status)return status;
  status=owner.capture(&value,&value,&expert,&value,&value,512e-9f,512u,0u);if(status)return status;
  return owner.finish(&value,&value,&down,&value,&value);
 };
 reset();assert(run(false,false)==0&&!operations&&!launches);
 reset();assert(run(true,false)==hipErrorInvalidValue&&!operations);
 reset();assert(run()==0&&allocations==2&&frees==2&&launches==2&&drains==1);const unsigned successful_operations=operations;
 for(unsigned step=1;step<=successful_operations;++step){reset(step);assert(run()==failure&&live.empty()&&pending.empty()&&allocations==frees);}
 for(unsigned corruption=1;corruption<=8;++corruption){reset(0,corruption);assert(run()==hipErrorInvalidValue&&live.empty()&&pending.empty()&&allocations==frees);}
 reset();{a::Owner owner(nullptr,&value);assert(owner.initialize(true,true)==0);assert(owner.finish(&value,&value,&down,&value,&value)==hipErrorInvalidValue);}
 reset();{a::Owner owner(nullptr,&value);assert(owner.initialize(true,true)==0);assert(owner.capture(nullptr,&value,&expert,&value,&value,512e-9f,512,0)==hipErrorInvalidValue);}
 reset();{a::Owner owner(nullptr,&value);assert(owner.initialize(true,true)==0);assert(owner.capture(&value,&value,&expert,&value,&value,512e-9f,512,0)==0);assert(owner.capture(&value,&value,&expert,&value,&value,512e-9f,512,0)==hipErrorInvalidValue);}
 reset();
 std::printf("moe_down_owner_host injected_failures=%u invalid_reports=8 actual_async_owner=1 drain_before_free=1 pass=1\n",successful_operations);
}
