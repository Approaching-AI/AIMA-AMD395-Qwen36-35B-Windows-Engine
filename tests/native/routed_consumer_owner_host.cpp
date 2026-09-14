#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using hipError_t = int;
using hipStream_t = void*;
constexpr int hipSuccess=0, hipErrorNotReady=1, hipErrorLaunchTimeOut=2,
 hipErrorInvalidValue=3, hipMemcpyDeviceToHost=4, failure=99;
unsigned calls=0, fail_call=0, allocated=0, freed=0, drained=0;
bool pending=false;
bool fail(){return ++calls==fail_call;}
int hipMalloc(void** p,size_t bytes){if(fail())return failure;*p=std::malloc(bytes);assert(*p);++allocated;return 0;}
int hipMemsetAsync(void* p,int v,size_t n,hipStream_t){if(fail())return failure;std::memset(p,v,n);pending=true;return 0;}
int hipStreamQuery(hipStream_t){if(fail())return failure;pending=false;return 0;}
int hipStreamSynchronize(hipStream_t){pending=false;++drained;return 0;}
int hipMemcpy(void* d,const void* s,size_t n,int kind){assert(!pending && kind==hipMemcpyDeviceToHost);if(fail())return failure;std::memcpy(d,s,n);return 0;}
int hipFree(void* p){assert(!pending);++freed;std::free(p);return fail()?failure:0;}
unsigned atomicAdd(unsigned* p,unsigned v){const auto old=*p;*p+=v;return old;}
unsigned atomicMin(unsigned* p,unsigned v){const auto old=*p;*p=std::min(old,v);return old;}
#define __device__
#include "triton_moe/routed_consumer_audit.h"
int main(){
 namespace a=qrt_routed_consumer_audit;
 namespace c=qrt_routed_consumer;
 unsigned cases=0;
 auto reset=[&](unsigned injected=0){assert(allocated==freed && !pending);calls=allocated=freed=drained=0;fail_call=injected;};
 auto run=[&](bool enabled=true,bool corrupt=false){
  a::Owner owner(nullptr,"host-test");auto status=owner.initialize(enabled);
  if(status)return status;
  if(corrupt){assert(owner.data());owner.data()[-1]=0;}
  return owner.finish();
 };
 reset();assert(run(false)==0 && !calls);++cases;
 reset();assert(run()==0 && allocated==1 && freed==1);++cases;
 const unsigned operations=calls;
 for(unsigned operation=1;operation<=operations;++operation){reset(operation);assert(run()!=0 && allocated==freed && !pending);++cases;}
 reset();assert(run(true,true)==hipErrorInvalidValue && allocated==freed && drained==1);++cases;
 // The actual observation code reports misses while preserving its input and
 // the existing correction. Deliberately supply a too-narrow empirical range.
 std::array<uint16_t,65536> table{};
 table.fill(0x3f80u);table[c::rounded(2.0f)]=0x4000u;
 std::array<unsigned,a::counters> stats{};stats[6]=stats[7]=UINT32_MAX;
 a::observe(false,1.0f,2.0f,0.0f,0,table.data(),stats.data(),71u);
 assert(stats[0]==1 && stats[1]==1 && stats[2]==1 && stats[3]==1 && stats[4]==1 && stats[5]==0 && stats[6]==71 && stats[7]==71);
 stats={};stats[6]=stats[7]=UINT32_MAX;
 a::observe(true,1.0f,1.0f,0.0f,0,table.data(),stats.data(),13u);
 assert(stats[0]==1 && stats[1]==1 && stats[2]==0 && stats[3]==0 && stats[4]==0 && stats[5]==0);
 std::printf("consumer_owner_host cases=%u injected_transport_failures=%u observed_range_and_endpoint_miss=1 pass=1\n",cases,operations);
}
