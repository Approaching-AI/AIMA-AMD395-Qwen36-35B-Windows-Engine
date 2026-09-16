#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <vector>
#include "q8192_matrix_producer_policy.h"
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorInvalidConfiguration,hipErrorUnknown};
using hipStream_t=void*;
constexpr unsigned hipMemcpyDeviceToHost=2u,red=128u;
struct dim3{unsigned x;explicit dim3(unsigned a):x(a){}};
struct Allocation{unsigned char* base;size_t bytes;};
std::vector<Allocation> allocations;std::deque<std::function<void()>> work;
unsigned steps=0,fail_at=0,launches=0,selected=0,scenario=0,converted=0,executed=0;
dim3 grid(1u),threads(1u);
uint16_t weight=17u,input=29u,norm=0u,output=0u;
uint8_t table=3u;float residual=3.0f,raw=-1.0f;
hipError_t touch(){return ++steps==fail_at?hipErrorUnknown:hipSuccess;}
void span(const void* pointer,size_t bytes){
 const uintptr_t p=reinterpret_cast<uintptr_t>(pointer);
 assert(std::any_of(allocations.begin(),allocations.end(),[&](const Allocation& a){const uintptr_t begin=reinterpret_cast<uintptr_t>(a.base+red);return p>=begin&&p-begin<=a.bytes&&bytes<=a.bytes-(p-begin);}));
}
hipError_t hipMalloc(void** pointer,size_t bytes){
 *pointer=nullptr;const auto status=touch();if(status!=hipSuccess)return status;
 auto* base=static_cast<unsigned char*>(std::malloc(bytes+2u*red));assert(base);
 std::memset(base,0xa5,red);std::memset(base+red+bytes,0xa5,red);allocations.push_back({base,bytes});*pointer=base+red;return hipSuccess;
}
hipError_t hipStreamSynchronize(hipStream_t){const auto status=touch();while(!work.empty()){auto fn=std::move(work.front());work.pop_front();fn();}return status;}
void free_device(void* pointer){
 assert(work.empty());const auto it=std::find_if(allocations.begin(),allocations.end(),[&](const Allocation& a){return a.base+red==pointer;});assert(it!=allocations.end());
 for(unsigned i=0;i<red;++i)assert(it->base[i]==0xa5u&&it->base[red+it->bytes+i]==0xa5u);
 std::free(it->base);allocations.erase(it);
}
hipError_t hipGetLastError(){return touch();}
hipError_t hipMemcpy(void* destination,const void* source,size_t bytes,unsigned kind){
 const auto status=touch();if(status!=hipSuccess)return status;
 assert(work.empty()&&kind==hipMemcpyDeviceToHost&&bytes==8192u*8u*4u);span(source,bytes);std::memcpy(destination,source,bytes);return hipSuccess;
}
#define hipLaunchKernelGGL(kernel,blocks,block,shared,stream,...) \
 do{const auto g=blocks,t=block;assert(!(shared));++launches;work.push_back([=]{grid=g;threads=t;assert(t.x==256u);kernel(__VA_ARGS__);});}while(0)
namespace qrt_sm121_staged_half_projection{struct Row{uint32_t word[9];};}
namespace qrt_sm121_scaled_half_projection{
void prepare_rows(const uint16_t* source,qrt_sm121_staged_half_projection::Row* target,unsigned rows,unsigned width){
 assert(width==4096u&&((rows==2048u&&source==&weight)||(rows==8192u&&source==&input)));
 const size_t n=size_t(rows)*(width/16u);span(target,n*sizeof(*target));assert(grid.x==(n+255u)/256u);target[0].word[0]=rows;target[n-1u].word[8]=width;
}}
bool resident_bf16_matrix_matmul_f32_output_with_heuristic_index(const uint16_t* w,const uint16_t* x,float* target,
 unsigned rows,unsigned width,unsigned tokens,unsigned algorithm,hipStream_t,const std::string&,std::string*,std::string*){
 assert(w==&weight&&x==&input&&target==&raw&&rows==2048u&&width==4096u&&tokens==8192u&&algorithm==0u);
 if(touch()!=hipSuccess)return false;work.push_back([]{raw=2.0f;});return true;
}
void bf16_row_l2_upper_bound_kernel(const uint16_t* source,float* target,unsigned count,unsigned width){
 assert(width==4096u&&grid.x==count&&((count==8192u&&source==&input)||(count==2048u&&source==&weight)));
 span(target,count*4u);target[0]=1.0f;target[count-1u]=2.0f;
}
namespace qrt_out_variance_replay{
void execute(const qrt_sm121_staged_half_projection::Row* w,const qrt_sm121_staged_half_projection::Row* x,
 float* target,const float* previous,const uint16_t* norms,const uint8_t* correction,const float* xn,const float* wn,
 unsigned tokens,unsigned width,unsigned radius,unsigned ppb,unsigned* reports){
 assert(grid.x==8192u&&tokens==8192u&&width==4096u&&radius==512u&&ppb==1000u);
 assert(target==&raw&&raw==2.0f&&previous==&residual&&norms==&norm&&correction==&table);
 span(w,2048u*256u*sizeof(*w));span(x,8192u*256u*sizeof(*x));span(xn,8192u*4u);span(wn,2048u*4u);span(reports,8192u*8u*4u);
 assert(w[0].word[0]==2048u&&w[2048u*256u-1u].word[8]==4096u&&x[0].word[0]==8192u&&xn[8191u]==2.0f&&wn[2047u]==2.0f);
 for(unsigned token=0;token<8192u;++token){
  auto* r=reports+token*8u;std::fill_n(r,8u,0u);
  r[0]=selected;r[1]=std::min(selected,64u);r[2]=scenario?selected-r[1]:(selected-r[1])/2u;r[3]=selected-r[1]-r[2];r[4]=scenario?18u:7u;r[5]=scenario?1u:0u;r[6]=scenario?0u:1u;r[7]=scenario==2u&&r[2]?1u:0u;
  if(scenario==3u)r[0]=2049u;
  if(scenario==4u)r[4]=0u;
  if(scenario==5u)r[3]=1u;
  if(scenario==6u)r[6]=2u;
 }
 ++executed;
}}
void f32_to_bf16_kernel(const float* source,uint16_t* target,size_t cells){assert(source==&raw&&target==&output&&cells==16777216u&&grid.x==cells/256u&&executed==1u);output=0x3f80u;++converted;}
void bf16_to_f32_kernel(const uint16_t* source,float* target,size_t cells){assert(source==&output&&target==&raw&&cells==16777216u&&grid.x==cells/256u&&converted==1u);raw=1.0f;}
#include "out_variance_replay_actual_owner.h"
void reset(unsigned failure=0u){assert(allocations.empty()&&work.empty());steps=0;fail_at=failure;launches=converted=executed=0;output=0;raw=-1.0f;}
int main(){
 using namespace qrt_out_variance_replay;
 assert(workspace_bytes==94674944u);
 for(const char* off:{static_cast<const char*>(nullptr),"","0"})assert(setting(off)==0);
 assert(setting("1")==1);for(const char* bad:{"2","-1","01","true"," 1"})assert(setting(bad)==-1);
 for(unsigned flags=0;flags<8u;++flags)assert(applicable(2048,8192,4096,512,1000,flags&1u,flags&2u,flags&4u)==(flags==3u));
 for(unsigned n:{0u,1u,7169u,8191u,8193u,16384u,262144u})assert(!applicable(2048,n,4096,512,1000,true,true,false));
 const char* options[]={"QRT_QWEN36_COARSE_LINEAR_OUT_PRODUCER","QRT_QWEN36_Q8192_OUT_MATRIX_SHADOW_AUDIT","QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT","QRT_QWEN36_Q8192_OUT_L1_BOUND","QRT_QWEN36_Q8192_OUT_RESIDUAL_FILTER","QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT","QRT_QWEN36_Q8192_OUT_VARIANCE_BUDGET_AUDIT","QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_BOUND"};
 for(auto* name:options)unsetenv(name);
 setenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM","4",1);setenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","qkv",1);assert(options_compatible());
 for(auto* name:options){for(const char* value:{"","0","1","2","01","invalid"}){setenv(name,value,1);assert(options_compatible()==(setting(value)==0));}unsetenv(name);}
 setenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","all",1);assert(!options_compatible());setenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE","qkv",1);
 unsigned successful=0,failed=0;Stats stats;
 for(unsigned mode=0;mode<3u;++mode)for(unsigned n:{0u,1u,63u,64u,65u,257u,2048u}){
  scenario=mode;selected=n;reset();assert(run(&weight,&input,&residual,&norm,&table,&output,&raw,nullptr,&stats)==hipSuccess);
  assert(stats.selected==uint64_t(n)*8192u&&stats.first==uint64_t(std::min(n,64u))*8192u&&stats.first+stats.additional+stats.skipped==stats.selected);
  assert(stats.certified_rows==(scenario?0u:8192u)&&stats.fallback_rows==(scenario?8192u:0u)&&stats.rounds==uint64_t(scenario?18u:7u)*8192u);
  assert(launches==7u&&converted==1u&&executed==1u&&raw==1.0f&&output==0x3f80u);++successful;const unsigned count=steps;
  for(unsigned failure=1u;failure<=count;++failure){reset(failure);assert(run(&weight,&input,&residual,&norm,&table,&output,&raw,nullptr,&stats)!=hipSuccess);assert(allocations.empty()&&work.empty());++failed;}
 }
 selected=257u;for(unsigned mode=3u;mode<=6u;++mode){scenario=mode;reset();assert(run(&weight,&input,&residual,&norm,&table,&output,&raw,nullptr,&stats)==hipErrorInvalidValue&&allocations.empty()&&work.empty());}
 for(unsigned missing=0;missing<9u;++missing){reset();assert(run(missing==0?nullptr:&weight,missing==1?nullptr:&input,missing==2?nullptr:&residual,missing==3?nullptr:&norm,missing==4?nullptr:&table,missing==5?nullptr:&output,missing==6?nullptr:missing==8?&residual:&raw,nullptr,missing==7?nullptr:&stats)==hipErrorInvalidValue&&!steps);}
 assert(weight==17u&&input==29u&&residual==3.0f&&norm==0u&&table==3u);
 std::printf("out_variance_replay_host_pass success_cases=%u injected_failures=%u invalid_reports=4 workspace_bytes=%zu deferred_owner_executed=1 native_kernel_executed=0\n",successful,failed,workspace_bytes);
}
