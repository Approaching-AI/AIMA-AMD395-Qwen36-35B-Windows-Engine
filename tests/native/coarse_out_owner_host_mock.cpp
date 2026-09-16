#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <vector>
enum hipError_t{hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
constexpr unsigned hipMemcpyDeviceToHost=2u,red=128u;
struct dim3{unsigned x,y;explicit dim3(unsigned a,unsigned b=1u):x(a),y(b){}};
struct Allocation{unsigned char* base;size_t bytes;};
std::vector<Allocation> allocations;std::deque<std::function<void()>> work;
unsigned steps=0u,fail_at=0u,selected=0u,replayed=0u,converted=0u,launches=0u;
unsigned maximum_grid=4096u;dim3 grid(1u),threads(1u);
uint16_t weight=17u,input=29u,output=0u;
float rounded_output=-1.0f;
hipError_t touch(){return ++steps==fail_at?hipErrorUnknown:hipSuccess;}
void span(const void* pointer,size_t bytes){
 const uintptr_t p=reinterpret_cast<uintptr_t>(pointer);
 assert(std::any_of(allocations.begin(),allocations.end(),[&](const Allocation& a){
  const uintptr_t begin=reinterpret_cast<uintptr_t>(a.base+red);
  return p>=begin&&p-begin<=a.bytes&&bytes<=a.bytes-(p-begin);
 }));
}
hipError_t hipMalloc(void** pointer,size_t bytes){
 *pointer=nullptr;const auto status=touch();if(status!=hipSuccess)return status;
 auto* base=static_cast<unsigned char*>(std::malloc(bytes+2u*red));assert(base);
 std::memset(base,0xa5,red);std::memset(base+red+bytes,0xa5,red);
 allocations.push_back({base,bytes});*pointer=base+red;return hipSuccess;
}
void drain(){while(!work.empty()){auto f=std::move(work.front());work.pop_front();f();}}
hipError_t hipStreamSynchronize(hipStream_t){const auto status=touch();drain();return status;}
void free_device(void* pointer){
 assert(work.empty());const auto it=std::find_if(allocations.begin(),allocations.end(),[&](const Allocation& a){return a.base+red==pointer;});assert(it!=allocations.end());
 for(unsigned i=0u;i<red;++i)assert(it->base[i]==0xa5u&&it->base[red+it->bytes+i]==0xa5u);
 std::free(it->base);allocations.erase(it);
}
hipError_t hipGetLastError(){return touch();}
hipError_t hipMemsetAsync(void* pointer,int value,size_t bytes,hipStream_t){
 const auto status=touch();if(status!=hipSuccess)return status;
 work.push_back([=]{span(pointer,bytes);std::memset(pointer,value,bytes);});return hipSuccess;
}
hipError_t hipMemcpy(void* destination,const void* source,size_t bytes,unsigned kind){
 const auto status=touch();if(status!=hipSuccess)return status;
 assert(work.empty()&&kind==hipMemcpyDeviceToHost&&bytes==sizeof(unsigned));span(source,bytes);std::memcpy(destination,source,bytes);return hipSuccess;
}
#define hipLaunchKernelGGL(kernel,blocks,block,shared,stream,...) \
 do{const auto g=blocks,t=block;assert(!(shared));++launches;work.push_back([=]{grid=g;threads=t;assert(t.x==256u);kernel(__VA_ARGS__);});}while(0)
namespace qrt_sm121_staged_half_projection{struct Row{uint32_t word[9];};}
namespace qrt_sm121_scaled_half_projection{
void prepare_rows(const uint16_t* source,qrt_sm121_staged_half_projection::Row* target,unsigned rows,unsigned width){
 assert(width==4096u&&((rows==2048u&&source==&weight)||(rows==8192u&&source==&input)));
 const size_t n=size_t(rows)*(width/16u);span(target,n*sizeof(*target));assert(grid.x==(n+255u)/256u);
 target[0].word[0]=rows;target[n-1u].word[8]=width;
}}
namespace qrt_sm121_coarse_projection_matrix{
void eligibility(const uint16_t* source,unsigned* target,unsigned rows,unsigned width){
 assert(width==4096u&&grid.x==rows&&((rows==2048u&&source==&weight)||(rows==8192u&&source==&input)));
 span(target,size_t(rows)*4u);target[0]=1u;target[rows-1u]=0u;
}
template<unsigned Chunk,unsigned Fragments>
void produce(const uint16_t* w,const uint16_t* x,const unsigned* wf,const unsigned* xf,float* center,float* error,unsigned rows,unsigned tokens,unsigned width){
 static_assert(Chunk==64u&&Fragments==1u);assert(w==&weight&&x==&input&&rows==2048u&&tokens==8192u&&width==4096u);
 assert(grid.x==16u&&grid.y==512u);span(wf,rows*4u);span(xf,tokens*4u);assert(wf[0]==1u&&xf[tokens-1u]==0u);
 span(center,size_t(rows)*tokens*4u);span(error,size_t(rows)*tokens*4u);assert(error==center+size_t(rows)*tokens);
 center[0]=1.0f;center[size_t(rows)*tokens-1u]=2.0f;error[0]=0.0f;error[size_t(rows)*tokens-1u]=3.0f;
}
void compact(const float* center,const float* error,float* target,unsigned* ids,unsigned* count,unsigned cells){
 assert(cells==16777216u&&grid.x==cells/256u&&center==target&&center[0]==1.0f&&error[cells-1u]==3.0f);
 span(ids,size_t(cells)*4u);span(count,4u);assert(!*count);*count=selected;
 if(selected&&selected<=cells){ids[0]=cells-1u;ids[selected-1u]=0u;}
}}
void selected_bf16_projection_hawkeye_staged_half_kernel(
 const qrt_sm121_staged_half_projection::Row* w,const qrt_sm121_staged_half_projection::Row* x,
 float* raw,unsigned rows,unsigned width,const unsigned* ids,unsigned offset,unsigned end){
 assert(rows==2048u&&width==4096u&&offset==replayed&&end>offset&&end<=selected&&end-offset<=maximum_grid*64u);
 assert(grid.x==(end-offset+63u)/64u&&grid.x<=maximum_grid);
 span(w,size_t(rows)*(width/16u)*sizeof(*w));span(x,size_t(8192u)*(width/16u)*sizeof(*x));
 assert(w[0].word[0]==rows&&w[size_t(rows)*(width/16u)-1u].word[8]==width&&x[0].word[0]==8192u);
 span(ids,size_t(selected)*4u);span(raw,size_t(16777216u)*4u);replayed=end;
}
void f32_to_bf16_kernel(const float* raw,uint16_t* target,size_t cells){
 assert(target==&output&&cells==16777216u&&grid.x==cells/256u&&replayed==selected);span(raw,cells*4u);
 output=0x3f80u;++converted;
}
void bf16_to_f32_kernel(const uint16_t* source,float* target,size_t cells){
 assert(source==&output&&target==&rounded_output&&cells==16777216u&&grid.x==cells/256u&&converted==1u&&output==0x3f80u);
 rounded_output=1.0f;
}
#include "q8192_coarse_out.h"
void reset(unsigned failure=0u){assert(allocations.empty()&&work.empty());steps=0u;fail_at=failure;replayed=converted=launches=0u;output=0u;rounded_output=-1.0f;}
int main(){
 assert(qrt_coarse_out::workspace_bytes==295739396u);
 for(const char* value:{static_cast<const char*>(nullptr),"","0"})assert(qrt_coarse_out::setting(value)==0);
 assert(qrt_coarse_out::setting("1")==1);
 for(const char* value:{"2","-1","01","true"," 1"})assert(qrt_coarse_out::setting(value)==-1);
 assert(qrt_coarse_out::applicable(2048u,8192u,4096u,512u,10000u));
 for(unsigned i=0u;i<5u;++i){unsigned p[]={2048u,8192u,4096u,512u,10000u};--p[i];assert(!qrt_coarse_out::applicable(p[0],p[1],p[2],p[3],p[4]));}
 for(unsigned flags=0u;flags<8u;++flags)
  assert(qrt_coarse_out::linear_applicable(2048u,8192u,4096u,512u,1000u,flags&1u,flags&2u,flags&4u)==(flags==3u));
 for(unsigned i=0u;i<5u;++i){unsigned p[]={2048u,8192u,4096u,512u,1000u};--p[i];assert(!qrt_coarse_out::linear_applicable(p[0],p[1],p[2],p[3],p[4],true,true,false));}
 for(unsigned n:{0u,1u,7169u,8191u,8193u,16384u,262144u})
  assert(!qrt_coarse_out::linear_applicable(2048u,n,4096u,512u,1000u,true,true,false));
 const char* options[]={"QRT_QWEN36_Q8192_OUT_MATRIX_SHADOW_AUDIT","QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT","QRT_QWEN36_Q8192_OUT_L1_BOUND","QRT_QWEN36_Q8192_OUT_RESIDUAL_FILTER","QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT","QRT_QWEN36_Q8192_LINEAR_OUT_VARIANCE_REPLAY","QRT_QWEN36_Q8192_OUT_VARIANCE_BUDGET_AUDIT"};
 for(const char* name:options)unsetenv(name);
 assert(qrt_coarse_out::linear_options_compatible());
 for(const char* name:options){
  for(const char* value:{"","0","1","2","01","invalid"}){
   assert(!setenv(name,value,1));assert(qrt_coarse_out::linear_options_compatible()==(qrt_coarse_out::setting(value)==0));
  }
  unsetenv(name);
 }
 unsigned failure_cases=0u,success_cases=0u;
 for(bool linear:{false,true})for(unsigned n:{0u,1u,63u,64u,65u,262144u,262145u,16777216u}){
  selected=n;maximum_grid=4096u;reset();qrt_coarse_out::Stats stats;
  assert(qrt_coarse_out::run(&weight,&input,&output,maximum_grid,nullptr,&stats,linear?&rounded_output:nullptr)==hipSuccess);
  assert(stats.candidates==selected&&stats.dispatches==(n+262143u)/262144u&&converted==1u&&output==0x3f80u&&allocations.empty()&&work.empty());
  assert(rounded_output==(linear?1.0f:-1.0f));
  const unsigned count=steps;++success_cases;
  for(unsigned failure=1u;failure<=count;++failure){reset(failure);assert(qrt_coarse_out::run(&weight,&input,&output,maximum_grid,nullptr,&stats,linear?&rounded_output:nullptr)!=hipSuccess);assert(allocations.empty()&&work.empty());++failure_cases;}
 }
 selected=65u;maximum_grid=1u;reset();qrt_coarse_out::Stats stats;
 assert(qrt_coarse_out::run(&weight,&input,&output,1u,nullptr,&stats)==hipSuccess&&stats.dispatches==2u);++success_cases;
 selected=16777217u;reset();assert(qrt_coarse_out::run(&weight,&input,&output,1u,nullptr,&stats)==hipErrorInvalidValue&&converted==0u&&allocations.empty()&&work.empty());
 for(unsigned missing=0u;missing<6u;++missing){reset();assert(qrt_coarse_out::run(missing==0u?nullptr:&weight,missing==1u?nullptr:&input,
  missing==2u?nullptr:&output,missing==4u?0u:missing==5u?4097u:1u,nullptr,missing==3u?nullptr:&stats)==hipErrorInvalidValue&&!steps);}
 assert(weight==17u&&input==29u);
 std::printf("coarse_out_owner_host_pass successful_shapes=%u injected_failures=%u maximum_selected=%u workspace_bytes=%zu native_matrix_executed=0\n",success_cases,failure_cases,16777216u,qrt_coarse_out::workspace_bytes);
}
