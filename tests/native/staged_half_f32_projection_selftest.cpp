#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_staged_half_f32_projection.h"
#include "strong_float_replay_cases.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace scaled=qrt_sm121_scaled_half_products;
namespace projection=qrt_sm121_staged_half_f32_projection;
namespace cases=qrt_strong_replay_cases;
constexpr unsigned guard=65u;
struct Result { float value;unsigned accepted,restarted; };
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
struct Device {
 void* pointer=nullptr;
 explicit Device(size_t bytes){check(hipMalloc(&pointer,bytes));check(hipMemset(pointer,0xa5,bytes));}
 ~Device(){if(pointer)(void)hipFree(pointer);}
 template<class T>T* data(){return static_cast<T*>(pointer)+guard;}
};
template<class T>std::vector<T> read(Device& d,size_t count){std::vector<T> out(count+2u*guard);check(hipMemcpy(out.data(),d.pointer,out.size()*sizeof(T),hipMemcpyDeviceToHost));return out;}
template<class T>void guards(const std::vector<T>& data){const auto* b=reinterpret_cast<const unsigned char*>(data.data());for(size_t i=0u;i<guard*sizeof(T);++i)if(b[i]!=0xa5u || b[(data.size()-guard)*sizeof(T)+i]!=0xa5u)throw std::runtime_error("projection redzone changed");}
void finish(){hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("projection completion deadline");std::this_thread::yield();}check(hipEventDestroy(event));}

template<unsigned StagingGroups,bool Audit>
__global__ void execute(const scaled::Row* a,const scaled::Row* b,Result* out,uint32_t* trace,unsigned rows,unsigned width){
 const unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(row>=rows)return;
 projection::Stats stats;const float value=projection::dot<StagingGroups,Audit>(a+size_t(row)*(width/16u),b+size_t(row)*(width/16u),width,
  Audit?trace+size_t(row)*(width/16u):nullptr,Audit?&stats:nullptr);
 if(!(threadIdx.x&3u))out[row]={value,stats.accepted_groups,unsigned(stats.restarted)};
}
template<unsigned StagingGroups>
void variant(unsigned rows,unsigned width,bool ordinary,Device& a,Device& b,Device& pa,Device& pb,Device& out,Device& trace,
 const std::vector<uint16_t>& left,const std::vector<uint16_t>& right,const std::vector<uint32_t>& expected,const std::vector<uint32_t>& endpoints){
 const size_t groups=size_t(rows)*(width/16u);
 check(hipMemset(out.pointer,0xa5,(rows+2u*guard)*sizeof(Result)));check(hipMemset(trace.pointer,0xa5,(groups+2u*guard)*4u));
 hipLaunchKernelGGL((execute<StagingGroups,true>),dim3((rows*4u+255u)/256u),dim3(256u),0u,nullptr,pa.data<scaled::Row>(),pb.data<scaled::Row>(),out.data<Result>(),trace.data<uint32_t>(),rows,width);check(hipGetLastError());finish();
 const auto values=read<Result>(out,rows);const auto actual=read<uint32_t>(trace,groups);guards(values);guards(actual);
 for(size_t i=0u;i<groups;++i)if(actual[guard+i]!=expected[i]){
  std::fprintf(stderr,"group=%zu actual=%08x expected=%08x\n",i,actual[guard+i],expected[i]);
  throw std::runtime_error("K16 FP32 endpoint differs from independent original primitive");
 }
 size_t accepted=0u,restarts=0u,full_fast=0u;
 for(unsigned row=0u;row<rows;++row){
  const auto& r=values[guard+row];if(projection::f32::bits(r.value)!=endpoints[row])throw std::runtime_error("final projection endpoint differs");
  if(r.accepted>width/16u || r.restarted>1u || bool(r.restarted)!=(r.accepted<width/16u))throw std::runtime_error("invalid completion/restart accounting");
  accepted+=r.accepted;restarts+=r.restarted;full_fast+=!r.restarted;
 }
 if(!full_fast || (!ordinary && !restarts))throw std::runtime_error("missing numerical path coverage");
 check(hipMemset(out.pointer,0xa5,(rows+2u*guard)*sizeof(Result)));
 hipLaunchKernelGGL((execute<StagingGroups,false>),dim3((rows*4u+255u)/256u),dim3(256u),0u,nullptr,pa.data<scaled::Row>(),pb.data<scaled::Row>(),out.data<Result>(),trace.data<uint32_t>(),rows,width);check(hipGetLastError());finish();
 const auto production=read<Result>(out,rows);guards(production);
 for(unsigned row=0u;row<rows;++row)if(projection::f32::bits(production[guard+row].value)!=endpoints[row])throw std::runtime_error("production and diagnostic differ");
 if(read<uint32_t>(trace,groups)!=actual)throw std::runtime_error("production modified diagnostic trace");
 for(unsigned side=0u;side<2u;++side){
  const auto& input=side?right:left;const auto raw=read<uint16_t>(side?b:a,input.size());const auto packed=read<scaled::Row>(side?pb:pa,groups);guards(raw);guards(packed);
  if(std::memcmp(raw.data()+guard,input.data(),input.size()*2u))throw std::runtime_error("original operand changed");
  for(size_t group=0u;group<groups;++group){const auto reference=scaled::prepare(input.data()+group*16u);if(std::memcmp(&reference,&packed[guard+group],sizeof(reference)))throw std::runtime_error("operand encoding differs or changed");for(unsigned i=0u;i<16u;++i)if(scaled::original(packed[guard+group],i)!=input[group*16u+i])throw std::runtime_error("lossless operand roundtrip failed");}
 }
 std::printf("{\"kind\":\"staged_half_f32_projection_safety\",\"lanes\":4,\"staging_groups\":%u,\"rows\":%u,\"width\":%u,\"ordinary_inputs\":%s,\"ordered_fp32_carry_endpoints\":%zu,\"accepted_groups_before_restart\":%zu,\"restarted_dots\":%zu,\"fully_fast_dots\":%zu,\"raw_bit_mismatches\":0,\"unaligned_operands\":true,\"production_diagnostic_parity\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",StagingGroups,rows,width,ordinary?"true":"false",groups,accepted,restarts,full_fast);std::fflush(stdout);
}
void run(unsigned rows,unsigned width,bool ordinary=false){
 const size_t words=size_t(rows)*width,groups=words/16u;
 std::vector<uint16_t> left(words),right(words);std::vector<uint32_t> expected(groups),endpoints(rows);
 for(unsigned row=0u;row<rows;++row){
  qrt_q1_moe_hawkeye::Value carry{0u,-133,false};
  for(unsigned group=0u;group<width/16u;++group){
   qrt_q1_moe_hawkeye::Value terms[17];terms[0]=carry;const size_t base=size_t(row)*width+group*16u;
   for(unsigned i=0u;i<16u;++i){
    auto input=cases::input(row,group,i);
    if(ordinary){const unsigned a=cases::random_word(unsigned(base+i)^0x3958192u),b=cases::random_word(a^0x8192395u);input={uint16_t((a&0x807fu)|((120u+a%12u)<<7u)),uint16_t((b&0x807fu)|((119u+b%14u)<<7u))};if((base+i)%37u==0u)input.x=uint16_t(a&0x8000u);}
    left[base+i]=input.x;right[base+i]=input.y;
    terms[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(input.x,input.y,-133);
   }
   carry=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);
   expected[base/16u]=projection::f32::bits(qrt_q1_moe_hawkeye::value_to_float(carry));
  }
  endpoints[row]=cases::output_bits(carry);
 }
 Device a((words+2u*guard)*2u),b((words+2u*guard)*2u),pa((groups+2u*guard)*sizeof(scaled::Row)),pb((groups+2u*guard)*sizeof(scaled::Row)),out((rows+2u*guard)*sizeof(Result)),trace((groups+2u*guard)*4u);
 check(hipMemcpy(a.data<uint16_t>(),left.data(),words*2u,hipMemcpyHostToDevice));check(hipMemcpy(b.data<uint16_t>(),right.data(),words*2u,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((groups+255u)/256u+1u),dim3(256u),0u,nullptr,a.data<uint16_t>(),pa.data<scaled::Row>(),rows,width);check(hipGetLastError());
 hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((groups+255u)/256u+1u),dim3(256u),0u,nullptr,b.data<uint16_t>(),pb.data<scaled::Row>(),rows,width);check(hipGetLastError());finish();
 variant<2u>(rows,width,ordinary,a,b,pa,pb,out,trace,left,right,expected,endpoints);
 variant<4u>(rows,width,ordinary,a,b,pa,pb,out,trace,left,right,expected,endpoints);
}
int main()try{
 hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
 run(257u,16u);run(4096u,272u);run(2048u,2048u);run(1024u,4096u);run(129u,4112u);run(129u,8192u);run(2048u,2048u,true);
 return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
