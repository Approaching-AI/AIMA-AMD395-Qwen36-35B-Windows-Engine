#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_folded_half_projection.h"
#include "folded_half_reference.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace scaled=qrt_sm121_folded_half_products;
namespace projection=qrt_sm121_folded_half_projection;
namespace cases=qrt_strong_replay_cases;
constexpr unsigned guard=65u;
struct Result { float value;unsigned folded,fallback; };
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
 constexpr unsigned Lanes=4u;
 const unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/Lanes;if(row>=rows)return;
 projection::Stats stats;const float value=projection::dot<StagingGroups,Audit>(a+size_t(row)*(width/16u),b+size_t(row)*(width/16u),width,
  Audit?trace+size_t(row)*(width/16u)*3u:nullptr,Audit?&stats:nullptr);
 if(!(threadIdx.x&(Lanes-1u)))out[row]={value,stats.folded,stats.fallback};
}
template<unsigned StagingGroups>
void variant(unsigned rows,unsigned width,Device& a,Device& b,Device& pa,Device& pb,Device& out,Device& trace,
 const std::vector<uint16_t>& left,const std::vector<uint16_t>& right,const std::vector<uint32_t>& expected,const std::vector<unsigned>& folded){
 constexpr unsigned Lanes=4u;
 const size_t groups=size_t(rows)*(width/16u);
 check(hipMemset(out.pointer,0xa5,(rows+2u*guard)*sizeof(Result)));check(hipMemset(trace.pointer,0xa5,(groups*3u+2u*guard)*4u));
 hipLaunchKernelGGL((execute<StagingGroups,true>),dim3((rows*Lanes+255u)/256u),dim3(256u),0u,nullptr,pa.data<scaled::Row>(),pb.data<scaled::Row>(),out.data<Result>(),trace.data<uint32_t>(),rows,width);check(hipGetLastError());finish();
 const auto values=read<Result>(out,rows);const auto actual=read<uint32_t>(trace,groups*3u);guards(values);guards(actual);
 if(std::memcmp(actual.data()+guard,expected.data(),expected.size()*4u))throw std::runtime_error("raw K16 carry differs from independent original primitive");
 unsigned floating=0u,original=0u;
 for(unsigned row=0u;row<rows;++row){
  const size_t base=((size_t(row)+1u)*(width/16u)-1u)*3u;
  const qrt_q1_moe_hawkeye::Value carry{expected[base],int16_t(int32_t(expected[base+1u])),expected[base+2u]!=0u};
  const float reference=qrt_q1_moe_hawkeye::value_to_float(qrt_q1_moe_hawkeye::group_sum<26,-133>(&carry,1u));
  if(std::memcmp(&reference,&values[guard+row].value,4u))throw std::runtime_error("final projection endpoint differs");
  if(values[guard+row].folded!=folded[row] || values[guard+row].fallback!=width/16u-folded[row])throw std::runtime_error("group path count differs from independent original-exponent predicate");
  floating+=values[guard+row].folded;original+=values[guard+row].fallback;
 }
 check(hipMemset(out.pointer,0xa5,(rows+2u*guard)*sizeof(Result)));
 hipLaunchKernelGGL((execute<StagingGroups,false>),dim3((rows*Lanes+255u)/256u),dim3(256u),0u,nullptr,pa.data<scaled::Row>(),pb.data<scaled::Row>(),out.data<Result>(),trace.data<uint32_t>(),rows,width);check(hipGetLastError());finish();
 const auto production=read<Result>(out,rows);guards(production);for(unsigned row=0u;row<rows;++row)if(std::memcmp(&production[guard+row].value,&values[guard+row].value,4u))throw std::runtime_error("production and audit differ");
 if(read<uint32_t>(trace,groups*3u)!=actual)throw std::runtime_error("production modified audit trace");
 for(unsigned side=0u;side<2u;++side){
  const auto& input=side?right:left;const auto raw=read<uint16_t>(side?b:a,input.size());const auto packed=read<scaled::Row>(side?pb:pa,groups);guards(raw);guards(packed);
  if(std::memcmp(raw.data()+guard,input.data(),input.size()*2u))throw std::runtime_error("original operand changed");
  for(size_t group=0u;group<groups;++group){const auto reference=scaled::prepare(input.data()+group*16u);if(std::memcmp(&reference,&packed[guard+group],sizeof(reference)))throw std::runtime_error("operand encoding differs or changed");for(unsigned i=0u;i<16u;++i)if(scaled::original(packed[guard+group],i)!=input[group*16u+i])throw std::runtime_error("lossless operand roundtrip failed");}
 }
 std::printf("{\"kind\":\"folded_half_projection_safety\",\"lanes\":%u,\"staging_groups\":%u,\"rows\":%u,\"width\":%u,\"ordered_raw_carry_states\":%zu,\"folded_groups\":%u,\"fallback_groups\":%u,\"raw_bit_mismatches\":0,\"unaligned_operands\":true,\"production_diagnostic_parity\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",Lanes,StagingGroups,rows,width,groups,floating,original);std::fflush(stdout);
 if(!floating || !original || floating+original!=groups)throw std::runtime_error("missing numerical path coverage");
}
void run(unsigned rows,unsigned width){
 const size_t words=size_t(rows)*width,groups=words/16u;std::vector<uint16_t> left(words),right(words);std::vector<uint32_t> expected(groups*3u);std::vector<unsigned> folded(rows,0u);
 for(unsigned row=0u;row<rows;++row){qrt_q1_moe_hawkeye::Value carry{0u,-133,false};for(unsigned group=0u;group<width/16u;++group){qrt_q1_moe_hawkeye::Value terms[17];terms[0]=carry;const size_t base=size_t(row)*width+group*16u;for(unsigned i=0u;i<16u;++i){const auto input=cases::input(row,group,i);left[base+i]=input.x;right[base+i]=input.y;terms[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(input.x,input.y,-133);}folded[row]+=qrt_folded_half_reference::expected(carry,left.data()+base,right.data()+base).allowed;carry=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);const size_t out=base/16u*3u;expected[out]=carry.significand;expected[out+1u]=uint32_t(int32_t(carry.exponent));expected[out+2u]=unsigned(carry.negative);}}
 Device a((words+2u*guard)*2u),b((words+2u*guard)*2u),pa((groups+2u*guard)*sizeof(scaled::Row)),pb((groups+2u*guard)*sizeof(scaled::Row)),out((rows+2u*guard)*sizeof(Result)),trace((groups*3u+2u*guard)*4u);
 check(hipMemcpy(a.data<uint16_t>(),left.data(),words*2u,hipMemcpyHostToDevice));check(hipMemcpy(b.data<uint16_t>(),right.data(),words*2u,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(projection::prepare_rows,dim3((groups+255u)/256u+1u),dim3(256u),0u,nullptr,a.data<uint16_t>(),pa.data<scaled::Row>(),rows,width);check(hipGetLastError());
 hipLaunchKernelGGL(projection::prepare_rows,dim3((groups+255u)/256u+1u),dim3(256u),0u,nullptr,b.data<uint16_t>(),pb.data<scaled::Row>(),rows,width);check(hipGetLastError());finish();
 variant<1u>(rows,width,a,b,pa,pb,out,trace,left,right,expected,folded);variant<2u>(rows,width,a,b,pa,pb,out,trace,left,right,expected,folded);variant<4u>(rows,width,a,b,pa,pb,out,trace,left,right,expected,folded);variant<8u>(rows,width,a,b,pa,pb,out,trace,left,right,expected,folded);
}
int main()try{hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");run(257u,16u);run(4096u,272u);run(2048u,2048u);run(1024u,4096u);run(129u,4112u);run(129u,8192u);return 0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
