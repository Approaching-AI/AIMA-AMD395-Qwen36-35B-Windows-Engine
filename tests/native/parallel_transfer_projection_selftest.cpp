#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_parallel_transfer_projection.h"
#include "dominant_half_cases.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace scaled=qrt_sm121_scaled_half_products;
namespace projection=qrt_sm121_parallel_transfer_projection;
namespace cases=qrt_strong_replay_cases;
constexpr unsigned guard=65u;
struct Result { float value;unsigned transformed,original,transfer_blocks,reused_groups,transition_replays; };
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
struct Device {
 void* pointer=nullptr;
 explicit Device(size_t bytes){check(hipMalloc(&pointer,bytes));check(hipMemset(pointer,0xa5,bytes));}
 ~Device(){if(pointer && hipFree(pointer)!=hipSuccess)std::abort();}
 template<class T>T* data(){return static_cast<T*>(pointer)+guard;}
};
template<class T>std::vector<T> read(Device& d,size_t count){std::vector<T> out(count+2u*guard);check(hipMemcpy(out.data(),d.pointer,out.size()*sizeof(T),hipMemcpyDeviceToHost));return out;}
template<class T>void guards(const std::vector<T>& data){const auto* b=reinterpret_cast<const unsigned char*>(data.data());for(size_t i=0u;i<guard*sizeof(T);++i)if(b[i]!=0xa5u || b[(data.size()-guard)*sizeof(T)+i]!=0xa5u)throw std::runtime_error("projection redzone changed");}
void finish(){hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("projection completion deadline");std::this_thread::yield();}check(hipEventDestroy(event));}
template<unsigned StagingGroups,bool Audit>
__global__ void execute(const scaled::Row* a,const scaled::Row* b,Result* out,uint32_t* trace,unsigned rows,unsigned width){
 constexpr unsigned Lanes=StagingGroups*4u;
 const unsigned row=(blockIdx.x*blockDim.x+threadIdx.x)/Lanes;if(row>=rows)return;
 projection::Stats stats;const float value=projection::dot<StagingGroups,Audit>(a+size_t(row)*(width/16u),b+size_t(row)*(width/16u),width,
  Audit?trace+size_t(row)*(width/16u)*3u:nullptr,Audit?&stats:nullptr);
 if(!(threadIdx.x&(Lanes-1u)))out[row]={value,stats.transformed,stats.original,stats.transfer_blocks,stats.reused_groups,stats.transition_replays};
}
template<unsigned StagingGroups>
void variant(unsigned rows,unsigned width,bool ordinary,Device& a,Device& b,Device& pa,Device& pb,Device& out,Device& trace,
 const std::vector<uint16_t>& left,const std::vector<uint16_t>& right,const std::vector<uint32_t>& expected,const std::vector<unsigned>& transformed){
 constexpr unsigned Lanes=StagingGroups*4u;
 const size_t groups=size_t(rows)*(width/16u);
 check(hipMemset(out.pointer,0xa5,(rows+2u*guard)*sizeof(Result)));check(hipMemset(trace.pointer,0xa5,(groups*3u+2u*guard)*4u));
 hipLaunchKernelGGL((execute<StagingGroups,true>),dim3((rows*Lanes+255u)/256u),dim3(256u),0u,nullptr,pa.data<scaled::Row>(),pb.data<scaled::Row>(),out.data<Result>(),trace.data<uint32_t>(),rows,width);check(hipGetLastError());finish();
 const auto values=read<Result>(out,rows);const auto actual=read<uint32_t>(trace,groups*3u);guards(values);guards(actual);
 if(std::memcmp(actual.data()+guard,expected.data(),expected.size()*4u))throw std::runtime_error("raw K16 carry differs from independent original primitive");
 unsigned floating=0u,original=0u,blocks=0u,reused=0u,transitions=0u;
 for(unsigned row=0u;row<rows;++row){
  const size_t base=((size_t(row)+1u)*(width/16u)-1u)*3u;
  const qrt_q1_moe_hawkeye::Value carry{expected[base],int16_t(int32_t(expected[base+1u])),expected[base+2u]!=0u};
  const float reference=qrt_q1_moe_hawkeye::value_to_float(qrt_q1_moe_hawkeye::group_sum<26,-133>(&carry,1u));
  if(std::memcmp(&reference,&values[guard+row].value,4u))throw std::runtime_error("final projection endpoint differs");
  if(values[guard+row].transformed!=transformed[row] || values[guard+row].original!=width/16u-transformed[row])throw std::runtime_error("group path count differs from independent row-range predicate");
  if(values[guard+row].transfer_blocks>(width/16u)/StagingGroups || values[guard+row].reused_groups>values[guard+row].transformed || values[guard+row].transition_replays>values[guard+row].transformed)throw std::runtime_error("invalid carry transfer counters");
  floating+=values[guard+row].transformed;original+=values[guard+row].original;blocks+=values[guard+row].transfer_blocks;reused+=values[guard+row].reused_groups;transitions+=values[guard+row].transition_replays;
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
 std::printf("{\"kind\":\"parallel_transfer_projection_safety\",\"lanes\":%u,\"staging_groups\":%u,\"rows\":%u,\"width\":%u,\"ordinary_inputs\":%s,\"ordered_raw_carry_states\":%zu,\"transformed_groups\":%u,\"original_groups\":%u,\"transfer_blocks\":%u,\"reused_groups\":%u,\"transition_replays\":%u,\"raw_bit_mismatches\":0,\"unaligned_operands\":true,\"production_diagnostic_parity\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",Lanes,StagingGroups,rows,width,ordinary?"true":"false",groups,floating,original,blocks,reused,transitions);std::fflush(stdout);
 if(!floating || (!ordinary && !original) || floating+original!=groups || reused>floating || (ordinary && !blocks))throw std::runtime_error("missing numerical path coverage");
}
void run(unsigned rows,unsigned width,bool ordinary=false){
 const size_t words=size_t(rows)*width,groups=words/16u;std::vector<uint16_t> left(words),right(words);std::vector<uint32_t> expected(groups*3u);std::vector<unsigned> transformed(rows,0u);
 for(unsigned row=0u;row<rows;++row){qrt_q1_moe_hawkeye::Value carry{0u,-133,false};for(unsigned group=0u;group<width/16u;++group){qrt_q1_moe_hawkeye::Value terms[17];terms[0]=carry;const size_t base=size_t(row)*width+group*16u;for(unsigned i=0u;i<16u;++i){const auto input=dominant_half_cases::input(row,group,i,ordinary);left[base+i]=input.x;right[base+i]=input.y;terms[i+1u]=qrt_q1_moe_hawkeye::multiply_bf16(input.x,input.y,-133);}transformed[row]+=dominant_half_cases::eligible_row(left.data()+base)&&dominant_half_cases::eligible_row(right.data()+base);carry=qrt_q1_moe_hawkeye::group_sum<26,-133>(terms,17u);const size_t out=base/16u*3u;expected[out]=carry.significand;expected[out+1u]=uint32_t(int32_t(carry.exponent));expected[out+2u]=unsigned(carry.negative);}}
 Device a((words+2u*guard)*2u),b((words+2u*guard)*2u),pa((groups+2u*guard)*sizeof(scaled::Row)),pb((groups+2u*guard)*sizeof(scaled::Row)),out((rows+2u*guard)*sizeof(Result)),trace((groups*3u+2u*guard)*4u);
 check(hipMemcpy(a.data<uint16_t>(),left.data(),words*2u,hipMemcpyHostToDevice));check(hipMemcpy(b.data<uint16_t>(),right.data(),words*2u,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((groups+255u)/256u+1u),dim3(256u),0u,nullptr,a.data<uint16_t>(),pa.data<scaled::Row>(),rows,width);check(hipGetLastError());
 hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((groups+255u)/256u+1u),dim3(256u),0u,nullptr,b.data<uint16_t>(),pb.data<scaled::Row>(),rows,width);check(hipGetLastError());finish();
 variant<2u>(rows,width,ordinary,a,b,pa,pb,out,trace,left,right,expected,transformed);
 variant<4u>(rows,width,ordinary,a,b,pa,pb,out,trace,left,right,expected,transformed);
 variant<8u>(rows,width,ordinary,a,b,pa,pb,out,trace,left,right,expected,transformed);
}
int main()try{hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));if(std::strncmp(p.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");run(257u,16u);run(4096u,272u);run(2048u,2048u);run(1024u,4096u);run(129u,4112u);run(129u,8192u);run(2048u,2048u,true);return 0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
