#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/sm121_cooperative_half_projection.h"
#include "strong_float_replay_cases.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>
namespace coop=qrt_sm121_cooperative_half_projection;
namespace half=qrt_sm121_scaled_half_products;
namespace original=qrt_q1_moe_hawkeye;
namespace cases=qrt_strong_replay_cases;
constexpr unsigned guard=65u,marker=0xa5a5a5a5u;
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
void require(bool v,const char* s){if(!v)throw std::runtime_error(s);}
struct Device {
 void* p=nullptr;size_t bytes;
 explicit Device(size_t b):bytes(b){check(hipMalloc(&p,b));reset();}
 ~Device(){if(p)(void)hipFree(p);}
 void reset(){check(hipMemset(p,0xa5,bytes));}
 template<class T>T* data(){return static_cast<T*>(p)+guard;}
};
template<class T>std::vector<T> read(Device& d,size_t count){std::vector<T> out(count+2u*guard);check(hipMemcpy(out.data(),d.p,out.size()*sizeof(T),hipMemcpyDeviceToHost));return out;}
template<class T>void guards(const std::vector<T>& data){const auto* b=reinterpret_cast<const unsigned char*>(data.data());for(size_t i=0u;i<guard*sizeof(T);++i)require(b[i]==0xa5u && b[(data.size()-guard)*sizeof(T)+i]==0xa5u,"redzone");}
void finish(){hipEvent_t event;check(hipEventCreate(&event));check(hipEventRecord(event));const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){const auto s=hipEventQuery(event);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);if(std::chrono::steady_clock::now()>=deadline)throw std::runtime_error("completion deadline");std::this_thread::yield();}check(hipEventDestroy(event));}
void run(unsigned rows,unsigned tokens,unsigned width){
 const size_t cells=size_t(rows)*tokens,groups=width/16u,wg=size_t(rows)*groups,ig=size_t(tokens)*groups,mw=(cells+31u)/32u,tw=cells*groups*3u;
 std::vector<uint16_t> weights(size_t(rows)*width),inputs(size_t(tokens)*width);
 for(unsigned row=0u;row<rows;++row)for(unsigned k=0u;k<width;++k)weights[size_t(row)*width+k]=cases::input(row+2u,k/16u,k%16u).y;
 for(unsigned token=0u;token<tokens;++token)for(unsigned k=0u;k<width;++k)inputs[size_t(token)*width+k]=cases::input(token+1u,k/16u,k%16u).x;
 Device w((weights.size()+2u*guard)*2u),in((inputs.size()+2u*guard)*2u),pw((wg+2u*guard)*sizeof(half::Row)),pi((ig+2u*guard)*sizeof(half::Row));
 check(hipMemcpy(w.data<uint16_t>(),weights.data(),weights.size()*2u,hipMemcpyHostToDevice));check(hipMemcpy(in.data<uint16_t>(),inputs.data(),inputs.size()*2u,hipMemcpyHostToDevice));
 hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,w.data<uint16_t>(),pw.data<half::Row>(),rows,width);check(hipGetLastError());
 hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,in.data<uint16_t>(),pi.data<half::Row>(),tokens,width);check(hipGetLastError());finish();
 Device mask((mw+2u*guard)*4u),out((cells+2u*guard)*4u),trace((tw+2u*guard)*4u);
 for(unsigned pattern=0u;pattern<4u;++pattern){
  std::vector<unsigned> selected,expected_mask(mw,0u),expected_output(cells,marker),expected_trace(tw,marker);
  for(unsigned cell=0u;cell<cells;++cell)if(pattern==3u || (pattern==2u && cell%4u==1u) || (pattern==1u && cell%37u==0u)){
   selected.push_back(cell);expected_mask[cell/32u]|=1u<<(cell&31u);original::Value carry{0u,-133,false};
   for(unsigned group=0u;group<groups;++group){original::Value values[17];values[0]=carry;
    for(unsigned i=0u;i<16u;++i)values[i+1u]=original::multiply_bf16(inputs[size_t(cell/rows)*width+group*16u+i],weights[size_t(cell%rows)*width+group*16u+i],-133);
    carry=original::group_sum<26,-133>(values,17u);const size_t idx=(size_t(cell)*groups+group)*3u;
    expected_trace[idx]=carry.significand;expected_trace[idx+1u]=uint32_t(int32_t(carry.exponent));expected_trace[idx+2u]=unsigned(carry.negative);
   }
   expected_output[cell]=cases::output_bits(carry);
  }
  Device index((selected.size()+2u*guard)*4u);
  if(!selected.empty())check(hipMemcpy(index.data<unsigned>(),selected.data(),selected.size()*4u,hipMemcpyHostToDevice));
  mask.reset();check(qrt_sm121_tiled_projection::mark(index.data<unsigned>(),unsigned(selected.size()),unsigned(cells),mask.data<unsigned>(),mw,nullptr));finish();
  const auto bits=read<unsigned>(mask,mw);guards(bits);require(!std::memcmp(bits.data()+guard,expected_mask.data(),mw*4u),"original candidate bitmap");
  for(unsigned variant=0u;variant<3u;++variant){
   out.reset();trace.reset();
   check(coop::launch(pw.data<half::Row>(),wg,pi.data<half::Row>(),ig,mask.data<unsigned>(),mw,out.data<float>(),cells,rows,tokens,width,variant,nullptr,trace.data<uint32_t>(),tw));finish();
   const auto actual=read<unsigned>(out,cells),states=read<unsigned>(trace,tw);guards(actual);guards(states);
   require(!std::memcmp(actual.data()+guard,expected_output.data(),cells*4u),"independent original output bits");
   require(!std::memcmp(states.data()+guard,expected_trace.data(),tw*4u),"independent original K16 states or inactive writes");
   out.reset();check(coop::launch(pw.data<half::Row>(),wg,pi.data<half::Row>(),ig,mask.data<unsigned>(),mw,out.data<float>(),cells,rows,tokens,width,variant,nullptr));finish();
   require(read<unsigned>(out,cells)==actual,"production and trace output differ");require(read<unsigned>(trace,tw)==states,"production changed trace");
   require(read<unsigned>(mask,mw)==bits,"mask modified");const auto indices=read<unsigned>(index,selected.size());guards(indices);require(selected.empty() || !std::memcmp(indices.data()+guard,selected.data(),selected.size()*4u),"indices modified");
   std::printf("{\"kind\":\"cooperative_half_projection\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"variant\":%u,\"pattern\":%u,\"candidates\":%zu,\"k16_states\":%zu,\"raw_bit_mismatches\":0,\"production_trace_parity\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",rows,tokens,width,variant,pattern,selected.size(),selected.size()*groups);
  }
 }
 for(unsigned side=0u;side<2u;++side){const auto& input=side?inputs:weights;const auto raw=read<uint16_t>(side?in:w,input.size());const auto packed=read<half::Row>(side?pi:pw,side?ig:wg);guards(raw);guards(packed);require(!std::memcmp(raw.data()+guard,input.data(),input.size()*2u),"operand changed");
  for(size_t g=0u;g<input.size()/16u;++g){const auto expected=half::prepare(input.data()+g*16u);require(!std::memcmp(&expected,&packed[guard+g],sizeof(expected)),"packed operand changed");for(unsigned i=0u;i<16u;++i)require(half::original(packed[guard+g],i)==input[g*16u+i],"lossless encoding");}}
}
int main(){try{const unsigned shapes[][3]={{1u,1u,16u},{17u,19u,272u},{33u,17u,2048u},{35u,33u,4096u},{17u,17u,8192u},{65u,3u,4112u}};for(const auto& s:shapes)run(s[0],s[1],s[2]);return 0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}}
