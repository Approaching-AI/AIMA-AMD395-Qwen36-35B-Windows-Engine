#include <hip/hip_runtime.h>
#include "../../native/providers/gdn/fused_state_output.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
namespace scalar=qrt_fla_blackwell_scalar;
namespace fused=qrt_fla_fused_state_output;
namespace original=qrt_q1_moe_hawkeye;
constexpr size_t guard=65u,state_cells=524288u;
void require(bool b,const char* text){if(!b)throw std::runtime_error(text);}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4u);return u;}
float value(uint32_t u){float f;std::memcpy(&f,&u,4u);return f;}
float from(uint16_t u){return value(uint32_t(u)<<16u);}
uint16_t rounded(float f){const uint32_t u=bits(f);return uint16_t(((u&0x7fffffffu)>0x7f800000u?(u|0x00400000u):u+0x7fffu+((u>>16u)&1u))>>16u);}
uint32_t random_word(uint32_t u){u^=u<<13u;u^=u>>17u;return u^(u<<5u);}
struct Device{
 void* p=nullptr;size_t bytes;
 explicit Device(size_t n):bytes(n){check(hipMalloc(&p,n));reset();}
 ~Device(){if(p)(void)hipFree(p);}
 void reset(){check(hipMemset(p,0xa5,bytes));}
 template<class T>T* data(){return static_cast<T*>(p)+guard;}
 template<class T>void upload(const std::vector<T>& input){check(hipMemcpy(data<T>(),input.data(),input.size()*sizeof(T),hipMemcpyHostToDevice));}
};
template<class T>std::vector<T> read(Device& d,size_t count){std::vector<T> out(count+2u*guard);check(hipMemcpy(out.data(),d.p,out.size()*sizeof(T),hipMemcpyDeviceToHost));return out;}
template<class T>void guards(const std::vector<T>& x){const auto* p=reinterpret_cast<const unsigned char*>(x.data());for(size_t i=0u;i<guard*sizeof(T);++i)require(p[i]==0xa5u&&p[(x.size()-guard)*sizeof(T)+i]==0xa5u,"memory guard");}
template<class T>void unchanged(Device& d,const std::vector<T>& expected){const auto actual=read<T>(d,expected.size());guards(actual);require(!std::memcmp(actual.data()+guard,expected.data(),expected.size()*sizeof(T)),"immutable source changed");}
void finish(){hipEvent_t e;check(hipEventCreate(&e));check(hipEventRecord(e));const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){auto s=hipEventQuery(e);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);require(std::chrono::steady_clock::now()<end,"GPU completion deadline");std::this_thread::yield();}check(hipEventDestroy(e));}
float dot(const uint16_t* a,const uint16_t* b,unsigned width){original::Value carry{0u,-133,false};for(unsigned base=0u;base<width;base+=16u){original::Value terms[17];terms[0]=carry;for(unsigned i=0u;i<16u;++i)terms[i+1u]=original::multiply_bf16(a[base+i],b[base+i],-133);carry=original::group_sum<26,-133>(terms,17u);}return original::value_to_float(qrt_sm121_group16::finish_accumulator(carry));}
size_t cpu_column(unsigned count,unsigned head,unsigned column,const std::vector<uint16_t>& q,const std::vector<uint16_t>& k,const std::vector<uint16_t>& u,const std::vector<uint16_t>& w,const std::vector<float>& g,const std::vector<uint16_t>& scores,const std::vector<float>& seed,const std::vector<unsigned char>& table,const std::vector<float>& output,const std::vector<float>& final,const std::vector<uint16_t>& h,const std::vector<uint16_t>& vn){
 float current[128];for(unsigned i=0u;i<128u;++i)current[i]=seed[(head*128u+column)*128u+i];
 size_t dots=0u;auto exp=[&](float x){return qrt_sm121_exp2::evaluate(table.data(),x*1.4426950408889634074f);};
 for(unsigned offset=0u;offset<count;offset+=64u){
  const unsigned valid=std::min(64u,count-offset);uint16_t old[128],updated[64]{},residual[64]{};
  for(unsigned i=0u;i<128u;++i){old[i]=rounded(current[i]);require(h[guard+size_t(offset/64u)*state_cells+(head*128u+column)*128u+i]==old[i],"independent CPU checkpoint");}
  for(unsigned row=0u;row<valid;++row){const size_t at=(size_t(offset+row)*32u+head)*128u+column;
   const float sum=dot(w.data()+(size_t(offset+row)*32u+head)*128u,old,128u);++dots;
   const float difference=from(u[at])-sum;updated[row]=rounded(difference);
   residual[row]=rounded(difference*exp(g[size_t(offset+valid-1u)*32u+head]-g[size_t(offset+row)*32u+head]));
   require(vn[guard+at]==updated[row],"independent CPU residual");
  }
  for(unsigned feature=0u;feature<128u;++feature){uint16_t keys[64]{};for(unsigned row=0u;row<valid;++row)keys[row]=k[(size_t(offset+row)*16u+head/2u)*128u+feature];
   const float sum=dot(keys,residual,64u);++dots;current[feature]=std::fma(current[feature],exp(g[size_t(offset+valid-1u)*32u+head]),sum);
  }
  for(unsigned row=0u;row<valid;++row){const float prior=dot(q.data()+(size_t(offset+row)*16u+head/2u)*128u,old,128u)*exp(g[size_t(offset+row)*32u+head]);
   const float local=dot(scores.data()+(size_t(offset+row)*32u+head)*64u,updated,64u);dots+=2u;constexpr float scale=0.08838834764831845f;
   const float expected=from(rounded(std::fma(local,scale,prior*scale)));
   require(bits(output[guard+(size_t(offset+row)*32u+head)*128u+column])==bits(expected),"independent CPU output");
  }
 }
 for(unsigned i=0u;i<128u;++i)require(bits(final[guard+(head*128u+column)*128u+i])==bits(current[i]),"independent CPU final state");
 return dots;
}
template<unsigned Columns,bool Capture>
void candidate(unsigned count,Device& q,Device& k,Device& u,Device& w,Device& g,Device& scores,Device& output,Device& h,Device& vn,Device& state,Device& table){
 hipLaunchKernelGGL((fused::kernel<Columns,Capture>),dim3(128u/Columns,32u),dim3(256u),0u,nullptr,q.data<uint16_t>(),k.data<uint16_t>(),u.data<uint16_t>(),w.data<uint16_t>(),g.data<float>(),scores.data<uint16_t>(),output.data<float>(),h.data<uint16_t>(),vn.data<uint16_t>(),state.data<float>(),count,reinterpret_cast<unsigned char*>(table.data<uint32_t>()));check(hipGetLastError());finish();
}
void run(unsigned count,unsigned mode,Device& table,const std::vector<unsigned char>& host_table){
 const size_t small=size_t(count)*2048u,large=size_t(count)*4096u,gate=size_t(count)*32u,checkpoints=size_t((count+63u)/64u)*state_cells;
 std::vector<uint16_t> q(small),k(small),u(large),w(large),scores(small);std::vector<float> g(gate),seed(state_cells);
 auto fill=[&](std::vector<uint16_t>& x,unsigned salt,unsigned exponent){for(size_t i=0u;i<x.size();++i){const unsigned r=random_word(unsigned(i)^salt);x[i]=mode?uint16_t((r&0x807fu)|((exponent+r%4u)<<7u)):uint16_t(r&0x8000u);if(mode==2u&&i%29u==0u)x[i]=uint16_t(r&0x807fu);}};
 fill(q,395u,115u);fill(k,8192u,115u);fill(u,35u,120u);fill(w,3u,114u);fill(scores,121u,112u);
 for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<32u;++head){g[size_t(row)*32u+head]=-float((row%64u+1u)*(head%7u+1u))*.0078125f;for(unsigned future=row%64u+1u;future<64u;++future)scores[(size_t(row)*32u+head)*64u+future]=0u;}
 for(size_t i=0u;i<state_cells;++i){const unsigned r=random_word(unsigned(i)^0x395u);seed[i]=mode==0u?from(uint16_t(r&0x8000u)):from(uint16_t((r&0x807fu)|((118u+r%4u)<<7u)));}
 Device dq((small+2u*guard)*2u),dk((small+2u*guard)*2u),du((large+2u*guard)*2u),dw((large+2u*guard)*2u),ds((small+2u*guard)*2u),dg((gate+2u*guard)*4u);
 dq.upload(q);dk.upload(k);du.upload(u);dw.upload(w);ds.upload(scores);dg.upload(g);
 Device output((large+2u*guard)*4u),state((state_cells+2u*guard)*4u),h((checkpoints+2u*guard)*2u),vn((large+2u*guard)*2u);state.upload(seed);
 hipLaunchKernelGGL((scalar::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,dk.data<uint16_t>(),du.data<uint16_t>(),dw.data<uint16_t>(),dg.data<float>(),h.data<uint16_t>(),vn.data<uint16_t>(),state.data<float>(),count,reinterpret_cast<unsigned char*>(table.data<uint32_t>()));check(hipGetLastError());
 hipLaunchKernelGGL(scalar::output_kernel,dim3(16u,32u,(count+63u)/64u),dim3(256u),0u,nullptr,dq.data<uint16_t>(),vn.data<uint16_t>(),h.data<uint16_t>(),dg.data<float>(),ds.data<uint16_t>(),output.data<float>(),count,reinterpret_cast<unsigned char*>(table.data<uint32_t>()));check(hipGetLastError());finish();
 const auto expected=read<float>(output,large),final=read<float>(state,state_cells);const auto old_h=read<uint16_t>(h,checkpoints),old_vn=read<uint16_t>(vn,large);guards(expected);guards(final);guards(old_h);guards(old_vn);
 size_t cpu_dots=0u;
 if(mode<2u)for(const auto& choice:std::vector<std::pair<unsigned,unsigned>>{{0u,0u},{3u,7u},{31u,127u}})cpu_dots+=cpu_column(count,choice.first,choice.second,q,k,u,w,g,scores,seed,host_table,expected,final,old_h,old_vn);
 for(unsigned variant=0u;variant<4u;++variant){
  output.reset();state.reset();state.upload(seed);h.reset();vn.reset();
  const auto untouched_h=read<uint16_t>(h,checkpoints),untouched_vn=read<uint16_t>(vn,large);
  if(variant==0u)candidate<4u,true>(count,dq,dk,du,dw,dg,ds,output,h,vn,state,table);
  if(variant==1u)candidate<8u,true>(count,dq,dk,du,dw,dg,ds,output,h,vn,state,table);
  if(variant==2u)candidate<4u,false>(count,dq,dk,du,dw,dg,ds,output,h,vn,state,table);
  if(variant==3u)candidate<8u,false>(count,dq,dk,du,dw,dg,ds,output,h,vn,state,table);
  const auto actual=read<float>(output,large),next=read<float>(state,state_cells);guards(actual);guards(next);
  require(!std::memcmp(actual.data(),expected.data(),actual.size()*4u),"fused output differs");require(!std::memcmp(next.data(),final.data(),next.size()*4u),"fused state differs");
  require(read<uint16_t>(h,checkpoints)==(variant<2u?old_h:untouched_h),"checkpoint capture or inactive surface changed");require(read<uint16_t>(vn,large)==(variant<2u?old_vn:untouched_vn),"residual capture or inactive surface changed");
  unchanged(dq,q);unchanged(dk,k);unchanged(du,u);unchanged(dw,w);unchanged(dg,g);unchanged(ds,scores);
  std::printf("{\"kind\":\"fused_state_output_safety\",\"tokens\":%u,\"mode\":%u,\"columns\":%u,\"capture_intermediates\":%s,\"output_cells\":%zu,\"state_cells\":%zu,\"checkpoint_cells\":%zu,\"cpu_dots\":%zu,\"raw_bit_mismatches\":0,\"intermediate_capture_and_untouched_buffers_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",count,mode,variant%2u?8u:4u,variant<2u?"true":"false",large,state_cells,checkpoints,cpu_dots);std::fflush(stdout);
 }
}
int main(int argc,char** argv)try{
 require(argc==2,"requires verified SM121 exponential table path");hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
 std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(file&&file.tellg()==std::streamoff(qrt_sm121_exp2::table_bytes),"table span");std::vector<unsigned char> table(qrt_sm121_exp2::table_bytes);file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());require(bool(file)&&qrt_sm121_exp2::valid_layout(table.data(),table.size()),"table layout");
 std::vector<uint32_t> table_words(table.size()/4u);require(table.size()%4u==0u,"table alignment");std::memcpy(table_words.data(),table.data(),table.size());
 Device dt((table_words.size()+2u*guard)*4u);dt.upload(table_words);
 for(unsigned count:{1u,63u,64u,65u,129u,1024u})for(unsigned mode=0u;mode<3u;++mode)run(count,mode,dt,table);
 unchanged(dt,table_words);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
