#include <hip/hip_runtime.h>
#include "../../native/providers/gdn/prepared_gate_matrices.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>
namespace scalar=qrt_fla_blackwell_scalar;
namespace compact=qrt_fla_prepared_gate;
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
 ~Device(){if(p && hipFree(p)!=hipSuccess)std::abort();}
 void reset(){check(hipMemset(p,0xa5,bytes));}
 template<class T>T* data(){return static_cast<T*>(p)+guard;}
 template<class T>void upload(const std::vector<T>& input){check(hipMemcpy(data<T>(),input.data(),input.size()*sizeof(T),hipMemcpyHostToDevice));}
};
template<class T>std::vector<T> read(Device& d,size_t count){std::vector<T> out(count+2u*guard);check(hipMemcpy(out.data(),d.p,out.size()*sizeof(T),hipMemcpyDeviceToHost));return out;}
template<class T>void guards(const std::vector<T>& x){const auto* p=reinterpret_cast<const unsigned char*>(x.data());for(size_t i=0u;i<guard*sizeof(T);++i)require(p[i]==0xa5u&&p[(x.size()-guard)*sizeof(T)+i]==0xa5u,"memory guard");}
template<class T>void unchanged(Device& d,const std::vector<T>& expected){const auto actual=read<T>(d,expected.size());guards(actual);require(!std::memcmp(actual.data()+guard,expected.data(),expected.size()*sizeof(T)),"immutable source changed");}
void finish(){hipEvent_t e;check(hipEventCreate(&e));check(hipEventRecord(e));const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){auto s=hipEventQuery(e);if(s==hipSuccess)break;if(s!=hipErrorNotReady)check(s);require(std::chrono::steady_clock::now()<end,"GPU completion deadline");std::this_thread::yield();}check(hipEventDestroy(e));}
struct PhaseEvents{
 hipEvent_t events[5]{};bool enabled;
 explicit PhaseEvents(bool active):enabled(active){if(enabled)for(auto& event:events)check(hipEventCreate(&event));}
 ~PhaseEvents(){for(auto event:events)if(event && hipEventDestroy(event)!=hipSuccess)std::abort();}
 void mark(unsigned index){if(enabled)check(hipEventRecord(events[index],nullptr));}
 float elapsed(unsigned phase){float ms=0.0f;if(enabled)check(hipEventElapsedTime(&ms,events[phase],events[phase+1u]));require(std::isfinite(ms)&&ms>=0.0f,"invalid phase duration");return ms;}
};
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

void launch(unsigned variant,unsigned count,Device& q,Device& k,Device& v,Device& beta,Device& inverse,Device& g,Device& scores,Device& u,Device& w,Device& output,Device& h,Device& vn,Device& state,Device& table,Device& gate_exp,Device& gate_remaining,bool alias,PhaseEvents& phases){
 const unsigned chunks=(count+63u)/64u;const auto* exp=reinterpret_cast<unsigned char*>(table.data<uint32_t>());
 auto* actual_u=alias?v.data<uint16_t>():u.data<uint16_t>();
 phases.mark(0u);
 const compact::GateView view{gate_exp.data<float>(),gate_remaining.data<float>()};
 if(variant)check(compact::prepare(g.data<float>(),gate_exp.data<float>(),gate_remaining.data<float>(),count,size_t(count)*32u,exp,nullptr));
 phases.mark(1u);
 if(!variant){
  hipLaunchKernelGGL(scalar::wu_kernel,dim3(16u,32u,chunks),dim3(256u),0u,nullptr,k.data<uint16_t>(),v.data<uint16_t>(),beta.data<uint16_t>(),inverse.data<uint16_t>(),g.data<float>(),w.data<uint16_t>(),actual_u,count,exp);check(hipGetLastError());
  phases.mark(2u);
  hipLaunchKernelGGL((scalar::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,k.data<uint16_t>(),actual_u,w.data<uint16_t>(),g.data<float>(),h.data<uint16_t>(),vn.data<uint16_t>(),state.data<float>(),count,exp);check(hipGetLastError());
  phases.mark(3u);
  hipLaunchKernelGGL(scalar::output_kernel,dim3(16u,32u,chunks),dim3(256u),0u,nullptr,q.data<uint16_t>(),vn.data<uint16_t>(),h.data<uint16_t>(),g.data<float>(),scores.data<uint16_t>(),output.data<float>(),count,exp);check(hipGetLastError());
  phases.mark(4u);
 }else{
  hipLaunchKernelGGL(compact::wu_kernel,dim3(16u,32u,chunks),dim3(256u),0u,nullptr,k.data<uint16_t>(),v.data<uint16_t>(),beta.data<uint16_t>(),inverse.data<uint16_t>(),g.data<float>(),w.data<uint16_t>(),actual_u,count,view);check(hipGetLastError());
  phases.mark(2u);
  hipLaunchKernelGGL((compact::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,k.data<uint16_t>(),actual_u,w.data<uint16_t>(),g.data<float>(),h.data<uint16_t>(),vn.data<uint16_t>(),state.data<float>(),count,view);check(hipGetLastError());
  phases.mark(3u);
  hipLaunchKernelGGL(compact::output_kernel,dim3(16u,32u,chunks),dim3(256u),0u,nullptr,q.data<uint16_t>(),vn.data<uint16_t>(),h.data<uint16_t>(),g.data<float>(),scores.data<uint16_t>(),output.data<float>(),count,view);check(hipGetLastError());
  phases.mark(4u);

 }
}
size_t cpu_wu(unsigned count,const std::vector<uint16_t>& k,const std::vector<uint16_t>& v,const std::vector<uint16_t>& beta,const std::vector<uint16_t>& inverse,const std::vector<float>& g,const std::vector<unsigned char>& table,const std::vector<uint16_t>& w,const std::vector<uint16_t>& u){
 size_t dots=0u;auto exp=[&](float x){return qrt_sm121_exp2::evaluate(table.data(),x*1.4426950408889634074f);};
 for(const auto& choice:std::vector<std::pair<unsigned,unsigned>>{{0u,0u},{3u,7u},{31u,127u}}){
  const unsigned head=choice.first,column=choice.second;
  for(unsigned offset=0u;offset<count;offset+=64u){
   const unsigned valid=std::min(64u,count-offset);uint16_t ks[64]{},vs[64]{};
   for(unsigned row=0u;row<valid;++row){const size_t token=offset+row;const float scale=from(beta[token*32u+head]);const uint16_t scaled=rounded(from(k[(token*16u+head/2u)*128u+column])*scale);ks[row]=rounded(from(scaled)*exp(g[token*32u+head]));vs[row]=rounded(from(v[(token*32u+head)*128u+column])*scale);}
   for(unsigned row=0u;row<valid;++row){uint16_t a[64]{};for(unsigned i=0u;i<valid;++i)a[i]=inverse[(size_t(offset+row)*32u+head)*64u+i];const size_t at=guard+(size_t(offset+row)*32u+head)*128u+column;require(w[at]==rounded(dot(a,ks,64u)) && u[at]==rounded(dot(a,vs,64u)),"independent CPU WU");dots+=2u;}
  }
 }
 return dots;
}
void run(unsigned count,unsigned mode,unsigned measured,Device& table,const std::vector<unsigned char>& host_table){
 const size_t small=size_t(count)*2048u,large=size_t(count)*4096u,gate=size_t(count)*32u,checkpoints=size_t((count+63u)/64u)*state_cells;
 std::vector<uint16_t> q(small),k(small),v(large),beta(gate),inverse(small),scores(small);std::vector<float> g(gate),seed(state_cells);
 auto fill=[&](std::vector<uint16_t>& x,unsigned salt,unsigned exponent){for(size_t i=0u;i<x.size();++i){const unsigned r=random_word(unsigned(i)^salt);x[i]=mode?uint16_t((r&0x807fu)|((exponent+r%4u)<<7u)):uint16_t(r&0x8000u);if(mode==2u && i%29u==0u)x[i]=uint16_t(r&0x807fu);}};
 fill(q,395u,115u);fill(k,8192u,115u);fill(v,35u,120u);fill(inverse,3u,116u);fill(scores,121u,112u);
 for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<32u;++head){
  beta[size_t(row)*32u+head]=uint16_t(0x3e80u|((row+head)&127u));g[size_t(row)*32u+head]=-float((row%64u+1u)*(head%7u+1u))*.0078125f;
  for(unsigned future=row%64u+1u;future<64u;++future){scores[(size_t(row)*32u+head)*64u+future]=0u;inverse[(size_t(row)*32u+head)*64u+future]=0u;}
  inverse[(size_t(row)*32u+head)*64u+row%64u]=0x3f80u;
 }
 if(mode==3u){const uint32_t edge[]={0u,0x80000000u,0xb0000000u,0xc3000000u,0xff800000u,0x7fc12345u,0x33800000u};for(size_t i=0;i<g.size();++i)g[i]=value(edge[i%7u]);}
 for(size_t i=0u;i<state_cells;++i){const unsigned r=random_word(unsigned(i)^0x395u);seed[i]=mode==0u?from(uint16_t(r&0x8000u)):from(uint16_t((r&0x807fu)|((118u+r%4u)<<7u)));}
 Device dq((small+2u*guard)*2u),dk((small+2u*guard)*2u),dv((large+2u*guard)*2u),db((gate+2u*guard)*2u),dinv((small+2u*guard)*2u),ds((small+2u*guard)*2u),dg((gate+2u*guard)*4u);
 dq.upload(q);dk.upload(k);db.upload(beta);dinv.upload(inverse);ds.upload(scores);dg.upload(g);
 Device du((large+2u*guard)*2u),dw((large+2u*guard)*2u),output((large+2u*guard)*4u),state((state_cells+2u*guard)*4u),h((checkpoints+2u*guard)*2u),vn((large+2u*guard)*2u);
 Device gate_exp((gate+2u*guard)*4u),gate_remaining((gate+2u*guard)*4u);
 std::vector<float> expected_exp(gate),expected_remaining(gate);
 for(unsigned token=0u;token<count;++token)for(unsigned head=0u;head<32u;++head){
  const size_t cell=size_t(token)*32u+head;const unsigned last=std::min((token/64u+1u)*64u,count)-1u;
  expected_exp[cell]=qrt_sm121_exp2::evaluate(host_table.data(),g[cell]*1.4426950408889634074f);
  expected_remaining[cell]=qrt_sm121_exp2::evaluate(host_table.data(),(g[size_t(last)*32u+head]-g[cell])*1.4426950408889634074f);
 }
 if(count==1u&&!mode){
  const auto* tab=reinterpret_cast<const unsigned char*>(table.data<uint32_t>());
  auto* ga=dg.data<float>();auto* ea=gate_exp.data<float>();auto* ra=gate_remaining.data<float>();
  require(compact::prepare(ga,ea,ra,0u,gate,tab,nullptr)==hipErrorInvalidValue,"zero gate extent");
  require(compact::prepare(ga,ea,ra,8193u,gate,tab,nullptr)==hipErrorInvalidValue,"large gate extent");
  require(compact::prepare(ga,ea,ra,count,gate-1u,tab,nullptr)==hipErrorInvalidValue,"short gate storage");
  require(compact::prepare(ga,ga,ra,count,gate,tab,nullptr)==hipErrorInvalidValue,"overlapping source gate storage");
  require(compact::prepare(ga,ea,ea+1u,count,gate,tab,nullptr)==hipErrorInvalidValue,"overlapping output gate storage");
  require(compact::prepare(ga,ea,ra,count,gate,nullptr,nullptr)==hipErrorInvalidValue,"missing gate table");
  require(compact::prepare(ga,reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(ea)+1u),ra,count,gate,tab,nullptr)==hipErrorInvalidValue,"misaligned gate storage");
 }
 std::vector<float> expected,final;std::vector<uint16_t> old_w,old_u,old_h,old_vn;size_t cpu_dots=0u;double samples[2][2][3]{};
 double phase_samples[2][2][4][3]{};PhaseEvents phases(measured!=0u);
 const unsigned attempts=measured+1u;
 for(unsigned attempt=0u;attempt<attempts;++attempt)for(unsigned position=0u;position<4u;++position){
  const unsigned choice=(attempt+position)%4u,variant=choice%2u;const bool alias=choice>=2u;
  dv.reset();dv.upload(v);du.reset();dw.reset();output.reset();state.reset();state.upload(seed);h.reset();vn.reset();gate_exp.reset();gate_remaining.reset();finish();
  const auto begin=std::chrono::steady_clock::now();launch(variant,count,dq,dk,dv,db,dinv,dg,ds,du,dw,output,h,vn,state,table,gate_exp,gate_remaining,alias,phases);finish();
  if(attempt){
   samples[unsigned(alias)][variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
   // Resolve all phase events after the existing final completion. No barrier is
   // inserted between kernels; event recording overhead remains in the host clock.
   for(unsigned phase=0u;phase<4u;++phase)phase_samples[unsigned(alias)][variant][phase][attempt-1u]=phases.elapsed(phase);
  }
  if(variant){unchanged(gate_exp,expected_exp);unchanged(gate_remaining,expected_remaining);}
  else for(auto* buffer:{&gate_exp,&gate_remaining}){
   const auto unused=read<unsigned>(*buffer,gate);require(std::all_of(unused.begin(),unused.end(),[](unsigned x){return x==0xa5a5a5a5u;}),"inactive gate buffer changed");
  }
  const auto actual=read<float>(output,large),next=read<float>(state,state_cells);
  const auto actual_w=read<uint16_t>(dw,large),actual_u=read<uint16_t>(alias?dv:du,large),actual_h=read<uint16_t>(h,checkpoints),actual_vn=read<uint16_t>(vn,large);
  guards(actual);guards(next);guards(actual_w);guards(actual_u);guards(actual_h);guards(actual_vn);
  if(!attempt && !choice){
   expected=actual;final=next;old_w=actual_w;old_u=actual_u;old_h=actual_h;old_vn=actual_vn;
   if(mode<2u){
    cpu_dots+=cpu_wu(count,k,v,beta,inverse,g,host_table,old_w,old_u);
    const std::vector<uint16_t> hw(old_w.begin()+guard,old_w.end()-guard),hu(old_u.begin()+guard,old_u.end()-guard);
    for(const auto& column:std::vector<std::pair<unsigned,unsigned>>{{0u,0u},{3u,7u},{31u,127u}})cpu_dots+=cpu_column(count,column.first,column.second,q,k,hu,hw,g,scores,seed,host_table,expected,final,old_h,old_vn);
   }
  }
  require(!expected.empty() && !std::memcmp(actual.data(),expected.data(),actual.size()*4u),"prepared-gate GDN output differs");require(!std::memcmp(next.data(),final.data(),next.size()*4u),"prepared-gate GDN state differs");
  require(actual_w==old_w && actual_u==old_u && actual_h==old_h && actual_vn==old_vn,"prepared-gate GDN intermediate or alias differs");
  unchanged(dq,q);unchanged(dk,k);unchanged(db,beta);unchanged(dinv,inverse);unchanged(dg,g);unchanged(ds,scores);
  if(!alias)unchanged(dv,v);else{const auto unused=read<uint16_t>(du,large);const auto* bytes=reinterpret_cast<const unsigned char*>(unused.data());require(std::all_of(bytes,bytes+unused.size()*2u,[](unsigned char x){return x==0xa5u;}),"inactive U buffer changed");}
 }
 for(unsigned alias=0u;alias<2u;++alias)for(unsigned variant=0u;variant<2u;++variant){
  double sorted[3]={samples[alias][variant][0],samples[alias][variant][1],samples[alias][variant][2]};std::sort(sorted,sorted+3u);
  std::printf("{\"kind\":\"prepared_gate_gdn_component\",\"tokens\":%u,\"mode\":%u,\"variant\":%u,\"lanes\":%u,\"u_aliases_v\":%s,\"output_cells\":%zu,\"state_cells\":%zu,\"checkpoint_cells\":%zu,\"wu_and_residual_cells\":%zu,\"independent_cpu_dots\":%zu,\"warmups\":1,\"measured_attempts\":%u,\"complete_preparation_wu_state_output_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"phase_events_enabled\":%s,\"phase_event_ms\":{",count,mode,variant,1u,alias?"true":"false",large,state_cells,checkpoints,large*3u,cpu_dots,measured,sorted[1],samples[alias][variant][0],samples[alias][variant][1],samples[alias][variant][2],measured?"true":"false");
  const char* names[4]={"gate_preparation","wu","state","output"};
  for(unsigned phase=0u;phase<4u;++phase){
   const auto* values=phase_samples[alias][variant][phase];double phase_sorted[3]={values[0],values[1],values[2]};std::sort(phase_sorted,phase_sorted+3u);
   std::printf("%s\"%s\":{\"median\":%.6f,\"samples\":[%.6f,%.6f,%.6f]}",phase?",":"",names[phase],phase_sorted[1],values[0],values[1],values[2]);
  }
  std::printf("},\"phase_events_same_stream\":true,\"intermediate_host_synchronization\":false,\"all_attempts_verified\":true,\"raw_bit_mismatches\":0,\"intermediate_and_alias_ownership_checked\":true,\"cpu_gate_values_checked\":true,\"redzones_pass\":true,\"immutable_nonaliased_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n");
 }
 std::fflush(stdout);
}
int main(int argc,char** argv)try{
 require(argc==3,"requires verified SM121 exponential table and safety or throughput");hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
 const bool timing=!std::strcmp(argv[2],"throughput");require(timing || !std::strcmp(argv[2],"safety"),"unknown action");
 std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(file&&file.tellg()==std::streamoff(qrt_sm121_exp2::table_bytes),"table span");std::vector<unsigned char> table(qrt_sm121_exp2::table_bytes);file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());require(bool(file)&&qrt_sm121_exp2::valid_layout(table.data(),table.size()),"table layout");
 std::vector<uint32_t> table_words(table.size()/4u);require(table.size()%4u==0u,"table alignment");std::memcpy(table_words.data(),table.data(),table.size());Device dt((table_words.size()+2u*guard)*4u);dt.upload(table_words);
 if(timing)run(8192u,1u,3u,dt,table);else for(unsigned count:{1u,63u,64u,65u,129u,1024u})for(unsigned mode=0u;mode<4u;++mode)run(count,mode,0u,dt,table);
 unchanged(dt,table_words);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
