// Reuse the original kernels, independent integer CPU oracle and guards.
#define main qrt_separate_state_prior_gdn_main
#include "tiled_scalar_gdn_selftest.cpp"
#undef main
#include "../../native/providers/gdn/blackwell_lifetime_matrices.h"
#include "../../native/providers/gdn/paired_score_matrices.h"
#include "../../native/providers/gdn/separate_state_replay.h"
#include "../../native/providers/moe_accumulator/sm121_subgroup.h"
#include <array>
#include <string>
#ifdef QRT_GDN_PROVIDER_REPLAY
#include "../../native/providers/gdn/blackwell_cooperative.h"
#endif
#ifdef QRT_GDN_HYBRID_RETRY
#include "../../native/providers/gdn/hybrid_state_replay.h"
constexpr unsigned state_variants=3u;
#ifdef QRT_GDN_PROVIDER_REPLAY
constexpr const char* state_component="state_replay_provider_component";
#else
constexpr const char* state_component="hybrid_state_gdn_component";
#endif
#else
constexpr unsigned state_variants=2u;
constexpr const char* state_component="separate_state_gdn_component";
#endif
namespace lifetime=qrt_fla_lifetime;
__global__ void capture_scores(const uint16_t*,const uint16_t*,const float*,uint16_t*,unsigned,const unsigned char*);
void paired_launch(unsigned variant,unsigned count,Device& q,Device& k,Device& v,Device& beta,
 Device& inverse,Device& g,Device& scores,Device& u,Device& w,Device& output,Device& h,Device& vn,
 Device& state,Device& table,Device& receipts,bool alias){
 require(variant<state_variants,"invalid separate-state GDN variant");
 const auto* exp=reinterpret_cast<unsigned char*>(table.data<uint32_t>());
 auto* actual_u=alias?v.data<uint16_t>():u.data<uint16_t>();
 for(unsigned offset=0u;offset<count;offset+=1024u){
  const unsigned n=std::min(1024u,count-offset),chunks=(n+63u)/64u;
#define QRT_SEPARATE_SCORE_ARGS q.data<uint16_t>()+size_t(offset)*2048u,k.data<uint16_t>()+size_t(offset)*2048u,g.data<float>()+size_t(offset)*32u,scores.data<uint16_t>()+size_t(offset)*2048u,n,exp
  hipLaunchKernelGGL(qrt_fla_paired_score::kernel,dim3(2u,16u,chunks*8u),dim3(256u),0u,nullptr,QRT_SEPARATE_SCORE_ARGS);
  check(hipGetLastError());
#undef QRT_SEPARATE_SCORE_ARGS
#define QRT_SEPARATE_WU_ARGS k.data<uint16_t>()+size_t(offset)*2048u,v.data<uint16_t>()+size_t(offset)*4096u,beta.data<uint16_t>()+size_t(offset)*32u,inverse.data<uint16_t>()+size_t(offset)*2048u,g.data<float>()+size_t(offset)*32u,w.data<uint16_t>()+size_t(offset)*4096u,actual_u+size_t(offset)*4096u,n,exp
  hipLaunchKernelGGL(scalar::wu_kernel,dim3(16u,32u,chunks),dim3(256u),0u,nullptr,QRT_SEPARATE_WU_ARGS);
  check(hipGetLastError());
#undef QRT_SEPARATE_WU_ARGS
#define QRT_SEPARATE_STATE_ARGS k.data<uint16_t>()+size_t(offset)*2048u,actual_u+size_t(offset)*4096u,w.data<uint16_t>()+size_t(offset)*4096u,g.data<float>()+size_t(offset)*32u,h.data<uint16_t>()+size_t(offset/64u)*state_cells,vn.data<uint16_t>()+size_t(offset)*4096u,state.data<float>(),n,exp
  if(variant){
   auto* flags=receipts.data<uint32_t>()+size_t(offset/1024u)*512u;
#ifdef QRT_GDN_PROVIDER_REPLAY
   check(qrt_fla_blackwell_cooperative::state_replay(QRT_SEPARATE_STATE_ARGS,nullptr,flags,int(variant)));
#else
   hipLaunchKernelGGL((qrt_fla_separate_state::fast_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,QRT_SEPARATE_STATE_ARGS,flags);check(hipGetLastError());
#ifdef QRT_GDN_HYBRID_RETRY
   if(variant==2u){hipLaunchKernelGGL((qrt_fla_hybrid_state::retry_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,QRT_SEPARATE_STATE_ARGS,flags);check(hipGetLastError());}
#endif
   hipLaunchKernelGGL((qrt_fla_separate_state::replay_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,QRT_SEPARATE_STATE_ARGS,flags);
#endif
  }
  else{hipLaunchKernelGGL((lifetime::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,nullptr,QRT_SEPARATE_STATE_ARGS);}
  check(hipGetLastError());
#undef QRT_SEPARATE_STATE_ARGS
#define QRT_SEPARATE_OUTPUT_ARGS q.data<uint16_t>()+size_t(offset)*2048u,vn.data<uint16_t>()+size_t(offset)*4096u,h.data<uint16_t>()+size_t(offset/64u)*state_cells,g.data<float>()+size_t(offset)*32u,scores.data<uint16_t>()+size_t(offset)*2048u,output.data<float>()+size_t(offset)*4096u,n,exp
  hipLaunchKernelGGL(lifetime::output_kernel,dim3(16u,32u,chunks),dim3(256u),0u,nullptr,QRT_SEPARATE_OUTPUT_ARGS);
  check(hipGetLastError());
#undef QRT_SEPARATE_OUTPUT_ARGS
 }
}
template<class T>std::vector<T> capture_file(const std::string& root,const char* name,size_t count){
 const std::string path=root+"/"+name;std::ifstream f(path,std::ios::binary|std::ios::ate);
 require(f && f.tellg()==std::streamoff(count*sizeof(T)),"capture file span");std::vector<T> result(count);f.seekg(0);f.read(reinterpret_cast<char*>(result.data()),count*sizeof(T));require(bool(f),"capture read");return result;
}
template<class T>void captured_rows(std::vector<T>& dest,const std::string& root,const char* name,unsigned width,unsigned count){
 const auto original=capture_file<T>(root,name,size_t(7169u)*width);
 for(unsigned row=0u;row<count;++row){const unsigned src=count==8192u && row>=7168u?row-7168u:row;
  std::copy_n(original.data()+size_t(src)*width,width,dest.data()+size_t(row)*width);}
}
// Extracted from blackwell_wu_output.cpp::score_kernel. Valid output arithmetic
// is unchanged; the observer omits padded stores to keep tight query guards.
// Only current_scores above is timed as the production score control.
__global__ void capture_scores(const uint16_t* q,const uint16_t* k,const float* g,
 uint16_t* scores,unsigned count,const unsigned char* table){
 using namespace qrt_fla_blackwell;
 const unsigned offset=blockIdx.z*64u;
 q+=size_t(offset)*2048u;k+=size_t(offset)*2048u;g+=size_t(offset)*32u;scores+=size_t(offset)*2048u;
 count=count-offset<64u?count-offset:64u;
 const unsigned cell=blockIdx.x*(kThreads/kGroup)+threadIdx.x/kGroup;
 const unsigned token=cell/64u,source=cell%64u,head=blockIdx.y,lane=threadIdx.x%kGroup;
 if(token>=count)return;
 const unsigned index=(token*32u+head)*64u+source;
 if(source>token){if(!lane)scores[index]=0u;return;}
 original::Value sum{0u,kZeroExponent,false};
 for(unsigned base=0u;base<128u;base+=kGroup){const unsigned d=base+lane;
  sum=accumulate(sum,q[(token*16u+head/2u)*128u+d],k[(source*16u+head/2u)*128u+d],lane);}
 if(!lane){sum=original::group_sum<26,kZeroExponent>(&sum,1u);
  scores[index]=to_bf16(original::value_to_float(sum)*scalar::exponential(g[token*32u+head]-g[source*32u+head],table));}
}
void paired_run(unsigned count,unsigned mode,unsigned measured,Device& table,const std::vector<unsigned char>& host_table,const char* capture_root=nullptr){
 const size_t small=size_t(count)*2048u,large=size_t(count)*4096u,gate=size_t(count)*32u,checkpoints=size_t((count+63u)/64u)*state_cells,score_cells=small;
 std::vector<uint16_t> q(small),k(small),v(large),beta(gate),inverse(small),scores(score_cells);std::vector<float> g(gate),seed(state_cells);
 auto fill=[&](std::vector<uint16_t>& x,unsigned salt,unsigned exponent){for(size_t i=0u;i<x.size();++i){const unsigned r=random_word(unsigned(i)^salt);x[i]=mode?uint16_t((r&0x807fu)|((exponent+r%4u)<<7u)):uint16_t(r&0x8000u);if(mode==2u && i%29u==0u)x[i]=uint16_t(r&0x807fu);}};
 fill(q,395u,115u);fill(k,8192u,115u);fill(v,35u,120u);fill(inverse,3u,116u);fill(scores,121u,112u);
 for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<32u;++head){
  beta[size_t(row)*32u+head]=uint16_t(0x3e80u|((row+head)&127u));g[size_t(row)*32u+head]=-float((row%64u+1u)*(head%7u+1u))*.0078125f;
  for(unsigned future=row%64u+1u;future<64u;++future){scores[(size_t(row)*32u+head)*64u+future]=0u;inverse[(size_t(row)*32u+head)*64u+future]=0u;}
  inverse[(size_t(row)*32u+head)*64u+row%64u]=0x3f80u;
 }
 for(size_t i=0u;i<state_cells;++i){const unsigned r=random_word(unsigned(i)^0x395u);seed[i]=mode==0u?from(uint16_t(r&0x8000u)):from(uint16_t((r&0x807fu)|((118u+r%4u)<<7u)));}

 if(mode==3u || mode==4u){
  for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<32u;++head)
   g[size_t(row)*32u+head]=-float(row%64u+1u)*(head%4u==0u?64.0f:head%4u==1u?2.0f:.0078125f);
 }
 if(mode==4u){
  for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<32u;++head)for(unsigned col=0u;col<64u;++col){
   const size_t at=(size_t(row)*32u+head)*64u+col;
   if(col!=row%64u && col%16u!=row%16u){inverse[at]&=0x8000u;scores[at]&=0x8000u;}
  }
 }
 if(mode==5u){
  for(size_t i=3u;i<q.size();i+=257u)q[i]=uint16_t(i%3u==0u?0x7f80u:i%3u==1u?0x7fc1u:0x0001u);
  for(size_t i=15u;i<scores.size();i+=509u)scores[i]=uint16_t(0x7f01u|(i&0x8000u));
 }
 if(mode==6u){
  // Both exact domain endpoints and nearby normal exclusions. Large Q has
  // no recurrent feedback; K's sparse tiny values keep the state finite.
  const unsigned exponents[]={95u,159u,94u,160u};
  for(unsigned row=0u;row<count;++row)for(unsigned head=0u;head<16u;++head){
   q[(size_t(row)*16u+head)*128u+3u]=uint16_t(exponents[(row+head)%4u]<<7u|37u);
   k[(size_t(row)*16u+head)*128u+7u]=uint16_t((row%2u?95u:94u)<<7u|11u);
  }
 }
 if(capture_root){
  require(count==7169u || count==8192u,"unsupported capture extension");const std::string root=capture_root;
  captured_rows(q,root,"full-q-normalized-bf16.bin",2048u,count);captured_rows(k,root,"full-k-normalized-bf16.bin",2048u,count);
  captured_rows(v,root,"full-v-bf16.bin",4096u,count);captured_rows(beta,root,"full-beta-bf16.bin",32u,count);
  captured_rows(inverse,root,"full-a-inverse-bf16.bin",2048u,count);captured_rows(g,root,"full-g-cumsum-f32.bin",32u,count);
  seed=capture_file<float>(root,"full-initial_state-f32.bin",state_cells);
 }
 Device dq((small+2u*guard)*2u),dk((small+2u*guard)*2u),dv((large+2u*guard)*2u),db((gate+2u*guard)*2u),dinv((small+2u*guard)*2u),ds((score_cells+2u*guard)*2u),dg((gate+2u*guard)*4u);
 dq.upload(q);dk.upload(k);db.upload(beta);dinv.upload(inverse);ds.upload(scores);dg.upload(g);

 {
  hipLaunchKernelGGL(capture_scores,dim3(256u,32u,(count+63u)/64u),dim3(256u),0u,nullptr,dq.data<uint16_t>(),dk.data<uint16_t>(),dg.data<float>(),ds.data<uint16_t>(),count,reinterpret_cast<unsigned char*>(table.data<uint32_t>()));check(hipGetLastError());finish();
  const auto observed=read<uint16_t>(ds,score_cells);guards(observed);scores.assign(observed.begin()+guard,observed.end()-guard);
  // Independent original CPU score samples cover every chunk, including the
  // partial final row and source extension. They cannot select replay work.
  if(mode<2u || mode==3u || mode==4u || mode==6u || capture_root)for(unsigned offset=0u;offset<count;offset+=64u)for(unsigned head:{0u,7u,31u}){
   const unsigned row=std::min(63u,count-offset-1u),src=row/2u;
   const float sum=dot(q.data()+(size_t(offset+row)*16u+head/2u)*128u,k.data()+(size_t(offset+src)*16u+head/2u)*128u,128u);
   const uint16_t expected=rounded(sum*qrt_sm121_exp2::evaluate(host_table.data(),(g[size_t(offset+row)*32u+head]-g[size_t(offset+src)*32u+head])*1.4426950408889634074f));
   require(scores[(size_t(offset+row)*32u+head)*64u+src]==expected,"independent captured score");
  }
 }
 Device du((large+2u*guard)*2u),dw((large+2u*guard)*2u),output((large+2u*guard)*4u),state((state_cells+2u*guard)*4u),h((checkpoints+2u*guard)*2u),vn((large+2u*guard)*2u);
 std::vector<float> expected,final;std::vector<uint16_t> old_w,old_u,old_h,old_vn;size_t cpu_dots=0u;double samples[2][state_variants][3]{};
 const size_t receipt_cells=size_t((count+1023u)/1024u)*512u;Device receipts((receipt_cells+2u*guard)*4u);
 size_t admitted_ctas[2][state_variants]{},retried_ctas[2][state_variants]{};
 const unsigned attempts=measured+1u;
 for(unsigned attempt=0u;attempt<attempts;++attempt)for(unsigned position=0u;position<2u*state_variants;++position){
  const unsigned choice=(attempt+position)%(2u*state_variants),variant=choice%state_variants;const bool alias=choice>=state_variants;
  dv.reset();dv.upload(v);du.reset();dw.reset();output.reset();state.reset();state.upload(seed);h.reset();vn.reset();ds.reset();receipts.reset();finish();
  const auto begin=std::chrono::steady_clock::now();paired_launch(variant,count,dq,dk,dv,db,dinv,dg,ds,du,dw,output,h,vn,state,table,receipts,alias);finish();
  if(attempt){
   samples[unsigned(alias)][variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();

  }
  const auto flags=read<uint32_t>(receipts,receipt_cells);guards(flags);
  if(variant){
   size_t admitted=0u,retried=0u;for(size_t i=0u;i<receipt_cells;++i){require(flags[guard+i]<=variant,"invalid state completion receipt");admitted+=flags[guard+i]==1u;retried+=flags[guard+i]==2u;}
   if(!attempt){admitted_ctas[unsigned(alias)][variant]=admitted;retried_ctas[unsigned(alias)][variant]=retried;}
   else require(admitted_ctas[unsigned(alias)][variant]==admitted && retried_ctas[unsigned(alias)][variant]==retried,"state admission changed between attempts");
  }else for(size_t i=0u;i<receipt_cells;++i)require(flags[guard+i]==0xa5a5a5a5u,"control changed candidate receipt");
  const auto actual_scores=read<uint16_t>(ds,score_cells);guards(actual_scores);
  require(std::equal(scores.begin(),scores.end(),actual_scores.begin()+guard),"complete original score differs");
  const auto actual=read<float>(output,large),next=read<float>(state,state_cells);
  const auto actual_w=read<uint16_t>(dw,large),actual_u=read<uint16_t>(alias?dv:du,large),actual_h=read<uint16_t>(h,checkpoints),actual_vn=read<uint16_t>(vn,large);
  guards(actual);guards(next);guards(actual_w);guards(actual_u);guards(actual_h);guards(actual_vn);
  if(!attempt && !choice){
   expected=actual;final=next;old_w=actual_w;old_u=actual_u;old_h=actual_h;old_vn=actual_vn;
   if(mode<2u || mode==3u || mode==4u || mode==6u || capture_root){
    cpu_dots+=cpu_wu(count,k,v,beta,inverse,g,host_table,old_w,old_u);
    const std::vector<uint16_t> hw(old_w.begin()+guard,old_w.end()-guard),hu(old_u.begin()+guard,old_u.end()-guard);
    for(const auto& column:std::vector<std::pair<unsigned,unsigned>>{{0u,0u},{3u,7u},{31u,127u}})cpu_dots+=cpu_column(count,column.first,column.second,q,k,hu,hw,g,scores,seed,host_table,expected,final,old_h,old_vn);
   }
  }
  if(capture_root && !attempt && !choice){
   const std::string root=capture_root;const size_t rows=count==8192u?7168u:7169u,cells=rows*4096u;
   const auto gout=capture_file<uint16_t>(root,"full-output-bf16.bin",size_t(7169u)*4096u);
   const auto gw=capture_file<uint16_t>(root,"full-w-bf16.bin",size_t(7169u)*4096u);
   const auto gu=capture_file<uint16_t>(root,"full-u-bf16.bin",size_t(7169u)*4096u);
   const auto gv=capture_file<uint16_t>(root,"full-v-new-bf16.bin",size_t(7169u)*4096u);
   const auto gh=capture_file<uint16_t>(root,"full-chunk-state-bf16.bin",size_t(113u)*state_cells);
   for(size_t i=0u;i<cells;++i){require(bits(expected[guard+i])==bits(from(gout[i])),"original GB10 output boundary");require(old_w[guard+i]==gw[i] && old_u[guard+i]==gu[i] && old_vn[guard+i]==gv[i],"original GB10 WU/residual boundary");}
   require(!std::memcmp(old_h.data()+guard,gh.data(),((rows+63u)/64u)*state_cells*2u),"original GB10 checkpoint boundary");
   if(count==7169u){const auto gs=capture_file<float>(root,"full-state-f32.bin",state_cells);require(!std::memcmp(final.data()+guard,gs.data(),state_cells*4u),"original GB10 final state boundary");}
  }

  require(!expected.empty() && !std::memcmp(actual.data(),expected.data(),actual.size()*4u),"separate-state GDN output differs");require(!std::memcmp(next.data(),final.data(),next.size()*4u),"separate-state GDN state differs");
  require(actual_w==old_w && actual_u==old_u && actual_h==old_h && actual_vn==old_vn,"separate-state GDN intermediate or alias differs");
  unchanged(dq,q);unchanged(dk,k);unchanged(db,beta);unchanged(dinv,inverse);unchanged(dg,g);
  if(!alias)unchanged(dv,v);else{const auto unused=read<uint16_t>(du,large);const auto* bytes=reinterpret_cast<const unsigned char*>(unused.data());require(std::all_of(bytes,bytes+unused.size()*2u,[](unsigned char x){return x==0xa5u;}),"inactive U buffer changed");}
 }
 for(unsigned alias=0u;alias<2u;++alias)for(unsigned variant=0u;variant<state_variants;++variant){
  double sorted[3]={samples[alias][variant][0],samples[alias][variant][1],samples[alias][variant][2]};std::sort(sorted,sorted+3u);
  std::printf("{\"kind\":\"%s\",\"tokens\":%u,\"mode\":%u,\"variant\":%u,\"separate_fast_and_replay\":%s,\"hybrid_retry_enabled\":%s,\"segment_tokens\":1024,\"segments\":%u,\"u_aliases_v\":%s,\"score_cells\":%zu,\"output_cells\":%zu,\"state_cells\":%zu,\"checkpoint_cells\":%zu,\"wu_and_residual_cells\":%zu,\"independent_cpu_dots\":%zu,\"state_ctas\":%zu,\"fast_state_ctas\":%zu,\"retried_state_ctas\":%zu,\"replayed_state_ctas\":%zu,\"warmups\":1,\"measured_attempts\":%u,\"complete_score_wu_state_output_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"captured_inputs\":%s,\"gb10_output_cells\":%zu,\"repeated_capture_rows\":%u,\"intermediate_host_synchronization\":false,\"all_attempts_verified\":true,\"raw_bit_mismatches\":0,\"receipt_guards_pass\":true,\"intermediate_and_alias_ownership_checked\":true,\"redzones_pass\":true,\"immutable_nonaliased_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",state_component,count,mode,variant,variant?"true":"false",variant==2u?"true":"false",(count+1023u)/1024u,alias?"true":"false",score_cells,large,state_cells,checkpoints,large*3u,cpu_dots,receipt_cells,variant?admitted_ctas[alias][variant]:0u,variant?retried_ctas[alias][variant]:0u,variant?receipt_cells-admitted_ctas[alias][variant]-retried_ctas[alias][variant]:0u,measured,sorted[1],samples[alias][variant][0],samples[alias][variant][1],samples[alias][variant][2],capture_root?"true":"false",capture_root?size_t(count==8192u?7168u:7169u)*4096u:0u,capture_root&&count==8192u?1024u:0u);
 }
 std::fflush(stdout);
}
int main(int argc,char** argv)try{
 require(argc==3 || argc==4,"requires verified SM121 exponential table and safety or throughput");hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
 const bool timing=!std::strcmp(argv[2],"throughput"),captured=!std::strcmp(argv[2],"q7169") || !std::strcmp(argv[2],"q8192");require(timing || captured || !std::strcmp(argv[2],"safety"),"unknown action");require(captured==(argc==4),"capture argument ownership");
 std::ifstream file(argv[1],std::ios::binary|std::ios::ate);require(file&&file.tellg()==std::streamoff(qrt_sm121_exp2::table_bytes),"table span");std::vector<unsigned char> table(qrt_sm121_exp2::table_bytes);file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());require(bool(file)&&qrt_sm121_exp2::valid_layout(table.data(),table.size()),"table layout");
 std::vector<uint32_t> table_words(table.size()/4u);require(table.size()%4u==0u,"table alignment");std::memcpy(table_words.data(),table.data(),table.size());Device dt((table_words.size()+2u*guard)*4u);dt.upload(table_words);
 if(captured)paired_run(!std::strcmp(argv[2],"q7169")?7169u:8192u,7u,3u,dt,table,argv[3]);else if(timing){paired_run(8192u,1u,3u,dt,table);paired_run(8192u,3u,3u,dt,table);}else for(unsigned count:{1u,63u,64u,65u,129u,1023u,1024u,1025u})for(unsigned mode=0u;mode<7u;++mode)paired_run(count,mode,0u,dt,table);
 unchanged(dt,table_words);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
