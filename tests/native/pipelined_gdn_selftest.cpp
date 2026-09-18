// The retained paired-score/state8/shared-arena kernels are unchanged. Reuse
// their complete fixture and independent CPU/GB10 boundaries through a policy.
#define QRT_PAIRED_SCORE_FIXTURE_ONLY
#include "paired_score_gdn_selftest.cpp"
#undef QRT_PAIRED_SCORE_FIXTURE_ONLY

namespace {
struct Pipeline {
 std::array<hipStream_t,3> streams{};
 std::array<hipEvent_t,8> prepared{},advanced{};
 hipEvent_t completed=nullptr;
 Pipeline(){
  try {
   for(auto& stream:streams)check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
   for(auto& event:prepared)check(hipEventCreateWithFlags(&event,hipEventDisableTiming));
   for(auto& event:advanced)check(hipEventCreateWithFlags(&event,hipEventDisableTiming));
   check(hipEventCreateWithFlags(&completed,hipEventDisableTiming));
  } catch(...) {release();throw;}
 }
 Pipeline(const Pipeline&)=delete;
 Pipeline& operator=(const Pipeline&)=delete;
 void release() noexcept {
  // Drain all submitted readers before destroying dependencies or buffers,
  // including when a launch or event operation failed partway through a call.
  for(auto stream:streams)if(stream)(void)hipStreamSynchronize(stream);
  for(auto& event:prepared)if(event){(void)hipEventDestroy(event);event=nullptr;}
  for(auto& event:advanced)if(event){(void)hipEventDestroy(event);event=nullptr;}
  if(completed){(void)hipEventDestroy(completed);completed=nullptr;}
  for(auto& stream:streams)if(stream){(void)hipStreamDestroy(stream);stream=nullptr;}
 }
 ~Pipeline(){release();}
};
Pipeline* active_pipeline=nullptr;

void pipelined_launch(unsigned variant,unsigned count,Device& q,Device& k,Device& v,Device& beta,
 Device& inverse,Device& g,Device& scores,Device& u,Device& w,Device& output,Device& h,Device& vn,
 Device& state,Device& table,bool alias){
 require(variant<=2u && count>0u && count<=8192u && active_pipeline,"invalid GDN pipeline request");
 if(!variant){paired_launch(2u,count,q,k,v,beta,inverse,g,scores,u,w,output,h,vn,state,table,alias);return;}
 auto& owner=*active_pipeline;
 const auto producer=owner.streams[0];
 const auto recurrence=owner.streams[variant==2u?1u:0u];
 const auto consumer=owner.streams[variant==2u?2u:1u];
 const auto* exp=reinterpret_cast<unsigned char*>(table.data<uint32_t>());
 auto* actual_u=alias?v.data<uint16_t>():u.data<uint16_t>();
 try {
  for(unsigned offset=0u;offset<count;offset+=1024u){
   const unsigned segment=offset/1024u,n=std::min(1024u,count-offset),chunks=(n+63u)/64u;
   // Score and WU touch only this segment. In U=V mode each original WU CTA
   // captures all of its V columns before stores; later segments are disjoint.
   hipLaunchKernelGGL(qrt_fla_paired_score::kernel,dim3(2u,16u,chunks*8u),dim3(256u),0u,producer,
    q.data<uint16_t>()+size_t(offset)*2048u,k.data<uint16_t>()+size_t(offset)*2048u,
    g.data<float>()+size_t(offset)*32u,scores.data<uint16_t>()+size_t(offset)*2048u,n,exp);
   check(hipGetLastError());
   hipLaunchKernelGGL(scalar::wu_kernel,dim3(16u,32u,chunks),dim3(256u),0u,producer,
    k.data<uint16_t>()+size_t(offset)*2048u,v.data<uint16_t>()+size_t(offset)*4096u,
    beta.data<uint16_t>()+size_t(offset)*32u,inverse.data<uint16_t>()+size_t(offset)*2048u,
    g.data<float>()+size_t(offset)*32u,w.data<uint16_t>()+size_t(offset)*4096u,
    actual_u+size_t(offset)*4096u,n,exp);
   check(hipGetLastError());
   if(variant==2u){
    check(hipEventRecord(owner.prepared[segment],producer));
    check(hipStreamWaitEvent(recurrence,owner.prepared[segment],0u));
   }
   // All state kernels remain on one ordered stream. Every segment consumes
   // the complete original FP32 state left by its immediate predecessor.
   hipLaunchKernelGGL((lifetime::state_kernel<8u>),dim3(16u,32u),dim3(256u),0u,recurrence,
    k.data<uint16_t>()+size_t(offset)*2048u,actual_u+size_t(offset)*4096u,
    w.data<uint16_t>()+size_t(offset)*4096u,g.data<float>()+size_t(offset)*32u,
    h.data<uint16_t>()+size_t(offset/64u)*state_cells,vn.data<uint16_t>()+size_t(offset)*4096u,
    state.data<float>(),n,exp);
   check(hipGetLastError());
   check(hipEventRecord(owner.advanced[segment],recurrence));
   check(hipStreamWaitEvent(consumer,owner.advanced[segment],0u));
   // This completion dependency also covers the score producer transitively.
   // H and Vnew are immutable per-segment spans until all consumers complete.
   hipLaunchKernelGGL(lifetime::output_kernel,dim3(16u,32u,chunks),dim3(256u),0u,consumer,
    q.data<uint16_t>()+size_t(offset)*2048u,vn.data<uint16_t>()+size_t(offset)*4096u,
    h.data<uint16_t>()+size_t(offset/64u)*state_cells,g.data<float>()+size_t(offset)*32u,
    scores.data<uint16_t>()+size_t(offset)*2048u,output.data<float>()+size_t(offset)*4096u,n,exp);
   check(hipGetLastError());
  }
  // The inherited bounded completion helper records an event on the default
  // stream; nonblocking streams have no implicit dependency on that event.
  // The final consumer covers every earlier stage transitively. Join it back
  // before the helper starts polling or any observer reads/reuses storage.
  check(hipEventRecord(owner.completed,consumer));
  check(hipStreamWaitEvent(nullptr,owner.completed,0u));
 } catch(...) {
  // The shared fixture's Device locals unwind before the outer Pipeline, so
  // waiting only in Pipeline's destructor would release their storage early.
  for(auto stream:owner.streams)(void)hipStreamSynchronize(stream);
  throw;
 }
 // The fixture's bounded default-stream event now covers the complete graph.
 // Events/streams are reused only after that completion and validation.
}
const PairedFixturePolicy pipeline_policy{
 "pipelined_gdn_component",pipelined_launch,{true,true,true},{1u,2u,3u}};
}

int main(int argc,char** argv)try{
 require(argc==3 || argc==4,"requires verified SM121 exponential table and action");
 hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,0));
 require(!std::strncmp(p.gcnArchName,"gfx1151",7u),"requires gfx1151");
 const bool timing=!std::strcmp(argv[2],"throughput");
 const bool captured=!std::strcmp(argv[2],"q7169") || !std::strcmp(argv[2],"q8192");
 require(timing || captured || !std::strcmp(argv[2],"safety"),"unknown action");
 require(captured==(argc==4),"capture argument ownership");
 std::ifstream file(argv[1],std::ios::binary|std::ios::ate);
 require(file && file.tellg()==std::streamoff(qrt_sm121_exp2::table_bytes),"table span");
 std::vector<unsigned char> table(qrt_sm121_exp2::table_bytes);
 file.seekg(0);file.read(reinterpret_cast<char*>(table.data()),table.size());
 require(bool(file) && qrt_sm121_exp2::valid_layout(table.data(),table.size()),"table layout");
 std::vector<uint32_t> words(table.size()/4u);require(table.size()%4u==0u,"table alignment");
 std::memcpy(words.data(),table.data(),table.size());
 Device device_table((words.size()+2u*guard)*4u);device_table.upload(words);
 Pipeline pipeline;active_pipeline=&pipeline;
 if(captured)paired_run(!std::strcmp(argv[2],"q7169")?7169u:8192u,6u,3u,device_table,table,argv[3],pipeline_policy);
 else if(timing){
  paired_run(8192u,1u,3u,device_table,table,nullptr,pipeline_policy);
  paired_run(8192u,3u,3u,device_table,table,nullptr,pipeline_policy);
 }else for(unsigned count:{1u,63u,64u,65u,129u,1023u,1024u,1025u})
  for(unsigned mode=0u;mode<6u;++mode)paired_run(count,mode,0u,device_table,table,nullptr,pipeline_policy);
 unchanged(device_table,words);active_pipeline=nullptr;return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
