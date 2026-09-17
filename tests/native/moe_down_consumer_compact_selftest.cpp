// Run the actual certificate, original collector/replay and production combine.
#define QRT_TRITON_MOE_BATCHED_HAWKEYE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_GATE 1
#define QRT_TRITON_MOE_NATIVE_WMMA_DOWN 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SPLIT_GATE_PASSES 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_GATE_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_SERIAL_DOWN_N32 1
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_LOAD_THREADS 192
#define QRT_TRITON_MOE_NATIVE_WMMA_LDS_B_M64_FUSED_OVERFLOW32 1
#include "../../native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp"
#include "../../native/providers/triton_moe/down_consumer_compact.h"
#include <stdexcept>
#include <array>
#include <vector>

namespace test {
namespace f = qrt_moe_down_consumer_filter;
namespace fused = qrt_moe_down_consumer_compact;
namespace c = qrt_routed_consumer;
namespace in = qrt_moe_down_consumer;
constexpr size_t guard=128;
hipStream_t stream=nullptr;
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
void ok(hipError_t result,const char* why){if(result!=hipSuccess)throw std::runtime_error(std::string(why)+": "+hipGetErrorString(result));}
void finish(){const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){auto result=hipStreamQuery(stream);if(result==hipSuccess)return;ok(result==hipErrorNotReady?hipSuccess:result,"query");require(std::chrono::steady_clock::now()<deadline,"deadline");std::this_thread::yield();}}
void finish_event(hipEvent_t event){const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){auto result=hipEventQuery(event);if(result==hipSuccess)return;ok(result==hipErrorNotReady?hipSuccess:result,"event query");require(std::chrono::steady_clock::now()<deadline,"event deadline");std::this_thread::yield();}}
template<class T>struct Buffer{
 std::vector<T> initial;T* storage=nullptr;size_t count;
 explicit Buffer(const std::vector<T>& values):initial(values.size()+2*guard,T(93)),count(values.size()){
  std::copy(values.begin(),values.end(),initial.begin()+guard);ok(hipMalloc(reinterpret_cast<void**>(&storage),initial.size()*sizeof(T)),"malloc");reset();
 }
 explicit Buffer(size_t n,T value=T{}):Buffer(std::vector<T>(n,value)){}
 ~Buffer(){if(storage){(void)hipStreamSynchronize(stream);(void)hipFree(storage);}}
 T* data(){return storage+guard;}
 void reset(){ok(hipMemcpy(storage,initial.data(),initial.size()*sizeof(T),hipMemcpyHostToDevice),"upload");}
 std::vector<T> read(bool immutable=false){std::vector<T> result(initial.size());ok(hipMemcpy(result.data(),storage,result.size()*sizeof(T),hipMemcpyDeviceToHost),"read");
  require(std::equal(result.begin(),result.begin()+guard,initial.begin())&&std::equal(result.end()-guard,result.end(),initial.end()-guard),"redzone");
  if(immutable)require(!std::memcmp(result.data(),initial.data(),result.size()*sizeof(T)),"immutable input");
  return std::vector<T>(result.begin()+guard,result.end()-guard);
 }
};
bool selected_candidate(float native,float contribution,float error,unsigned radius,unsigned exponent){
 const unsigned low=c::bits(contribution)&65535u,distance=low>=32768u?low-32768u:32768u-low;
 return (radius&&distance<=radius)||(exponent&&((c::bits(native)>>23u)&255u)<=exponent)||qrt_bf16_midpoint::within_error(contribution,error);
}
void invalid_calls(){
 Buffer<float> floats(64u);Buffer<unsigned> storage(64u);Buffer<uint16_t> bf16s(64u);Buffer<int32_t> experts(64u);
 fused::View view{floats.data(),floats.data(),experts.data(),floats.data(),floats.data(),bf16s.data(),floats.data(),512e-9f,0u,0u,1u};
 unsigned rejected=0u;
 auto call=[&](fused::View v,unsigned first,unsigned count,size_t capacity,unsigned* ids,unsigned* number,unsigned* stats){
  require(fused::launch(v,first,count,ids,capacity,number,stats,stream)==hipErrorInvalidValue,"invalid fused view accepted");++rejected;
 };
 for(unsigned which=0u;which<7u;++which){auto v=view;switch(which){case 0:v.native=nullptr;break;case 1:v.weights=nullptr;break;case 2:v.experts=nullptr;break;case 3:v.input_l2=nullptr;break;case 4:v.weight_l2=nullptr;break;case 5:v.shared_down=nullptr;break;case 6:v.shared_gate=nullptr;break;}call(v,0,1,16384,storage.data(),storage.data(),storage.data());}
 call(view,0,1,16384,nullptr,storage.data(),storage.data());call(view,0,1,16384,storage.data(),nullptr,storage.data());call(view,0,1,16384,storage.data(),storage.data(),nullptr);
 call(view,0,0,16384,storage.data(),storage.data(),storage.data());call(view,0,257,16384,storage.data(),storage.data(),storage.data());
 call(view,1,1,16384,storage.data(),storage.data(),storage.data());call(view,0,2,32768,storage.data(),storage.data(),storage.data());call(view,0,1,16383,storage.data(),storage.data(),storage.data());
 auto v=view;v.tokens=8193u;call(v,0,1,16384,storage.data(),storage.data(),storage.data());v=view;v.tokens=0u;call(v,0,1,16384,storage.data(),storage.data(),storage.data());
 for(float scale:{0.0f,-1.0f,INFINITY,c::value(0x7fc00000u)}){v=view;v.error_scale=scale;call(v,0,1,16384,storage.data(),storage.data(),storage.data());}
 finish();floats.read(true);storage.read(true);bf16s.read(true);experts.read(true);
 require(rejected==21u,"invalid-call count");std::printf("{\"kind\":\"moe_down_compact_invalid_calls\",\"rejected\":%u,\"no_memory_changes\":true}\n",rejected);
}

void run(unsigned tokens,unsigned mode,bool throughput=false){
 const unsigned routes=tokens*8u,columns=2048u,width=512u;const size_t cells=size_t(tokens)*columns,outputs=size_t(routes)*columns;
 const unsigned expert_count=throughput?256u:2u;
 std::vector<uint16_t> input(size_t(routes)*width),weights(size_t(expert_count)*columns*width),shared(cells);
 std::vector<float> topk(routes),gate(tokens),residual(cells),raw(outputs),cpu_exact(outputs),input_norm(routes,mode==3?1e8f:0.0f),weight_norm(256u*columns,1.0f);
 std::vector<int32_t> experts(routes);
 for(unsigned r=0;r<routes;++r){experts[r]=int32_t(r%expert_count);topk[r]=std::ldexp((r%3u==0u?-1.0f:1.0f),-int(r%8u));
  for(unsigned k=0;k<width;++k)input[size_t(r)*width+k]=c::rounded(float(int((r%7u*13u+k*3u)%61u)-30)/32.0f);
 }
 for(unsigned e=0;e<2;++e)for(unsigned n=0;n<columns;++n)for(unsigned k=0;k<width;++k)
  weights[(size_t(e)*columns+n)*width+k]=c::rounded(float(int((e*17u+n%11u*7u+k*5u)%47u)-23)/64.0f);
 for(unsigned e=2u;e<expert_count;++e)std::memcpy(weights.data()+size_t(e)*columns*width,weights.data()+size_t(e%2u)*columns*width,size_t(columns)*width*2u);
 float reference[7][2][11]{};
 for(unsigned r=0;r<std::min(routes,7u);++r)for(unsigned e=0;e<2;++e)for(unsigned n=0;n<11;++n)
  reference[r][e][n]=qrt_q1_moe_hawkeye::dot_bf16_hopper(input.data()+size_t(r)*width,weights.data()+(size_t(e)*columns+n)*width,width);
 for(unsigned r=0;r<routes;++r)for(unsigned n=0;n<columns;++n){
  const size_t index=size_t(r)*columns+n;const float exact=reference[r%7u][r%2u][n%11u];cpu_exact[index]=exact;
  raw[index]=exact;
  if(mode&&exact!=0.0f&&(mode!=2u||n%3u==0u))raw[index]=c::value((uint32_t(c::rounded(exact))<<16u)+0x8000u);
  const auto interval=c::range(in::multiply(topk[r],raw[index]),input_norm[r]*512e-9f*fabsf(topk[r]));
  if(interval.valid)require(c::contains(interval,in::multiply(topk[r],exact)),"fixture envelope");
 }
 for(unsigned t=0;t<tokens;++t){gate[t]=t%2u?-0.5f:1.0f;
  for(unsigned n=0;n<columns;++n){const size_t index=size_t(t)*columns+n;
   shared[index]=c::rounded(n%4u==0u?0.0f:n%4u==1u?128.0f:n%4u==2u?-64.0f:0.5f);
   residual[index]=c::widen(c::rounded(float(int((t*3u+n)%17u)-8)/8.0f));
  }
 }
 Buffer<uint16_t> di(input),dw(weights),ds(shared);Buffer<float> dt(topk),dg(gate),dh(residual),dn(raw),dinput(input_norm),dweight(weight_norm);
 const unsigned window_blocks=throughput?kMaximumMoeCompactionBlocks:kMoeCompactionBlocks;
 Buffer<int32_t> de(experts);Buffer<uint8_t> masks(cells);Buffer<unsigned> stats(f::CounterCount),indices(size_t(window_blocks)*kNativeThreads),count(1u);
 Buffer<float> output(cells);std::vector<float> control,control_routes;
 g_state.compact_routed_hawkeye=true;g_state.sm121_moe_absolute_error_ppb=512u;g_state.moe_compaction_blocks=window_blocks;
 g_state.moe_compacted_indices=indices.data();g_state.moe_compacted_count=count.data();
 g_state.moe_l2[size_t(MoeL2::RoutedActivated)]=dinput.data();g_state.moe_l2[size_t(MoeL2::RoutedDown)]=dweight.data();

 const size_t ig=size_t(routes)*width/16u,wg=size_t(expert_count)*columns*width/16u;
 Buffer<uint16_t> prepared_input(throughput?ig*18u:0u),prepared_weight(throughput?wg*18u:0u);
 Buffer<unsigned> input_flags(throughput?routes:0u),weight_flags(throughput?expert_count*columns:0u);
 Buffer<float> norm_scratch(throughput?size_t(expert_count)*columns:0u);
 Buffer<unsigned> order_storage(throughput?size_t(window_blocks)*kNativeThreads+qrt_moe_expert_order::metadata_words:0u);
 g_state.prevalidated_float_active=throughput;g_state.staged_half_replay_active=throughput;g_state.moe_expert_order_active=throughput;
 g_state.prepared_replay_inputs=prepared_input.data();g_state.prepared_replay_weights=prepared_weight.data();
 g_state.prepared_replay_input_rows=input_flags.data();g_state.prepared_replay_weight_rows=weight_flags.data();
 g_state.moe_expert_order_storage=order_storage.data();g_state.topk_ids=de.data();
 if(throughput){
  for(unsigned side=0u;side<2u;++side){
   const uint16_t* source=side?dw.data():di.data();const unsigned rows=side?expert_count*columns:routes;
   auto* destination=reinterpret_cast<qrt_sm121_staged_half_projection::Row*>(side?prepared_weight.data():prepared_input.data());
   hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((size_t(rows)*(width/16u)+255u)/256u),dim3(256u),0,stream,source,destination,rows,width);ok(hipGetLastError(),"prepare retained staged operands");
   for(unsigned first=0u;first<rows;first+=4096u){
    hipLaunchKernelGGL(HIP_KERNEL_NAME(moe_bf16_row_l2_prepared_kernel<true>),dim3(std::min(4096u,rows-first)),dim3(256u),0,stream,source,norm_scratch.data(),nullptr,side?weight_flags.data():input_flags.data(),rows,width,first);ok(hipGetLastError(),"classify retained replay rows");
   }
  }
  finish();
 }
 const unsigned radius=mode==1?32768u:mode==2?128u:0u;
 std::vector<unsigned> actual_stats,reference_stats;size_t cpu_compared=0,changed_raw=0;
 double samples[4][3]{};const unsigned attempts=throughput?4u:1u;
 for(unsigned attempt=0;attempt<attempts;++attempt)for(unsigned position=0;position<4;++position){
  const unsigned filtered=(attempt+position)%4u;
  if(!throughput){std::fprintf(stderr,"COMPACT_CASE tokens=%u mode=%u variant=%u phase=reset\n",tokens,mode,filtered);std::fflush(stderr);}
  dn.reset();output.reset();stats.reset();
  fused::Owner owner(stream);ok(owner.initialize(filtered==3u,true),"actual compact owner initialization");
  hipEvent_t shared_done=nullptr;ok(hipEventCreate(&shared_done),"shared completion create");ok(hipEventRecord(shared_done,stream),"same-call shared completion record");
  if(!throughput){std::fprintf(stderr,"COMPACT_CASE tokens=%u mode=%u variant=%u phase=shared_recorded\n",tokens,mode,filtered);std::fflush(stderr);}
  finish_event(shared_done);finish();
  const auto begin=std::chrono::steady_clock::now();
  if(!throughput){std::fprintf(stderr,"COMPACT_CASE tokens=%u mode=%u variant=%u phase=launch\n",tokens,mode,filtered);std::fflush(stderr);}
  if(filtered==1u){hipLaunchKernelGGL(f::certify,dim3(throughput?4096u:17u),dim3(256),0,stream,dn.data(),dt.data(),de.data(),dinput.data(),dweight.data(),512e-9f,radius,0u,ds.data(),dg.data(),masks.data(),stats.data(),tokens);ok(hipGetLastError(),"certificate");}
  using Phase=MoeCorrectionPhase;
  if(filtered<2u){
  ok(launch_moe_routed_correction<false>(routed_down_batched_hawkeye_correction_kernel<Phase::Local>,routed_down_batched_hawkeye_correction_kernel<Phase::Collect>,routed_down_batched_hawkeye_correction_kernel<Phase::Replay>,routed_down_batched_hawkeye_correction_kernel<Phase::Local>,unsigned((outputs+255u)/256u),stream,MoeL2::RoutedActivated,MoeL2::RoutedDown,nullptr,filtered?masks.data():nullptr,filtered?stats.data():nullptr,dn.data(),dt.data(),de.data(),di.data(),dw.data(),routes,radius,0u),"original replay");
  }else if(filtered==2u){
   fused::View view{dn.data(),dt.data(),de.data(),dinput.data(),dweight.data(),ds.data(),dg.data(),512e-9f,radius,0u,tokens};
   const unsigned window_tokens=window_blocks*kNativeThreads/(8u*2048u);
   for(unsigned first=0u;first<tokens;first+=window_tokens){
    const unsigned n=std::min(window_tokens,tokens-first);
    if(!throughput)indices.reset();
    ok(hipMemsetAsync(count.data(),0,sizeof(unsigned),stream),"fused count reset");
    ok(fused::launch(view,first,n,indices.data(),size_t(window_blocks)*kNativeThreads,count.data(),stats.data(),stream),"fused certificate and collect");
    if(!throughput){
     finish();const auto queue=indices.read(),num=count.read();const auto old_mask=masks.read();
     require(num[0]<=n*8u*2048u,"fused queue capacity");
     std::vector<unsigned> expected;
     for(unsigned r=first*8u;r<(first+n)*8u;++r)for(unsigned col=0u;col<2048u;++col){
      const size_t index=size_t(r)*2048u+col;
      const bool chosen=selected_candidate(raw[index],in::multiply(topk[r],raw[index]),input_norm[r]*512e-9f*fabsf(topk[r]),radius,0u);
      const bool omitted=(old_mask[size_t(r/8u)*2048u+col]>>(r%8u))&1u;
      if(chosen&&!omitted)expected.push_back(unsigned(index));
     }
     std::vector<unsigned> actual_queue(queue.begin(),queue.begin()+num[0]);std::sort(actual_queue.begin(),actual_queue.end());
     require(actual_queue==expected,"fused complete window permutation");
     for(size_t i=num[0];i<queue.size();++i)require(queue[i]==0u,"fused unused queue tail");
    }
    MoeCorrectionBounds bounds{dinput.data(),dweight.data(),512e-9f,first*64u,indices.data(),count.data()};
    bounds.down_consumer_filter_stats=stats.data();
    if(throughput){
     bounds.prevalidated_float=true;bounds.staged_half_replay=true;
     bounds.prepared_input=prepared_input.data();bounds.prepared_weights=prepared_weight.data();
     bounds.prepared_input_rows=input_flags.data();bounds.prepared_weight_rows=weight_flags.data();
     ok(qrt_moe_expert_order::launch(indices.data(),count.data(),de.data(),2048u,{order_storage.data(),size_t(window_blocks)*kNativeThreads},stream),"unchanged expert permutation");
     bounds.compacted_indices=order_storage.data();
    }
    hipLaunchKernelGGL(routed_down_batched_hawkeye_correction_kernel<Phase::Replay>,dim3(std::min(kMoeCompactionBlocks,n*64u)),dim3(kNativeThreads),0,stream,dn.data(),dt.data(),de.data(),di.data(),dw.data(),routes,radius,0u,bounds);ok(hipGetLastError(),"unchanged original replay");
   }
  }else{
   ok(owner.prepare({dn.data(),dt.data(),de.data(),dinput.data(),dweight.data(),ds.data(),dg.data(),512e-9f,radius,0u,tokens},shared_done),"actual compact owner dependency");
   ok(launch_moe_down_compacted(owner,dn.data(),dt.data(),de.data(),di.data(),dw.data(),routes,radius,0u,stream),"actual provider compact launcher");
  }

  hipLaunchKernelGGL(full_v3_fused_combine_residual_kernel,dim3(unsigned((cells+kNativeThreads*kFusedCombineWidth-1)/(kNativeThreads*kFusedCombineWidth))),dim3(kNativeThreads),0,stream,dn.data(),dt.data(),de.data(),di.data(),dw.data(),ds.data(),dg.data(),dh.data(),output.data(),true,true,true,true,true,false,0u,cells);ok(hipGetLastError(),"production combine");finish();
  ok(owner.finish(),"actual compact owner completion");
  if(throughput&&attempt)samples[filtered][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
  auto actual=output.read(),actual_routes=dn.read();
  if(!filtered){control=actual;control_routes=actual_routes;
   cpu_compared=0u;
   for(unsigned r=0;r<routes;++r)for(unsigned n=0;n<columns;++n){const size_t index=size_t(r)*columns+n;
    if(selected_candidate(raw[index],in::multiply(topk[r],raw[index]),input_norm[r]*512e-9f*fabsf(topk[r]),radius,0u)){
     require(c::bits(actual_routes[index])==c::bits(cpu_exact[index]),"independent CPU K16");++cpu_compared;
    }
   }
  }else{
   require(!std::memcmp(actual.data(),control.data(),cells*sizeof(float)),"complete production residual");
   auto bitmap=masks.read();actual_stats=stats.read();changed_raw=0u;
   if(filtered==3u)ok(hipMemcpy(actual_stats.data(),owner.stats(),actual_stats.size()*sizeof(unsigned),hipMemcpyDeviceToHost),"actual compact counters");
   if(filtered==1u)reference_stats=actual_stats;else require(actual_stats==reference_stats,"fused and old counter totals differ");
   unsigned selected=0,skipped=0,invariant=0;
   for(size_t index=0;index<cells;++index){unsigned mask=bitmap[index];if(mask)++invariant;skipped+=unsigned(__builtin_popcount(mask));}
   for(unsigned r=0;r<routes;++r)for(unsigned n=0;n<columns;++n){const size_t index=size_t(r)*columns+n;
    const bool chosen=selected_candidate(raw[index],in::multiply(topk[r],raw[index]),input_norm[r]*512e-9f*fabsf(topk[r]),radius,0u);
    selected+=chosen;const bool skip=(bitmap[size_t(r/8u)*columns+n]>>(r%8u))&1u;
    require(!skip||chosen,"removed noncandidate");
    require(c::bits(actual_routes[index])==c::bits(skip?raw[index]:control_routes[index]),"actual skip or original replay");
    changed_raw+=c::bits(actual_routes[index])!=c::bits(control_routes[index]);
   }
   require(actual_stats[f::Cells]==cells&&actual_stats[f::Candidates]==selected&&actual_stats[f::Collected]==selected,"selected identities");
   require(actual_stats[f::Removable]==skipped&&actual_stats[f::Omitted]==skipped&&actual_stats[f::Replayed]==selected-skipped,"replay identities");
   require(actual_stats[f::InvariantCells]==invariant&&!actual_stats[f::InvalidValues],"certificate identities");
   if(mode==1)require(skipped>0&&skipped<selected&&changed_raw>0,"missing removed and retained work");
   if(mode==3)require(skipped==0&&selected>0,"wide interval must retain all");
  }
  ok(hipEventDestroy(shared_done),"shared completion destroy");
 }
 di.read(true);dw.read(true);ds.read(true);dt.read(true);dg.read(true);dh.read(true);de.read(true);dinput.read(true);dweight.read(true);indices.read();count.read();
 if(throughput){
  order_storage.read();norm_scratch.read();
  for(unsigned side=0u;side<2u;++side){
   const auto packed=(side?prepared_weight:prepared_input).read();const auto flags=(side?weight_flags:input_flags).read();
   const auto& raw=side?weights:input;const size_t groups=raw.size()/16u;
   for(size_t g=0u;g<groups;++g){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+g*16u);require(!std::memcmp(packed.data()+g*18u,&expected,sizeof(expected)),"retained prepared representation");}
   for(auto flag:flags)require(flag==1u,"finite generated replay classification");
  }
 }
 std::printf("{\"kind\":\"moe_down_consumer_compact\",\"staged_and_expert_order\":%s,\"per_window_permutation_checked\":%s,\"tokens\":%u,\"mode\":%u,\"output_cells\":%zu,\"selected\":%u,\"skipped\":%u,\"replayed\":%u,\"changed_raw_route_values\":%zu,\"independent_cpu_dots\":%zu,\"production_f32_mismatches\":0,\"actual_original_replay_checked\":true,\"counter_identities_pass\":true,\"immutable_inputs\":true,\"redzones_pass\":true,\"inference_acceptance\":false}\n",throughput?"true":"false",throughput?"false":"true",tokens,mode,cells,actual_stats[f::Candidates],actual_stats[f::Omitted],actual_stats[f::Replayed],changed_raw,cpu_compared);
 if(throughput)for(unsigned variant=0u;variant<4u;++variant){
  std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  std::printf("{\"kind\":\"moe_down_compact_timing\",\"tokens\":%u,\"mode\":%u,\"variant\":%u,\"complete_certificate_collect_replay_combine_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"warmups\":1,\"measured_attempts\":3,\"production_f32_mismatches\":0,\"all_attempts_verified\":true,\"allocation_and_upload_excluded\":true,\"model_loaded\":false,\"gb10_qualified\":false}\n",tokens,mode,variant,ordered[1],samples[variant][0],samples[variant][1],samples[variant][2]);std::fflush(stdout);
 }

}
}
int main(int argc,char** argv){try{
 struct StreamScope {StreamScope(){test::ok(hipStreamCreateWithFlags(&test::stream,hipStreamNonBlocking),"test stream create");}~StreamScope(){(void)hipStreamSynchronize(test::stream);(void)hipStreamDestroy(test::stream);test::stream=nullptr;}} stream_scope;
 hipDeviceProp_t p{};test::ok(hipGetDeviceProperties(&p,0),"device");test::require(!std::strncmp(p.gcnArchName,"gfx1151",7),"requires gfx1151");
 if(argc==2&&!std::strcmp(argv[1],"--throughput")){test::run(8192u,1u,true);return 0;}
 test::require(argc==1,"unsupported argument");test::invalid_calls();
 for(unsigned tokens:{1u,3u,17u,33u,129u,257u})for(unsigned mode=0;mode<4;++mode)test::run(tokens,mode);
 return 0;}catch(const std::exception& e){std::fprintf(stderr,"moe_down_consumer_compact_selftest: %s\n",e.what());return 1;}}
