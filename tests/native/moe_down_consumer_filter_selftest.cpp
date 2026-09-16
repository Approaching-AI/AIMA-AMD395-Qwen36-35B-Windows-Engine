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
#include <stdexcept>
#include <vector>

namespace test {
namespace f = qrt_moe_down_consumer_filter;
namespace c = qrt_routed_consumer;
namespace in = qrt_moe_down_consumer;
constexpr size_t guard=128;
void require(bool condition,const char* why){if(!condition)throw std::runtime_error(why);}
void ok(hipError_t result,const char* why){if(result!=hipSuccess)throw std::runtime_error(std::string(why)+": "+hipGetErrorString(result));}
void finish(){const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){auto result=hipStreamQuery(nullptr);if(result==hipSuccess)return;ok(result==hipErrorNotReady?hipSuccess:result,"query");require(std::chrono::steady_clock::now()<deadline,"deadline");std::this_thread::yield();}}
template<class T>struct Buffer{
 std::vector<T> initial;T* storage=nullptr;size_t count;
 explicit Buffer(const std::vector<T>& values):initial(values.size()+2*guard,T(93)),count(values.size()){
  std::copy(values.begin(),values.end(),initial.begin()+guard);ok(hipMalloc(reinterpret_cast<void**>(&storage),initial.size()*sizeof(T)),"malloc");reset();
 }
 explicit Buffer(size_t n,T value=T{}):Buffer(std::vector<T>(n,value)){}
 ~Buffer(){if(storage){(void)hipStreamSynchronize(nullptr);(void)hipFree(storage);}}
 T* data(){return storage+guard;}
 void reset(){ok(hipMemcpy(storage,initial.data(),initial.size()*sizeof(T),hipMemcpyHostToDevice),"upload");}
 std::vector<T> read(bool immutable=false){std::vector<T> result(initial.size());ok(hipMemcpy(result.data(),storage,result.size()*sizeof(T),hipMemcpyDeviceToHost),"read");
  require(std::equal(result.begin(),result.begin()+guard,initial.begin())&&std::equal(result.end()-guard,result.end(),initial.end()-guard),"redzone");
  if(immutable)require(!std::memcmp(result.data(),initial.data(),result.size()*sizeof(T)),"immutable input");
  return std::vector<T>(result.begin()+guard,result.end()-guard);
 }
};
bool selected(float native,float contribution,float error,unsigned radius,unsigned exponent){
 const unsigned low=c::bits(contribution)&65535u,distance=low>=32768u?low-32768u:32768u-low;
 return (radius&&distance<=radius)||(exponent&&((c::bits(native)>>23u)&255u)<=exponent)||qrt_bf16_midpoint::within_error(contribution,error);
}
void run(unsigned tokens,unsigned mode){
 const unsigned routes=tokens*8u,columns=2048u,width=512u;const size_t cells=size_t(tokens)*columns,outputs=size_t(routes)*columns;
 std::vector<uint16_t> input(size_t(routes)*width),weights(size_t(2)*columns*width),shared(cells);
 std::vector<float> topk(routes),gate(tokens),residual(cells),raw(outputs),cpu_exact(outputs),input_norm(routes,mode==3?1e8f:0.0f),weight_norm(256u*columns,1.0f);
 std::vector<int32_t> experts(routes);
 for(unsigned r=0;r<routes;++r){experts[r]=int32_t(r%2u);topk[r]=std::ldexp((r%3u==0u?-1.0f:1.0f),-int(r%8u));
  for(unsigned k=0;k<width;++k)input[size_t(r)*width+k]=c::rounded(float(int((r%7u*13u+k*3u)%61u)-30)/32.0f);
 }
 for(unsigned e=0;e<2;++e)for(unsigned n=0;n<columns;++n)for(unsigned k=0;k<width;++k)
  weights[(size_t(e)*columns+n)*width+k]=c::rounded(float(int((e*17u+n%11u*7u+k*5u)%47u)-23)/64.0f);
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
 Buffer<int32_t> de(experts);Buffer<uint8_t> masks(cells);Buffer<unsigned> stats(f::CounterCount),indices(size_t(kMoeCompactionBlocks)*kNativeThreads),count(1u);
 Buffer<float> output(cells);std::vector<float> control,control_routes;
 g_state.compact_routed_hawkeye=true;g_state.sm121_moe_absolute_error_ppb=512u;g_state.moe_compaction_blocks=kMoeCompactionBlocks;
 g_state.moe_compacted_indices=indices.data();g_state.moe_compacted_count=count.data();
 g_state.moe_l2[size_t(MoeL2::RoutedActivated)]=dinput.data();g_state.moe_l2[size_t(MoeL2::RoutedDown)]=dweight.data();
 const unsigned radius=mode==1?32768u:mode==2?128u:0u;
 std::vector<unsigned> actual_stats;size_t cpu_compared=0,changed_raw=0;
 for(unsigned filtered=0;filtered<2;++filtered){
  dn.reset();output.reset();
  if(filtered){hipLaunchKernelGGL(f::certify,dim3(17),dim3(256),0,nullptr,dn.data(),dt.data(),de.data(),dinput.data(),dweight.data(),512e-9f,radius,0u,ds.data(),dg.data(),masks.data(),stats.data(),tokens);ok(hipGetLastError(),"certificate");}
  using Phase=MoeCorrectionPhase;
  ok(launch_moe_routed_correction<false>(routed_down_batched_hawkeye_correction_kernel<Phase::Local>,routed_down_batched_hawkeye_correction_kernel<Phase::Collect>,routed_down_batched_hawkeye_correction_kernel<Phase::Replay>,routed_down_batched_hawkeye_correction_kernel<Phase::Local>,unsigned((outputs+255u)/256u),nullptr,MoeL2::RoutedActivated,MoeL2::RoutedDown,nullptr,filtered?masks.data():nullptr,filtered?stats.data():nullptr,dn.data(),dt.data(),de.data(),di.data(),dw.data(),routes,radius,0u),"original replay");
  hipLaunchKernelGGL(full_v3_fused_combine_residual_kernel,dim3(unsigned((cells+kNativeThreads*kFusedCombineWidth-1)/(kNativeThreads*kFusedCombineWidth))),dim3(kNativeThreads),0,nullptr,dn.data(),dt.data(),de.data(),di.data(),dw.data(),ds.data(),dg.data(),dh.data(),output.data(),true,true,true,true,true,false,0u,cells);ok(hipGetLastError(),"production combine");finish();
  auto actual=output.read(),actual_routes=dn.read();
  if(!filtered){control=actual;control_routes=actual_routes;
   for(unsigned r=0;r<routes;++r)for(unsigned n=0;n<columns;++n){const size_t index=size_t(r)*columns+n;
    if(selected(raw[index],in::multiply(topk[r],raw[index]),input_norm[r]*512e-9f*fabsf(topk[r]),radius,0u)){
     require(c::bits(actual_routes[index])==c::bits(cpu_exact[index]),"independent CPU K16");++cpu_compared;
    }
   }
  }else{
   require(!std::memcmp(actual.data(),control.data(),cells*sizeof(float)),"complete production residual");
   auto bitmap=masks.read();actual_stats=stats.read();
   unsigned selected=0,skipped=0,invariant=0;
   for(size_t index=0;index<cells;++index){unsigned mask=bitmap[index];if(mask)++invariant;skipped+=unsigned(__builtin_popcount(mask));}
   for(unsigned r=0;r<routes;++r)for(unsigned n=0;n<columns;++n){const size_t index=size_t(r)*columns+n;
    const bool chosen=selected(raw[index],in::multiply(topk[r],raw[index]),input_norm[r]*512e-9f*fabsf(topk[r]),radius,0u);
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
 }
 di.read(true);dw.read(true);ds.read(true);dt.read(true);dg.read(true);dh.read(true);de.read(true);dinput.read(true);dweight.read(true);indices.read();count.read();
 std::printf("{\"kind\":\"moe_down_consumer_filter\",\"tokens\":%u,\"mode\":%u,\"output_cells\":%zu,\"selected\":%u,\"skipped\":%u,\"replayed\":%u,\"changed_raw_route_values\":%zu,\"independent_cpu_dots\":%zu,\"production_f32_mismatches\":0,\"actual_original_replay_checked\":true,\"counter_identities_pass\":true,\"immutable_inputs\":true,\"redzones_pass\":true,\"inference_acceptance\":false}\n",tokens,mode,cells,actual_stats[f::Candidates],actual_stats[f::Omitted],actual_stats[f::Replayed],changed_raw,cpu_compared);
}
}
int main(){try{for(unsigned tokens:{1u,3u,17u,33u})for(unsigned mode=0;mode<4;++mode)test::run(tokens,mode);return 0;}catch(const std::exception& e){std::fprintf(stderr,"moe_down_consumer_filter_selftest: %s\n",e.what());return 1;}}
