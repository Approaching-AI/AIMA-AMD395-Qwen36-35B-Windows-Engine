#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_staged_half_f32_projection.h"
namespace projection_safety_test {
namespace staged_f32=qrt_sm121_staged_half_f32_projection;
using StagedF32Row=staged_f32::Row;
template<unsigned Variant>
__global__ void staged_half_f32_replay_kernel(const StagedF32Row* w,const StagedF32Row* x,
 const unsigned* indices,float* output,unsigned rows,unsigned width,unsigned count){
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
 const unsigned cell=indices[slot];const auto* left=x+size_t(cell/rows)*(width/16u);
 const auto* right=w+size_t(cell%rows)*(width/16u);float value;
 if constexpr(!Variant)value=qrt_sm121_staged_half_projection::dot<2u>(left,right,width);
 else value=staged_f32::dot<(Variant==1u?2u:4u)>(left,right,width);
 if(!(threadIdx.x&3u))output[cell]=value;
}
void run_staged_half_f32_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& selected,
 unsigned rows,unsigned tokens,unsigned width){
 const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
 constexpr unsigned marker=0xa5a5a5a5u;StagedF32Row guard;std::memset(&guard,0xa5,sizeof(guard));
 std::vector<StagedF32Row> wp(wg+2u*kGuard,guard),ip(ig+2u*kGuard,guard);DeviceBuffer<StagedF32Row> pw(wp),pi(ip);
 std::vector<unsigned> indices(selected.size()+2u*kGuard,marker);
 std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);DeviceBuffer<unsigned> ids(indices);
 std::vector<float> control;double samples[3][3]{};
 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<3u;++position){
  const unsigned variant=(position+attempt)%3u;
  hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"staged_f32_reset_output");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"staged_f32_prepare_weights");
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"staged_f32_prepare_inputs");
  if(!selected.empty()){
   const dim3 grid((unsigned(selected.size())*4u+255u)/256u);
#define QRT_STAGED_F32_CASE(v) if(variant==v)hipLaunchKernelGGL(staged_half_f32_replay_kernel<v>,grid,dim3(256u),0u,nullptr,pw.data(),pi.data(),ids.data(),dout.data(),rows,width,unsigned(selected.size()))
   QRT_STAGED_F32_CASE(0u);QRT_STAGED_F32_CASE(1u);QRT_STAGED_F32_CASE(2u);
#undef QRT_STAGED_F32_CASE
   hip_ok(hipGetLastError(),"staged_f32_replay");
  }
  complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;
  require(!control.empty(),"staged f32 original control missing");
  for(size_t cell=0u;cell<cells;++cell){
   require(std::isfinite(output[kGuard+cell]),"staged f32 nonfinite");
   require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"staged f32 raw candidate or inactive output differs");
   require(bf16(output[kGuard+cell])==reference[kGuard+cell],"staged f32 GB10 endpoint differs");
  }
  auto raw_w=weights,raw_i=inputs;dw.read(raw_w);di.read(raw_i);
  require(raw_w==weights&&raw_i==inputs,"staged f32 original operands modified");
  auto after_indices=indices;ids.read(after_indices);require(after_indices==indices,"staged f32 candidate identity changed");
  pw.read(wp);pi.read(ip);
  for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
   for(size_t group=0u;group<count;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"staged f32 prepared row changed");}
   for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&guard,sizeof(guard))&&!std::memcmp(&packed[kGuard+count+i],&guard,sizeof(guard)),"staged f32 operand guard");
  }
  for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"staged f32 output guard");
 }
 for(unsigned variant=0u;variant<3u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  std::cout<<"{\"type\":\"staged_half_f32_projection_real_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"lanes\":4,\"staging_groups\":"<<(variant==2u?4u:2u)<<",\"fp32_carry\":"<<(variant?"true":"false")
   <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_and_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
} // namespace projection_safety_test
