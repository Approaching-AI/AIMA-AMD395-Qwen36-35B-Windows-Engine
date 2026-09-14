#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_cooperative_half_projection.h"
namespace projection_safety_test {
namespace cooperative=qrt_sm121_cooperative_half_projection;
namespace cooperative_staged=qrt_sm121_staged_half_projection;
using CooperativeRow=cooperative_staged::Row;
__global__ void cooperative_control_kernel(const CooperativeRow* w,const CooperativeRow* x,const unsigned* indices,
 float* output,unsigned rows,unsigned width,unsigned count){
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
 const unsigned cell=indices[slot];const float value=cooperative_staged::dot<2u>(x+size_t(cell/rows)*(width/16u),w+size_t(cell%rows)*(width/16u),width);
 if(!(threadIdx.x&3u))output[cell]=value;
}
void run_cooperative_half_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& selected,
 unsigned rows,unsigned tokens,unsigned width){
 const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u),mw=(cells+31u)/32u;
 constexpr unsigned marker=0xa5a5a5a5u;CooperativeRow guard;std::memset(&guard,0xa5,sizeof(guard));
 std::vector<CooperativeRow> wp(wg+2u*kGuard,guard),ip(ig+2u*kGuard,guard);DeviceBuffer<CooperativeRow> pw(wp),pi(ip);
 std::vector<unsigned> indices(selected.size()+2u*kGuard,marker),mask(mw+2u*kGuard,marker),expected_mask=mask;
 std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);std::fill(expected_mask.begin()+kGuard,expected_mask.end()-kGuard,0u);
 for(unsigned cell:selected)expected_mask[kGuard+cell/32u]|=1u<<(cell&31u);
 DeviceBuffer<unsigned> ids(indices),bits(mask);std::vector<float> control;double samples[4][3]{};
 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<4u;++position){
  const unsigned variant=(position+attempt)%4u;
  hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"cooperative_reset_output");
  hip_ok(hipMemset(bits.base,0xa5,mask.size()*4u),"cooperative_reset_bitmap");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"cooperative_prepare_weights");
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"cooperative_prepare_inputs");
  if(!variant){
   if(!selected.empty()){hipLaunchKernelGGL(cooperative_control_kernel,dim3((unsigned(selected.size())*4u+255u)/256u),dim3(256u),0u,nullptr,pw.data(),pi.data(),ids.data(),dout.data(),rows,width,unsigned(selected.size()));hip_ok(hipGetLastError(),"cooperative_staged_control");}
  }else{
   hip_ok(qrt_sm121_tiled_projection::mark(ids.data(),unsigned(selected.size()),unsigned(cells),bits.data(),mw,nullptr),"cooperative_original_bitmap");
   hip_ok(cooperative::launch(pw.data(),wg,pi.data(),ig,bits.data(),mw,dout.data(),cells,rows,tokens,width,variant-1u,nullptr),"cooperative_shared_replay");
  }
  complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;require(!control.empty(),"cooperative original control missing");
  for(size_t cell=0u;cell<cells;++cell){require(std::isfinite(output[kGuard+cell]),"cooperative nonfinite");require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"cooperative raw candidate or inactive output differs");require(bf16(output[kGuard+cell])==reference[kGuard+cell],"cooperative GB10 endpoint differs");}
  auto raw_w=weights,raw_i=inputs;dw.read(raw_w);di.read(raw_i);require(raw_w==weights&&raw_i==inputs,"cooperative original operands modified");
  auto after_indices=indices;ids.read(after_indices);require(after_indices==indices,"cooperative candidate identity changed");bits.read(mask);
  require(mask==(variant?expected_mask:std::vector<unsigned>(mw+2u*kGuard,marker)),"cooperative bitmap or guard changed");
  pw.read(wp);pi.read(ip);
  for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
   for(size_t group=0u;group<count;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"cooperative prepared row changed");}
   for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&guard,sizeof(guard))&&!std::memcmp(&packed[kGuard+count+i],&guard,sizeof(guard)),"cooperative operand guard");
  }
  for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"cooperative output guard");
 }
 for(unsigned variant=0u;variant<4u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  const unsigned tile_rows=variant==1u?16u:32u,k_groups=variant==3u?16u:8u,row_tiles=(rows+tile_rows-1u)/tile_rows;
  std::vector<unsigned> counts(size_t(row_tiles)*((tokens+15u)/16u),0u);for(unsigned cell:selected)++counts[size_t((cell/rows)/16u)*row_tiles+(cell%rows)/tile_rows];
  size_t active=0u,occupied=0u;for(unsigned n:counts){active+=(n+63u)/64u;occupied+=n!=0u;}
  std::cout<<"{\"type\":\"cooperative_half_projection_real_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"tile_rows\":"<<(variant?tile_rows:0u)<<",\"tile_tokens\":"<<(variant?16u:0u)<<",\"k_tile\":"<<(variant?k_groups*16u:0u)<<",\"shared_bytes\":"<<(variant?(tile_rows+16u)*k_groups*sizeof(CooperativeRow)+65u*4u:0u)<<",\"active_ctas\":"<<(variant?active:(selected.size()+63u)/64u)<<",\"occupied_tiles\":"<<(variant?occupied:0u)<<",\"mask_bytes\":"<<(variant?mw*4u:0u)
   <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_bitmap_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
} // namespace projection_safety_test
