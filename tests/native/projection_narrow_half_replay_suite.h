#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_narrow_half_projection.h"
namespace projection_safety_test {
namespace narrow_half=qrt_sm121_narrow_half_projection;
using NarrowHalfRow=narrow_half::Row;
template<unsigned Variant>
__global__ void narrow_half_replay_kernel(const NarrowHalfRow* w,const NarrowHalfRow* x,
 const unsigned* wf,const unsigned* xf,const unsigned* indices,float* output,unsigned rows,unsigned width,unsigned count){
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
 const unsigned cell=indices[slot];const auto* left=x+size_t(cell/rows)*(width/16u);
 const auto* right=w+size_t(cell%rows)*(width/16u);float value;
 if constexpr(!Variant)value=qrt_sm121_staged_half_projection::dot<2u>(left,right,width);
 else value=narrow_half::dot<(Variant==1u?2u:4u)>(left,right,width,wf[cell%rows]&&xf[cell/rows]);
 if(!(threadIdx.x&3u))output[cell]=value;
}
void run_narrow_half_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& selected,
 unsigned rows,unsigned tokens,unsigned width){
 const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
 constexpr unsigned marker=0xa5a5a5a5u;NarrowHalfRow guard;std::memset(&guard,0xa5,sizeof(guard));
 std::vector<NarrowHalfRow> wp(wg+2u*kGuard,guard),ip(ig+2u*kGuard,guard);DeviceBuffer<NarrowHalfRow> pw(wp),pi(ip);
 std::vector<unsigned> indices(selected.size()+2u*kGuard,marker);
 std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);DeviceBuffer<unsigned> ids(indices);
 std::vector<unsigned> wh(rows+2u*kGuard,marker),xh(tokens+2u*kGuard,marker);DeviceBuffer<unsigned> fw(wh),fx(xh);
 size_t fast_candidates=0u,fast_rows[2]{};
 std::vector<float> control;double samples[3][3]{};
 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<3u;++position){
  const unsigned variant=(position+attempt)%3u;
  hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"narrow_half_reset_output");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"narrow_half_prepare_weights");
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"narrow_half_prepare_inputs");
  if(variant){
   hipLaunchKernelGGL(narrow_half::classify_rows,dim3(rows),dim3(128u),0u,nullptr,dw.data(),pw.data(),fw.data(),rows,width);hip_ok(hipGetLastError(),"narrow_half_classify_weights");
   hipLaunchKernelGGL(narrow_half::classify_rows,dim3(tokens),dim3(128u),0u,nullptr,di.data(),pi.data(),fx.data(),tokens,width);hip_ok(hipGetLastError(),"narrow_half_classify_inputs");
  }
  if(!selected.empty()){
   const dim3 grid((unsigned(selected.size())*4u+255u)/256u);
#define QRT_NARROW_HALF_CASE(v) if(variant==v)hipLaunchKernelGGL(narrow_half_replay_kernel<v>,grid,dim3(256u),0u,nullptr,pw.data(),pi.data(),fw.data(),fx.data(),ids.data(),dout.data(),rows,width,unsigned(selected.size()))
   QRT_NARROW_HALF_CASE(0u);QRT_NARROW_HALF_CASE(1u);QRT_NARROW_HALF_CASE(2u);
#undef QRT_NARROW_HALF_CASE
   hip_ok(hipGetLastError(),"narrow_half_replay");
  }
  complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;
  require(!control.empty(),"narrow half original control missing");
  for(size_t cell=0u;cell<cells;++cell){
   require(std::isfinite(output[kGuard+cell]),"narrow half nonfinite");
   require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"narrow half raw candidate or inactive output differs");
   require(bf16(output[kGuard+cell])==reference[kGuard+cell],"narrow half GB10 endpoint differs");
  }
  auto raw_w=weights,raw_i=inputs;dw.read(raw_w);di.read(raw_i);
  require(raw_w==weights&&raw_i==inputs,"narrow half original operands modified");
  auto after_indices=indices;ids.read(after_indices);require(after_indices==indices,"narrow half candidate identity changed");
  pw.read(wp);pi.read(ip);
  for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
   for(size_t group=0u;group<count;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"narrow half prepared row changed");}
   for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&guard,sizeof(guard))&&!std::memcmp(&packed[kGuard+count+i],&guard,sizeof(guard)),"narrow half operand guard");
  }
  if(variant){
   fw.read(wh);fx.read(xh);fast_rows[0]=fast_rows[1]=0u;
   for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& flags=side?xh:wh;const unsigned row_count=side?tokens:rows;
    for(unsigned row=0u;row<row_count;++row){bool eligible=width<=8192u;
     for(unsigned g=0u;g<width/16u;++g){unsigned low=255u,high=0u;
      for(unsigned i=0u;i<16u;++i){const uint16_t x=raw[kGuard+size_t(row)*width+g*16u+i];if(x&0x7fffu){const unsigned e=(x>>7u)&255u;eligible&=e>=95u&&e<=159u;low=std::min(low,e);high=std::max(high,e);}}
      eligible&=!high||high-low<=29u;
     }
     require(flags[kGuard+row]==unsigned(eligible),"narrow half whole-row flag differs");fast_rows[side]+=eligible;
    }
    for(size_t i=0u;i<kGuard;++i)require(flags[i]==marker&&flags[kGuard+row_count+i]==marker,"narrow half flag guard");
   }
   fast_candidates=0u;for(unsigned cell:selected)fast_candidates+=wh[kGuard+cell%rows]&&xh[kGuard+cell/rows];
  }
  for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"narrow half output guard");
 }
 for(unsigned variant=0u;variant<3u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  std::cout<<"{\"type\":\"narrow_half_projection_real_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"lanes\":4,\"staging_groups\":"<<(variant==2u?4u:2u)<<",\"fp32_carry\":"<<(variant?"true":"false")<<",\"admitted_candidates\":"<<(variant?fast_candidates:0u)<<",\"original_candidates\":"<<(variant?selected.size()-fast_candidates:selected.size())<<",\"admitted_weight_rows\":"<<fast_rows[0]<<",\"admitted_input_rows\":"<<fast_rows[1]
   <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_classification_and_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true,\"whole_row_flags_checked\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
} // namespace projection_safety_test
