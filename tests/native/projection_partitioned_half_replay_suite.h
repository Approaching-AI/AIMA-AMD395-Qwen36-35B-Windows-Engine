#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_partitioned_half_projection.h"
namespace projection_safety_test {
namespace partitioned_half_projection=qrt_sm121_partitioned_half_projection;
using PartitionedHalfRow=partitioned_half_projection::Row;
template<unsigned Path>
__global__ __launch_bounds__(256) void partitioned_half_replay_kernel(const PartitionedHalfRow* w,const PartitionedHalfRow* x,
 const unsigned* wf,const unsigned* xf,const unsigned* indices,float* output,unsigned rows,unsigned width,unsigned count){
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
 const unsigned cell=indices[slot],classification=wf[cell%rows]&xf[cell/rows];
 if constexpr(Path==1u){if(!classification)return;}
 if constexpr(Path==2u){if(classification!=3u)return;}
 if constexpr(Path==3u){if(classification!=1u)return;}
 if constexpr(Path==4u){if(classification)return;}
 const auto* left=x+size_t(cell/rows)*(width/16u);const auto* right=w+size_t(cell%rows)*(width/16u);float value;
 if constexpr(Path==0u||Path==4u)value=qrt_sm121_staged_half_projection::dot<2u>(left,right,width);
 else value=partitioned_half_projection::dot<Path==2u>(left,right,width);
 if(!(threadIdx.x&3u))output[cell]=value;
}
unsigned partitioned_half_cpu_class(const uint16_t* row,unsigned width){
 unsigned result=3u;
 for(unsigned group=0u;group<width/16u;++group){
  unsigned lo=255u,hi=0u;bool nonzero=true,any=false,eligible=true;
  for(unsigned i=0u;i<16u;++i){const auto v=row[group*16u+i];if(!(v&0x7fffu)){nonzero=false;continue;}
   const unsigned exponent=(v>>7u)&255u;eligible &= exponent!=0u&&exponent!=255u;lo=std::min(lo,exponent);hi=std::max(hi,exponent);any=true;}
  eligible &= !any||hi-lo<=29u;result &= eligible?(nonzero?3u:1u):0u;
 }
 return result;
}
void run_partitioned_half_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& selected,
 unsigned rows,unsigned tokens,unsigned width){
 const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
 constexpr unsigned marker=0xa5a5a5a5u;PartitionedHalfRow guard;std::memset(&guard,0xa5,sizeof(guard));
 std::vector<PartitionedHalfRow> wp(wg+2u*kGuard,guard),ip(ig+2u*kGuard,guard);DeviceBuffer<PartitionedHalfRow> pw(wp),pi(ip);
 std::vector<unsigned> indices(selected.size()+2u*kGuard,marker);
 std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);DeviceBuffer<unsigned> ids(indices);
 std::vector<unsigned> wf(rows+2u*kGuard,marker),xf(tokens+2u*kGuard,marker),expected_w(rows),expected_x(tokens);
 DeviceBuffer<unsigned> dwf(wf),dxf(xf);size_t pair_classes[3]{};
 for(unsigned row=0u;row<rows;++row)expected_w[row]=partitioned_half_cpu_class(weights.data()+kGuard+size_t(row)*width,width);
 for(unsigned token=0u;token<tokens;++token)expected_x[token]=partitioned_half_cpu_class(inputs.data()+kGuard+size_t(token)*width,width);
 for(const unsigned cell:selected){const unsigned c=expected_w[cell%rows]&expected_x[cell/rows];++pair_classes[c==3u?2u:c==1u?1u:0u];}
 std::vector<float> control;double samples[3][3]{};
 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<3u;++position){
  const unsigned variant=(position+attempt)%3u;
  std::cout<<"{\"type\":\"partitioned_half_projection_attempt\",\"attempt\":"<<attempt<<",\"variant\":"<<variant<<",\"candidates\":"<<selected.size()<<"}"<<std::endl;
  hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"transfer_reset_output");
  hip_ok(hipMemset(dwf.base,0xa5,wf.size()*4u),"partitioned_reset_weight_flags");hip_ok(hipMemset(dxf.base,0xa5,xf.size()*4u),"partitioned_reset_input_flags");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),pw.data(),rows,width);hip_ok(hipGetLastError(),"transfer_prepare_weights");
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),pi.data(),tokens,width);hip_ok(hipGetLastError(),"transfer_prepare_inputs");
  if(variant){
   hipLaunchKernelGGL(partitioned_half_projection::classify_rows,dim3(rows),dim3(256u),0u,nullptr,pw.data(),dwf.data(),rows,width);hip_ok(hipGetLastError(),"partitioned_classify_weights");
   hipLaunchKernelGGL(partitioned_half_projection::classify_rows,dim3(tokens),dim3(256u),0u,nullptr,pi.data(),dxf.data(),tokens,width);hip_ok(hipGetLastError(),"partitioned_classify_inputs");
  }
  for(size_t offset=0u;offset<selected.size();offset+=262144u){
   const unsigned count=unsigned(std::min<size_t>(262144u,selected.size()-offset));
   const dim3 grid((count+63u)/64u);
#define QRT_PARTITIONED_HALF_PATH(p) hipLaunchKernelGGL(partitioned_half_replay_kernel<p>,grid,dim3(256u),0u,nullptr,pw.data(),pi.data(),dwf.data(),dxf.data(),ids.data()+offset,dout.data(),rows,width,count);hip_ok(hipGetLastError(),"partitioned_replay")
   if(variant==0u){QRT_PARTITIONED_HALF_PATH(0u);}
   if(variant==1u){QRT_PARTITIONED_HALF_PATH(1u);QRT_PARTITIONED_HALF_PATH(4u);}
   if(variant==2u){QRT_PARTITIONED_HALF_PATH(2u);QRT_PARTITIONED_HALF_PATH(3u);QRT_PARTITIONED_HALF_PATH(4u);}
#undef QRT_PARTITIONED_HALF_PATH
  }
  complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;
  require(!control.empty(),"partitioned half original control missing");
  for(size_t cell=0u;cell<cells;++cell){
   require(std::isfinite(output[kGuard+cell]),"partitioned half nonfinite");
   require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"partitioned half raw candidate or inactive output differs");
   require(bf16(output[kGuard+cell])==reference[kGuard+cell],"partitioned half GB10 endpoint differs");
  }
  auto raw_w=weights,raw_i=inputs;dw.read(raw_w);di.read(raw_i);
  require(raw_w==weights&&raw_i==inputs,"partitioned half original operands modified");
  auto after_indices=indices;ids.read(after_indices);require(after_indices==indices,"partitioned half candidate identity changed");
  pw.read(wp);pi.read(ip);
  dwf.read(wf);dxf.read(xf);
  for(unsigned side=0u;side<2u;++side){const auto& flags=side?xf:wf;const auto& expected=side?expected_x:expected_w;
   for(size_t row=0u;row<expected.size();++row)require(flags[kGuard+row]==(variant?expected[row]:marker),"partitioned classification differs or changed");
   for(size_t i=0u;i<kGuard;++i)require(flags[i]==marker&&flags[kGuard+expected.size()+i]==marker,"partitioned flag guard");
  }
  for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;const auto& packed=side?ip:wp;const size_t count=side?ig:wg;
   for(size_t group=0u;group<count;++group){const auto expected=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u);require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(expected)),"partitioned half prepared row changed");}
   for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&guard,sizeof(guard))&&!std::memcmp(&packed[kGuard+count+i],&guard,sizeof(guard)),"partitioned half operand guard");
  }
  for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"partitioned half output guard");
 }
 for(unsigned variant=0u;variant<3u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  std::cout<<"{\"type\":\"partitioned_half_projection_real_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"lanes\":4,\"staging_groups\":2,\"partitioned\":"<<(variant?"true":"false")
   <<",\"original_pair_count\":"<<pair_classes[0]<<",\"sparse_pair_count\":"<<pair_classes[1]<<",\"nonzero_pair_count\":"<<pair_classes[2]
   <<",\"maximum_candidates_per_dispatch\":262144,\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_and_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"independent_row_classification_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
}
