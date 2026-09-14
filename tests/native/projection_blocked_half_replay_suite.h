#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_blocked_half_projection.h"
namespace projection_safety_test {
namespace blocked_staged=qrt_sm121_staged_half_projection;
namespace blocked_half=qrt_sm121_blocked_half_projection;
using BlockedRow=blocked_staged::Row;
template<unsigned Variant>
__global__ void blocked_half_projection_replay_kernel(const uint16_t* weights,const uint16_t* inputs,
 const unsigned* weight_flags,const unsigned* input_flags,const BlockedRow* packed_weights,const BlockedRow* packed_inputs,
 const unsigned* indices,float* output,float* unrounded,unsigned rows,unsigned width,unsigned count){
 constexpr unsigned lanes=4u;
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/lanes;if(slot>=count)return;
 const unsigned cell=indices[slot],row=cell%rows,token=cell/rows;float value;
 if constexpr(Variant==0u)value=qrt_sm121_scalar_projection::validated_dot<4u>(inputs+size_t(token)*width,weights+size_t(row)*width,width,weight_flags[row]&&input_flags[token]);
 else if constexpr(Variant==1u)value=blocked_staged::dot<2u>(packed_inputs+size_t(token)*(width/16u),packed_weights+size_t(row)*(width/16u),width);
 else value=blocked_half::dot(packed_inputs+size_t(token)*(width/16u),packed_weights,rows,row,width);
 if(!(threadIdx.x&(lanes-1u))){unrounded[slot]=value;output[cell]=device_bf16_round_to_float(value);}
}
void run_blocked_half_projection_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& original_output,
 const std::vector<unsigned>& selected,unsigned rows,unsigned tokens,unsigned width){
 const size_t elements=size_t(rows)*tokens,wgroups=size_t(rows)*(width/16u),igroups=size_t(tokens)*(width/16u);
 const size_t weight_records=qrt_sm121_blocked_half_layout::records(rows,width);
 require(weight_records==wgroups,"captured blocked shape must need no padding");
 constexpr unsigned marker=0xa5a5a5a5u;
 std::vector<unsigned> wf(rows+2u*kGuard,marker),inf(tokens+2u*kGuard,marker),index(selected.size()+2u*kGuard,marker);
 std::copy(selected.begin(),selected.end(),index.begin()+kGuard);DeviceBuffer<unsigned> dfw(wf),dfi(inf),indices(index);
 BlockedRow row_guard;std::memset(&row_guard,0xa5,sizeof(row_guard));
 std::vector<BlockedRow> pw(wgroups+2u*kGuard,row_guard),pi(igroups+2u*kGuard,row_guard);
 DeviceBuffer<BlockedRow> dpw(pw),dpi(pi);
 std::vector<float> raw_initial(selected.size()+2u*kGuard,kF32Guard);
 std::fill(raw_initial.begin()+kGuard,raw_initial.end()-kGuard,std::numeric_limits<float>::quiet_NaN());
 DeviceBuffer<float> raw(raw_initial);std::vector<float> control,raw_control;
 double samples[3][3]{};unsigned eligible_counts[2]{};size_t unsupported_groups[2]{};
 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<3u;++position){
  const unsigned variant=(position+attempt)%3u;
  hip_ok(hipMemcpy(dout.base,original_output.data(),original_output.size()*4u,hipMemcpyHostToDevice),"blocked_half_reset_output");
  hip_ok(hipMemcpy(raw.base,raw_initial.data(),raw_initial.size()*4u,hipMemcpyHostToDevice),"blocked_half_reset_raw");
  hip_ok(hipMemset(dpw.base,0xa5,pw.size()*sizeof(BlockedRow)),"blocked_half_reset_weights");hip_ok(hipMemset(dpi.base,0xa5,pi.size()*sizeof(BlockedRow)),"blocked_half_reset_inputs");
  hip_ok(hipMemset(dfw.base,0xa5,wf.size()*4u),"blocked_half_reset_weight_flags");hip_ok(hipMemset(dfi.base,0xa5,inf.size()*4u),"blocked_half_reset_input_flags");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  if(variant){
   if(variant==2u)hip_ok(blocked_half::prepare(dw.data(),dpw.data(),weight_records,rows,width,nullptr),"blocked_half_encode_weights");
   else {hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wgroups+255u)/256u),dim3(256u),0u,nullptr,dw.data(),dpw.data(),rows,width);hip_ok(hipGetLastError(),"blocked_half_encode_weights");}
   hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((igroups+255u)/256u),dim3(256u),0u,nullptr,di.data(),dpi.data(),tokens,width);hip_ok(hipGetLastError(),"blocked_half_encode_inputs");
  }else{
   hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(rows),dim3(256u),0u,nullptr,dw.data(),dfw.data(),rows,width);hip_ok(hipGetLastError(),"blocked_half_control_weights");
   hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel,dim3(tokens),dim3(256u),0u,nullptr,di.data(),dfi.data(),tokens,width);hip_ok(hipGetLastError(),"blocked_half_control_inputs");
  }
  const unsigned lanes=4u;const dim3 grid((unsigned(selected.size())*lanes+255u)/256u);
#define QRT_BLOCKED_HALF_REPLAY_CASE(v) if(variant==v)hipLaunchKernelGGL(blocked_half_projection_replay_kernel<v>,grid,dim3(256u),0u,nullptr,dw.data(),di.data(),dfw.data(),dfi.data(),dpw.data(),dpi.data(),indices.data(),dout.data(),raw.data(),rows,width,unsigned(selected.size()))
  QRT_BLOCKED_HALF_REPLAY_CASE(0u);QRT_BLOCKED_HALF_REPLAY_CASE(1u);QRT_BLOCKED_HALF_REPLAY_CASE(2u);
#undef QRT_BLOCKED_HALF_REPLAY_CASE
  hip_ok(hipGetLastError(),"blocked_half_replay");complete_strong_projection();
  if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=original_output;dout.read(output);auto unrounded=raw_initial;raw.read(unrounded);
  if(!attempt && !variant){control=output;raw_control=unrounded;}
  require(!control.empty(),"blocked half missing original control");
  for(size_t i=0u;i<selected.size();++i){require(std::isfinite(unrounded[kGuard+i]),"blocked half unwritten raw value");require(!std::memcmp(&unrounded[kGuard+i],&raw_control[kGuard+i],4u),"blocked half unrounded candidate differs");}
  for(size_t i=0u;i<elements;++i){require(std::isfinite(output[kGuard+i]),"blocked half nonfinite output");require(!std::memcmp(&output[kGuard+i],&control[kGuard+i],4u),"blocked half corrected output differs");require(bf16(output[kGuard+i])==reference[kGuard+i],"blocked half output differs from GB10");}
  dfw.read(wf);dfi.read(inf);dpw.read(pw);dpi.read(pi);
  for(unsigned side=0u;side<2u;++side){
   const auto& source=side?inputs:weights;const auto& flags=side?inf:wf;const auto& packed=side?pi:pw;const unsigned count=side?tokens:rows;const size_t groups=size_t(count)*(width/16u);
   unsigned eligible=0u;size_t unsupported=0u;
   for(unsigned row=0u;row<count;++row){bool good=true;for(unsigned k=0u;k<width;++k){const uint16_t x=source[kGuard+size_t(row)*width+k];const unsigned e=(x>>7u)&255u;good &= !(x&0x7fffu)||(e>=64u&&e<=190u);}eligible+=good;require(flags[kGuard+row]==(variant?marker:unsigned(good)),"blocked half control flag changed");}
   for(size_t group=0u;group<groups;++group){
    const size_t index=variant==2u && !side?qrt_sm121_blocked_half_layout::offset(rows,width,unsigned(group/(width/16u)),unsigned(group%(width/16u))):group;
    require(index<groups,"blocked weight permutation bounds");
    if(variant){const auto expected=qrt_sm121_scaled_half_products::prepare(source.data()+kGuard+group*16u);require(!std::memcmp(&expected,&packed[kGuard+index],sizeof(BlockedRow)),"blocked half encoded row differs");unsupported+=qrt_sm121_scaled_half_products::unit(expected)==-32768;for(unsigned i=0u;i<16u;++i)require(qrt_sm121_scaled_half_products::original(packed[kGuard+index],i)==source[kGuard+group*16u+i],"blocked half original operand not recoverable");}
    else require(!std::memcmp(&packed[kGuard+index],&row_guard,sizeof(row_guard)),"control wrote scaled operands");
   }
   for(size_t i=0u;i<kGuard;++i){require(flags[i]==marker&&flags[kGuard+count+i]==marker,"blocked half flag guard");require(!std::memcmp(&packed[i],&row_guard,sizeof(row_guard))&&!std::memcmp(&packed[kGuard+groups+i],&row_guard,sizeof(row_guard)),"blocked half operand guard");}
   eligible_counts[side]=eligible;if(variant)unsupported_groups[side]=unsupported;
  }
  auto after_w=weights,after_i=inputs;dw.read(after_w);di.read(after_i);auto after_indices=index;indices.read(after_indices);
  require(after_w==weights&&after_i==inputs&&after_indices==index,"blocked half immutable inputs changed");
  for(size_t i=0u;i<kGuard;++i){require(output[i]==kF32Guard&&output[kGuard+elements+i]==kF32Guard,"blocked half output guard");require(unrounded[i]==kF32Guard&&unrounded[kGuard+selected.size()+i]==kF32Guard,"blocked half raw guard");}
 }
 for(unsigned variant=0u;variant<3u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  std::cout<<"{\"type\":\"blocked_half_projection_real_replay\",\"variant\":"<<variant<<",\"blocked_weights\":"<<(variant==2u)<<",\"weight_tile_rows\":32,\"weight_tile_groups\":2,\"lanes\":"<<4u<<",\"staging_groups\":"<<(variant?2u:0u)<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<elements<<",\"candidates\":"<<selected.size()
   <<",\"control_eligible_weight_rows\":"<<eligible_counts[0]<<",\"control_eligible_input_rows\":"<<eligible_counts[1]<<",\"unsupported_weight_k16_rows\":"<<unsupported_groups[0]<<",\"unsupported_input_k16_rows\":"<<unsupported_groups[1]<<",\"packed_operand_bytes\":"<<(variant?(wgroups+igroups)*sizeof(BlockedRow):0u)
   <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_and_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"]"
   <<",\"warmup_sequences\":1,\"timed_sequences\":3,\"all_attempts_verified\":true,\"rotated_variant_order\":true,\"raw_capture_writes_included\":true,\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
} // namespace projection_safety_test
