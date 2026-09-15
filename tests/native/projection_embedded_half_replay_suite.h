#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_embedded_half_projection.h"
namespace projection_safety_test {
namespace embedded=qrt_sm121_embedded_half_projection;
using EmbeddedRow=embedded::Row;
using OriginalHalfRow=qrt_sm121_staged_half_projection::Row;
template<unsigned Variant>
__global__ void embedded_half_replay_kernel(const OriginalHalfRow* old_w,const OriginalHalfRow* old_x,
 const EmbeddedRow* w,const EmbeddedRow* x,const uint16_t* raw_w,const uint16_t* raw_x,
 const unsigned* wf,const unsigned* xf,const unsigned* indices,float* output,
 unsigned rows,unsigned width,unsigned count){
 const unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;if(slot>=count)return;
 const unsigned cell=indices[slot],row=cell%rows,token=cell/rows,groups=width/16u;
 float value;
 if constexpr(!Variant)value=qrt_sm121_staged_half_projection::dot<2u>(old_x+size_t(token)*groups,old_w+size_t(row)*groups,width);
 else{
  const unsigned words=qrt_sm121_embedded_half::flag_words(width);
  value=embedded::dot<(Variant==1u?2u:4u)>(x+size_t(token)*groups,w+size_t(row)*groups,
   raw_x+size_t(token)*width,raw_w+size_t(row)*width,xf+size_t(token)*words,wf+size_t(row)*words,width);
 }
 if(!(threadIdx.x&3u))output[cell]=value;
}
void run_embedded_half_replays(DeviceBuffer<uint16_t>& dw,DeviceBuffer<uint16_t>& di,
 DeviceBuffer<float>& dout,const std::vector<uint16_t>& weights,const std::vector<uint16_t>& inputs,
 const std::vector<uint16_t>& reference,const std::vector<float>& initial,const std::vector<unsigned>& selected,
 unsigned rows,unsigned tokens,unsigned width){
 const size_t cells=size_t(rows)*tokens,wg=size_t(rows)*(width/16u),ig=size_t(tokens)*(width/16u);
 const unsigned words=qrt_sm121_embedded_half::flag_words(width);
 constexpr unsigned marker=0xa5a5a5a5u;EmbeddedRow compact_guard;OriginalHalfRow original_guard;
 std::memset(&compact_guard,0xa5,sizeof(compact_guard));std::memset(&original_guard,0xa5,sizeof(original_guard));
 std::vector<EmbeddedRow> wp(wg+2u*kGuard,compact_guard),ip(ig+2u*kGuard,compact_guard);
 std::vector<OriginalHalfRow> wo(wg+2u*kGuard,original_guard),io(ig+2u*kGuard,original_guard);
 std::vector<unsigned> wf(size_t(rows)*words+2u*kGuard,marker),inf(size_t(tokens)*words+2u*kGuard,marker);
 DeviceBuffer<EmbeddedRow> pw(wp),pi(ip);DeviceBuffer<OriginalHalfRow> ow(wo),oi(io);
 DeviceBuffer<unsigned> fw(wf),fi(inf);
 // Independent host encoding is a verification oracle only, never GPU input.
 for(unsigned side=0u;side<2u;++side){const auto& raw=side?inputs:weights;auto& packed=side?ip:wp;auto& old=side?io:wo;auto& flags=side?inf:wf;const unsigned count=side?tokens:rows;
  std::fill(flags.begin()+kGuard,flags.end()-kGuard,0u);
  for(unsigned row=0u;row<count;++row){bool all=true;for(unsigned group=0u;group<width/16u;++group){
   const size_t at=size_t(row)*(width/16u)+group;old[kGuard+at]=qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+at*16u);
   const bool valid=qrt_sm121_embedded_half::prepare(raw.data()+kGuard+at*16u,&packed[kGuard+at]);all&=valid;
   if(valid)flags[kGuard+size_t(row)*words+1u+group/32u]|=1u<<(group&31u);
  }flags[kGuard+size_t(row)*words]=all;}
 }
 auto prepare_original=[&](){
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((wg+255u)/256u),dim3(256u),0u,nullptr,dw.data(),ow.data(),rows,width);hip_ok(hipGetLastError(),"embedded_control_weights");
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((ig+255u)/256u),dim3(256u),0u,nullptr,di.data(),oi.data(),tokens,width);hip_ok(hipGetLastError(),"embedded_control_inputs");
 };
 auto prepare_compact=[&](){
  hip_ok(embedded::prepare(dw.data(),size_t(rows)*width,pw.data(),wg,fw.data(),size_t(rows)*words,rows,width,nullptr),"embedded_weights");
  hip_ok(embedded::prepare(di.data(),size_t(tokens)*width,pi.data(),ig,fi.data(),size_t(tokens)*words,tokens,width,nullptr),"embedded_inputs");
 };
 prepare_original();prepare_compact();complete_strong_projection();
 std::vector<unsigned> indices(selected.size()+2u*kGuard,marker);
 std::copy(selected.begin(),selected.end(),indices.begin()+kGuard);DeviceBuffer<unsigned> ids(indices);
 auto verify_buffer=[](auto& device,const auto& expected){auto actual=expected;device.read(actual);require(!std::memcmp(actual.data(),expected.data(),expected.size()*sizeof(expected[0])),"embedded prepared operand, bitmap, padding or guard changed");};
 std::vector<float> control;double samples[3][3]{};
 for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<3u;++position){
  const unsigned variant=(position+attempt)%3u;
  hip_ok(hipMemcpy(dout.base,initial.data(),initial.size()*4u,hipMemcpyHostToDevice),"embedded_reset_output");
  complete_strong_projection();const auto start=std::chrono::steady_clock::now();
  if(variant)prepare_compact();else prepare_original();
  if(!selected.empty()){
   const dim3 grid((unsigned(selected.size())*4u+255u)/256u);
#define QRT_EMBEDDED_CASE(v) if(variant==v)hipLaunchKernelGGL(embedded_half_replay_kernel<v>,grid,dim3(256u),0u,nullptr,ow.data(),oi.data(),pw.data(),pi.data(),dw.data(),di.data(),fw.data(),fi.data(),ids.data(),dout.data(),rows,width,unsigned(selected.size()))
   QRT_EMBEDDED_CASE(0u);QRT_EMBEDDED_CASE(1u);QRT_EMBEDDED_CASE(2u);
#undef QRT_EMBEDDED_CASE
   hip_ok(hipGetLastError(),"embedded_replay");
  }
  complete_strong_projection();if(attempt)samples[variant][attempt-1u]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  auto output=initial;dout.read(output);if(!attempt&&!variant)control=output;
  require(!control.empty(),"embedded original control missing");
  for(size_t cell=0u;cell<cells;++cell){
   require(std::isfinite(output[kGuard+cell]),"embedded nonfinite");
   require(!std::memcmp(&output[kGuard+cell],&control[kGuard+cell],4u),"embedded raw candidate or inactive output differs");
   require(bf16(output[kGuard+cell])==reference[kGuard+cell],"embedded GB10 endpoint differs");
  }
  verify_buffer(dw,weights);verify_buffer(di,inputs);verify_buffer(ids,indices);
  verify_buffer(pw,wp);verify_buffer(pi,ip);verify_buffer(ow,wo);verify_buffer(oi,io);verify_buffer(fw,wf);verify_buffer(fi,inf);
  for(size_t i=0u;i<kGuard;++i)require(output[i]==kF32Guard&&output[kGuard+cells+i]==kF32Guard,"embedded output guard");
 }
 for(unsigned variant=0u;variant<3u;++variant){std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(ordered.begin(),ordered.end());
  const size_t prepared_bytes=(wg+ig)*(variant?sizeof(EmbeddedRow):sizeof(OriginalHalfRow))+(variant?size_t(rows+tokens)*words*4u:0u);
  std::cout<<"{\"type\":\"embedded_half_projection_real_replay\",\"variant\":"<<variant<<",\"rows\":"<<rows<<",\"tokens\":"<<tokens<<",\"k\":"<<width<<",\"elements\":"<<cells<<",\"candidates\":"<<selected.size()<<",\"lanes\":4,\"staging_groups\":"<<(variant==2u?4u:2u)<<",\"embedded_metadata\":"<<(variant?"true":"false")<<",\"prepared_operand_and_flag_bytes\":"<<prepared_bytes
   <<",\"raw_bit_mismatches\":0,\"bf16_mismatches\":0,\"unrounded_candidate_bit_mismatches\":0,\"preparation_and_replay_host_ms\":"<<ordered[1]<<",\"completed_host_samples_ms\":["<<samples[variant][0]<<","<<samples[variant][1]<<","<<samples[variant][2]<<"],\"warmup_sequences\":1,\"timed_sequences\":3,\"rotated_variant_order\":true,\"all_attempts_verified\":true,\"all_encoded_words_checked\":true,\"support_bitmap_and_padding_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}"<<std::endl;
 }
}
} // namespace projection_safety_test
