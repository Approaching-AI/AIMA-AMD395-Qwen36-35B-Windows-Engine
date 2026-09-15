#pragma once
#include "out_consumer_interval.h"

namespace qrt_out_consumer_audit {
namespace interval=qrt_out_consumer_interval;
constexpr unsigned rows=2048u,tokens=8192u,width=4096u,guard=128u,fields=15u,maximum_variance_span=4096u;
// Diagnostic classification only. For cells whose residual cannot be proved
// invariant, the already completed original output stands in for a prospective
// first replay pass. That pass is NOT executed or timed here. All model-owned
// buffers are read-only; no external oracle value enters this kernel.
__global__ __launch_bounds__(256) void classify(const float* before,const uint16_t* control,
 const float* residual,const uint16_t* norm,const uint8_t* rsqrt,
 const float* input_norm,const float* weight_norm,unsigned radius,unsigned ppb,
 bool exact_terminal,unsigned* reports) {
 const unsigned token=blockIdx.x,lane=threadIdx.x;
 __shared__ float partial[256],inverse_low[256],inverse_high[256];
 __shared__ unsigned reduction[256],row_report[fields],span;
 float lo_abs[8],hi_abs[8],original_unrounded[8],residual_value[8],original_residual[8];
 unsigned local[fields]{};bool row_norm_same=true;
 for(unsigned item=0u;item<8u;++item) {
  const unsigned row=lane*8u+item,cell=token*rows+row;
  const float raw=before[cell],exact=interval::value(uint32_t(control[cell])<<16u);
  const float previous=device_bf16_round_to_float(residual[cell]);
  const float original=__fadd_rn(previous,exact);
  original_unrounded[item]=original;original_residual[item]=device_bf16_round_to_float(original);
  const bool selected=(exact_terminal&&token==tokens-1u)||selected_bf16_projection_hawkeye_candidate(
   raw,cell,rows,radius,0u,ppb,nullptr,input_norm,weight_norm);
  local[0]+=selected;local[4]+=!selected&&interval::bf16(raw)!=control[cell];
  local[6]+=!interval::finite(raw)||!interval::finite(exact)||!interval::finite(previous);
  interval::Interval projected{device_bf16_round_to_float(raw),device_bf16_round_to_float(raw)};
  bool collapse=false;
  if(selected) {
   if((exact_terminal&&token==tokens-1u)||!interval::projection(raw,input_norm[token]*weight_norm[row],ppb,radius,&projected))collapse=true;
   else {
    local[5]+=exact<projected.lower||exact>projected.upper;
    const float a=__fadd_rn(previous,projected.lower),b=__fadd_rn(previous,projected.upper);
    collapse=interval::bf16(a)!=interval::bf16(b);
    if(!collapse) {
     if(interval::bits(projected.lower)==interval::bits(projected.upper))++local[2];
     else ++local[3];
    }
   }
   if(collapse){++local[1];projected={exact,exact};}
  }
  const interval::Interval unrounded{__fadd_rn(previous,projected.lower),__fadd_rn(previous,projected.upper)};
  local[7]+=interval::bf16(unrounded.lower)!=interval::bf16(unrounded.upper);
  const auto magnitude=interval::absolute_range(unrounded);
  lo_abs[item]=magnitude.lower;hi_abs[item]=magnitude.upper;
  residual_value[item]=device_bf16_round_to_float(unrounded.lower);
  local[11]+=interval::bits(residual_value[item])!=interval::bits(original_residual[item]);
 }
 // Reuse the actual consumer's FMA/add sequence and reduction tree. Taking
 // absolute endpoints preserves bounds through every nonnegative operation.
 const float low_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(lo_abs),partial,lane);__syncthreads();
 const float high_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(hi_abs),partial,lane);__syncthreads();
 const float actual_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(original_unrounded),partial,lane);__syncthreads();
 const float low_var=__fadd_rn(low_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
 const float high_var=__fadd_rn(high_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
 const float actual_var=__fadd_rn(actual_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
 const bool valid_variance=interval::finite(low_var)&&interval::finite(high_var)&&low_var>0.0f&&high_var>=low_var;
 if(!lane) {
  span=valid_variance?interval::bits(high_var)-interval::bits(low_var):UINT_MAX;
  row_report[8]=!interval::finite(actual_var)||actual_var<low_var||actual_var>high_var;
  row_report[12]=span>maximum_variance_span;row_report[13]=span;
 }
 __syncthreads();
 float lower_inverse=INFINITY,upper_inverse=-INFINITY;bool inverse_finite=true;
 // Enumerate every actual float variance in the bounded range. No monotonic
 // assumption about the hardware reciprocal-square-root or correction table
 // is required. Wide ranges retain prospective original replay for the row.
 if(span<=maximum_variance_span)for(unsigned offset=lane;offset<=span;offset+=256u) {
  const float inverse=device_sm121_rsqrt_from_gfx1151(interval::value(interval::bits(low_var)+offset),rsqrt);
  inverse_finite=inverse_finite&&interval::finite(inverse);
  lower_inverse=fminf(lower_inverse,inverse);upper_inverse=fmaxf(upper_inverse,inverse);
 }
 inverse_low[lane]=lower_inverse;inverse_high[lane]=upper_inverse;__syncthreads();
 for(unsigned stride=128u;stride;stride>>=1u) {
  if(lane<stride){inverse_low[lane]=fminf(inverse_low[lane],inverse_low[lane+stride]);inverse_high[lane]=fmaxf(inverse_high[lane],inverse_high[lane+stride]);}
  __syncthreads();
 }
 const float actual_inverse=device_sm121_rsqrt_from_gfx1151(actual_var,rsqrt);
 row_norm_same=span<=maximum_variance_span&&inverse_finite&&interval::finite(inverse_low[0])&&interval::finite(inverse_high[0]);
 for(unsigned item=0u;item<8u;++item) {
  const float factor=1.0f+device_bf16_to_float(norm[lane*8u+item]);
  const float a=residual_value[item]*inverse_low[0]*factor,b=residual_value[item]*inverse_high[0]*factor;
  const float actual=original_residual[item]*actual_inverse*factor;
  const bool same=interval::finite(a)&&interval::finite(b)&&interval::bf16(a)==interval::bf16(b);
  row_norm_same=row_norm_same&&same;
  local[10]+=same&&interval::bf16(a)!=interval::bf16(actual);
 }
 local[9]=row_norm_same?0u:1u;
 for(unsigned field=0u;field<fields;++field) {
  if(field==8u||field==12u||field==13u||field==14u)continue;
  reduction[lane]=local[field];__syncthreads();
  for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride)reduction[lane]+=reduction[lane+stride];__syncthreads();}
  if(!lane)row_report[field]=reduction[0];__syncthreads();
 }
 if(!lane){row_report[9]=row_report[9]==0u;row_report[14]=row_report[9]?row_report[3]:0u;}
 __syncthreads();if(lane<fields)reports[token*fields+lane]=row_report[lane];
}

inline hipError_t run(const uint16_t* weights,const uint16_t* inputs,const uint16_t* control,
 const float* residual,const uint16_t* norm,const uint8_t* rsqrt,unsigned token_count,
 unsigned radius,unsigned ppb,hipStream_t stream,const std::string& stage,bool compatible,bool exact_terminal=false) {
 const char* setting=std::getenv("QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT");
 if(!setting||!*setting||!std::strcmp(setting,"0"))return hipSuccess;
 if(std::strcmp(setting,"1"))return hipErrorInvalidValue;
 if(token_count!=tokens)return hipSuccess;
 if(!compatible||!weights||!inputs||!control||!residual||!norm||!rsqrt||!ppb||radius>32768u)return hipErrorInvalidValue;
#if !defined(QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER)
 return hipErrorInvalidConfiguration;
#else
 unsigned choice=99u;
 if(!qrt_q8192_matrix_producer::resolve(std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM"),rows,width,tokens,true,&choice,
      std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE"))||choice!=0u||qrt_out_l1_policy::selected_factor(std::getenv("QRT_QWEN36_Q8192_OUT_L1_BOUND"))!=0)
  return hipErrorInvalidValue;
 const std::array<size_t,3> counts={size_t(rows)*tokens,size_t(rows+tokens),size_t(tokens)*fields};
 std::array<void*,3> owned{};std::vector<unsigned> host(counts[2]);
 const auto start=std::chrono::steady_clock::now();
 auto complete=[&](){const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){const auto s=hipStreamQuery(stream);if(s!=hipErrorNotReady)return s;if(std::chrono::steady_clock::now()>=deadline)return hipErrorLaunchTimeOut;std::this_thread::yield();}};
 hipError_t status=complete();if(status!=hipSuccess)return status;
 status=[&]()->hipError_t {
  hipError_t s;
  for(unsigned i=0u;i<3u;++i){const size_t bytes=(counts[i]+2u*guard)*4u;if((s=hipMalloc(&owned[i],bytes))!=hipSuccess)return s;if((s=hipMemsetAsync(owned[i],0xa5,bytes,stream))!=hipSuccess)return s;}
  auto* raw=static_cast<float*>(owned[0])+guard;auto* xn=static_cast<float*>(owned[1])+guard;auto* wn=xn+tokens;auto* reports=static_cast<unsigned*>(owned[2])+guard;
  std::string failed_stage,failure;
  if(!resident_bf16_matrix_matmul_f32_output_with_heuristic_index(weights,inputs,raw,rows,width,tokens,0u,stream,stage+"_consumer_original_matrix",&failed_stage,&failure))return hipErrorInvalidConfiguration;
  hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(tokens),dim3(256u),0u,stream,inputs,xn,tokens,width);if((s=hipGetLastError())!=hipSuccess)return s;
  hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(rows),dim3(256u),0u,stream,weights,wn,rows,width);if((s=hipGetLastError())!=hipSuccess|| (s=complete())!=hipSuccess)return s;
  hipLaunchKernelGGL(classify,dim3(tokens),dim3(256u),0u,stream,raw,control,residual,norm,rsqrt,xn,wn,radius,ppb,exact_terminal,reports);
  if((s=hipGetLastError())!=hipSuccess||(s=complete())!=hipSuccess||(s=hipMemcpy(host.data(),reports,host.size()*4u,hipMemcpyDeviceToHost))!=hipSuccess)return s;
  std::array<unsigned char,guard*4u> redzone{};
  for(unsigned i=0u;i<3u;++i)for(size_t offset:{size_t(0u),(counts[i]+guard)*4u}) {
   if((s=hipMemcpy(redzone.data(),static_cast<unsigned char*>(owned[i])+offset,redzone.size(),hipMemcpyDeviceToHost))!=hipSuccess)return s;
   for(auto byte:redzone)if(byte!=0xa5u)return hipErrorInvalidValue;
  }
  return hipSuccess;
 }();
 const auto drained=hipStreamSynchronize(stream);if(status==hipSuccess)status=drained;
 for(auto allocation:owned)if(allocation){const auto freed=hipFree(allocation);if(status==hipSuccess)status=freed;}
 if(status!=hipSuccess)return status;
 std::array<uint64_t,fields> totals{};unsigned max_span=0u;
 for(unsigned token=0u;token<tokens;++token)for(unsigned field=0u;field<fields;++field){totals[field]+=host[token*fields+field];if(field==13u)max_span=(std::max)(max_span,host[token*fields+field]);}
 const size_t workspace=(counts[0]+counts[1]+counts[2]+6u*guard)*4u;
 const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 std::fprintf(stderr,"BATCH_MARK out_consumer_audit stage=%s rows=%u tokens=%u k=%u radius=%u ppb=%u original_candidates=%llu prospective_first_replay=%llu projection_constant=%llu residual_constant_pending=%llu unselected_differences=%llu projection_undercoverage=%llu nonfinite=%llu residual_interval_failures=%llu variance_undercoverage=%llu normalized_rows_certified=%llu normalized_certificate_false_cells=%llu residual_certificate_false_cells=%llu variance_span_rejected_rows=%llu maximum_variance_span=%u residual_pending_certified=%llu prospective_total_replay=%llu workspace_bytes=%zu audit_wall_ms=%.6f existing_control_supplies_hypothetical_first_pass=1 candidate_replay_executed=0 inference_inputs_unchanged=1 redzones_pass=1 completed=1\n",
  stage.c_str(),rows,tokens,width,radius,ppb,(unsigned long long)totals[0],(unsigned long long)totals[1],(unsigned long long)totals[2],(unsigned long long)totals[3],(unsigned long long)totals[4],(unsigned long long)totals[5],(unsigned long long)totals[6],(unsigned long long)totals[7],(unsigned long long)totals[8],(unsigned long long)totals[9],(unsigned long long)totals[10],(unsigned long long)totals[11],(unsigned long long)totals[12],max_span,(unsigned long long)totals[14],(unsigned long long)(totals[1]+totals[3]-totals[14]),workspace,wall);
 std::fflush(stderr);return hipSuccess;
#endif
}
} // namespace qrt_out_consumer_audit
