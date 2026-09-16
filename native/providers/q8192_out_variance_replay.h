#pragma once
#include "out_consumer_interval.h"

// Included after original matrix/replay kernels and the descriptor memory
// owner. Each token's CTA owns its entire residual/normalization boundary.
namespace qrt_out_variance_replay {
namespace interval=qrt_out_consumer_interval;
using Row=qrt_sm121_staged_half_projection::Row;
constexpr unsigned rows=2048u,tokens=8192u,width=4096u,fields=8u;
constexpr size_t cells=size_t(rows)*tokens;
constexpr size_t prepared_groups=size_t(rows+tokens)*(width/16u);
constexpr size_t workspace_bytes=prepared_groups*sizeof(Row)+size_t(rows+tokens+tokens*fields)*4u;

__device__ __forceinline__ void publish(unsigned* queue,unsigned* count,unsigned row,bool live) {
 const unsigned lane=threadIdx.x&31u,mask=unsigned(__ballot(live));
 unsigned base=0u;
 if(!lane&&mask)base=atomicAdd(count,unsigned(__popc(mask)));
 base=__shfl(base,0,32);
 if(live)queue[base+unsigned(__popc(mask&((uint32_t(1u)<<lane)-1u)))]=row;
}

__device__ __forceinline__ void replay_queue(const Row* weight,const Row* input,
 float* output,const unsigned* queue,unsigned count,unsigned token,unsigned reduction) {
 const unsigned groups=reduction/16u;
 for(unsigned slot=threadIdx.x/4u;slot<count;slot+=64u) {
  const unsigned row=queue[slot];
  const float value=qrt_sm121_staged_half_projection::dot<2u>(
      input+size_t(token)*groups,weight+size_t(row)*groups,reduction);
  if(!(threadIdx.x&3u))output[size_t(token)*rows+row]=device_bf16_round_to_float(value);
 }
}

// The width/count parameters permit numerical component checks at cheaper
// reductions. The product owner below admits only 2048x8192x4096.
__global__ __launch_bounds__(256) void execute(const Row* weight,const Row* input,
 float* output,const float* residual,const uint16_t* norm,const uint8_t* rsqrt,
 const float* input_norm,const float* weight_norm,unsigned token_count,unsigned reduction,
 unsigned radius,unsigned ppb,unsigned* reports) {
 const unsigned token=blockIdx.x,lane=threadIdx.x;
 if(token>=token_count)return;
 __shared__ unsigned queue[rows],queued,original_count,first_count,extra_count,certified;
 __shared__ unsigned reduction_scratch[256],rounds,complete_fallback;
 __shared__ float partial[256],inverse_low[256],inverse_high[256],maximum_cost;
 __shared__ unsigned span,boundary_failures;
 float low[8],high[8],previous[8],residual_value[8],importance[8];
 unsigned pending=0u,first=0u,local_original=0u;
 if(!lane){queued=extra_count=rounds=complete_fallback=boundary_failures=0u;}
 __syncthreads();
#pragma unroll
 for(unsigned item=0u;item<8u;++item) {
  const unsigned row=lane*8u+item,cell=token*rows+row;
  const float raw=output[cell];previous[item]=device_bf16_round_to_float(residual[cell]);
  const bool selected=selected_bf16_projection_hawkeye_candidate(
      raw,cell,rows,radius,0u,ppb,nullptr,input_norm,weight_norm);
  local_original+=selected;
  interval::Interval projected{device_bf16_round_to_float(raw),device_bf16_round_to_float(raw)};
  bool required=false;
  if(selected) {
   required=!interval::projection(raw,input_norm[token]*weight_norm[row],ppb,radius,&projected);
   if(!required) {
    const float a=__fadd_rn(previous[item],projected.lower),b=__fadd_rn(previous[item],projected.upper);
    required=!interval::finite(a)||!interval::finite(b)||interval::bf16(a)!=interval::bf16(b);
    if(!required)pending|=1u<<item;
   }
  }
  if(required)first|=1u<<item;
  low[item]=projected.lower;high[item]=projected.upper;
  publish(queue,&queued,row,required);
 }
 reduction_scratch[lane]=local_original;__syncthreads();
 for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride)reduction_scratch[lane]+=reduction_scratch[lane+stride];__syncthreads();}
 if(!lane){original_count=reduction_scratch[0];first_count=queued;}
 __syncthreads();
 replay_queue(weight,input,output,queue,queued,token,reduction);
 __syncthreads();
 float lane_max=0.0f;
#pragma unroll
 for(unsigned item=0u;item<8u;++item) {
  const unsigned cell=token*rows+lane*8u+item;
  if(first&(1u<<item))low[item]=high[item]=device_bf16_round_to_float(output[cell]);
  const interval::Interval unrounded{__fadd_rn(previous[item],low[item]),__fadd_rn(previous[item],high[item])};
  residual_value[item]=device_bf16_round_to_float(unrounded.lower);
  const auto magnitude=interval::absolute_range(unrounded);low[item]=magnitude.lower;high[item]=magnitude.upper;
  const float cost=(high[item]-low[item])*(high[item]+low[item]);
  importance[item]=(pending&(1u<<item))?(interval::finite(cost)&&cost>=0.0f?cost:INFINITY):0.0f;
  lane_max=fmaxf(lane_max,importance[item]);
 }
 inverse_high[lane]=lane_max;__syncthreads();
 for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride)inverse_high[lane]=fmaxf(inverse_high[lane],inverse_high[lane+stride]);__syncthreads();}
 if(!lane)maximum_cost=inverse_high[0];
 __syncthreads();
 for(unsigned round=0u;round<18u;++round) {
  if(round) {
   if(!lane)queued=0u;
   __syncthreads();
   const float threshold=maximum_cost*interval::value((128u-round)<<23u);
   unsigned chosen=0u;
#pragma unroll
   for(unsigned item=0u;item<8u;++item) {
    const bool live=(pending&(1u<<item))&&(round==17u||importance[item]>=threshold);
    if(live)chosen|=1u<<item;
    publish(queue,&queued,lane*8u+item,live);
   }
   __syncthreads();
   replay_queue(weight,input,output,queue,queued,token,reduction);
   __syncthreads();
#pragma unroll
   for(unsigned item=0u;item<8u;++item)if(chosen&(1u<<item)) {
    const float exact=device_bf16_round_to_float(output[token*rows+lane*8u+item]);
    const float value=__fadd_rn(previous[item],exact),magnitude=std::fabs(value);
    if(!interval::finite(value)||magnitude<low[item]||magnitude>high[item]||
       interval::bf16(value)!=interval::bf16(residual_value[item]))atomicAdd(&boundary_failures,1u);
    low[item]=high[item]=magnitude;
   }
   pending&=~chosen;
   if(!lane){extra_count+=queued;complete_fallback=round==17u;}
   __syncthreads();
  }
  const float low_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(low),partial,lane);__syncthreads();
  const float high_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(high),partial,lane);__syncthreads();
  const float low_var=__fadd_rn(low_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
  const float high_var=__fadd_rn(high_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
  const bool valid=interval::finite(low_var)&&interval::finite(high_var)&&low_var>0.0f&&high_var>=low_var;
  if(!lane){span=valid?interval::bits(high_var)-interval::bits(low_var):UINT_MAX;rounds=round+1u;}
  __syncthreads();
  float inv_lo=INFINITY,inv_hi=-INFINITY;bool finite=true;
  if(span<=4096u)for(unsigned offset=lane;offset<=span;offset+=256u) {
   const float value=device_sm121_rsqrt_from_gfx1151(interval::value(interval::bits(low_var)+offset),rsqrt);
   finite=finite&&interval::finite(value);inv_lo=fminf(inv_lo,value);inv_hi=fmaxf(inv_hi,value);
  }
  inverse_low[lane]=inv_lo;inverse_high[lane]=inv_hi;__syncthreads();
  for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride){inverse_low[lane]=fminf(inverse_low[lane],inverse_low[lane+stride]);inverse_high[lane]=fmaxf(inverse_high[lane],inverse_high[lane+stride]);}__syncthreads();}
  bool same=!boundary_failures&&span<=4096u&&finite&&interval::finite(inverse_low[0])&&interval::finite(inverse_high[0]);
#pragma unroll
  for(unsigned item=0u;item<8u;++item) {
   const float factor=1.0f+device_bf16_to_float(norm[lane*8u+item]);
   const float a=residual_value[item]*inverse_low[0]*factor,b=residual_value[item]*inverse_high[0]*factor;
   same=same&&interval::finite(a)&&interval::finite(b)&&interval::bf16(a)==interval::bf16(b);
  }
  reduction_scratch[lane]=same?0u:1u;__syncthreads();
  for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride)reduction_scratch[lane]+=reduction_scratch[lane+stride];__syncthreads();}
  if(!lane)certified=reduction_scratch[0]==0u;
  __syncthreads();if(certified)break;
 }
 // If a nonfinite consumer cannot be certified, the bounded final round has
 // replayed every original selected pending value. Keep that original route.
 if(!lane) {
  unsigned* r=reports+token*fields;
  r[0]=original_count;r[1]=first_count;r[2]=extra_count;
  r[3]=original_count-first_count-extra_count;r[4]=rounds;r[5]=complete_fallback;r[6]=certified;r[7]=boundary_failures;
 }
}

struct Stats {uint64_t selected=0,first=0,additional=0,skipped=0,rounds=0,fallback_rows=0,certified_rows=0,boundary_failures=0;double completed_ms=0.0;const char* operation="arguments";};
inline int setting(const char* value) {
 if(!value||!*value||!std::strcmp(value,"0"))return 0;
 return !std::strcmp(value,"1")?1:-1;
}
inline bool applicable(unsigned r,unsigned t,unsigned k,unsigned radius,unsigned ppb,
 bool corrected,bool consumer,bool diagnostics) {
 return r==rows&&t==tokens&&k==width&&radius==512u&&ppb==1000u&&corrected&&consumer&&!diagnostics;
}
inline bool options_compatible() {
 for(const char* name:{"QRT_QWEN36_COARSE_LINEAR_OUT_PRODUCER","QRT_QWEN36_Q8192_OUT_MATRIX_SHADOW_AUDIT",
  "QRT_QWEN36_Q8192_OUT_L1_SHADOW_AUDIT","QRT_QWEN36_Q8192_OUT_L1_BOUND","QRT_QWEN36_Q8192_OUT_RESIDUAL_FILTER",
  "QRT_QWEN36_Q8192_OUT_CONSUMER_AUDIT","QRT_QWEN36_Q8192_OUT_VARIANCE_BUDGET_AUDIT",
  "QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_BOUND"})if(setting(std::getenv(name))!=0)return false;
 unsigned algorithm=99u;
 return qrt_q8192_matrix_producer::resolve(std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_ALGORITHM"),rows,width,tokens,true,&algorithm,
     std::getenv("QRT_QWEN36_Q8192_MATRIX_PRODUCER_SCOPE"))&&algorithm==0u;
}

inline hipError_t run(const uint16_t* weights,const uint16_t* inputs,const float* residual,
 const uint16_t* norm,const uint8_t* rsqrt,uint16_t* output,float* raw,hipStream_t stream,Stats* stats) {
 if(!weights||!inputs||!residual||!norm||!rsqrt||!output||!raw||!stats||raw==residual||!options_compatible())return hipErrorInvalidValue;
 *stats=Stats{};const auto start=std::chrono::steady_clock::now();void* storage=nullptr;
 std::vector<unsigned> host(size_t(tokens)*fields);
 hipError_t status=[&]()->hipError_t {
  hipError_t result;stats->operation="allocate";
  if((result=hipMalloc(&storage,workspace_bytes))!=hipSuccess)return result;
  auto* pw=static_cast<Row*>(storage);auto* px=pw+size_t(rows)*(width/16u);
  auto* xn=reinterpret_cast<float*>(px+size_t(tokens)*(width/16u));auto* wn=xn+tokens;
  auto* reports=reinterpret_cast<unsigned*>(wn+rows);
  static_assert(sizeof(Row)==36u&&alignof(Row)<=alignof(unsigned));
  stats->operation="producer";std::string failed_stage,failure;
  if(!resident_bf16_matrix_matmul_f32_output_with_heuristic_index(weights,inputs,raw,rows,width,tokens,0u,stream,
      "linear_out_variance_replay_matrix",&failed_stage,&failure))return hipErrorInvalidConfiguration;
  stats->operation="prepare_weights";
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((size_t(rows)*(width/16u)+255u)/256u),dim3(256u),0u,stream,weights,pw,rows,width);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="prepare_inputs";
  hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,dim3((size_t(tokens)*(width/16u)+255u)/256u),dim3(256u),0u,stream,inputs,px,tokens,width);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="input_norm";
  hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(tokens),dim3(256u),0u,stream,inputs,xn,tokens,width);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="weight_norm";
  hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel,dim3(rows),dim3(256u),0u,stream,weights,wn,rows,width);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="adaptive_replay";
  hipLaunchKernelGGL(execute,dim3(tokens),dim3(256u),0u,stream,pw,px,raw,residual,norm,rsqrt,xn,wn,tokens,width,512u,1000u,reports);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="bf16_endpoint";
  hipLaunchKernelGGL(f32_to_bf16_kernel,dim3(cells/256u),dim3(256u),0u,stream,raw,output,cells);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="rounded_f32_carrier";
  hipLaunchKernelGGL(bf16_to_f32_kernel,dim3(cells/256u),dim3(256u),0u,stream,output,raw,cells);
  if((result=hipGetLastError())!=hipSuccess)return result;
  stats->operation="completion";
  if((result=hipStreamSynchronize(stream))!=hipSuccess)return result;
  stats->operation="reports";
  if((result=hipMemcpy(host.data(),reports,host.size()*sizeof(unsigned),hipMemcpyDeviceToHost))!=hipSuccess)return result;
  for(unsigned token=0u;token<tokens;++token) {
   const auto* r=host.data()+size_t(token)*fields;
   if(r[0]>rows||r[1]>r[0]||r[2]>r[0]-r[1]||r[3]!=r[0]-r[1]-r[2]||!r[4]||r[4]>18u||r[5]>1u||r[6]>1u||r[7]>r[2]||(r[7]&&r[6])||(!r[6]&&(!r[5]||r[3])))return hipErrorInvalidValue;
   stats->selected+=r[0];stats->first+=r[1];stats->additional+=r[2];stats->skipped+=r[3];stats->rounds+=r[4];stats->fallback_rows+=r[5];stats->certified_rows+=r[6];stats->boundary_failures+=r[7];
  }
  return hipSuccess;
 }();
 const hipError_t drained=hipStreamSynchronize(stream);if(status==hipSuccess)status=drained;
 if(storage)free_device(storage);
 stats->completed_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
 return status;
}
} // namespace qrt_out_variance_replay
