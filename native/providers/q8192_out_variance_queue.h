#pragma once

// Included after the fused kernel. Long original K16 dots run independently
// of live certificate arrays and token-wide barriers. All communication stays
// on the caller's stream; the host does not read per-round candidate counts.
namespace qrt_out_variance_queue {
namespace interval=qrt_out_consumer_interval;
using Row=qrt_sm121_staged_half_projection::Row;
constexpr unsigned rows=2048u,threads=256u;
constexpr size_t additional_workspace_bytes=(size_t(8192u)*threads+8192u+size_t(rows)*8192u+1u)*4u;

__global__ __launch_bounds__(256) void initialize(
 float* bounds,uint16_t* output,const float* residual,const float* xn,const float* wn,
 unsigned tokens,unsigned radius,unsigned ppb,unsigned* states,float* maximum,
 unsigned* queue,unsigned* count,unsigned* reports) {
 const unsigned token=blockIdx.x,lane=threadIdx.x;
 if(token>=tokens)return;
 __shared__ unsigned selected_sum[threads],first_sum[threads];
 __shared__ float maximum_cost[threads];
 __shared__ unsigned local_queue[rows],local_count,global_base;
 if(!lane)local_count=0u;
 __syncthreads();
 unsigned pending=0u,selected_count=0u,first_count=0u;float largest=0.0f;
#pragma unroll
 for(unsigned item=0;item<8u;++item) {
  const unsigned row=lane*8u+item,cell=token*rows+row;
  const float center=bounds[cell],previous=device_bf16_round_to_float(residual[cell]);
  const bool selected=selected_bf16_projection_hawkeye_candidate(center,cell,rows,radius,0u,ppb,nullptr,xn,wn);
  selected_count+=selected;
  interval::Interval projected{device_bf16_round_to_float(center),device_bf16_round_to_float(center)};
  bool required=false;
  if(selected) {
   required=!interval::projection(center,xn[token]*wn[row],ppb,radius,&projected);
   if(!required) {
    const float a=__fadd_rn(previous,projected.lower),b=__fadd_rn(previous,projected.upper);
    required=!interval::finite(a)||!interval::finite(b)||interval::bf16(a)!=interval::bf16(b);
    if(!required) {
     pending|=1u<<item;
     const auto magnitude=interval::absolute_range({a,b});
     const float cost=(magnitude.upper-magnitude.lower)*(magnitude.upper+magnitude.lower);
     largest=fmaxf(largest,interval::finite(cost)&&cost>=0.0f?cost:INFINITY);
    }
   }
  }
  first_count+=required;
  output[cell]=device_float_to_bf16(center);
  // Both endpoints are BF16. Their packed bits replace the no-longer-used
  // approximate F32 matrix, while exact replays write the BF16 output buffer.
  const unsigned packed=unsigned(interval::bf16(projected.lower))|
      (unsigned(interval::bf16(projected.upper))<<16u);
  bounds[cell]=interval::value(packed);
  qrt_out_variance_replay::publish(local_queue,&local_count,cell,required);
 }
 states[token*threads+lane]=pending;
 selected_sum[lane]=selected_count;first_sum[lane]=first_count;maximum_cost[lane]=largest;
 __syncthreads();
 if(!lane)global_base=local_count?atomicAdd(count,local_count):0u;
 __syncthreads();
 for(unsigned i=lane;i<local_count;i+=threads)queue[global_base+i]=local_queue[i];
 for(unsigned stride=128u;stride;stride>>=1u) {
  if(lane<stride){selected_sum[lane]+=selected_sum[lane+stride];first_sum[lane]+=first_sum[lane+stride];maximum_cost[lane]=fmaxf(maximum_cost[lane],maximum_cost[lane+stride]);}
  __syncthreads();
 }
 if(!lane) {
  unsigned* r=reports+token*8u;
  r[0]=selected_sum[0];r[1]=first_sum[0];r[2]=0u;r[3]=r[0]-r[1];
  r[4]=r[5]=r[6]=r[7]=0u;maximum[token]=maximum_cost[0];
 }
}

__global__ __launch_bounds__(256) void replay(const Row* weight,const Row* input,
 uint16_t* output,const unsigned* queue,const unsigned* count,unsigned reduction) {
 const unsigned total=*count,groups=reduction/16u;
 for(unsigned slot=(blockIdx.x*blockDim.x+threadIdx.x)/4u;slot<total;
     slot+=gridDim.x*blockDim.x/4u) {
  const unsigned cell=queue[slot],token=cell/rows,row=cell%rows;
  const float value=qrt_sm121_staged_half_projection::dot<2u>(
      input+size_t(token)*groups,weight+size_t(row)*groups,reduction);
  if(!(threadIdx.x&3u))output[cell]=device_float_to_bf16(value);
 }
}

__global__ __launch_bounds__(256) void certify(
 const float* bounds,const uint16_t* output,const float* residual,
 const uint16_t* norm,const uint8_t* rsqrt,unsigned tokens,unsigned round,
 unsigned* states,const float* maximum,unsigned* queue,unsigned* count,unsigned* reports) {
 const unsigned token=blockIdx.x,lane=threadIdx.x;
 if(token>=tokens)return;
 unsigned* r=reports+token*8u;
 if(r[6])return;
 __shared__ unsigned reduction_scratch[threads],span,certified,boundary_failures;
 __shared__ float partial[threads],inverse_low[threads],inverse_high[threads];
 __shared__ unsigned local_queue[rows],local_count,global_base;
 if(!lane)local_count=0u;
 __syncthreads();
 float low[8],high[8],residual_value[8],importance[8];
 const unsigned prior=states[token*threads+lane],chosen=prior>>8u;
 unsigned pending=prior&255u,failures=0u;
#pragma unroll
 for(unsigned item=0u;item<8u;++item) {
  const unsigned cell=token*rows+lane*8u+item;
  const float previous=device_bf16_round_to_float(residual[cell]);
  const float actual=__fadd_rn(previous,device_bf16_to_float(output[cell]));
  const unsigned packed=interval::bits(bounds[cell]);
  const interval::Interval old_sum{
      __fadd_rn(previous,interval::value((packed&65535u)<<16u)),
      __fadd_rn(previous,interval::value(packed&0xffff0000u))};
  const auto old_magnitude=interval::absolute_range(old_sum);
  if(chosen&(1u<<item)) {
   const float magnitude=std::fabs(actual);
   failures+=!interval::finite(actual)||magnitude<old_magnitude.lower||magnitude>old_magnitude.upper||
       interval::bf16(actual)!=interval::bf16(old_sum.lower);
  }
  const bool remains=(pending&(1u<<item))&&!(chosen&(1u<<item));
  low[item]=remains?old_magnitude.lower:std::fabs(actual);
  high[item]=remains?old_magnitude.upper:std::fabs(actual);
  residual_value[item]=device_bf16_round_to_float(remains?old_sum.lower:actual);
  const float cost=(high[item]-low[item])*(high[item]+low[item]);
  importance[item]=remains?(interval::finite(cost)&&cost>=0.0f?cost:INFINITY):0.0f;
 }
 pending&=~chosen;
 reduction_scratch[lane]=failures;__syncthreads();
 for(unsigned stride=128u;stride;stride>>=1u){if(lane<stride)reduction_scratch[lane]+=reduction_scratch[lane+stride];__syncthreads();}
 if(!lane){r[7]+=reduction_scratch[0];boundary_failures=r[7];r[4]=round+1u;}
 __syncthreads();
 const float low_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(low),partial,lane);__syncthreads();
 const float high_sum=vllm_triton_reduce_sumsq(vllm_triton_lane8_sumsq(high),partial,lane);__syncthreads();
 const float low_var=__fadd_rn(low_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
 const float high_var=__fadd_rn(high_sum/float(rows),QRT_QWEN36_RMS_NORM_EPSILON);
 const bool valid=interval::finite(low_var)&&interval::finite(high_var)&&low_var>0.0f&&high_var>=low_var;
 if(!lane)span=valid?interval::bits(high_var)-interval::bits(low_var):UINT_MAX;
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
 if(!lane){certified=reduction_scratch[0]==0u;r[6]=certified;}
 __syncthreads();
 unsigned next=0u;
 if(!certified&&round<17u) {
  const float threshold=maximum[token]*interval::value((127u-round)<<23u);
#pragma unroll
  for(unsigned item=0u;item<8u;++item) {
   const bool live=(pending&(1u<<item))&&(round==16u||importance[item]>=threshold);
   if(live)next|=1u<<item;
   qrt_out_variance_replay::publish(local_queue,&local_count,token*rows+lane*8u+item,live);
  }
 }
 states[token*threads+lane]=pending|(next<<8u);
 __syncthreads();
 if(!lane) {
  global_base=local_count?atomicAdd(count,local_count):0u;
  r[2]+=local_count;r[3]=r[0]-r[1]-r[2];
  if(!certified&&round==16u)r[5]=1u;
 }
 __syncthreads();
 for(unsigned i=lane;i<local_count;i+=threads)queue[global_base+i]=local_queue[i];
}
} // namespace qrt_out_variance_queue
