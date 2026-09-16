#pragma once
#include "sm121_prefix_replay_projection.h"
#include "sm121_cooperative_norm_metadata.h"
#include "sm121_whole_dot_bound.h"

// Isolated whole-dot producer and exact continuation. Internal launches use
// nonempty dimensions, width divisible by16 and at most8192, complete operand
// and metadata spans, unique candidate IDs and full continuation capacity.
namespace qrt_sm121_whole_dot_projection {
namespace original=qrt_sm121_prefix_replay_projection;
namespace staged=qrt_sm121_staged_half_projection;
namespace bound=qrt_sm121_whole_dot_bound;
namespace meta=qrt_sm121_cooperative_norm_metadata;
using Summary=meta::Summary;
using Row=original::Row;
using Work=original::Work;
using Value=original::Value;
using B16=original::B16;
using F8=original::F8;

// Four quarter-dot summaries allow each continuation to bound only its own
// remaining matrix error. At width8192 each lane adds at most64 exact BF16
// squares and five reduction terms; gamma69 is below4.2ppm. The20ppm norm
// allowance includes this sum, square root and multiplication rounding.
__global__ void prepare(const uint16_t* input,Summary* output,unsigned rows,unsigned width){
 const unsigned lane=threadIdx.x%32u;
 const size_t entry=size_t(blockIdx.x)*(blockDim.x/32u)+threadIdx.x/32u;
 if(entry>=size_t(rows)*4u)return;
 const unsigned row=unsigned(entry/4u),part=unsigned(entry%4u),groups=width/16u;
 const unsigned first=(part*groups/4u)*16u,end=((part+1u)*groups/4u)*16u;
 float sum=0.0f;unsigned maximum=0u,invalid=0u;
 for(unsigned k=first+lane;k<end;k+=32u)meta::add_word(sum,maximum,invalid,input[size_t(row)*width+k]);
 for(unsigned offset=16u;offset;offset/=2u){
  sum+=__shfl_down(sum,offset,32);
  const unsigned other=__shfl_down(maximum,offset,32);maximum=other>maximum?other:maximum;
  invalid|=__shfl_down(invalid,offset,32);
 }
 if(!lane)output[entry]=meta::finish(sum,maximum,invalid);
}

template<unsigned Parts,bool VectorLoads=false>
__global__ __launch_bounds__(256) void produce(const uint16_t* weights,const uint16_t* inputs,
 const unsigned* weight_ok,const unsigned* input_ok,const Summary* weight_norms,const Summary* input_norms,
 float* centers,float* errors,unsigned rows,unsigned tokens,unsigned width,
 bound::State* snapshots,float* step_errors){
 static_assert(Parts==1u || Parts==2u || Parts==4u);
 const unsigned lane=threadIdx.x%32u,wave=threadIdx.x/32u,source=lane%16u;
 const unsigned row=blockIdx.x*128u+wave*16u+source,first_token=blockIdx.y*16u,groups=width/16u;
 const size_t cells=size_t(rows)*tokens;
 const bool valid_weight=row<rows&&weight_ok[row];
 const uint16_t* vector_weight=nullptr;const uint16_t* vector_input=nullptr;bool vector_input_ok=false;
 if constexpr(VectorLoads){
  const unsigned token=first_token+source;vector_input_ok=token<tokens&&input_ok[token];
  if(valid_weight)vector_weight=weights+size_t(row)*width;
  if(vector_input_ok)vector_input=inputs+size_t(token)*width;
 }
 F8 sum{},maximum{};
 for(unsigned part=0u;part<Parts;++part){
  const unsigned first=part*groups/Parts,end=(part+1u)*groups/Parts;
#pragma unroll 1
  for(unsigned group=first;group<end;++group){
   B16 w{},x{};
   if constexpr(VectorLoads){
    w=qrt_sm121_wmma_operand_load::read<B16>(vector_weight,group*16u,valid_weight);
    x=qrt_sm121_wmma_operand_load::read<B16>(vector_input,group*16u,vector_input_ok);
   }else{const unsigned token=first_token+source;const bool valid_input=token<tokens&&input_ok[token];
#pragma unroll
    for(unsigned k=0u;k<16u;++k){
     w[k]=valid_weight?weights[size_t(row)*width+group*16u+k]:0u;
     x[k]=valid_input?inputs[size_t(token)*width+group*16u+k]:0u;
    }
   }
   sum+=__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(x,w,F8{});
#pragma unroll
   for(unsigned item=0u;item<8u;++item){const float absolute=bound::scalar::absolute(sum[item]);maximum[item]=absolute>maximum[item]?absolute:maximum[item];}
  }
  if constexpr(Parts>1u)if(part+1u<Parts&&row<rows){
#pragma unroll
   for(unsigned item=0u;item<8u;++item){const unsigned token=first_token+2u*item+lane/16u;
    if(token<tokens)snapshots[size_t(part)*cells+size_t(token)*rows+row].center=sum[item];}
  }
 }
 if(row>=rows)return;
#pragma unroll
 for(unsigned item=0u;item<8u;++item){
  const unsigned token=first_token+2u*item+lane/16u;if(token>=tokens)continue;
  const size_t cell=size_t(token)*rows+row;float norms[4],absolute=0.0f,product=0.0f;
#pragma unroll
  for(unsigned part=0u;part<4u;++part){
   const auto w=weight_norms[size_t(row)*4u+part],x=input_norms[size_t(token)*4u+part];
   norms[part]=meta::bound::absolute_bound(w,x);absolute=bound::scalar::upper(absolute+norms[part]);
   const float candidate=meta::bound::product_bound(w,x);product=candidate>product?candidate:product;
  }
  auto envelope=bound::build(groups,sum[item],maximum[item],absolute,product);
  if(!valid_weight||!input_ok[token]){envelope.value.error=bound::scalar::infinity();envelope.step_error=bound::scalar::infinity();}
  centers[cell]=envelope.value.center;errors[cell]=envelope.value.error;
  if constexpr(Parts>1u){
   step_errors[cell]=envelope.step_error;
#pragma unroll
   for(unsigned part=0u;part<Parts-1u;++part){
    float remaining_norm=0.0f;
#pragma unroll
    for(unsigned segment=0u;segment<4u;++segment)if(segment>=(part+1u)*4u/Parts)
     remaining_norm=bound::scalar::upper(remaining_norm+norms[segment]);
    const unsigned remaining=groups-(part+1u)*groups/Parts;
    snapshots[size_t(part)*cells+cell].error=valid_weight&&input_ok[token]?
     bound::matrix_error(remaining_norm,remaining):bound::scalar::infinity();
   }
  }
 }
}

template<unsigned Parts,bool Audit=false>
__global__ void phase(const Row* weights,const Row* inputs,const unsigned* indices,
 const Work* previous,const float* centers,const float* step_errors,const bound::State* snapshots,
 float* output,Work* next,unsigned* next_count,unsigned rows,unsigned tokens,unsigned width,
 unsigned stage,unsigned offset,unsigned count,unsigned* finished_stage=nullptr){
 static_assert(Parts==2u || Parts==4u);
 const unsigned rank=offset+blockIdx.x*64u+threadIdx.x/4u,lane=threadIdx.x&3u;
 const bool live=rank<offset+count;
 const unsigned groups=width/16u,first=stage*groups/Parts,end=(stage+1u)*groups/Parts;
 const size_t cells=size_t(rows)*tokens;
 unsigned cell=0u;Value value{0u,-133,false};bool selected=false;
 if(live){
  if(!stage)cell=indices[rank];else{const auto item=previous[rank];cell=item.cell;value=original::carry(item);}
  const auto* w=weights+size_t(cell%rows)*groups;const auto* x=inputs+size_t(cell/rows)*groups;
#pragma unroll 1
  for(unsigned base=first;base<end;base+=2u){
   const auto a=staged::load(w[base],x[base]);
   if(base+1u<end){const auto b=staged::load(w[base+1u],x[base+1u]);value=staged::accumulate(value,a);value=staged::accumulate(value,b);}
   else value=staged::accumulate(value,a);
  }
  if(!lane){
   const float exact=qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(value));
   bool done=stage+1u==Parts;float representative=exact;
   if(!done&&(!value.significand||(value.significand>=0x800000u&&value.exponent>=-126&&value.exponent<=127))){
    const auto state=snapshots[size_t(stage)*cells+cell];
    const auto estimate=bound::suffix(centers[cell],state.center,exact,state.error,step_errors[cell],groups-end);
    done=bound::base::certified(estimate);if(done)representative=estimate.center;
   }
   if(done){output[cell]=representative;if constexpr(Audit)finished_stage[cell]=stage+1u;}
   else selected=true;
  }
 }
 if(stage+1u==Parts)return;
 const unsigned wave=threadIdx.x/32u,wlane=threadIdx.x%32u,mask=__ballot(selected);
 __shared__ unsigned offsets[8],begin;
 if(!wlane)offsets[wave]=__popc(mask);__syncthreads();
 if(!threadIdx.x){unsigned n=0u;for(unsigned i=0u;i<8u;++i){const unsigned old=offsets[i];offsets[i]=n;n+=old;}begin=atomicAdd(next_count,n);}
 __syncthreads();
 if(selected)next[begin+offsets[wave]+__popc(mask&((1u<<wlane)-1u))]=original::work(cell,value);
}
} // namespace qrt_sm121_whole_dot_projection
