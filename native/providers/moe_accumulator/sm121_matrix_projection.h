#pragma once
#include <hip/hip_runtime.h>
#include "sm121_integer_core.h"
#include "sm121_wave16.h"

// Component only. Each CTA owns a16x16 projection tile. Whole waves produce
// independent K16 IU8 products before one scalar lane per cell consumes them
// in original order. Encode raw BF16 directly into LDS: there is no global
// expanded operand allocation or preparation pass. Original selector bits
// remain authoritative and inactive outputs are untouched.
namespace qrt_sm121_matrix_projection {
namespace core=qrt_sm121_integer_core;
using Value=qrt_q1_moe_hawkeye::Value;
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
constexpr unsigned threads=256u;
template<unsigned Groups,bool Trace>
__global__ __launch_bounds__(256) void replay_kernel(const uint16_t* weights,const uint16_t* inputs,
 const unsigned* mask,float* output,uint32_t* trace,unsigned rows,unsigned tokens,unsigned width){
 static_assert(Groups==4u || Groups==8u);
 __shared__ core::Row operands[Groups][32];
 __shared__ int64_t products[Groups][256];
 __shared__ unsigned any;
 const unsigned tid=threadIdx.x,lane=tid%32u,wave=tid/32u;
 const unsigned row_tiles=(rows+15u)/16u,first_row=(blockIdx.x%row_tiles)*16u,first_token=(blockIdx.x/row_tiles)*16u;
 const unsigned row=first_row+tid%16u,token=first_token+tid/16u,total_groups=width/16u;
 const size_t cell=size_t(token)*rows+row;
 const bool active=row<rows && token<tokens && ((mask[cell/32u]>>(cell&31u))&1u);
 if(!tid)any=0u;
 __syncthreads();if(active)atomicOr(&any,1u);__syncthreads();if(!any)return;
 Value carry{0u,-133,false};
 for(unsigned base=0u;base<total_groups;base+=Groups){
  const unsigned count=total_groups-base<Groups?total_groups-base:Groups;
  for(unsigned item=tid;item<count*32u;item+=threads){
   const unsigned local=item/32u,side=item%32u,index=side%16u;
   const unsigned source_row=side<16u?first_token+index:first_row+index;
   const bool valid=source_row<(side<16u?tokens:rows);
   core::Row packed{};
#pragma unroll
   for(unsigned i=0u;i<16u;++i)if(valid)packed.original[i]=(side<16u?inputs:weights)[size_t(source_row)*width+(base+local)*16u+i];
   core::prepare(packed);operands[local][side]=packed;
  }
  __syncthreads();
  for(unsigned local=wave;local<count;local+=8u){
   const auto& a=operands[local][lane%16u];const auto& b=operands[local][16u+lane%16u];
   I4 ah{},al{},bh{},bl{};
#pragma unroll
   for(unsigned i=0u;i<4u;++i){ah[i]=a.high[i];al[i]=a.low[i];bh[i]=b.high[i];bl[i]=b.low[i];}
   const I8 zero{};
   const auto hh=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,true,bh,zero,false);
   const auto hl=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true,ah,false,bl,zero,false);
   const auto lh=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,true,bh,zero,false);
   const auto ll=__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,al,false,bl,zero,false);
#pragma unroll
   for(unsigned i=0u;i<8u;++i){const unsigned index=(2u*i+lane/16u)*16u+lane%16u;
    products[local][index]=int64_t(hh[i])*65536+(int64_t(hl[i])+lh[i])*256+ll[i];}
  }
  __syncthreads();
  if(active)for(unsigned local=0u;local<count;++local){
   const auto& a=operands[local][tid/16u];const auto& b=operands[local][16u+tid%16u];
   qrt_sm121_group16::AlignedSum sum;
   if(!core::sum_integer_product(carry,a,b,products[local][tid],&sum)){
    uint32_t raw[16];
#pragma unroll
    for(unsigned i=0u;i<16u;++i)raw[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a.original[i],b.original[i],-133));
    sum=qrt_sm121_group16::sum_packed(carry,raw);
   }
   carry=qrt_sm121_wave16::normalize(sum.value.magnitude,sum.value.negative,sum.max_exponent);
   if constexpr(Trace){const size_t at=(cell*total_groups+base+local)*3u;
    trace[at]=carry.significand;trace[at+1u]=uint32_t(int32_t(carry.exponent));trace[at+2u]=unsigned(carry.negative);}
  }
  if(base+count<total_groups)__syncthreads();
 }
 if(active)output[cell]=qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
template<unsigned Groups>
inline hipError_t dispatch(const uint16_t* weights,const uint16_t* inputs,const unsigned* mask,
 float* output,uint32_t* trace,unsigned rows,unsigned tokens,unsigned width,hipStream_t stream){
 const unsigned blocks=((rows+15u)/16u)*((tokens+15u)/16u);
 if(trace){hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<Groups,true>),dim3(blocks),dim3(threads),0u,stream,weights,inputs,mask,output,trace,rows,tokens,width);}
 else{hipLaunchKernelGGL(HIP_KERNEL_NAME(replay_kernel<Groups,false>),dim3(blocks),dim3(threads),0u,stream,weights,inputs,mask,output,trace,rows,tokens,width);}
 return hipGetLastError();
}
inline hipError_t launch(const uint16_t* weights,size_t weight_words,const uint16_t* inputs,size_t input_words,
 const unsigned* mask,size_t mask_words,float* output,size_t output_cells,unsigned rows,unsigned tokens,
 unsigned width,unsigned variant,hipStream_t stream,uint32_t* trace=nullptr,size_t trace_words=0u){
 const size_t cells=size_t(rows)*tokens;
 if(!weights || !inputs || !mask || !output || !rows || rows>16384u || !tokens || tokens>8192u ||
  !width || width>8192u || width%16u || variant>1u || weight_words<size_t(rows)*width ||
  input_words<size_t(tokens)*width || mask_words<(cells+31u)/32u || output_cells<cells ||
  (trace && trace_words<cells*(width/16u)*3u) || (!trace && trace_words))return hipErrorInvalidValue;
 if(!variant)return dispatch<4u>(weights,inputs,mask,output,trace,rows,tokens,width,stream);
 return dispatch<8u>(weights,inputs,mask,output,trace,rows,tokens,width,stream);
}
} // namespace qrt_sm121_matrix_projection
