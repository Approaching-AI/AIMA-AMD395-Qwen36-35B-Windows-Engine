#pragma once
#include "sm121_macro_norm_bound.h"
#if defined(__HIPCC__)
#define QRT_COOPERATIVE_NORM_INLINE __host__ __device__ __forceinline__
#else
#define QRT_COOPERATIVE_NORM_INLINE inline
#endif

// Isolated metadata preparation for the unchanged macro norm producer.
// A wave reads one block contiguously. BF16 squares are exact in F32 within
// the admitted exponent domain (normal squares and finite sums through1024).
// Each lane adds at most32 positive terms, followed by five tree additions.
// Their gamma(37) bound is below2.3ppm. Including sqrt and final multiplication,
// the existing20ppm allowance still bounds the exact norm from above.
namespace qrt_sm121_cooperative_norm_metadata {
namespace bound=qrt_sm121_macro_norm_bound;
namespace scalar=bound::scalar;
using Summary=bound::Summary;
template<class Number> QRT_COOPERATIVE_NORM_INLINE void add_word(
 Number& sum,unsigned& maximum,unsigned& invalid,uint16_t word){
 if(!bound::base::eligible(word)){invalid=1u;return;}
 const unsigned magnitude=unsigned(word)&0x7fffu;
 maximum=magnitude>maximum?magnitude:maximum;
 const Number value=Number(scalar::value(magnitude<<16u));sum+=value*value;
}
QRT_COOPERATIVE_NORM_INLINE Summary finish(float sum,unsigned maximum,unsigned invalid){
 if(invalid)return {scalar::infinity(),scalar::infinity()};
 return {::sqrtf(sum)*1.00002f,scalar::value(maximum<<16u)};
}
QRT_COOPERATIVE_NORM_INLINE Summary finish(double sum,unsigned maximum,unsigned invalid){
 if(invalid)return {scalar::infinity(),scalar::infinity()};
 return {float(::sqrt(sum))*1.00002f,scalar::value(maximum<<16u)};
}

// Host model of the same lane-strided scan and reduction tree. The native
// harness independently checks each GPU norm against original F64 squares.
template<class Number> inline Summary simulate(const uint16_t* input,unsigned count){
 if(!input || !count || count>1024u)return {scalar::infinity(),scalar::infinity()};
 Number sums[32]{};unsigned maxima[32]{},invalid[32]{};
 for(unsigned lane=0u;lane<32u;++lane)for(unsigned k=lane;k<count;k+=32u)
  add_word(sums[lane],maxima[lane],invalid[lane],input[k]);
 for(unsigned offset=16u;offset;offset/=2u)for(unsigned lane=0u;lane<offset;++lane){
  sums[lane]+=sums[lane+offset];
  maxima[lane]=maxima[lane]>maxima[lane+offset]?maxima[lane]:maxima[lane+offset];
  invalid[lane]|=invalid[lane+offset];
 }
 return finish(sums[0],maxima[0],invalid[0]);
}

#if defined(__HIPCC__)
template<unsigned Groups,class Number>
__global__ void prepare(const uint16_t* input,Summary* output,unsigned rows,unsigned width){
 static_assert(Groups==8u || Groups==16u || Groups==32u || Groups==64u);
 constexpr unsigned chunk=Groups*16u;
 const unsigned chunks=(width+chunk-1u)/chunk;
 const unsigned lane=threadIdx.x%32u;
 const size_t entry=size_t(blockIdx.x)*(blockDim.x/32u)+threadIdx.x/32u;
 if(entry>=size_t(rows)*chunks)return;
 const unsigned row=unsigned(entry/chunks),base=unsigned(entry%chunks)*chunk;
 const unsigned count=min(chunk,width-base);
 Number sum=0.0;unsigned maximum=0u,invalid=0u;
 for(unsigned k=lane;k<count;k+=32u)add_word(sum,maximum,invalid,input[size_t(row)*width+base+k]);
 for(unsigned offset=16u;offset;offset/=2u){
  sum+=__shfl_down(sum,offset,32);
  const unsigned other=__shfl_down(maximum,offset,32);maximum=other>maximum?other:maximum;
  invalid|=__shfl_down(invalid,offset,32);
 }
 if(!lane)output[entry]=finish(sum,maximum,invalid);
}
#endif
} // namespace qrt_sm121_cooperative_norm_metadata
#undef QRT_COOPERATIVE_NORM_INLINE
