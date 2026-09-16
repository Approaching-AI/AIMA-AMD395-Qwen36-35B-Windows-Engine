#pragma once
#include "sm121_coarse_projection_bound.h"
#include <cmath>
#if defined(__HIPCC__)
#define QRT_MACRO_NORM_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MACRO_NORM_INLINE inline
#endif
namespace qrt_sm121_macro_norm_bound {
namespace base=qrt_sm121_coarse_projection_bound;
namespace scalar=base::scalar;
using State=base::State;
struct Summary {float norm,maximum;};
static_assert(sizeof(Summary)==8u);
// BF16 squares in the admitted exponent domain are exactly representable in
// F64. At most1024 positive additions and the square root are far inside the
// existing20-ppm outward allowance, which also covers final F32 rounding.
// This is metadata only; all unsupported rows retain original exact replay.
QRT_MACRO_NORM_INLINE Summary prepare(const uint16_t* input,unsigned count){
 if(!input || !count || count>1024u)return {scalar::infinity(),scalar::infinity()};
 double squares=0.0;uint16_t maximum=0u;
 for(unsigned i=0u;i<count;++i){
  const uint16_t word=input[i];
  if(!base::eligible(word))return {scalar::infinity(),scalar::infinity()};
  const uint16_t magnitude=word&0x7fffu;maximum=magnitude>maximum?magnitude:maximum;
  const double value=scalar::value(uint32_t(magnitude)<<16u);squares+=value*value;
 }
 return {float(::sqrt(squares))*1.00002f,scalar::value(uint32_t(maximum)<<16u)};
}
QRT_MACRO_NORM_INLINE float absolute_bound(Summary a,Summary b){return scalar::upper(a.norm*b.norm);}
QRT_MACRO_NORM_INLINE float product_bound(Summary a,Summary b){return scalar::upper(a.maximum*b.maximum);}

// Isolated component: norm_upper bounds the entire block absolute dot;
// product_upper bounds each original BF16 product. Both are comparison-free
// operand metadata. Partial and prefix maxima include every native K16 step.
template<unsigned Groups> QRT_MACRO_NORM_INLINE State advance(State before,float partial,float norm_upper,
 float prefix_max,float partial_max,float product_upper,float* barrier=nullptr){
 static_assert(Groups==8u || Groups==16u || Groups==32u || Groups==64u);
 State result{before.center+partial,scalar::infinity()};
 if(!scalar::finite(before.center)||!scalar::finite(before.error)||before.error<0.0f||!scalar::finite(partial)||!scalar::finite(norm_upper)||norm_upper<0.0f||!scalar::finite(prefix_max)||prefix_max<0.0f||!scalar::finite(partial_max)||partial_max<scalar::absolute(partial)||!scalar::finite(product_upper)||product_upper<0.0f||!scalar::finite(result.center)||prefix_max<scalar::absolute(before.center)||prefix_max<scalar::absolute(result.center))return result;
 constexpr float epsilon=0x1p-19f,floor=float(Groups)*0x1p-118f;
 // Every exact scalar addition preceding rounding lies within one ULP of
 // its observed rounded result. This bounds all explicit native-output sums
 // without charging their much larger absolute-product sum for each add.
 const float partial_barrier=scalar::upper(scalar::upper(partial_max+base::unit(partial_max,23u))+floor);
 const float native_error=scalar::upper(scalar::upper(norm_upper*epsilon)+scalar::upper(float(Groups)*base::unit(partial_barrier,23u))+floor);
 const float prefix_add_error=base::unit(scalar::upper(scalar::absolute(before.center)+partial_max),23u);
 const float prefix_bound=scalar::upper(scalar::upper(prefix_max+before.error)+scalar::upper(native_error+prefix_add_error));
 const float minimum=prefix_bound>product_upper?prefix_bound:product_upper;
 constexpr float growth=1.0f/(1.0f-float(21u*Groups)*0x1p-25f);
 const float magnitude=scalar::upper(scalar::upper(minimum*growth)+floor);
 const float canonical_error=scalar::upper(float(21u*Groups)*base::unit(magnitude,25u)+floor);
 const float addition_error=base::unit(scalar::upper(scalar::absolute(before.center)+scalar::absolute(partial)),23u);
 if(barrier)*barrier=magnitude;
 result.error=scalar::upper(scalar::upper(before.error+native_error)+scalar::upper(canonical_error+addition_error));
 return result;
}
}

#undef QRT_MACRO_NORM_INLINE
