#pragma once
#include "sm121_coarse_projection_bound.h"
#if defined(__HIPCC__)
#define QRT_OBSERVED_PREFIX_INLINE __host__ __device__ __forceinline__
#else
#define QRT_OBSERVED_PREFIX_INLINE inline
#endif
// Isolated block bound. Callers supply the maximum of the rounded entry
// center plus EVERY native partial prefix, including the entry center, and
// the largest zero-C absolute K16 matrix result. Native error2^-19 remains
// conditional; observed prefix magnitudes do not prove the hardware bound.
namespace qrt_sm121_observed_prefix_bound {
namespace original=qrt_sm121_coarse_projection_bound;
namespace scalar=original::scalar;
using State=original::State;
template<unsigned Groups> QRT_OBSERVED_PREFIX_INLINE State advance(State before,float partial,float absolute,float prefix_max,float group_absolute_max,float* barrier=nullptr){
 static_assert(Groups==4u || Groups==8u || Groups==16u || Groups==32u);
 State result{before.center+partial,scalar::infinity()};
 if(!scalar::finite(before.center)||!scalar::finite(before.error)||before.error<0.0f||!scalar::finite(partial)||!scalar::finite(absolute)||absolute<0.0f||!scalar::finite(prefix_max)||prefix_max<0.0f||!scalar::finite(group_absolute_max)||group_absolute_max<0.0f||!scalar::finite(result.center)||prefix_max<scalar::absolute(before.center)||prefix_max<scalar::absolute(result.center))return result;
 constexpr float epsilon=0x1p-19f,floor=float(Groups)*0x1p-118f;
 constexpr float inflation=1.0f/(1.0f-epsilon-float(Groups)*0x1p-23f);
 const float positive=scalar::upper(scalar::upper(absolute*inflation)+floor);
 const float native_magnitude=scalar::upper(positive*inflation);
 const float native_error=scalar::upper(scalar::upper(positive*epsilon)+scalar::upper(float(Groups)*original::unit(native_magnitude,23u))+floor);
 // Every observed prefix is fl(before.center+the accumulated native K16
 // outputs). This charge encloses all such rounded additions, not just the
 // final addition. The total native bound also covers all partial prefixes.
 const float prefix_add_error=original::unit(scalar::upper(scalar::absolute(before.center)+native_magnitude),23u);
 const float prefix_bound=scalar::upper(scalar::upper(prefix_max+before.error)+scalar::upper(native_error+prefix_add_error));
 const float product_bound=scalar::upper(scalar::upper(group_absolute_max*inflation)+floor);
 const float base=prefix_bound>product_bound?prefix_bound:product_bound;
 // Barrier argument: canonical loss over the block is at most21*Groups
 // units at T, and unit(T,25)<=T*2^-25. Choosing this outward fixed-point
 // envelope encloses every original carry, aligned pre-normalization sum
 // and product exponent, despite the dependence of the loss on those values.
 constexpr float growth=1.0f/(1.0f-float(21u*Groups)*0x1p-25f);
 const float magnitude=scalar::upper(scalar::upper(base*growth)+floor);
 const float canonical_error=scalar::upper(float(21u*Groups)*original::unit(magnitude,25u)+floor);
 const float addition_error=original::unit(scalar::upper(scalar::absolute(before.center)+scalar::absolute(partial)),23u);
 if(barrier)*barrier=magnitude;
 result.error=scalar::upper(scalar::upper(before.error+native_error)+scalar::upper(canonical_error+addition_error));
 return result;
}
}

#undef QRT_OBSERVED_PREFIX_INLINE
