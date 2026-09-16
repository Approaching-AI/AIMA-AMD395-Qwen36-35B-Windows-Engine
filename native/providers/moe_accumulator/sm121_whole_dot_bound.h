#pragma once
#include "sm121_coarse_projection_bound.h"
#if defined(__HIPCC__)
#define QRT_WHOLE_DOT_INLINE __host__ __device__ __forceinline__
#else
#define QRT_WHOLE_DOT_INLINE inline
#endif
namespace qrt_sm121_whole_dot_bound {
namespace base=qrt_sm121_coarse_projection_bound;
namespace scalar=base::scalar;
using State=base::State;
struct Envelope {State value;float matrix_error,step_error,barrier;};
QRT_WHOLE_DOT_INLINE float matrix_error(float norm_upper,unsigned groups){
 if(!groups || groups>512u || !scalar::finite(norm_upper) || norm_upper<0.0f)return scalar::infinity();
 return scalar::upper(scalar::upper(norm_upper*0x1p-19f)+float(groups)*0x1p-118f);
}

// One complete dot uses only zero-C native K16 products and explicit F32
// additions. Maximum includes every rounded prefix. Operand metadata bounds
// the complete absolute dot and every individual original BF16 product.
// The native2^-19 coefficient is conditional, not a hardware proof.
QRT_WHOLE_DOT_INLINE Envelope build(unsigned groups,float sum,float maximum,
 float norm_upper,float product_upper){
 const float inf=scalar::infinity();Envelope result{{sum,inf},inf,inf,inf};
 if(!groups || groups>512u || !scalar::finite(sum) || !scalar::finite(maximum) ||
  maximum<scalar::absolute(sum) || !scalar::finite(norm_upper) || norm_upper<0.0f ||
  !scalar::finite(product_upper) || product_upper<0.0f)return result;
 const float count=float(groups),floor=count*0x1p-118f;
 const float partial_barrier=scalar::upper(scalar::upper(maximum+base::unit(maximum,23u))+floor);
 const float native_unit=base::unit(partial_barrier,23u);
 result.matrix_error=matrix_error(norm_upper,groups);
 const float native_error=scalar::upper(result.matrix_error+scalar::upper(count*native_unit+floor));
 const float prefix_bound=scalar::upper(maximum+native_error);
 const float minimum=prefix_bound>product_upper?prefix_bound:product_upper;
 // Round the positive denominator downward, including odd group counts.
 // Outward reciprocal and multiplication then enclose the fixed point.
 const float denominator=scalar::value(scalar::bits(1.0f-float(21u*groups)*0x1p-25f)-1u);
 const float growth=scalar::upper(1.0f/denominator);
 result.barrier=scalar::upper(scalar::upper(minimum*growth)+floor);
 result.step_error=scalar::upper(scalar::upper(native_unit+float(21u)*base::unit(result.barrier,25u))+0x1p-117f);
 result.value.error=scalar::upper(result.matrix_error+scalar::upper(count*result.step_error));
 return result;
}

// The complete-dot barrier encloses the original path's every carry and
// product, including the unreplayed suffix. Its per-step charge is valid for
// any suffix of that same path. The matrix-error argument can retain the full
// dot's allowance, or use independent metadata for exactly the remaining
// segments. No error budget for the prefix is subtracted from the full bound.
QRT_WHOLE_DOT_INLINE State suffix(float final,float prefix,float exact_prefix,
 float matrix_error,float step_error,unsigned remaining){
 State result{0.0f,scalar::infinity()};
 if(!remaining || remaining>512u || !scalar::finite(final) || !scalar::finite(prefix) ||
  !scalar::finite(exact_prefix) || !scalar::finite(matrix_error) || matrix_error<0.0f ||
  !scalar::finite(step_error) || step_error<0.0f)return result;
 const float tail=final-prefix;result.center=exact_prefix+tail;
 if(!scalar::finite(tail)||!scalar::finite(result.center))return {0.0f,scalar::infinity()};
 const float subtract_error=base::unit(scalar::upper(scalar::absolute(final)+scalar::absolute(prefix)),23u);
 const float addition_error=base::unit(scalar::upper(scalar::absolute(exact_prefix)+scalar::absolute(tail)),23u);
 result.error=scalar::upper(scalar::upper(matrix_error+scalar::upper(float(remaining)*step_error))+
  scalar::upper(subtract_error+addition_error));
 return result;
}
}
#undef QRT_WHOLE_DOT_INLINE
