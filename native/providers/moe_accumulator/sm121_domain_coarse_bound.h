#pragma once
#include "sm121_coarse_projection_bound.h"

#if defined(__HIPCC__)
#define QRT_DOMAIN_BOUND_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DOMAIN_BOUND_INLINE inline
#endif
namespace qrt_sm121_domain_coarse_bound {
namespace base=qrt_sm121_coarse_projection_bound;
namespace scalar=base::scalar;

// Only the C64 producer with zeroed ineligible rows and width<=8192 may
// call this recurrence. Admitted BF16 operands are zero or have exponents
// 80..174, hence |operand|<2^48. Each zero-C K16 signed/absolute result is
// bounded by the sum of16 products, with ample slack below2^101. Four
// ascending additions give |partial|,absolute<2^103. At most128 C64 blocks
// keep |center|<2^111. With error<2^104, every magnitude formed below is
// <2^113, each error increment is <2^96, and128 increments keep error
// <2^104. These deliberately loose bounds include FP32 rounding/outward
// steps. Thus every upper() argument is finite, nonnegative, far below
// overflow. Every unit() argument is strictly positive due to upper().
// Replacing their exceptional branches does NOT change any float operation,
// either two-ULP outward step, subnormal unit or native error coefficient.
// The native2^-19 correctness premise itself remains conditional.
QRT_DOMAIN_BOUND_INLINE bool width_supported(unsigned width) {
    return width&&width<=8192u&&!(width&15u);
}
QRT_DOMAIN_BOUND_INLINE float upper(float x) {
    return scalar::value(scalar::bits(x)+2u);
}
template<unsigned Fractional>
QRT_DOMAIN_BOUND_INLINE float positive_unit(float magnitude) {
    static_assert(Fractional==23u||Fractional==25u);
    const unsigned exponent=scalar::bits(magnitude)>>23u;
    if(exponent>Fractional)return scalar::value((exponent-Fractional)<<23u);
    const int shift=int(exponent)+22-int(Fractional);
    return scalar::value(1u<<unsigned(shift>0?shift:0));
}
QRT_DOMAIN_BOUND_INLINE base::State advance(base::State before,float partial,float absolute) {
    constexpr float epsilon=0x1p-19f,floor=4.0f*0x1p-118f;
    constexpr float inflation=1.0f/(1.0f-epsilon-4.0f*0x1p-23f);
    base::State result{before.center+partial,0.0f};
    const float positive=upper(upper(absolute*inflation)+floor);
    const float native_magnitude=upper(positive*inflation);
    const float native_error=upper(upper(positive*epsilon)+
        upper(4.0f*positive_unit<23u>(native_magnitude))+floor);
    const float magnitude=upper(upper(scalar::absolute(before.center)+before.error)+positive);
    const float canonical_error=upper(84.0f*positive_unit<25u>(magnitude)+floor);
    const float addition_error=positive_unit<23u>(upper(scalar::absolute(before.center)+scalar::absolute(partial)));
    result.error=upper(upper(before.error+native_error)+upper(canonical_error+addition_error));
    return result;
}
} // namespace qrt_sm121_domain_coarse_bound
#undef QRT_DOMAIN_BOUND_INLINE
