#ifndef QRT_SM121_DYADIC_CARRY_SCAN_H
#define QRT_SM121_DYADIC_CARRY_SCAN_H
#include "q1_moe_hawkeye_bf16_accumulator.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DYADIC_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DYADIC_INLINE inline
#endif

namespace qrt_sm121_dyadic_carry_scan {
using Value=qrt_q1_moe_hawkeye::Value;
// F(x)=floor((x+b)/2^s)*2^s+c in unsigned modulo2^64 arithmetic.
// The canonical representation has s<64 and 0<=b<2^s.
struct Function {uint64_t b=0u,c=0u;unsigned s=0u;};
QRT_DYADIC_INLINE uint64_t low(unsigned s) {return (uint64_t(1u)<<s)-1u;}
QRT_DYADIC_INLINE uint64_t evaluate(Function f,uint64_t x) {
    return ((x+f.b)&~low(f.s))+f.c;
}
// Return outer(inner(x)). Dyadic grids are nested. If the outer grid is
// coarser, round inner.c+outer.b to the inner grid before absorbing it into
// b. Otherwise the inner grid already lies on the outer grid. Normalizing b
// gives an associative representation, including arbitrary exponent changes.
QRT_DYADIC_INLINE Function compose(Function inner,Function outer) {
    const uint64_t d=inner.c+outer.b;
    if(outer.s>=inner.s) {
        const uint64_t b=inner.b+(d&~low(inner.s));
        return {b&low(outer.s),outer.c+(b&~low(outer.s)),outer.s};
    }
    return {inner.b,outer.c+(d&~low(outer.s)),inner.s};
}
QRT_DYADIC_INLINE bool regular(Value v) {
    return !v.significand ? v.exponent==-133 :
        v.significand>=0x800000u && v.significand<=0xffffffu &&
        v.exponent>=-126 && v.exponent<=127;
}
QRT_DYADIC_INLINE bool same_class(Value actual,Value predicted) {
    return actual.exponent==predicted.exponent &&
        bool(actual.significand)==bool(predicted.significand) &&
        (!actual.significand || actual.negative==predicted.negative);
}
// A K16 step is incoming RZ alignment, exact aligned product sum, then RZ
// normalization. Predictions specify quantizers only; they never supply a
// numerical answer. Caller computes the true maximum of product exponents
// and predicted incoming exponent, and the original per-product aligned sum.
// Every prefix must later pass from_integer and same_class. By induction,
// this checks each actual incoming sign/exponent and every output quantizer.
QRT_DYADIC_INLINE bool make_step(Value incoming,Value outgoing,int maximum,
    int64_t products,int base,Function* output) {
    constexpr int64_t limit=int64_t(16u)*255u*255u*2048u;
    if(!output || !regular(incoming) || !regular(outgoing) || base < -149 ||
        maximum < incoming.exponent || maximum < -133 || maximum > 127 ||
        products < -limit || products > limit) return false;
    const int align=maximum-25-base;
    const int normalize=outgoing.significand ? outgoing.exponent-23-base : 0;
    if(align<0 || align>30 || normalize<0 || normalize>62) return false;
    const Function alignment{incoming.negative?low(unsigned(align)):0u,
        uint64_t(products)<<unsigned(align),unsigned(align)};
    const Function normalization{outgoing.negative?low(unsigned(normalize)):0u,
        0u,unsigned(normalize)};
    *output=compose(alignment,normalization);
    return true;
}
// The alignment bound above makes each product sum smaller than2^61, and
// each correctly classified incoming carry smaller than2^56. Their sum is
// strictly within signed64, so modular composition cannot hide an overflow
// on a trajectory whose every classification passes. Reject nonnormal and
// non-FP32-representable endpoints before checking those classifications.
QRT_DYADIC_INLINE bool from_integer(uint64_t bits,int base,Value* output) {
    if(!output || base < -149) return false;
    const bool negative=bool(bits>>63u);
    const uint64_t magnitude=negative?0u-bits:bits;
    if(!magnitude) {*output={0u,-133,false};return true;}
    const unsigned width=qrt_q1_moe_hawkeye::bit_width_u64(magnitude);
    const int exponent=base+int(width)-1;
    if(width>62u || exponent < -126 || exponent > 127) return false;
    uint32_t significand;
    if(width>24u) {
        if(magnitude&low(width-24u)) return false;
        significand=uint32_t(magnitude>>(width-24u));
    } else significand=uint32_t(magnitude)<<(24u-width);
    *output={significand,int16_t(exponent),negative};return true;
}
QRT_DYADIC_INLINE bool to_integer(Value value,int base,uint64_t* output) {
    if(!output || !regular(value) || base < -149) return false;
    if(!value.significand) {*output=0u;return true;}
    const int shift=value.exponent-23-base;
    if(shift<0 || shift>38) return false;
    const uint64_t magnitude=uint64_t(value.significand)<<unsigned(shift);
    *output=value.negative?0u-magnitude:magnitude;return true;
}
} // namespace qrt_sm121_dyadic_carry_scan
#undef QRT_DYADIC_INLINE
#endif
