#ifndef QRT_FLA_CONSUMER_INTERVAL_H
#define QRT_FLA_CONSUMER_INTERVAL_H
#include "../moe_accumulator/sm121_projection_interval.h"
#include <cmath>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_GDN_INTERVAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_GDN_INTERVAL_INLINE inline
#endif
namespace qrt_fla_consumer_interval {
namespace interval=qrt_sm121_projection_interval;
namespace bound=qrt_sm121_pv_bound;
using interval::Interval;
QRT_GDN_INTERVAL_INLINE float multiply(float a,float b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fmul_rn(a,b);
#else
    volatile float result=a*b;return result;
#endif
}
QRT_GDN_INTERVAL_INLINE float subtract(float a,float b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fsub_rn(a,b);
#else
    volatile float result=a-b;return result;
#endif
}
QRT_GDN_INTERVAL_INLINE float fused(float a,float b,float c) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return fmaf(a,b,c);
#else
    return std::fma(a,b,c);
#endif
}
// The exponential table supplies positive values or positive zero. Negative
// zero is outside that domain and is conservatively sent to exact replay.
QRT_GDN_INTERVAL_INLINE bool nonnegative(float x){return bound::bits(x)<0x7f800000u;}
QRT_GDN_INTERVAL_INLINE bool rounded(Interval value,uint16_t* result) {
    if(!interval::same_bf16(value))return false;
    *result=bound::bf16(value.lower);return true;
}
// Each consumer is monotone in its dot arguments at the original FP32
// rounding boundaries. Equal BF16 endpoints certify every enclosed FP32 dot.
// Both state residuals must agree; the unrounded FP32 state update is excluded.
QRT_GDN_INTERVAL_INLINE bool residual(Interval dot,float u,float decay,
    uint16_t* updated,uint16_t* scaled) {
    if(!interval::valid(dot)||!bound::finite(u)||!nonnegative(decay))return false;
    const Interval difference{subtract(u,dot.upper),subtract(u,dot.lower)};
    const Interval product{multiply(difference.lower,decay),multiply(difference.upper,decay)};
    uint16_t a,b;
    if(!rounded(difference,&a)||!rounded(product,&b))return false;
    *updated=a;*scaled=b;return true;
}
QRT_GDN_INTERVAL_INLINE bool output(Interval prior,Interval local,float decay,uint16_t* result) {
    if(!interval::valid(prior)||!interval::valid(local)||!nonnegative(decay))return false;
    constexpr float scale=0.08838834764831845f;
    const float lower=multiply(multiply(prior.lower,decay),scale);
    const float upper=multiply(multiply(prior.upper,decay),scale);
    return rounded({fused(local.lower,scale,lower),fused(local.upper,scale,upper)},result);
}
}
#undef QRT_GDN_INTERVAL_INLINE
#endif
