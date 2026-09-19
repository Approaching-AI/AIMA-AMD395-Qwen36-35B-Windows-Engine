#ifndef QRT_FLA_ABSOLUTE_DOT_BOUND_H
#define QRT_FLA_ABSOLUTE_DOT_BOUND_H
#include "consumer_interval.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_GDN_NORM_INLINE __host__ __device__ __forceinline__
#else
#define QRT_GDN_NORM_INLINE inline
#endif

namespace qrt_fla_absolute_dot {
namespace bits=qrt_sm121_pv_bound;
namespace interval=qrt_sm121_projection_interval;
struct Row { float sum=0.0f, maximum=0.0f; unsigned count=0u; bool valid=true; };
// count == 0 denotes a rejected row, including an empty row.
struct Norm { float sum, maximum; unsigned count; };
QRT_GDN_NORM_INLINE float add(float a,float b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fadd_rn(a,b);
#else
    volatile float result=a+b;return result;
#endif
}
QRT_GDN_NORM_INLINE void include(Row& row,uint16_t word) {
    ++row.count;
    const unsigned magnitude=word&0x7fffu,exponent=magnitude>>7u;
    // Cap magnitudes below 2^48 so 128-term sums and norm products cannot
    // overflow. Replace BF16 subnormals by the larger FP32 minimum normal:
    // norm arithmetic stays normal even on devices that flush subnormals.
    row.valid=row.valid && row.count<=128u &&
        exponent<=174u;
    if(!row.valid)return;
    const float value=bits::value(magnitude && !exponent?0x00800000u:magnitude<<16u);
    row.sum=add(row.sum,value);
    row.maximum=value>row.maximum?value:row.maximum;
}
QRT_GDN_NORM_INLINE Norm finish(const Row& row) {
    if(!row.valid || !row.count || row.count>128u)return {0.0f,0.0f,0u};
    if(row.maximum==0.0f)return {0.0f,0.0f,row.count};
    // Positive FP32 summation has relative error at most gamma(127).
    // 1+2^-16 exceeds 1/(1-gamma(127)); two outward FP32 steps also
    // cover rounding of this inflation and the subsequent norm product.
    const float sum=bits::upper(qrt_fla_consumer_interval::multiply(row.sum,1.0f+0x1p-16f));
    return {sum,row.maximum,row.count};
}
QRT_GDN_NORM_INLINE float upper_product(float a,float b) {
    const float product=qrt_fla_consumer_interval::multiply(a,b);
    // A positive norm product rounded or flushed below the minimum normal
    // is enclosed by the minimum normal. No subnormal result is relied on.
    if(bits::bits(product)<0x00800000u)return bits::value(0x00800000u);
    return bits::upper(product);
}
QRT_GDN_NORM_INLINE interval::Interval enclose(Norm a,Norm b) {
    if(!a.count || a.count!=b.count || a.count>128u)return interval::invalid();
    // The original final accumulator clears the sign of an integer zero.
    if(a.maximum==0.0f || b.maximum==0.0f)return {0.0f,0.0f};
    const float first=upper_product(a.sum,b.maximum);
    const float second=upper_product(b.sum,a.maximum);
    const float magnitude=first<second?first:second;
    // Each original K16 alignment truncates operand magnitudes; its integer
    // reduction is exact and normalization truncates again. Inductively,
    // the final magnitude is at most sum(abs(a[i]*b[i])). Both L1/Linf
    // products enclose that sum, independent of cancellation or signs.
    return {-magnitude,magnitude};
}
}
#undef QRT_GDN_NORM_INLINE
#endif
