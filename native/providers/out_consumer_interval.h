#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>
#if defined(__HIPCC__)
#define QRT_OUT_CONSUMER_HD __host__ __device__
#else
#define QRT_OUT_CONSUMER_HD
#endif
namespace qrt_out_consumer_interval {
QRT_OUT_CONSUMER_HD inline uint32_t bits(float value) {uint32_t out;__builtin_memcpy(&out,&value,4u);return out;}
QRT_OUT_CONSUMER_HD inline float value(uint32_t bits) {float out;__builtin_memcpy(&out,&bits,4u);return out;}
QRT_OUT_CONSUMER_HD inline bool finite(float x) {return (bits(x)&0x7fffffffu)<0x7f800000u;}
QRT_OUT_CONSUMER_HD inline float next(float x,bool positive) {
 uint32_t b=bits(x);if(!(b&0x7fffffffu))return value(positive?1u:0x80000001u);
 b+=(((b&0x80000000u)==0u)==positive)?1u:uint32_t(-1);return value(b);
}
QRT_OUT_CONSUMER_HD inline uint16_t bf16(float x) {const uint32_t b=bits(x);return uint16_t((b+32767u+((b>>16u)&1u))>>16u);}
QRT_OUT_CONSUMER_HD inline float rounded(float x) {return value(uint32_t(bf16(x))<<16u);}
struct Interval {float lower,upper;};
// This transports the existing empirical producer envelope; it does not
// establish that the PPB coefficient bounds arbitrary matrix arithmetic.
// Tiny/nonfinite endpoints retain mandatory original replay. The diagnostic
// separately checks containment against the completed original projection.
QRT_OUT_CONSUMER_HD inline bool projection(float center,float cauchy,unsigned ppb,
 unsigned radius,Interval* result) {
 const unsigned exponent=(bits(center)>>23u)&255u;
 if(!result || !ppb || radius>32768u || exponent<32u || !finite(center) ||
    !finite(cauchy) || cauchy<0.0f)return false;
 const float absolute=next(cauchy*(float(ppb)*1.0e-9f),true);
 const float radial=next(float(radius+1u)*value((exponent-23u)<<23u),true);
 const float error=absolute>radial?absolute:radial;
 if(!finite(error))return false;
 const float lower=next(center-error,false),upper=next(center+error,true);
 if(!finite(lower)||!finite(upper))return false;
 *result={rounded(lower),rounded(upper)};
 return finite(result->lower)&&finite(result->upper)&&result->lower<=result->upper;
}
QRT_OUT_CONSUMER_HD inline Interval absolute_range(Interval x) {
 const float a=std::fabs(x.lower),b=std::fabs(x.upper);
 return {x.lower<=0.0f&&x.upper>=0.0f?0.0f:(a<b?a:b),a>b?a:b};
}
} // namespace qrt_out_consumer_interval
#undef QRT_OUT_CONSUMER_HD
