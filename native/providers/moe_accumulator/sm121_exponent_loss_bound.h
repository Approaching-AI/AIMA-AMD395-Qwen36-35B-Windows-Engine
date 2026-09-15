#ifndef QRT_SM121_EXPONENT_LOSS_BOUND_H
#define QRT_SM121_EXPONENT_LOSS_BOUND_H
#include "sm121_coarse_projection_bound.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_EXPONENT_LOSS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_EXPONENT_LOSS_INLINE inline
#endif

// Isolated C64 projection envelope. Native WMMA's conditional2^-19 bound is
// unchanged. Metadata only reduces the number of canonical product alignment
// losses that must be charged; it never supplies an inference value.
namespace qrt_sm121_exponent_loss_bound {
namespace coarse=qrt_sm121_coarse_projection_bound;
namespace scalar=coarse::scalar;
struct Summary { uint32_t first=255u,second=0u; };
static_assert(sizeof(Summary)==8u);
constexpr unsigned width=64u,bins=8u;
QRT_EXPONENT_LOSS_INLINE unsigned maximum(Summary s){return s.first&255u;}
QRT_EXPONENT_LOSS_INLINE unsigned nonzero(Summary s){return s.second&255u;}
QRT_EXPONENT_LOSS_INLINE unsigned count(Summary s,unsigned bin){
    return ((bin<4u?s.first:s.second)>>(8u+6u*(bin%4u)))&63u;
}
QRT_EXPONENT_LOSS_INLINE bool valid(Summary s){
    const unsigned e=maximum(s),n=nonzero(s);
    return (!e&&!n)||(e>=80u&&e<=174u&&n>=1u&&n<=width);
}
QRT_EXPONENT_LOSS_INLINE Summary summarize(const uint16_t* words,unsigned length=width){
    if(!words||length>width)return {};
    unsigned top=0u,n=0u,counts[bins]{};
    for(unsigned i=0u;i<length;++i){
        const uint16_t x=words[i];if(!coarse::eligible(x))return {};
        if(x&0x7fffu){const unsigned e=(x>>7u)&255u;top=e>top?e:top;++n;}
    }
    for(unsigned i=0u;i<length;++i){
        if(!(words[i]&0x7fffu))continue;
        const int e=int((words[i]>>7u)&255u);
        for(unsigned b=0u;b<bins;++b)counts[b]+=unsigned(e<int(top)-int(2u*b));
    }
    // Strictly below a threshold no larger than the row maximum means each
    // count is at most63. Four six-bit counters fit above each eight-bit field.
    Summary result{top,n};
    for(unsigned b=0u;b<bins;++b){
        const unsigned value=counts[b]<<(8u+6u*(b%4u));
        if(b<4u)result.first|=value;else result.second|=value;
    }
    return result;
}
// Round the requested threshold upward to a stored threshold, so a missing
// histogram bin only overcounts. Zeros never require product alignment loss.
QRT_EXPONENT_LOSS_INLINE unsigned below(Summary s,int threshold){
    if(!valid(s))return width;
    if(!nonzero(s))return 0u;
    const int top=int(maximum(s));
    if(threshold>top)return nonzero(s);
    const unsigned distance=unsigned(top-threshold);
    const unsigned bin=distance>=14u?7u:distance/2u;
    return count(s,bin);
}
QRT_EXPONENT_LOSS_INLINE unsigned possible_products(Summary left,Summary right,int alignment_exponent){
    if(!valid(left)||!valid(right))return width;
    unsigned result=nonzero(left)<nonzero(right)?nonzero(left):nonzero(right);
    if(!result)return 0u;
    // BF16 products carry11 zero low bits in the canonical26-bit alignment.
    // Loss is possible only if ea+eb < alignment_exponent-11. The stored
    // exponents include the two127 biases, hence the threshold+243 below.
    const int threshold=alignment_exponent+243;
    for(unsigned b=0u;b<bins;++b){
        const int cut=int(maximum(left))-int(2u*b);
        const unsigned upper=count(left,b)+below(right,threshold-cut);
        result=upper<result?upper:result;
    }
    return result;
}
QRT_EXPONENT_LOSS_INLINE coarse::State advance(coarse::State before,float partial,float absolute,
    Summary left,Summary right,unsigned* charged_products=nullptr){
    coarse::State result{before.center+partial,scalar::infinity()};
    if(charged_products)*charged_products=width;
    if(!valid(left)||!valid(right)||!scalar::finite(before.center)||!scalar::finite(before.error)||before.error<0.0f||
        !scalar::finite(partial)||!scalar::finite(absolute)||absolute<0.0f)return result;
    constexpr float epsilon=0x1p-19f,floor=4.0f*0x1p-118f;
    constexpr float inflation=1.0f/(1.0f-epsilon-4.0f*0x1p-23f);
    const float positive=scalar::upper(scalar::upper(absolute*inflation)+floor);
    const float native_magnitude=scalar::upper(positive*inflation);
    const float native_error=scalar::upper(scalar::upper(positive*epsilon)+
        scalar::upper(4.0f*coarse::unit(native_magnitude,23u))+floor);
    const float magnitude=scalar::upper(scalar::upper(scalar::absolute(before.center)+before.error)+positive);
    if(!scalar::finite(magnitude)||magnitude<=0.0f)return result;
    const int exponent=int((scalar::bits(magnitude)>>23u)&255u)-127;
    const unsigned losses=possible_products(left,right,exponent);
    if(charged_products)*charged_products=losses;
    // Triangle bounds cover every interior carried magnitude. Each potentially
    // inexact product costs one unit; each of four carries costs one unit, and
    // each of four normalizations costs four units. Products excluded by the
    // metadata are exact even at this upper alignment exponent. Taking the
    // minimum of eight union bounds remains conservative for any correlation.
    const float canonical_error=scalar::upper(float(losses+20u)*coarse::unit(magnitude,25u)+floor);
    const float addition_error=coarse::unit(scalar::upper(scalar::absolute(before.center)+scalar::absolute(partial)),23u);
    result.error=scalar::upper(scalar::upper(before.error+native_error)+scalar::upper(canonical_error+addition_error));
    return result;
}
} // namespace qrt_sm121_exponent_loss_bound
#undef QRT_EXPONENT_LOSS_INLINE
#endif
