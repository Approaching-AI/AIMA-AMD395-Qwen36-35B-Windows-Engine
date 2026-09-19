#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_BF16_FMA_INLINE __host__ __device__ __forceinline__
#else
#define QRT_BF16_FMA_INLINE inline
#endif

namespace qrt_sm121_bf16_fma {
QRT_BF16_FMA_INLINE unsigned leading(uint64_t x) {
    unsigned result=0;
    while(x>>1u){x>>=1u;++result;}
    return result;
}
QRT_BF16_FMA_INLINE uint64_t align(uint64_t value,int shift) {
    if(shift>=0)return value<<unsigned(shift);
    const unsigned right=unsigned(-shift);
    if(right>=64u)return value?1u:0u;
    return (value>>right)|uint64_t((value&((uint64_t(1)<<right)-1u))!=0u);
}
QRT_BF16_FMA_INLINE uint64_t rounded_shift(uint64_t value,int shift) {
    if(shift<=0)return value<<unsigned(-shift);
    if(shift>64)return 0u;
    if(shift==64)return uint64_t(value>(uint64_t(1)<<63u));
    const unsigned right=unsigned(shift);
    const uint64_t half=uint64_t(1)<<(right-1u),rest=value&((half<<1u)-1u);
    const uint64_t result=value>>right;
    return result+uint64_t(rest>half||(rest==half&&(result&1u)));
}
// Direct round-to-nearest-even BF16 FMA. Each finite operand has at most
// eight significant bits; the product has at most sixteen. Align the larger
// term's leading bit at bit62, leaving at least47 exact low bits. If the
// smaller term needs a right shift, its lost bits become sticky. Such terms
// are too far apart to cancel, so the sticky bit preserves the direction at
// every BF16 rounding boundary for both addition and subtraction.
QRT_BF16_FMA_INLINE uint16_t exact(uint16_t a,uint16_t b,uint16_t c) {
    const unsigned ae=(a>>7u)&255u,be=(b>>7u)&255u,ce=(c>>7u)&255u;
    const unsigned af=a&127u,bf=b&127u,cf=c&127u;
    const bool ps=((a^b)&0x8000u)!=0u,cs=(c&0x8000u)!=0u;
    if((ae==255u&&af)||(be==255u&&bf)||(ce==255u&&cf))return 0x7fc0u;
    if(ae==255u||be==255u){
        if((a&0x7fffu)==0u||(b&0x7fffu)==0u||(ce==255u&&ps!=cs))return 0x7fc0u;
        return uint16_t((ps?0x8000u:0u)|0x7f80u);
    }
    if(ce==255u)return c;
    const uint64_t product=uint64_t(af+(ae?128u:0u))*(bf+(be?128u:0u));
    const uint64_t addend=cf+(ce?128u:0u);
    if(!product&&!addend)return uint16_t(ps&&cs?0x8000u:0u);
    const int pe=(ae?int(ae)-134:-133)+(be?int(be)-134:-133);
    const int cexp=ce?int(ce)-134:-133;
    const int pt=product?pe+int(leading(product)):-1000;
    const int ct=addend?cexp+int(leading(addend)):-1000;
    const int top=pt>ct?pt:ct,base=top-62;
    const uint64_t p=product?align(product,pe-base):0u;
    const uint64_t q=addend?align(addend,cexp-base):0u;
    const bool negative=ps==cs?ps:(p>q?ps:cs);
    const uint64_t magnitude=ps==cs?p+q:(p>q?p-q:q-p);
    if(!magnitude)return 0u;
    const int highest=base+int(leading(magnitude));
    const uint16_t sign=negative?0x8000u:0u;
    if(highest>127)return uint16_t(sign|0x7f80u);
    if(highest< -134)return sign;
    const int step=highest>=-126?highest-7:-133;
    const uint64_t mantissa=rounded_shift(magnitude,step-base);
    const uint64_t encoded=uint64_t(step+133)*128u+mantissa;
    return uint16_t(sign|uint16_t(encoded>=0x7f80u?0x7f80u:encoded));
}
QRT_BF16_FMA_INLINE float widen(uint16_t x) {
    const uint32_t bits=uint32_t(x)<<16u;float result;
    std::memcpy(&result,&bits,sizeof(result));return result;
}
QRT_BF16_FMA_INLINE uint16_t round(uint16_t a,uint16_t b,uint16_t c) {
    const unsigned am=a&0x7fffu,bm=b&0x7fffu,cm=c&0x7fffu;
    // Preserve subnormals even on devices configured to flush FP32 denormals.
    if((am&&am<128u)||(bm&&bm<128u)||(cm&&cm<128u)||
       am>=0x7f80u||bm>=0x7f80u||cm>=0x7f80u)return exact(a,b,c);
    const float result=fmaf(widen(a),widen(b),widen(c));
    uint32_t bits;std::memcpy(&bits,&result,sizeof(bits));
    const unsigned exponent=(bits>>23u)&255u;
    // A FP32 result away from a BF16 midpoint has the same BF16 endpoint.
    // At the midpoint, rounding FP32 first may discard the decisive tail.
    if(!exponent||exponent==255u||(bits&65535u)==32768u)return exact(a,b,c);
    return uint16_t((bits+0x7fffu+((bits>>16u)&1u))>>16u);
}
QRT_BF16_FMA_INLINE float rounded(float a,float b,float c) {
    uint32_t x,y,z;std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);std::memcpy(&z,&c,4);
    // Legacy rotary tables can contain full FP32 coefficients. Keep their
    // declared FP32 arithmetic; only BF16 operands use the BF16 FMA contract.
    if((x|y|z)&65535u){
        const float value=fmaf(a,b,c);uint32_t bits;std::memcpy(&bits,&value,4);
        return widen(uint16_t((bits+0x7fffu+((bits>>16u)&1u))>>16u));
    }
    return widen(round(uint16_t(x>>16u),uint16_t(y>>16u),uint16_t(z>>16u)));
}
} // namespace qrt_sm121_bf16_fma
#undef QRT_BF16_FMA_INLINE
