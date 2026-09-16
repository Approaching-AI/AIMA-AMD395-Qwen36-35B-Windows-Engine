#ifndef QRT_SM121_CRT_INTEGER_CORE_H
#define QRT_SM121_CRT_INTEGER_CORE_H
#include <cstdint>
#if defined(__HIPCC__)
#define QRT_CRT_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CRT_INLINE inline
#endif

// Component experiment: one FP16 matrix estimates an H7 integer dot, and
// exact IU8 matrices supply its residues modulo255 and256. Their coprime
// product65280 uniquely determines the nearest integer when error<32640.
// The observed native pair model conditionally bounds H7 error by3920;
// this arithmetic implication does not prove universal hardware semantics.
namespace qrt_sm121_crt_integer {
constexpr int64_t maximum_dot=int64_t(16)*32640*32640;
constexpr int modulus=65280,half_modulus=32640;
struct Prepared{uint16_t values[16];uint32_t residue255[4],residue256[4];};
static_assert(sizeof(Prepared)==64u);
QRT_CRT_INLINE unsigned highest(unsigned x){
#if defined(__HIP_DEVICE_COMPILE__)
    return 31u-__clz(x|1u);
#else
    unsigned result=0u;for(x>>=1u;x;x>>=1u)++result;return result;
#endif
}
QRT_CRT_INLINE bool eligible(int value){
    const unsigned m=unsigned(value<0?-value:value),e=highest(m);
    return m<=32640u && (e<=7u || !(m&((1u<<(e-7u))-1u)));
}
// H7 cores have at most eight significant bits. Their FP16 encoding is
// exact, including small integers obtained by truncating original BF16 bits.
QRT_CRT_INLINE uint16_t half_bits(int value){
    if(!eligible(value))return 0x7e00u;
    const unsigned m=unsigned(value<0?-value:value),e=highest(m);
    const unsigned significand=e<=10u?m<<(10u-e):m>>(e-10u);
    return m?uint16_t((value<0?0x8000u:0u)|((e+15u)<<10u)|(significand&1023u)):0u;
}
template<class Row>
QRT_CRT_INLINE Prepared prepare(const Row& row){
    Prepared result{};
    for(unsigned i=0u;i<16u;++i){
        const unsigned word=i/4u,shift=i%4u*8u;
        const unsigned h=(uint32_t(row.high[word])>>shift)&255u;
        const unsigned l=(uint32_t(row.low[word])>>shift)&255u;
        const int high=int(h)-(h&128u?256:0),value=high*256+int(l);
        int residue=high+int(l);
        if(residue<0)residue+=255;
        if(residue>=255)residue-=255;
        result.values[i]=half_bits(value);
        result.residue255[word]|=uint32_t(residue)<<shift;
        result.residue256[word]|=uint32_t(l)<<shift;
    }
    return result;
}
QRT_CRT_INLINE uint32_t reduce255(uint32_t value){
    value=(value&65535u)+(value>>16u);
    value=(value&255u)+(value>>8u);
    value=(value&255u)+(value>>8u);
    return value>=255u?value-255u:value;
}
// The caller supplies a base within maximum_dot+65536, so its signed high
// word lies in[-4,3]. Since2^32=1 modulo255, one correction on each side is
// enough after adding that signed high word to the reduced low word.
QRT_CRT_INLINE int base_residue(int64_t base){
    const uint64_t bits=uint64_t(base);
    const uint32_t high=uint32_t(bits>>32u);
    const int signed_high=high&0x80000000u?int(int64_t(high)-4294967296ll):int(high);
    int result=int(reduce255(uint32_t(bits)))+signed_high;
    if(result<0)result+=255;
    if(result>=255)result-=255;
    return result;
}
QRT_CRT_INLINE bool recover(float approximate,uint32_t residue255,uint32_t residue256,int64_t* output){
    constexpr float limit=float(maximum_dot+65536);
    if(!(approximate>=-limit && approximate<=limit))return false;
    const int64_t truncated=int64_t(approximate);
    const int64_t base=truncated-int64_t(uint64_t(truncated)&255u)+int64_t(residue256&255u);
    int adjustment=int(reduce255(residue255))-base_residue(base);
    if(adjustment<0)adjustment+=255;
    if(adjustment>127)adjustment-=255;
    int64_t candidate=base+int64_t(adjustment)*256;
    const int64_t distance=candidate-truncated;
    if(distance>half_modulus)candidate-=modulus;
    else if(distance<-half_modulus)candidate+=modulus;
    else if(distance==half_modulus || distance==-half_modulus){
        const float fraction=approximate-float(truncated);
        if(fraction==0.0f)return false;
        if(distance==half_modulus && fraction<0.0f)candidate-=modulus;
        if(distance==-half_modulus && fraction>0.0f)candidate+=modulus;
    }
    if(candidate < -maximum_dot || candidate > maximum_dot)return false;
    *output=candidate;return true;
}
#if defined(__HIPCC__)
using H16=_Float16 __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
using I4=int __attribute__((ext_vector_type(4)));
using I8=int __attribute__((ext_vector_type(8)));
struct Parts{F8 approximate;I8 residue255,residue256;};
__device__ __forceinline__ Parts products(const Prepared& a,const Prepared& b){
    H16 av{},bv{};I4 a255{},a256{},b255{},b256{};
#pragma unroll
    for(unsigned i=0u;i<16u;++i){av[i]=__builtin_bit_cast(_Float16,a.values[i]);bv[i]=__builtin_bit_cast(_Float16,b.values[i]);}
#pragma unroll
    for(unsigned i=0u;i<4u;++i){a255[i]=int(a.residue255[i]);a256[i]=int(a.residue256[i]);b255[i]=int(b.residue255[i]);b256[i]=int(b.residue256[i]);}
    return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(av,bv,F8{}),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,a255,false,b255,I8{},false),
        __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false,a256,false,b256,I8{},false)};
}
#endif
}
#undef QRT_CRT_INLINE
#endif
