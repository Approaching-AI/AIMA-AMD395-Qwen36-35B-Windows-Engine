#ifndef QRT_SM121_PREPARED_INTEGER_PAIRS_H
#define QRT_SM121_PREPARED_INTEGER_PAIRS_H
#include "sm121_group16_modulo.h"
#include "sm121_canonical_normalize.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PREPARED_PAIRS_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PREPARED_PAIRS_INLINE inline
#endif
namespace qrt_sm121_prepared_integer_pairs {
// Two unsigned exponent/significand halfwords per word. A separate sign mask
// retains all original BF16 encodings, including zero and subnormal inputs.
// This is a lossless operand view, not a reduced-precision representation.
struct Row { uint32_t pairs[8];uint32_t negative; };
static_assert(sizeof(Row)==36u); // Odd nine-dword LDS row stride.
struct Products { uint32_t significands,exponents; };

QRT_PREPARED_PAIRS_INLINE uint16_t encode(uint16_t value) {
    const unsigned exponent=(value>>7u)&255u;
    const unsigned significand=(value&127u)|(exponent?128u:0u);
    return uint16_t(((exponent?exponent:1u)<<8u)|significand);
}
QRT_PREPARED_PAIRS_INLINE Row prepare(const uint16_t* input) {
    Row row{};
    for(unsigned i=0u;i<16u;++i) {
        row.pairs[i/2u]|=uint32_t(encode(input[i]))<<(i%2u*16u);
        row.negative|=uint32_t((input[i]&0x8000u)!=0u)<<i;
    }
    return row;
}
QRT_PREPARED_PAIRS_INLINE Products multiply(uint32_t left,uint32_t right) {
    const uint32_t a=left&0x00ff00ffu,b=right&0x00ff00ffu;
    uint32_t significands;
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(significands) : "v"(a),"v"(b));
#else
    significands=((a&65535u)*(b&65535u))|(((a>>16u)*(b>>16u))<<16u);
#endif
    const uint32_t nonzero=((significands|((significands&0x7fff7fffu)+0x7fff7fffu))&0x80008000u)>>15u;
    const uint32_t mask=nonzero*65535u;
    const uint32_t exponents=((left>>8u)&0x00ff00ffu)+((right>>8u)&0x00ff00ffu);
    return {significands,(exponents&mask)|(0x00790079u&~mask)};
}
QRT_PREPARED_PAIRS_INLINE uint32_t maximum_pair(uint32_t left,uint32_t right) {
#if defined(__HIP_DEVICE_COMPILE__) && defined(__AMDGCN__)
    uint32_t result;
    asm("v_pk_max_u16 %0, %1, %2" : "=v"(result) : "v"(left),"v"(right));
    return result;
#else
    const uint32_t lo=(left&65535u)>(right&65535u)?left&65535u:right&65535u;
    const uint32_t hi=(left>>16u)>(right>>16u)?left>>16u:right>>16u;
    return lo|(hi<<16u);
#endif
}
QRT_PREPARED_PAIRS_INLINE qrt_q1_moe_hawkeye::Value accumulate(
    qrt_q1_moe_hawkeye::Value carry,const Row& left,const Row& right) {
    Products products[8];
    const unsigned initial=unsigned(carry.exponent>-133?int(carry.exponent)+254:121);
    uint32_t maximum=initial*0x00010001u;
#pragma unroll
    for(unsigned i=0u;i<8u;++i) {
        products[i]=multiply(left.pairs[i],right.pairs[i]);
        maximum=maximum_pair(maximum,products[i].exponents);
    }
    const int biased_maximum=int((maximum&65535u)>(maximum>>16u)?maximum&65535u:maximum>>16u);
    const int exponent=biased_maximum-254;
    const unsigned carry_shift=unsigned(exponent-carry.exponent);
    const uint32_t aligned=carry_shift>=32u?0u:(carry.significand<<2u)>>carry_shift;
    uint32_t total=carry.negative?0u-aligned:aligned;
    const uint32_t signs=left.negative^right.negative;
#pragma unroll
    for(unsigned i=0u;i<8u;++i) {
        const unsigned low_shift=unsigned(biased_maximum-int(products[i].exponents&65535u));
        const unsigned high_shift=unsigned(biased_maximum-int(products[i].exponents>>16u));
        const uint32_t low=low_shift>=32u?0u:((products[i].significands&65535u)<<11u)>>low_shift;
        const uint32_t high=high_shift>=32u?0u:((products[i].significands>>16u)<<11u)>>high_shift;
        total+=(signs&(1u<<(2u*i)))?0u-low:low;
        total+=(signs&(2u<<(2u*i)))?0u-high:high;
    }
    // The proven bounded K16 decoder handles sums beyond INT32_MAX exactly.
    const auto signed_sum=qrt_sm121_group16::decode_modulo_sum(total,(signs&1u)!=0u);
    return qrt_sm121_canonical::normalize(signed_sum.magnitude,signed_sum.negative,exponent);
}
} // namespace qrt_sm121_prepared_integer_pairs
#undef QRT_PREPARED_PAIRS_INLINE
#endif
