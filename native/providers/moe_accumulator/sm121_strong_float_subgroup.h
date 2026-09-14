#pragma once
#include "sm121_float_alignment.h"
#include "sm121_canonical_normalize.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "sm121_float_subgroup.h"
#define QRT_STRONG_INLINE __host__ __device__ __forceinline__
#else
#define QRT_STRONG_INLINE inline
#endif

namespace qrt_sm121_strong_float {
namespace alignment = qrt_sm121_float_alignment;
using Value = qrt_q1_moe_hawkeye::Value;

// For two certified rows every nonzero product is a multiple of 2^-100:
// (84-127-7)*2 = -100. Alignment and canonical truncation preserve this
// lattice, including cancellation followed by all-zero K16 groups. With
// K<=4096 and input exponents<=174 the absolute sum is below 2^108.
// Thus a nonzero group's maximum stays in [-100,108], within the finite
// normal-scale range [-101,151]. Exponent80 is insufficient: four products
// (a+d)(b+d)-a(b+d)-(a+d)b+ab leave 2^-108, then zero groups require fallback.
QRT_STRONG_INLINE bool eligible(uint16_t x) {
    const unsigned exponent = (x >> 7u) & 255u;
    return !(x & 0x7fffu) || (exponent >= 84u && exponent <= 174u);
}
struct Product { float value; int exponent; };
QRT_STRONG_INLINE Product prepare(uint16_t a, uint16_t b) {
    return {alignment::from_bits(uint32_t(a) << 16u) * alignment::from_bits(uint32_t(b) << 16u),
        (a & 0x7fffu) && (b & 0x7fffu) ? int((a >> 7u) & 255u) + int((b >> 7u) & 255u) - 254 : -133};
}
QRT_STRONG_INLINE bool negative(float value) {
    uint32_t bits; __builtin_memcpy(&bits, &value, 4u); return (bits >> 31u) != 0u;
}
QRT_STRONG_INLINE Value finish(Value carry, uint32_t modulo, int maximum, bool first_negative) {
    const unsigned shift = unsigned(maximum - carry.exponent);
    const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
    modulo += carry.negative ? 0u - aligned : aligned;
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, first_negative);
    return qrt_sm121_canonical::normalize(sum.magnitude, sum.negative, maximum);
}
// Serial host/device form for independent numerical comparisons. Callers
// certify both complete rows and K<=4096 before entering either arithmetic loop.
QRT_STRONG_INLINE Value group(Value carry, const Product* products) {
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    for (unsigned i = 0u; i < 16u; ++i)
        maximum = products[i].exponent > maximum ? products[i].exponent : maximum;
    if (maximum == -133 && !carry.significand) return {0u, -133, false};
    const float scale = alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = 0u;
    for (unsigned i = 0u; i < 16u; ++i) modulo += uint32_t(int32_t(products[i].value * scale));
    return finish(carry, modulo, maximum, negative(products[0].value));
}

#if defined(__HIPCC__) || defined(__CUDACC__)
template<unsigned Lanes>
__device__ __forceinline__ Value accumulate(Value carry, const Product* products) {
    constexpr unsigned items = 16u / Lanes;
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
#pragma unroll
    for (unsigned i = 0u; i < items; ++i)
        maximum = products[i].exponent > maximum ? products[i].exponent : maximum;
    maximum = qrt_sm121_lane_reduce::maximum<Lanes>(maximum);
    if (maximum == -133 && !carry.significand) return {0u, -133, false};
    const float scale = alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = 0u;
#pragma unroll
    for (unsigned i = 0u; i < items; ++i) modulo += uint32_t(int32_t(products[i].value * scale));
    modulo = qrt_sm121_lane_reduce::sum<Lanes>(modulo);
    return finish(carry, modulo, maximum, negative(products[0].value));
}

template<unsigned Lanes, unsigned Staging = 1u>
__device__ __forceinline__ float dot(const uint16_t* left, const uint16_t* right,
    unsigned count, bool certified_rows, uint32_t* trace = nullptr) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    static_assert(Staging == 1u || Staging == 4u || Staging == 8u);
    if (!certified_rows || count > 4096u)
        return qrt_sm121_float_subgroup::dot<Lanes, Staging>(left, right, count, trace);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < count; base += 16u * Staging) {
        Product products[Staging][items];
#pragma unroll
        for (unsigned g = 0u; g < Staging; ++g) if (base + g * 16u < count) {
            using Packed = typename std::conditional<Lanes == 4u, uint64_t,
                typename std::conditional<Lanes == 8u, uint32_t, uint16_t>::type>::type;
            Packed a, b;
            __builtin_memcpy(&a, left + base + g * 16u + lane * items, sizeof(a));
            __builtin_memcpy(&b, right + base + g * 16u + lane * items, sizeof(b));
#pragma unroll
            for (unsigned i = 0u; i < items; ++i)
                products[g][i] = prepare(uint16_t(a >> (i * 16u)), uint16_t(b >> (i * 16u)));
        }
#pragma unroll
        for (unsigned g = 0u; g < Staging; ++g) if (base + g * 16u < count) {
            carry = accumulate<Lanes>(carry, products[g]);
            if (trace && !lane) {
                const float value = qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
                __builtin_memcpy(trace + base / 16u + g, &value, 4u);
            }
        }
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
#endif
} // namespace qrt_sm121_strong_float
#undef QRT_STRONG_INLINE
