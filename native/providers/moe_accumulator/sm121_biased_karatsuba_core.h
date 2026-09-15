#ifndef QRT_SM121_BIASED_KARATSUBA_CORE_H
#define QRT_SM121_BIASED_KARATSUBA_CORE_H
#include "sm121_karatsuba_core.h"
#include "sm121_positive_karatsuba.h"
#if defined(__HIPCC__)
#define QRT_BIASED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_BIASED_INLINE inline
#endif
namespace qrt_sm121_biased_karatsuba {
using qrt_sm121_karatsuba::Digits;
using qrt_sm121_karatsuba::split;
using qrt_sm121_karatsuba::reconstruct;
constexpr float bias = 12582912.0f;
struct Prepared { uint16_t high[16], low[16]; };
static_assert(sizeof(Prepared) == 64u);
QRT_BIASED_INLINE uint16_t signed_half(int value) {
    const unsigned magnitude = unsigned(value < 0 ? -value : value);
    return uint16_t((value < 0 ? 0x8000u : 0u) |
        qrt_sm121_positive_karatsuba::positive_half_bits(magnitude));
}
template<class Row>
QRT_BIASED_INLINE uint16_t core(const Row& row, unsigned i) {
    const unsigned word = i / 4u, shift = i % 4u * 8u;
    return uint16_t((((uint32_t(row.high[word]) >> shift) & 255u) << 8u) |
        ((uint32_t(row.low[word]) >> shift) & 255u));
}
template<class Row>
QRT_BIASED_INLINE Prepared prepare(const Row& row) {
    Prepared result{};
    for (unsigned i = 0u; i < 16u; ++i) {
        const auto digits = split(core(row, i));
        result.high[i] = signed_half(digits.high);
        result.low[i] = signed_half(digits.low);
    }
    return result;
}
#if defined(__HIPCC__)
using Bf16x16 = unsigned short __attribute__((ext_vector_type(16)));
using F16x16 = _Float16 __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
struct Parts { F32x8 high, low, combined; };
__device__ __forceinline__ F32x8 carry_bias() {
    F32x8 result;
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) result[i] = bias;
    return result;
}
// Component experiment only. Fixed biased C is part of each WMMA, and exact
// FP32 subtraction removes it. Balanced digits and their sums fit [-256,256];
// the mathematical biased dot stays in the exponent-23 unit-spaced interval.
// That range and Sterbenz subtraction do not prove the hardware dot itself.
// Native raw partial/reconstruction checks and real captures must qualify it;
// no nearest-integer repair or product dispatcher is introduced here.
template<class Row>
__device__ __forceinline__ Parts products(const Row& left, const Row& right) {
    Bf16x16 ah{}, al{}, as{}, bh{}, bl{}, bs{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        const auto x = split(core(left, i)), y = split(core(right, i));
        ah[i] = qrt_sm121_karatsuba::small_bf16(x.high);
        al[i] = qrt_sm121_karatsuba::small_bf16(x.low);
        as[i] = qrt_sm121_karatsuba::small_bf16(x.high + x.low);
        bh[i] = qrt_sm121_karatsuba::small_bf16(y.high);
        bl[i] = qrt_sm121_karatsuba::small_bf16(y.low);
        bs[i] = qrt_sm121_karatsuba::small_bf16(y.high + y.low);
    }
    const auto carry = carry_bias();
    return {__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(ah, bh, carry) - carry,
        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(al, bl, carry) - carry,
        __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(as, bs, carry) - carry};
}
__device__ __forceinline__ Parts products_prepared(const Prepared& left, const Prepared& right) {
    F16x16 ah{}, al{}, bh{}, bl{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        ah[i] = __builtin_bit_cast(_Float16, left.high[i]); al[i] = __builtin_bit_cast(_Float16, left.low[i]);
        bh[i] = __builtin_bit_cast(_Float16, right.high[i]); bl[i] = __builtin_bit_cast(_Float16, right.low[i]);
    }
    const auto carry = carry_bias();
    const auto as = ah + al, bs = bh + bl;
    return {__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(ah, bh, carry) - carry,
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(al, bl, carry) - carry,
        __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(as, bs, carry) - carry};
}
#endif
}
#undef QRT_BIASED_INLINE
#endif
