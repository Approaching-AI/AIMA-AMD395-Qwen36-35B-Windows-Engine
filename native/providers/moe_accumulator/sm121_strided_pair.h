#ifndef QRT_SM121_STRIDED_PAIR_H
#define QRT_SM121_STRIDED_PAIR_H

#include "sm121_group16_modulo.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "sm121_wave16.h"
#define QRT_SM121_PAIR_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SM121_PAIR_INLINE inline
#endif

namespace qrt_sm121_strided_pair {
constexpr unsigned kDistance = 16u;
constexpr unsigned kProducts = 8u;
constexpr unsigned kCellsPer256Threads = 128u;

// Each half-wave reads sixteen adjacent keys or value columns. The other
// half-wave owns the remaining eight products of the same sixteen dots.
QRT_SM121_PAIR_INLINE unsigned cell(unsigned thread) {
    return (thread / 32u) * 16u + thread % 16u;
}
QRT_SM121_PAIR_INLINE unsigned part(unsigned thread) {
    return (thread / 16u) & 1u;
}
QRT_SM121_PAIR_INLINE int maximum(
    qrt_q1_moe_hawkeye::Value accumulator, const uint32_t (&products)[kProducts]) {
    int result = accumulator.exponent > -133 ? accumulator.exponent : -133;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < kProducts; ++i) {
        const int exponent = qrt_sm121_group16::packed_exponent(products[i]);
        result = exponent > result ? exponent : result;
    }
    return result;
}
QRT_SM121_PAIR_INLINE uint32_t modulo_products(
    int max_exponent, const uint32_t (&products)[kProducts]) {
    uint32_t sum = 0u;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < kProducts; ++i) {
        const uint32_t product = products[i];
        const unsigned shift = unsigned(max_exponent - qrt_sm121_group16::packed_exponent(product));
        const uint32_t magnitude = shift >= 32u ? 0u : ((product & 0xffffu) << 11u) >> shift;
        sum += (product & 0x80000000u) ? 0u - magnitude : magnitude;
    }
    return sum;
}

#if defined(__HIPCC__) || defined(__CUDACC__)
// Carry is replicated across the pair. Exchange the maximum exponent and
// the product sum once each, then add exactly one carry in both lanes.
__device__ __forceinline__ qrt_q1_moe_hawkeye::Value accumulate(
    qrt_q1_moe_hawkeye::Value accumulator, const uint32_t (&products)[kProducts]) {
    int max_exponent = maximum(accumulator, products);
    const int other = __shfl_xor(max_exponent, kDistance, 32u);
    max_exponent = other > max_exponent ? other : max_exponent;
    uint32_t sum = modulo_products(max_exponent, products);
    sum += __shfl_xor(sum, kDistance, 32u);
    const unsigned shift = unsigned(max_exponent - accumulator.exponent);
    const uint32_t carried = shift >= 32u ? 0u : (accumulator.significand << 2u) >> shift;
    sum += accumulator.negative ? 0u - carried : carried;
    // Either half's first product resolves the overlapping modulo interval:
    // a magnitude in that interval requires all sixteen product signs equal.
    const auto decoded = qrt_sm121_group16::decode_modulo_sum(sum, (products[0] & 0x80000000u) != 0u);
    return qrt_sm121_wave16::normalize(decoded.magnitude, decoded.negative, max_exponent);
}
#endif
} // namespace qrt_sm121_strided_pair

#undef QRT_SM121_PAIR_INLINE
#endif
