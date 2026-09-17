#pragma once
#include "sm121_f32_carry.h"
#include "sm121_float_row_bounds.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_CERTIFIED_GROUP_INLINE __host__ __device__ __forceinline__
#else
#define QRT_CERTIFIED_GROUP_INLINE inline
#endif

namespace qrt_sm121_certified_f32_group {
namespace bounds = qrt_sm121_float_row_bounds;
namespace carry_math = qrt_sm121_f32_carry;

// Reuse the exact row-grid certificate with a normal FP32 carry. A higher
// alignment grid is admitted only if it discards no product or carry bits;
// alternatively a dominating carry determines the original maximum exactly.
// Rejection requests the original paired-exponent scan for this group.
QRT_CERTIFIED_GROUP_INLINE bool exponent(float carry, uint32_t left,
    uint32_t right, int* output) {
    const uint32_t raw = carry_math::bits(carry), absolute = raw & 0x7fffffffu;
    const unsigned biased = absolute >> 23u;
    if (absolute && (!biased || biased == 255u)) return false;
    const qrt_q1_moe_hawkeye::Value value{
        absolute ? (absolute & 0x7fffffu) | 0x800000u : 0u,
        int16_t(absolute ? int(biased) - 127 : -133), bool(raw >> 31u)};
    int maximum;
    if (!bounds::exponent(value, left, right, &maximum) || maximum > 127)
        return false;
    *output = maximum;
    return true;
}

// Preconditions: exponent() accepted these same operands and incoming carry.
// Scalar BF16 products are exact in the admitted range. Unsigned modulo
// reduction and the original normalizer retain the original K16 endpoint.
QRT_CERTIFIED_GROUP_INLINE bool accumulate(float carry, const float* products,
    bool first_negative, int maximum, float* output) {
    const float scale = qrt_sm121_float_alignment::from_bits(
        uint32_t(152 - maximum) << 23u);
    uint32_t modulo = uint32_t(int32_t(carry * scale));
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i)
        modulo += uint32_t(int32_t(products[i] * scale));
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, first_negative);
    return carry_math::normalize<0u>(sum.magnitude, sum.negative, maximum, output);
}
} // namespace qrt_sm121_certified_f32_group
#undef QRT_CERTIFIED_GROUP_INLINE
