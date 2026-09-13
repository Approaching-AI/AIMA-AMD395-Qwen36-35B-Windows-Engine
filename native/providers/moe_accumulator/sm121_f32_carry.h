#ifndef QRT_SM121_F32_CARRY_H
#define QRT_SM121_F32_CARRY_H
#include "sm121_float_alignment.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_F32_CARRY_INLINE __host__ __device__ __forceinline__
#else
#define QRT_F32_CARRY_INLINE inline
#endif

namespace qrt_sm121_f32_carry {
namespace alignment = qrt_sm121_float_alignment;
namespace original = qrt_q1_moe_hawkeye;

// Component experiment: retain an ordered K16 carry in one FP32 register.
// The product/exponent alignment and unsigned modulo reduction are unchanged.
// Only zero or normal finite FP32 endpoints enter the next fast group. A
// rejected group must restart the original dot, including its extended carry.
QRT_F32_CARRY_INLINE uint32_t bits(float value) {
    uint32_t result; __builtin_memcpy(&result, &value, 4u); return result;
}
QRT_F32_CARRY_INLINE unsigned leading(uint32_t magnitude) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __clz(magnitude | 1u);
#else
    return 32u - original::bit_width_u64(magnitude | 1u);
#endif
}

// Both variants discard exactly the low max(width-24,0) integer bits.
// Variant 0 encodes the resulting significand directly. Variant 1 uses the
// device's explicitly round-toward-zero unsigned conversion; changing its
// exponent bits then applies an exact power-of-two scale. Neither changes
// the wave's floating-point rounding mode. Subnormal and overflow endpoints
// select the original implementation rather than relying on FP32 scaling.
template<unsigned Method>
QRT_F32_CARRY_INLINE bool normalize(uint32_t magnitude, bool negative,
    int maximum, float* output) {
    static_assert(Method < 2u);
    if (!magnitude) { *output = 0.0f; return true; }
    uint32_t mantissa; int exponent;
    if constexpr (Method == 0u) {
        const unsigned shift = leading(magnitude);
        mantissa = ((magnitude << shift) >> 8u) & 0x7fffffu;
        exponent = maximum + 6 - int(shift);
    } else {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
        const uint32_t converted = bits(__uint2float_rz(magnitude));
#else
        const unsigned shift = leading(magnitude);
        const uint32_t truncated = shift < 8u ? (magnitude >> (8u - shift)) << (8u - shift) : magnitude;
        const uint32_t converted = bits(float(truncated));
#endif
        mantissa = converted & 0x7fffffu;
        exponent = int(converted >> 23u) - 127 + maximum - 25;
    }
    if (exponent < -126 || exponent > 127) return false;
    *output = alignment::from_bits((negative ? 0x80000000u : 0u) |
        (uint32_t(exponent + 127) << 23u) | mantissa);
    return true;
}

// Preconditions: the caller prevalidates all BF16 operands using eligible(),
// and carry is the zero/normal result of the preceding accepted group.
template<unsigned Method>
QRT_F32_CARRY_INLINE bool accumulate(float carry, const alignment::Group& group,
    float* output) {
    const uint32_t absolute_carry = bits(carry) & 0x7fffffffu;
    const int carry_exponent = absolute_carry ? int(absolute_carry >> 23u) - 127 : -133;
    const int maximum = group.maximum > carry_exponent ? group.maximum : carry_exponent;
    if (maximum == -133 && !absolute_carry) { *output = 0.0f; return true; }
    if (maximum < -101 || maximum > 127) return false;
    const float scale = alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = uint32_t(int32_t(carry * scale));
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i)
        modulo += uint32_t(int32_t(group.products[i] * scale));
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, group.first_negative);
    return normalize<Method>(sum.magnitude, sum.negative, maximum, output);
}
} // namespace qrt_sm121_f32_carry
#undef QRT_F32_CARRY_INLINE
#endif
