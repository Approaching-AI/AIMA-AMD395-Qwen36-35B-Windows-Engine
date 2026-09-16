#pragma once
#include "sm121_f32_carry.h"

// Isolated scalar experiment. The caller owns a wave-uniform FP32 RZ scope;
// only explicitly bounded zero/normal finite endpoints enter this route.
namespace qrt_sm121_native_rz_carry {
namespace alignment = qrt_sm121_float_alignment;

#if defined(__HIPCC__)
__device__ __forceinline__ unsigned enter() {
    unsigned saved;
    asm volatile("s_getreg_b32 %0, hwreg(HW_REG_MODE, 0, 2)\n\t"
                 "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 3\n\t"
                 : "=&s"(saved) : : "memory");
    return saved;
}
__device__ __forceinline__ void leave(unsigned saved) {
    asm volatile("s_setreg_b32 hwreg(HW_REG_MODE, 0, 2), %0\n\t"
                 : : "s"(saved) : "memory");
}
// Volatile arithmetic remains ordered after leave(), including exceptional
// subnormal endpoints. This avoids relying on C++ floating-environment rules.
__device__ __forceinline__ float restored_scale(float value, float scale) {
    float result;
    asm volatile("v_mul_f32_e64 %0, %1, %2\n\t"
                 : "=v"(result) : "v"(value), "v"(scale) : "memory");
    return result;
}
#define QRT_NATIVE_RZ_INLINE __host__ __device__ __forceinline__
#else
#define QRT_NATIVE_RZ_INLINE inline
#endif

// A nonzero uint32 has exponent 0..31. After scaling by 2^(maximum-25),
// maximum -101..121 therefore guarantees exponent -126..127. This bound
// prevents RZ overflow saturation and underflow; inspecting the result for
// infinity alone would not detect overflow in RZ mode. Zero always clears sign.
QRT_NATIVE_RZ_INLINE bool normalize(uint32_t magnitude, bool negative,
    int maximum, float* output) {
    if (maximum < -101 || maximum > 121) return false;
    float scaled;
#if defined(__HIP_DEVICE_COMPILE__)
    float converted;
    const int power = maximum - 25;
    asm volatile("v_cvt_f32_u32 %0, %2\n\t"
                 "v_ldexp_f32 %1, %0, %3\n\t"
                 : "=&v"(converted), "=&v"(scaled)
                 : "v"(magnitude), "v"(power) : "memory");
#else
    // Host specification; the native instruction and wave state are checked
    // separately on gfx1151 against independent canonical normalization.
    if (!qrt_sm121_f32_carry::normalize<0u>(magnitude, false, maximum, &scaled)) return false;
#endif
    *output = alignment::from_bits(qrt_sm121_f32_carry::bits(scaled) |
        (negative && magnitude ? 0x80000000u : 0u));
    return true;
}

QRT_NATIVE_RZ_INLINE bool accumulate(float carry, const alignment::Group& group,
    float* output) {
    const uint32_t absolute = qrt_sm121_f32_carry::bits(carry) & 0x7fffffffu;
    const int exponent = absolute ? int(absolute >> 23u) - 127 : -133;
    const int maximum = group.maximum > exponent ? group.maximum : exponent;
    if (maximum == -133 && !absolute) { *output = 0.0f; return true; }
    if (maximum < -101 || maximum > 121) return false;
    const float scale = alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = uint32_t(int32_t(carry * scale));
#if defined(__HIP_DEVICE_COMPILE__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i)
        modulo += uint32_t(int32_t(group.products[i] * scale));
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo, group.first_negative);
    return normalize(sum.magnitude, sum.negative, maximum, output);
}
#undef QRT_NATIVE_RZ_INLINE
} // namespace qrt_sm121_native_rz_carry
