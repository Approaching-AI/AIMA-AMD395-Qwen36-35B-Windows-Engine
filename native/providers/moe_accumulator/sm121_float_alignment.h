#ifndef QRT_SM121_FLOAT_ALIGNMENT_H
#define QRT_SM121_FLOAT_ALIGNMENT_H
#include "sm121_group16_modulo.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_FLOAT_ALIGN_INLINE __host__ __device__ __forceinline__
#else
#define QRT_FLOAT_ALIGN_INLINE inline
#endif

namespace qrt_sm121_float_alignment {
// Scalar FP32 multiplication of two BF16 significands is exact (at most
// 16 significant bits). This range keeps every nonzero product normal and
// finite. Matrix/DOT2 floating instructions are deliberately not used.
QRT_FLOAT_ALIGN_INLINE bool eligible(uint16_t value) {
    const unsigned exponent = (value >> 7u) & 255u;
    return !(value & 0x7fffu) || (exponent >= 64u && exponent <= 190u);
}
QRT_FLOAT_ALIGN_INLINE float from_bits(uint32_t bits) {
    float value; __builtin_memcpy(&value, &bits, 4u); return value;
}
struct Group {
    float products[16];
    int maximum = -133;
    bool first_negative = false;

    // The caller establishes eligibility before entering the product loop.
    QRT_FLOAT_ALIGN_INLINE void set(unsigned i, uint16_t left, uint16_t right) {
        products[i] = from_bits(uint32_t(left) << 16u) * from_bits(uint32_t(right) << 16u);
        const int exponent = (left & 0x7fffu) && (right & 0x7fffu) ?
            int((left >> 7u) & 255u) + int((right >> 7u) & 255u) - 254 : -133;
        maximum = exponent > maximum ? exponent : maximum;
        if (!i) first_negative = ((left ^ right) & 0x8000u) != 0u;
    }
};

QRT_FLOAT_ALIGN_INLINE bool sum(qrt_q1_moe_hawkeye::Value carry,
    const Group& group, qrt_sm121_group16::AlignedSum* output) {
    const int maximum = group.maximum > carry.exponent ? group.maximum : carry.exponent;
    if (maximum == -133 && !carry.significand) {
        *output = {{0u, false}, -133}; return true;
    }
    // The power-of-two scale is a finite normal FP32 value. An overflowing
    // carried Value cannot be converted to scalar FP32 and uses the original
    // integer path. Tiny groups likewise retain the original implementation.
    if (maximum < -101 || maximum > 127 || carry.exponent > 127) return false;
    const float scale = from_bits(uint32_t(127 + 25 - maximum) << 23u);
    // p * 2^(25-maximum) equals the original product significand << 11
    // followed by exponent alignment. Power-of-two multiplication is exact
    // for every nonzero integer result; underflow can only produce |x|<1.
    // C++ float-to-int conversion truncates toward zero, matching the
    // original signed-magnitude shift. Each term fits int32; only their
    // sum requires the existing unsigned modulo decoder.
    uint32_t modulo = uint32_t(int32_t(qrt_q1_moe_hawkeye::value_to_float(carry) * scale));
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i)
        modulo += uint32_t(int32_t(group.products[i] * scale));
    *output = {qrt_sm121_group16::decode_modulo_sum(modulo, group.first_negative), maximum};
    return true;
}
}  // namespace qrt_sm121_float_alignment
#undef QRT_FLOAT_ALIGN_INLINE
#endif
