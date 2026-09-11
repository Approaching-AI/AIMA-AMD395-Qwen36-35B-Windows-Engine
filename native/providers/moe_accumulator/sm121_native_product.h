#ifndef QRT_SM121_NATIVE_PRODUCT_H
#define QRT_SM121_NATIVE_PRODUCT_H

#include "sm121_group16_modulo.h"
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_NATIVE_PRODUCT_INLINE __host__ __device__ __forceinline__
#else
#define QRT_NATIVE_PRODUCT_INLINE inline
#endif

namespace qrt_sm121_native_product {
static_assert(qrt_sm121_group16::kMaxAlignedProduct < UINT32_C(0x7fffffff));

QRT_NATIVE_PRODUCT_INLINE uint32_t float_bits(float value) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
#endif
}
QRT_NATIVE_PRODUCT_INLINE float from_bits(uint32_t bits) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __uint_as_float(bits);
#else
    float value; std::memcpy(&value, &bits, sizeof(value)); return value;
#endif
}

// A normal BF16 product has at most sixteen significant bits and is exact
// in FP32. Its lowest eight FP32 bits are zero. Bit zero records the carry
// between the original, unnormalized BF16 exponent and the FP32 exponent.
// Bit one instead tags the original integer representation for subnormals,
// exceptional operands, or products outside the safe normal FP32 range.
QRT_NATIVE_PRODUCT_INLINE uint32_t pack(uint16_t left, uint16_t right) {
    const uint32_t sign = static_cast<uint32_t>((left ^ right) & 0x8000u) << 16u;
    if ((left & 0x7fffu) == 0u || (right & 0x7fffu) == 0u) return sign;
    const int le = (left >> 7u) & 0xffu, re = (right >> 7u) & 0xffu;
    const int exponent = le + re - 254;
    if (le == 0 || re == 0 || le == 255 || re == 255 || exponent < -126 || exponent > 126) {
        const auto product = qrt_q1_moe_hawkeye::multiply_bf16(left, right, -133);
        return ((product.significand >> 9u) << 2u) |
            (static_cast<uint32_t>(static_cast<int>(product.exponent) + 254) << 18u) |
            sign | 2u;
    }
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    const float product = __fmul_rn(from_bits(static_cast<uint32_t>(left) << 16u),
                                   from_bits(static_cast<uint32_t>(right) << 16u));
#else
    const float product = from_bits(static_cast<uint32_t>(left) << 16u) *
                          from_bits(static_cast<uint32_t>(right) << 16u);
#endif
    const uint32_t bits = float_bits(product);
    const uint32_t carry = static_cast<uint32_t>(static_cast<int>((bits >> 23u) & 0xffu) - 127 - exponent);
    return bits | carry;
}

QRT_NATIVE_PRODUCT_INLINE int normal_exponent(uint32_t product) {
    return (product & 0x7ffffffeu) == 0u ? -133 :
        static_cast<int>((product >> 23u) & 0xffu) - 127 - static_cast<int>(product & 1u);
}

QRT_NATIVE_PRODUCT_INLINE uint32_t original_pack(uint32_t product) {
    if ((product & 2u) != 0u) {
        return ((product >> 2u) & 0xffffu) |
            (((product >> 18u) & 0x1ffu) << 16u) | (product & 0x80000000u);
    }
    const uint32_t significand = (product & 0x7ffffffeu) == 0u ? 0u :
        (((product & 0x007ffffeu) | 0x00800000u) << (product & 1u)) >> 9u;
    return significand | (static_cast<uint32_t>(normal_exponent(product) + 254) << 16u) |
        (product & 0x80000000u);
}

QRT_NATIVE_PRODUCT_INLINE qrt_sm121_group16::AlignedSum sum(
    qrt_q1_moe_hawkeye::Value accumulator, const uint32_t (&products)[16]
) {
    uint32_t tags = 0u;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned int i = 0u; i < 16u; ++i) tags |= products[i];
    if ((tags & 2u) != 0u) {
        uint32_t original[16];
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (unsigned int i = 0u; i < 16u; ++i) original[i] = original_pack(products[i]);
        return qrt_sm121_group16::sum_packed(accumulator, original);
    }
    int max_exponent = accumulator.exponent > -133 ? accumulator.exponent : -133;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned int i = 0u; i < 16u; ++i) {
        const int exponent = normal_exponent(products[i]);
        max_exponent = exponent > max_exponent ? exponent : max_exponent;
    }
    const unsigned int shift = static_cast<unsigned int>(max_exponent - accumulator.exponent);
    const uint32_t aligned = shift >= 32u ? 0u : (accumulator.significand << 2u) >> shift;
    uint32_t modulo = accumulator.negative ? 0u - aligned : aligned;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned int i = 0u; i < 16u; ++i) {
        const uint32_t product = products[i];
        const int scaled_exponent = static_cast<int>((product >> 23u) & 0xffu) + 25 - max_exponent;
        if ((product & 0x7ffffffeu) != 0u && scaled_exponent > 0) {
            // Scaling by an exponent edit is exact. Each product is bounded
            // by 255*255*2^11, below INT32_MAX. Conversion truncates each signed
            // contribution toward zero at the original 26-bit alignment grid.
            const float scaled = from_bits((product & 0x807ffffeu) |
                (static_cast<uint32_t>(scaled_exponent) << 23u));
            modulo += static_cast<uint32_t>(static_cast<int32_t>(scaled));
        }
    }
    return {qrt_sm121_group16::decode_modulo_sum(modulo, (products[0] & 0x80000000u) != 0u), max_exponent};
}

}  // namespace qrt_sm121_native_product
#undef QRT_NATIVE_PRODUCT_INLINE
#endif
