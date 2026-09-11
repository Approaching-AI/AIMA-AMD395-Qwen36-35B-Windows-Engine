#ifndef QRT_SM121_GROUP16_MODULO_H
#define QRT_SM121_GROUP16_MODULO_H

#include <cstdint>
#include "q1_moe_hawkeye_bf16_accumulator.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SM121_GROUP16_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SM121_GROUP16_INLINE inline
#endif

namespace qrt_sm121_group16 {

// K16, internal width 26: BF16 products have (8 x 8)-bit significands
// shifted by 9, followed by the two alignment guard bits. The carried FP32
// accumulator has at most 24 significand bits. Exponent alignment only
// decreases these bounds.
constexpr uint32_t kMaxAlignedProduct = (255u * 255u) << 11u;
constexpr uint32_t kMaxAlignedAccumulator = 0x00ffffffu << 2u;
constexpr uint32_t kMaxMagnitude =
    16u * kMaxAlignedProduct + kMaxAlignedAccumulator;
constexpr uint32_t kMinNegativeModulo = 0u - kMaxMagnitude;
constexpr uint32_t kMaxMixedProductSignsMagnitude =
    15u * kMaxAlignedProduct + kMaxAlignedAccumulator;
static_assert(kMaxMagnitude == 2197848060u);
static_assert(kMaxMagnitude > 0x7fffffffu);
static_assert(kMaxMixedProductSignsMagnitude < kMinNegativeModulo);

struct SignedMagnitude {
    uint32_t magnitude;
    bool negative;
};

// The exact sum may exceed INT32_MAX, so a signed 32-bit reduction is unsafe.
// Unsigned addition instead preserves it modulo 2^32. Positive and negative
// representations overlap only in [2^32 - max, max]. A sum in that interval
// exceeds the mixed-product-sign bound, so all 16 products have the same
// sign. Any product's sign resolves the overlap, including lane zero's.
// Outside it the representation uniquely determines the sign. This is for
// exactly 16 bounded products plus one bounded accumulator, not a generic
// replacement for signed 64-bit reductions.
QRT_SM121_GROUP16_INLINE SignedMagnitude decode_modulo_sum(
    uint32_t modulo_sum,
    bool first_product_negative
) {
    const bool negative = modulo_sum >= kMinNegativeModulo &&
        (modulo_sum > kMaxMagnitude || first_product_negative);
    return SignedMagnitude{
        negative ? 0u - modulo_sum : modulo_sum,
        negative
    };
}

// BF16 products carry sixteen significant product bits and nine zero bits.
// Retain the product, its biased exponent and sign in one register per K cell
// while a lane computes sixteen independent products before their common
// exponent is known. The input must come from multiply_bf16(..., -133).
QRT_SM121_GROUP16_INLINE uint32_t pack_product(qrt_q1_moe_hawkeye::Value product) {
    return (product.significand >> 9u) |
        (static_cast<uint32_t>(static_cast<int>(product.exponent) + 254) << 16u) |
        (product.negative ? 0x80000000u : 0u);
}

QRT_SM121_GROUP16_INLINE int packed_exponent(uint32_t product) {
    return static_cast<int>((product >> 16u) & 0x01ffu) - 254;
}

struct AlignedSum {
    SignedMagnitude value;
    int max_exponent;
};

QRT_SM121_GROUP16_INLINE AlignedSum sum_packed(
    qrt_q1_moe_hawkeye::Value accumulator, const uint32_t (&products)[16]
) {
    int max_exponent = accumulator.exponent > -133 ? accumulator.exponent : -133;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned int item = 0u; item < 16u; ++item) {
        const int exponent = packed_exponent(products[item]);
        max_exponent = exponent > max_exponent ? exponent : max_exponent;
    }
    const unsigned int accumulator_shift = static_cast<unsigned int>(
        max_exponent - accumulator.exponent);
    const uint32_t aligned = accumulator_shift >= 32u
        ? 0u : (accumulator.significand << 2u) >> accumulator_shift;
    uint32_t modulo_sum = accumulator.negative ? 0u - aligned : aligned;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned int item = 0u; item < 16u; ++item) {
        const uint32_t product = products[item];
        const unsigned int shift = static_cast<unsigned int>(
            max_exponent - packed_exponent(product));
        const uint32_t magnitude = shift >= 32u
            ? 0u : ((product & 0xffffu) << 11u) >> shift;
        modulo_sum += (product & 0x80000000u) ? 0u - magnitude : magnitude;
    }
    return {decode_modulo_sum(modulo_sum, (products[0] & 0x80000000u) != 0u),
            max_exponent};
}

// A final one-value group preserves an already normalized FP32 accumulator,
// except that its integer zero sum clears a negative zero produced by
// underflow. Keep that endpoint without repeating exponent normalization.
QRT_SM121_GROUP16_INLINE qrt_q1_moe_hawkeye::Value finish_accumulator(
    qrt_q1_moe_hawkeye::Value value
) {
    if (value.significand == 0u) value.negative = false;
    return value;
}

}  // namespace qrt_sm121_group16

#undef QRT_SM121_GROUP16_INLINE
#endif
