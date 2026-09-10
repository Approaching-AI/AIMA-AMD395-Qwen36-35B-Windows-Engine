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
