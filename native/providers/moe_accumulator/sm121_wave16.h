#ifndef QRT_SM121_WAVE16_H
#define QRT_SM121_WAVE16_H

#include <hip/hip_runtime.h>
#include "q1_moe_hawkeye_bf16_accumulator.h"
#include "sm121_group16_modulo.h"

// Shared exact K16 / internal-width-26 arithmetic for attention, projection,
// routed MoE and recurrent-state kernels. Only subgroup lane zero owns the
// canonical accumulator. The other lanes contribute one BF16 product each.
namespace qrt_sm121_wave16 {
constexpr unsigned int kGroup = 16u;
constexpr int16_t kZeroExponent = -133;

__device__ __forceinline__ qrt_q1_moe_hawkeye::Value
normalize(
    uint32_t magnitude,
    bool negative,
    int max_exponent
) {
    constexpr int kInternalSignificandWidth = 26;
    constexpr int kInternalToFp32Shift =
        kInternalSignificandWidth - 24;
    constexpr int16_t kFp32MinNonzeroExponent = -126;

    const unsigned int width =
        magnitude == 0u ? 0u : 32u - static_cast<unsigned int>(__clz(magnitude));
    if (width == 0u) {
        return qrt_q1_moe_hawkeye::Value{
            0u,
            kZeroExponent,
            negative
        };
    }

    int exponent = max_exponent + static_cast<int>(width) -
        kInternalSignificandWidth;
    uint32_t normalized = magnitude;
    if (width > static_cast<unsigned int>(kInternalSignificandWidth)) {
        normalized >>= width -
            static_cast<unsigned int>(kInternalSignificandWidth);
    } else {
        normalized <<= static_cast<unsigned int>(kInternalSignificandWidth) -
            width;
    }
    if (exponent < kFp32MinNonzeroExponent) {
        const unsigned int underflow_shift = static_cast<unsigned int>(
            kFp32MinNonzeroExponent - exponent
        );
        normalized = underflow_shift >= 32u
            ? 0u
            : normalized >> underflow_shift;
        exponent = kFp32MinNonzeroExponent;
    }
    normalized >>= kInternalToFp32Shift;
    if (normalized == 0u) {
        return qrt_q1_moe_hawkeye::Value{
            0u,
            kZeroExponent,
            negative
        };
    }
    return qrt_q1_moe_hawkeye::Value{
        static_cast<uint32_t>(normalized),
        static_cast<int16_t>(exponent),
        negative
    };
}

/*
 * One wave64 owns four independent Blackwell K16 groups.  A 16-lane subgroup
 * computes the same max-exponent alignment and signed integer sum as
 * group_sum<26, -133>, but distributes the 16 BF16 products across its lanes.
 * Unsigned reduction preserves the exact signed sum modulo 2^32. The bounded
 * K16 decoder recovers its sign even where the magnitude exceeds INT32_MAX;
 * no signed overflow or 64-bit wave shuffle is needed.
 */
__device__ __forceinline__ qrt_q1_moe_hawkeye::Value
accumulate(
    qrt_q1_moe_hawkeye::Value accumulator,
    uint16_t left,
    uint16_t right,
    unsigned int subgroup_lane
) {
    constexpr int kInternalToFp32Shift = 2;

    const qrt_q1_moe_hawkeye::Value product =
        qrt_q1_moe_hawkeye::multiply_bf16(
            left,
            right,
            kZeroExponent
        );
    const uint32_t accumulator_packed = __shfl(
        accumulator.significand |
            (accumulator.negative ? 0x80000000u : 0u),
        0,
        kGroup
    );
    const int accumulator_exponent = __shfl(
        static_cast<int>(accumulator.exponent),
        0,
        kGroup
    );
    const uint32_t accumulator_significand = accumulator_packed & 0x00ffffffu;
    const bool accumulator_negative = (accumulator_packed & 0x80000000u) != 0u;
    int max_exponent = static_cast<int>(product.exponent) >
            accumulator_exponent
        ? static_cast<int>(product.exponent)
        : accumulator_exponent;
    for (unsigned int lane_mask = kGroup / 2u;
         lane_mask != 0u;
         lane_mask >>= 1u) {
        const int other_exponent = __shfl_xor(
            max_exponent,
            lane_mask,
            kGroup
        );
        if (other_exponent > max_exponent) {
            max_exponent = other_exponent;
        }
    }

    const int product_shift =
        max_exponent - static_cast<int>(product.exponent);
    const uint32_t product_aligned = product_shift >= 32
        ? 0u
        : (product.significand << kInternalToFp32Shift) >>
              static_cast<unsigned int>(product_shift);
    uint32_t modulo_significand = product.negative
        ? 0u - product_aligned
        : product_aligned;
    if (subgroup_lane == 0u) {
        const int accumulator_shift =
            max_exponent - accumulator_exponent;
        const uint32_t accumulator_aligned = accumulator_shift >= 32
            ? 0u
            : (accumulator_significand << kInternalToFp32Shift) >>
                  static_cast<unsigned int>(accumulator_shift);
        modulo_significand += accumulator_negative
            ? 0u - accumulator_aligned
            : accumulator_aligned;
    }
    for (unsigned int lane_mask = kGroup / 2u;
         lane_mask != 0u;
         lane_mask >>= 1u) {
        modulo_significand += __shfl_down(
            modulo_significand,
            lane_mask,
            kGroup
        );
    }
    if (subgroup_lane == 0u) {
        const qrt_sm121_group16::SignedMagnitude sum =
            qrt_sm121_group16::decode_modulo_sum(
                modulo_significand,
                product.negative
            );
        accumulator = normalize(
            sum.magnitude,
            sum.negative,
            max_exponent
        );
    }
    return accumulator;
}

}  // namespace qrt_sm121_wave16
#endif
