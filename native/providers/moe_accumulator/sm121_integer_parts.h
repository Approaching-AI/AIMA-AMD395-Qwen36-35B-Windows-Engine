#ifndef QRT_SM121_INTEGER_PARTS_H
#define QRT_SM121_INTEGER_PARTS_H
#include "sm121_group16_modulo.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_INTEGER_INLINE __host__ __device__ __forceinline__
#else
#define QRT_INTEGER_INLINE inline
#endif

namespace qrt_sm121_integer_parts {

// A normal BF16 row spanning at most seven exponents is exactly a signed
// 16-bit integer row times a common power of two: |integer| <= 255*128.
// Split it into an unsigned low byte and a signed high byte. Four integer
// matrix products reconstruct the complete signed dot without FP rounding.
QRT_INTEGER_INLINE int row_minimum(const uint16_t (&row)[18]) {
    int minimum = 255, maximum = 0;
    bool valid = true;
    for (unsigned i = 0u; i < 16u; ++i) {
        if ((row[i] & 0x7fffu) == 0u) continue;
        const int exponent = (row[i] >> 7u) & 255u;
        valid = valid && exponent != 0 && exponent != 255;
        minimum = exponent < minimum ? exponent : minimum;
        maximum = exponent > maximum ? exponent : maximum;
    }
    return !valid || maximum - minimum > 7 ? -1 : (minimum == 255 ? 127 : minimum);
}

QRT_INTEGER_INLINE uint16_t encode(uint16_t value, int minimum) {
    if (minimum < 0 || (value & 0x7fffu) == 0u) return 0u;
    const unsigned shift = static_cast<unsigned>(((value >> 7u) & 255u) - minimum);
    const uint32_t magnitude = (128u | (value & 127u)) << shift;
    return static_cast<uint16_t>((value & 0x8000u) ? 0u - magnitude : magnitude);
}

QRT_INTEGER_INLINE bool sum(qrt_q1_moe_hawkeye::Value carry, const uint32_t (&pairs)[16],
    const int32_t (&partials)[4], int left_minimum, int right_minimum,
    qrt_sm121_group16::AlignedSum* output) {
    if (left_minimum < 0 || right_minimum < 0) return false;
    int maximum = carry.exponent > -133 ? carry.exponent : -133;
    bool any = false;
    int exponents[16];
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t a = static_cast<uint16_t>(pairs[i]), b = static_cast<uint16_t>(pairs[i] >> 16u);
        const bool nonzero = (a & 0x7fffu) != 0u && (b & 0x7fffu) != 0u;
        const int exponent = nonzero ? int((a >> 7u) & 255u) + int((b >> 7u) & 255u) - 254 : -1000;
        exponents[i] = exponent;
        maximum = exponent > maximum ? exponent : maximum;
        any = any || nonzero;
    }
    const int minimum = left_minimum + right_minimum - 254;
    const int shift = maximum - minimum - 11;
    // Each original encoded product is smaller than 2^30. At shifts >=30
    // every individually truncated product is zero, irrespective of its sign.
    int64_t products = 0;
    if (any && shift < 30) {
        const int64_t mathematical = int64_t(partials[0]) * 65536 +
            (int64_t(partials[1]) + partials[2]) * 256 + partials[3];
        if (shift <= 0) {
            products = mathematical * (int64_t(1) << (-shift));
        } else {
            int64_t discarded = 0;
            const uint32_t mask = (1u << shift) - 1u;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
            for (unsigned i = 0u; i < 16u; ++i) {
                if (exponents[i] == -1000) continue;
                const uint16_t a = static_cast<uint16_t>(pairs[i]), b = static_cast<uint16_t>(pairs[i] >> 16u);
                const uint32_t magnitude = ((128u | (a & 127u)) * (128u | (b & 127u))) << (exponents[i] - minimum);
                const int32_t remainder = static_cast<int32_t>(magnitude & mask);
                discarded += ((a ^ b) & 0x8000u) ? -remainder : remainder;
            }
            // The compensated integer is a multiple of 2^shift. Explicit
            // signed magnitude avoids implementation-defined negative shifts.
            const int64_t compensated = mathematical - discarded;
            products = compensated < 0 ? -int64_t(uint64_t(-compensated) >> shift)
                                       : int64_t(uint64_t(compensated) >> shift);
        }
    }
    const unsigned carry_shift = static_cast<unsigned>(maximum - carry.exponent);
    const uint32_t aligned = carry_shift >= 32u ? 0u : (carry.significand << 2u) >> carry_shift;
    const uint32_t modulo = static_cast<uint32_t>(products) + (carry.negative ? 0u - aligned : aligned);
    const bool negative = ((pairs[0] ^ (pairs[0] >> 16u)) & 0x8000u) != 0u;
    *output = {qrt_sm121_group16::decode_modulo_sum(modulo, negative), maximum};
    return true;
}
}  // namespace qrt_sm121_integer_parts
#undef QRT_INTEGER_INLINE
#endif
