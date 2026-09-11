#ifndef QRT_SM121_MANTISSA_PARTS_H
#define QRT_SM121_MANTISSA_PARTS_H

#include "sm121_native_product.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MANTISSA_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MANTISSA_INLINE inline
#endif

namespace qrt_sm121_mantissa_parts {

QRT_MANTISSA_INLINE uint16_t high(uint16_t value) { return value & 0xfff0u; }
QRT_MANTISSA_INLINE uint16_t low(uint16_t value) {
    const float remainder = qrt_sm121_native_product::from_bits(static_cast<uint32_t>(value) << 16u) -
        qrt_sm121_native_product::from_bits(static_cast<uint32_t>(high(value)) << 16u);
    // The difference is itself an exact BF16 value with at most four bits.
    return static_cast<uint16_t>(qrt_sm121_native_product::float_bits(remainder) >> 16u);
}
QRT_MANTISSA_INLINE double power_of_two(int exponent) {
    const uint64_t bits = static_cast<uint64_t>(exponent + 1023) << 52u;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(bits));
#else
    double value; std::memcpy(&value, &bits, sizeof(value)); return value;
#endif
}

// Each partial sums sixteen products of four-bit significands. If the raw
// product exponents span at most twelve, each intermediate sum needs at most
// 8 + 12 + 4 = 24 bits, so all four zero-C FP32 matrix products are exact.
// Inputs with wider/exotic ranges are explicitly left to the original path.
// pairs[i] contains the original left BF16 in its low half and right in high.
QRT_MANTISSA_INLINE bool sum(
    qrt_q1_moe_hawkeye::Value accumulator, const uint32_t (&pairs)[16],
    const float (&partials)[4], qrt_sm121_group16::AlignedSum *output
) {
    // Padded matrix cells can still encounter a nonfinite opposite operand;
    // use the original explicitly padded pair list for those outputs.
    for (unsigned int i = 0u; i < 4u; ++i)
        if ((qrt_sm121_native_product::float_bits(partials[i]) & 0x7f800000u) == 0x7f800000u)
            return false;
    int maximum = -133, minimum = 1000;
    bool valid = true;
    int exponents[16];
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
    for (unsigned int i = 0u; i < 16u; ++i) {
        const uint16_t a = static_cast<uint16_t>(pairs[i]), b = static_cast<uint16_t>(pairs[i] >> 16u);
        const int ae = (a >> 7u) & 0xffu, be = (b >> 7u) & 0xffu;
        valid = valid && ae != 255 && be != 255;
        if ((a & 0x7fffu) == 0u || (b & 0x7fffu) == 0u) {
            exponents[i] = -1000;
        } else {
            valid = valid && ae > 0 && be > 0;
            // A nonzero low part must stay normal as a BF16 matrix operand.
            // Native BF16 instructions may flush a subnormal split operand.
            valid = valid && (ae > 7 || (a & 0xfu) == 0u) &&
                (be > 7 || (b & 0xfu) == 0u);
            const int exponent = ae + be - 254;
            exponents[i] = exponent;
            maximum = exponent > maximum ? exponent : maximum;
            minimum = exponent < minimum ? exponent : minimum;
        }
    }
    if (!valid) return false;
    const int max_exponent = accumulator.exponent > maximum ? accumulator.exponent : maximum;
    const bool empty = minimum == 1000;
    const int fraction_bits = !empty && max_exponent - minimum > 11
        ? max_exponent - minimum - 11 : 0;
    if (!empty && (maximum - minimum > 12 || minimum < -112 || maximum > 122 || fraction_bits > 16))
        return false;

    int32_t discarded = 0;
    if (fraction_bits != 0) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
#pragma unroll
#endif
        for (unsigned int i = 0u; i < 16u; ++i) {
            if (exponents[i] == -1000) continue;
            const int bits = max_exponent - exponents[i] - 11;
            if (bits <= 0) continue;
            const uint16_t a = static_cast<uint16_t>(pairs[i]), b = static_cast<uint16_t>(pairs[i] >> 16u);
            const uint32_t product = (0x80u | (a & 0x7fu)) * (0x80u | (b & 0x7fu));
            const uint32_t remainder = (product & ((1u << bits) - 1u)) << (fraction_bits - bits);
            discarded += ((a ^ b) & 0x8000u) ? -static_cast<int32_t>(remainder) : static_cast<int32_t>(remainder);
        }
    }
    // Four exact partials need at most 32 significant bits in combination.
    // The signed discarded fraction has at most twenty bits. Both fit exactly
    // in FP64; scaling by a power of two adds no rounding error. Subtracting
    // the fraction recovers the sum of individually truncated K16 products.
    const double mathematical = (static_cast<double>(partials[0]) + partials[1]) +
        (static_cast<double>(partials[2]) + partials[3]);
    const double aligned_products = empty ? 0.0 : mathematical * power_of_two(25 - max_exponent) -
        static_cast<double>(discarded) * power_of_two(-fraction_bits);
    const int32_t products = static_cast<int32_t>(aligned_products);
    const unsigned int shift = static_cast<unsigned int>(max_exponent - accumulator.exponent);
    const uint32_t aligned_accumulator = shift >= 32u ? 0u : (accumulator.significand << 2u) >> shift;
    const uint32_t modulo = static_cast<uint32_t>(products) +
        (accumulator.negative ? 0u - aligned_accumulator : aligned_accumulator);
    const bool negative = ((pairs[0] ^ (pairs[0] >> 16u)) & 0x8000u) != 0u;
    *output = {qrt_sm121_group16::decode_modulo_sum(modulo, negative), max_exponent};
    return true;
}

}  // namespace qrt_sm121_mantissa_parts
#undef QRT_MANTISSA_INLINE
#endif
