#pragma once
#include "sm121_scaled_half_products.h"
#include "sm121_f32_carry.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_HALF_F32_INLINE __host__ __device__ __forceinline__
#else
#define QRT_HALF_F32_INLINE inline
#endif

namespace qrt_sm121_half_f32_carry {
namespace half = qrt_sm121_scaled_half_products;
namespace f32 = qrt_sm121_f32_carry;
using Row = half::Row;

// Lossless normal-half operands, with the same FP32 carry representation as
// prepared decoded QK. A rejected group leaves output untouched; callers must
// restart the original dot so an extended original carry is never discarded.
QRT_HALF_F32_INLINE bool accumulate(float carry, const Row& left,
    const Row& right, float* output) {
    const uint32_t absolute = f32::bits(carry) & 0x7fffffffu;
    const unsigned biased = absolute >> 23u;
    if ((absolute && (!biased || biased == 255u)) ||
        half::unit(left) == -32768 || half::unit(right) == -32768) return false;
    const unsigned active = (left.control & right.control) >> 16u;
    if (!active) { *output = absolute ? carry : 0.0f; return true; }
    float products[16];
    uint32_t paired_maximum = 0u;
#pragma unroll
    for (unsigned i = 0u; i < 8u; ++i) {
        const uint32_t a = left.pairs[i], b = right.pairs[i];
        products[2u * i] = half::product<false>(a, b);
        products[2u * i + 1u] = half::product<true>(a, b);
        uint32_t exponents = ((a >> 10u) & 0x001f001fu) + ((b >> 10u) & 0x001f001fu);
        const unsigned bits = (active >> (2u * i)) & 3u;
        exponents &= ((bits & 1u) | ((bits & 2u) << 15u)) * 65535u;
        paired_maximum = qrt_sm121_prepared_integer_pairs::maximum_pair(paired_maximum, exponents);
    }
    const int combined_unit = half::unit(left) + half::unit(right);
    const int product_maximum = int((paired_maximum & 65535u) > (paired_maximum >> 16u)
        ? paired_maximum & 65535u : paired_maximum >> 16u) - 30 + combined_unit;
    const int carry_exponent = absolute ? int(biased) - 127 : -133;
    const int maximum = product_maximum > carry_exponent ? product_maximum : carry_exponent;
    if (maximum < -101 || maximum > 127) return false;
    const int power = 25 - maximum + combined_unit;
    if (power > 127) return false;
    const float carry_scale = qrt_sm121_float_alignment::from_bits(uint32_t(152 - maximum) << 23u);
    uint32_t modulo = uint32_t(int32_t(carry * carry_scale));
    // Every normal-half product has magnitude below 2^32. A product scale
    // below 2^-126 therefore contributes exactly zero to integer alignment.
    if (power >= -126) {
        const float product_scale = qrt_sm121_float_alignment::from_bits(uint32_t(127 + power) << 23u);
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i)
            modulo += uint32_t(int32_t(products[i] * product_scale));
    }
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo,
        ((left.pairs[0] ^ right.pairs[0]) & 0x8000u) != 0u);
    return f32::normalize<0u>(sum.magnitude, sum.negative, maximum, output);
}
} // namespace qrt_sm121_half_f32_carry
#undef QRT_HALF_F32_INLINE
