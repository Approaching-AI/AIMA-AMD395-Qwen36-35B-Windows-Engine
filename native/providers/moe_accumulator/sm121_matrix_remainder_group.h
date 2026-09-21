#pragma once
#include "sm121_compact_matrix_group.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MATRIX_REMAINDER_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MATRIX_REMAINDER_INLINE inline
#endif

namespace qrt_sm121_matrix_remainder_group {
namespace group = qrt_sm121_compact_matrix_group;

// Keep the established zero-remainder certificate. Otherwise, recover each
// product's signed discarded bits before aligning the complete matrix sum.
// This is the original compact-integer remainder calculation, with the exact
// signed16 mathematical dot supplied by four IU8 matrix instructions.
// The caller supplies complete exponent95..159/signed-zero rows and K<=8192,
// exactly as required by group::accumulate. Rejection leaves output untouched.
QRT_MATRIX_REMAINDER_INLINE bool accumulate(float carry, const group::Row& a,
    const group::Row& b, int64_t mathematical, float* output) {
    if (group::accumulate(carry, a, b, mathematical, output)) return true;
    if (!group::compact::unit(a.encoded) || !group::compact::unit(b.encoded)) return false;
    int maximum = group::product_maximum(a, b);
    const int exponent = int((group::f32::bits(carry) & 0x7fffffffu) >> 23u) - 127;
    maximum = maximum > exponent ? maximum : exponent;
    maximum = maximum > -89 ? maximum : -89;
    const int shift = maximum - (int(group::compact::unit(a.encoded) +
        group::compact::unit(b.encoded)) - 254) - 11;
    if (shift <= 0 || shift > 16) return false;
    const uint32_t mask = (1u << unsigned(shift)) - 1u;
    int32_t discarded = 0;
#pragma unroll
    for (unsigned pair = 0u; pair < 8u; ++pair) {
        const uint32_t x = a.encoded.pairs[pair], y = b.encoded.pairs[pair];
        const uint32_t residue = qrt_sm121_core_remainder::multiply_low16(x, y);
        const uint32_t lo = residue & mask, hi = (residue >> 16u) & mask;
        discarded += int32_t(lo) - (((x ^ y) & 0x8000u) && lo ? int32_t(mask + 1u) : 0);
        discarded += int32_t(hi) - (((x ^ y) & 0x80000000u) && hi ? int32_t(mask + 1u) : 0);
    }
    mathematical -= discarded;
    const int64_t aligned = mathematical < 0
        ? -int64_t(uint64_t(-mathematical) >> unsigned(shift))
        : int64_t(uint64_t(mathematical) >> unsigned(shift));
    const float scale = group::f32::alignment::from_bits(uint32_t(152 - maximum) << 23u);
    const uint32_t modulo = uint32_t(aligned) + uint32_t(int32_t(carry * scale));
    const auto sum = qrt_sm121_group16::decode_modulo_sum(modulo,
        ((a.encoded.pairs[0] ^ b.encoded.pairs[0]) & 0x8000u) != 0u);
    *output = group::normalize(sum.magnitude, sum.negative, maximum);
    return true;
}
} // namespace qrt_sm121_matrix_remainder_group
#undef QRT_MATRIX_REMAINDER_INLINE
