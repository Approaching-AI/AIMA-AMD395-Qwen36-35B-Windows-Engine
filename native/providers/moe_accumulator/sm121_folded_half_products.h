#pragma once
#include "sm121_scaled_half_products.h"

#if defined(__HIPCC__)
#define QRT_FOLDED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_FOLDED_INLINE inline
#endif
namespace qrt_sm121_folded_half_products {
namespace half = qrt_sm121_scaled_half_products;
using Value = qrt_q1_moe_hawkeye::Value;
// Keep the original 36-byte extent. Valid row units are -141..112, encoded
// in signed nine bits. Bits9..13 store the minimum normal FP16 exponent;
// bit15 marks original BF16 storage. The upper16 nonzero mask is unchanged.
struct Row { uint32_t pairs[8], control; };
static_assert(sizeof(Row) == sizeof(half::Row));
QRT_FOLDED_INLINE int unit(uint32_t control) {
    if (control & 0x8000u) return -32768;
    const int value = int(control & 511u);
    return value >= 256 ? value - 512 : value;
}
QRT_FOLDED_INLINE unsigned minimum(uint32_t control) { return (control >> 9u) & 31u; }
QRT_FOLDED_INLINE uint32_t legacy_control(uint32_t control) {
    return (control & 0xffff0000u) | uint16_t(unit(control));
}
QRT_FOLDED_INLINE Row prepare(const uint16_t* input) {
    const auto legacy = half::prepare(input);
    Row result{};
    unsigned low = 31u;
    for (unsigned pair = 0u; pair < 8u; ++pair) {
        const uint32_t word = legacy.pairs[pair];
        result.pairs[pair] = word;
        for (unsigned i = 0u; i < 2u; ++i) {
            const unsigned x = (word >> (i * 16u)) & 65535u;
            if (x & 0x7fffu) {
                const unsigned exponent = (x >> 10u) & 31u;
                low = exponent < low ? exponent : low;
            }
        }
    }
    const int scale = half::unit(legacy);
    result.control = (legacy.control & 0xffff0000u) | (scale == -32768
        ? 0x8000u : ((uint32_t(scale) & 511u) | ((low == 31u ? 0u : low) << 9u)));
    return result;
}
QRT_FOLDED_INLINE uint16_t original(const Row& row, unsigned i) {
    const uint16_t x = uint16_t(row.pairs[i / 2u] >> (i % 2u * 16u));
    const int scale = unit(row.control);
    if (scale == -32768) return x;
    if (!(x & 0x7fffu)) return uint16_t(x & 0x8000u);
    return uint16_t((x & 0x8000u) |
        (unsigned(int((x >> 10u) & 31u) + 112 + scale) << 7u) | ((x & 1023u) >> 3u));
}
struct Plan { int maximum, power; uint32_t delta; bool left, allowed; };
QRT_FOLDED_INLINE Plan plan(Value carry, uint32_t left, uint32_t right, int half_maximum) {
    const int combined = unit(left) + unit(right);
    int maximum = half_maximum - 30 + combined;
    maximum = maximum > carry.exponent ? maximum : carry.exponent;
    maximum = maximum > -133 ? maximum : -133;
    const int power = 25 - maximum + combined;
    const unsigned a = minimum(left), b = minimum(right), low = a >= b ? a : b;
    const bool allowed = !(left & 0x8000u) && !(right & 0x8000u) &&
        ((left & right) >> 16u) == 65535u && power <= 0 && int(low) + power >= 1;
    // Both exponents remain normal. Multiplication by65537 accounts for the
    // low-half carry, applying the same exact shift to both packed halves.
    const uint32_t delta = allowed ? uint32_t(power * 1024) * 65537u : 0u;
    return {maximum, power, delta, a >= b, allowed};
}
}
#undef QRT_FOLDED_INLINE
