#ifndef QRT_BF16_POSITIVE_SUM_BOUND_H
#define QRT_BF16_POSITIVE_SUM_BOUND_H
#include <cstdint>
#include <cstring>
#if defined(__HIPCC__)
#define QRT_SUM_BOUND_HD __host__ __device__
#else
#define QRT_SUM_BOUND_HD
#endif

namespace qrt_bf16_positive_sum_bound {
QRT_SUM_BOUND_HD inline float value(uint32_t bits) {
    float result; __builtin_memcpy(&result, &bits, sizeof(result)); return result;
}
QRT_SUM_BOUND_HD inline uint32_t bits(float value) {
    uint32_t result; __builtin_memcpy(&result, &value, sizeof(result)); return result;
}
QRT_SUM_BOUND_HD inline float next_up(float positive) {
    const uint32_t word = bits(positive);
    return value(word < 0x7f800000u ? word + 1u : 0x7f800000u);
}
// Selector metadata only. Inflate the nonnegative FP32 matrix sum by8K ulps
// at unit scale, then take outward steps on multiplication and addition.
// A K*min-normal floor covers a lost subnormal contribution per product.
// The matrix caller admits only signed zero or BF16 exponent64..191 rows;
// any excluded row receives infinity and remains eligible for exact replay.
QRT_SUM_BOUND_HD inline float finish(float approximate, unsigned count) {
    if (bits(approximate) >= 0x7f800000u) return value(0x7f800000u);
    const float inflation = 1.0f + float(count) * 0x1p-20f;
    const float floor = float(count) * 0x1p-126f;
    return next_up(next_up(approximate * inflation) + floor);
}
}
#undef QRT_SUM_BOUND_HD
#endif
