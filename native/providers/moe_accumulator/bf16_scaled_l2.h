#ifndef QRT_BF16_SCALED_L2_H
#define QRT_BF16_SCALED_L2_H
#include <cmath>
#include <cstdint>
#include <cstring>

#if defined(__HIPCC__)
#define QRT_L2_HD __host__ __device__
#else
#define QRT_L2_HD
#endif

// Conservative selector metadata, never replacement model activations.
// Scaling by the largest BF16 exponent keeps every retained square normal in
// FP32. Smaller terms are raised to exponent -56 relative to that scale; even
// a subnormal BF16 significand then has a square >= 2^-126. These squares have
// at most sixteen significant bits and are exactly representable in FP32.
namespace qrt_bf16_scaled_l2 {
QRT_L2_HD inline uint32_t bits(float value) {
    uint32_t result;
    __builtin_memcpy(&result, &value, sizeof(result));
    return result;
}
QRT_L2_HD inline float value(uint32_t raw) {
    float result;
    __builtin_memcpy(&result, &raw, sizeof(result));
    return result;
}
QRT_L2_HD inline unsigned exponent(uint16_t raw) {
    if ((raw & 0x7fffu) == 0u) return 0u;
    const unsigned e = (raw >> 7u) & 255u;
    return e == 0u ? 1u : e;
}
QRT_L2_HD inline float next_up(float positive) {
    const uint32_t raw = bits(positive);
    return value(raw < 0x7f800000u ? raw + 1u : 0x7f800000u);
}
QRT_L2_HD inline float add_up(float left, float right) {
    if (bits(left) == 0u) return right;
    if (bits(right) == 0u) return left;
    return next_up(left + right);
}
QRT_L2_HD inline float square(uint16_t raw, unsigned maximum_exponent) {
    if ((raw & 0x7fffu) == 0u) return 0.0f;
    const unsigned e = (raw >> 7u) & 255u;
    const unsigned significand = (raw & 127u) | (e == 0u ? 0u : 128u);
    int relative = int(e == 0u ? 1u : e) - int(maximum_exponent);
    if (relative < -56) relative = -56;
    const float scaled = float(significand) * value(unsigned(127 + relative - 7) << 23u);
    return scaled * scaled;
}

// Integer exponent adjustment also rounds subnormal results upward. No
// floating subnormal arithmetic is needed, including on devices using FTZ.
QRT_L2_HD inline float scale_up(float normal, int shift) {
    const uint32_t raw = bits(normal);
    if (raw >= 0x7f800000u) return value(0x7f800000u);
    const int e = int((raw >> 23u) & 255u) + shift;
    if (e >= 255) return value(0x7f800000u);
    if (e > 0) return value((raw & 0x7fffffu) | (unsigned(e) << 23u));
    const unsigned down = unsigned(1 - e);
    if (down >= 32u) return value(1u);
    const uint32_t significand = (raw & 0x7fffffu) | 0x800000u;
    return value((significand + ((uint32_t(1) << down) - 1u)) >> down);
}
QRT_L2_HD inline float finish(float sum_upper, unsigned maximum_exponent) {
    if (maximum_exponent == 0u) return 0.0f;
    if (maximum_exponent == 255u) return value(0x7f800000u);
    // Retain the existing 1.00002 inflation. Two outward steps cover the
    // sqrt endpoint, then one covers its multiplication by the inflation.
    const float root = next_up(next_up(sqrtf(sum_upper)));
    const float inflated = next_up(root * 1.00002f);
    return scale_up(inflated, int(maximum_exponent) - 127);
}
}  // namespace qrt_bf16_scaled_l2
#undef QRT_L2_HD
#endif
