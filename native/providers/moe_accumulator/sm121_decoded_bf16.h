#ifndef QRT_SM121_DECODED_BF16_H
#define QRT_SM121_DECODED_BF16_H
#include "sm121_float_alignment.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_DECODED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_DECODED_INLINE inline
#endif
namespace qrt_sm121_decoded_bf16 {
// Original eligible values have unbiased exponent -63..63. A zero sentinel
// below -196 ensures that even zero times the largest eligible value stays
// below Group's original -133 initial maximum. Product zero checks can then
// move out of every score dot without changing its maximum or signed zero.
constexpr int16_t zero_exponent = -512;
QRT_DECODED_INLINE int16_t exponent(uint16_t value) {
    return (value & 0x7fffu) ? int16_t(int((value >> 7u) & 255u) - 127) : zero_exponent;
}
QRT_DECODED_INLINE float value(uint16_t input) {
    return qrt_sm121_float_alignment::from_bits(uint32_t(input) << 16u);
}
QRT_DECODED_INLINE void set(qrt_sm121_float_alignment::Group& group, unsigned i,
    float left, float right, int16_t left_exponent, int16_t right_exponent) {
    const float product = left * right;
    group.products[i] = product;
    const int e = int(left_exponent) + int(right_exponent);
    group.maximum = e > group.maximum ? e : group.maximum;
    if (!i) {
        uint32_t bits; __builtin_memcpy(&bits, &product, sizeof(bits));
        group.first_negative = (bits & 0x80000000u) != 0u;
    }
}
} // namespace qrt_sm121_decoded_bf16
#undef QRT_DECODED_INLINE
#endif
