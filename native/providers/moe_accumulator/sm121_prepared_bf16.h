#ifndef QRT_SM121_PREPARED_BF16_H
#define QRT_SM121_PREPARED_BF16_H
#include <cstdint>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PREPARED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PREPARED_INLINE inline
#endif
namespace qrt_sm121_prepared_bf16 {
// A finite normal BF16 with biased exponent 64..191 fits a lossless alternate
// halfword: sign, seven relative exponent bits and eight explicit significand
// bits. Signed zero has a zero significand. Range checks happen once when
// staging operands; an ineligible block reloads its original BF16 values.
QRT_PREPARED_INLINE bool eligible(uint16_t x) {
    const unsigned exponent = (x >> 7u) & 255u;
    return (x & 0x7fffu) == 0u || (exponent >= 64u && exponent <= 191u);
}
QRT_PREPARED_INLINE uint16_t encode(uint16_t x) {
    if ((x & 0x7fffu) == 0u) return x;
    return uint16_t((x & 0x8000u) | ((((x >> 7u) & 255u) - 64u) << 8u) |
                    128u | (x & 127u));
}
QRT_PREPARED_INLINE uint32_t multiply(uint16_t left, uint16_t right) {
    const uint32_t magnitude = uint32_t(left & 255u) * (right & 255u);
    const uint32_t exponent = magnitude ? ((left >> 8u) & 127u) + ((right >> 8u) & 127u) + 128u : 121u;
    return magnitude | (exponent << 16u) | (uint32_t((left ^ right) & 0x8000u) << 16u);
}
}  // namespace qrt_sm121_prepared_bf16
#undef QRT_PREPARED_INLINE
#endif
