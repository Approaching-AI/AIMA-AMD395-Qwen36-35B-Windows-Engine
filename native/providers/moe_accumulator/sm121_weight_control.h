#pragma once
#include <cstdint>
#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_WEIGHT_CONTROL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_WEIGHT_CONTROL_INLINE inline
#endif
namespace qrt_sm121_weight_control_projection {
QRT_WEIGHT_CONTROL_INLINE unsigned control(const uint16_t* input) {
    unsigned maximum = 0u, minimum = 255u, nonzero = 0u;
    bool valid = true;
    for (unsigned i = 0u; i < 16u; ++i) if (input[i] & 0x7fffu) {
        const unsigned e = (input[i] >> 7u) & 255u;
        maximum = e > maximum ? e : maximum;
        minimum = e < minimum ? e : minimum;
        valid = valid && e && e < 255u;
        nonzero |= 1u << i;
    }
    valid = valid && (!nonzero || maximum - minimum <= 29u);
    const int scale = valid ? (nonzero ? int(maximum) - 142 : -15) : -32768;
    return (nonzero << 16u) | uint16_t(scale);
}

QRT_WEIGHT_CONTROL_INLINE unsigned encode_pair(unsigned original, unsigned metadata, unsigned pair) {
    const int scale = int(int16_t(metadata));
    if (scale == -32768) return original;
    const unsigned active = (metadata >> (16u + pair * 2u)) & 3u;
    const unsigned bias = unsigned(112 + scale) << 10u;
    // Independent halfword subtraction avoids cross-element carries. Modulo
    // masking is exact because the cached supported domain maps to FP16 1..30.
    const unsigned a = active & 1u ? (((original & 0x7fffu) << 3u) - bias) & 0x7fffu : 0u;
    const unsigned b = active & 2u ? ((((original >> 16u) & 0x7fffu) << 3u) - bias) & 0x7fffu : 0u;
    return (original & 0x80008000u) | a | (b << 16u);
}

} // namespace qrt_sm121_weight_control_projection
#undef QRT_WEIGHT_CONTROL_INLINE
