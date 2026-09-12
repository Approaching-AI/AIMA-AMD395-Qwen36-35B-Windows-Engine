#ifndef QRT_SM121_PV_ERROR_BOUND_H
#define QRT_SM121_PV_ERROR_BOUND_H

#include <cstdint>
#include <cstring>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_PV_BOUND_INLINE __host__ __device__ __forceinline__
#else
#define QRT_PV_BOUND_INLINE inline
#endif

namespace qrt_sm121_pv_bound {
QRT_PV_BOUND_INLINE uint32_t bits(float value) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __float_as_uint(value);
#else
    uint32_t out; std::memcpy(&out, &value, sizeof(out)); return out;
#endif
}
QRT_PV_BOUND_INLINE float value(uint32_t encoded) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __uint_as_float(encoded);
#else
    float out; std::memcpy(&out, &encoded, sizeof(out)); return out;
#endif
}
QRT_PV_BOUND_INLINE float absolute(float x) { return value(bits(x) & 0x7fffffffu); }
QRT_PV_BOUND_INLINE bool finite(float x) { return (bits(x) & 0x7fffffffu) < 0x7f800000u; }
QRT_PV_BOUND_INLINE float infinity() { return value(0x7f800000u); }

// Inflate each nonnegative computed bound by two FP32 representable steps.
// Exceptional arithmetic selects exact replay instead of admitting an output.
QRT_PV_BOUND_INLINE float upper(float x) {
    const uint32_t b = bits(x);
    if ((b & 0x80000000u) || b >= 0x7f7ffffeu) return infinity();
    return value(b + 2u);
}

// A K16 reference group aligns 16 exact BF16 products and one FP32 carry
// at 26 bits, then truncates to FP32. Its alignment loss is <17*2^-25 times
// the largest operand magnitude, followed by <2^-23 of the sum magnitude.
// Allow another gamma(16) for native FP32 additions. 32*2^-24 covers their
// combined coefficients. The positive matrix dot bounds sum(abs(P*V));
// inflate it before use and include a floor for possible normal/subnormal
// flushing. This is a candidate envelope, independently checked against
// canonical groups and captured GPU inputs; model acceptance stays GB10-based.
QRT_PV_BOUND_INLINE float group(float error, float carry, float absolute_dot) {
    if (!finite(error) || !finite(carry) || !finite(absolute_dot) ||
        error < 0.0f || absolute_dot < 0.0f) return infinity();
    const float products = upper(absolute_dot * (1.0f + 0x1p-19f) + 0x1p-118f);
    const float magnitude = upper(absolute(carry) + error + products);
    return upper(error + upper(magnitude * 0x1p-19f) + 0x1p-118f);
}

QRT_PV_BOUND_INLINE float rescale(float error, float carry, float alpha) {
    if (!finite(error) || !finite(carry) || !finite(alpha) ||
        error < 0.0f || alpha < 0.0f) return infinity();
    if (alpha == 0.0f) return 0.0f;
    if (alpha == 1.0f) return error;
    const float magnitude = upper((absolute(carry) + error) * alpha);
    return upper(upper(error * alpha) + upper(magnitude * 0x1p-22f) + 0x1p-118f);
}

QRT_PV_BOUND_INLINE float finish(float error, float carry, float reciprocal) {
    if (!finite(error) || !finite(carry) || !finite(reciprocal) ||
        error < 0.0f || reciprocal < 0.0f) return infinity();
    const float magnitude = upper((absolute(carry) + error) * reciprocal);
    return upper(upper(error * reciprocal) + upper(magnitude * 0x1p-22f) + 0x1p-118f);
}

QRT_PV_BOUND_INLINE uint16_t bf16(float x) {
    const uint32_t b = bits(x);
    return static_cast<uint16_t>((b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u);
}
QRT_PV_BOUND_INLINE float next(float x, bool positive) {
    uint32_t b = bits(x);
    if ((b & 0x7fffffffu) == 0u) return value(positive ? 1u : 0x80000001u);
    b += ((b & 0x80000000u) == 0u) == positive ? 1u : uint32_t(-1);
    return value(b);
}
QRT_PV_BOUND_INLINE bool same_bf16(float center, float error) {
    if (!finite(center) || !finite(error) || error < 0.0f) return false;
    const float lower = next(center - error, false);
    const float upper_value = next(center + error, true);
    return finite(lower) && finite(upper_value) && bf16(lower) == bf16(upper_value);
}
}  // namespace qrt_sm121_pv_bound

#undef QRT_PV_BOUND_INLINE
#endif
