#pragma once
#include "sm121_q1_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_Q2_GATED_INLINE __host__ __device__ __forceinline__
#else
#define QRT_Q2_GATED_INLINE inline
#endif

namespace qrt_sm121_q2 {
// The original short-row gated RMSNorm gives each of 32 lanes four adjacent
// BF16 core values: product 1, then FMA 0, 2, 3. Each head has 128 values.
QRT_Q2_GATED_INLINE float gated_lane_sum(const uint16_t* core, unsigned lane) {
    using namespace qrt_sm121_q1;
    const auto* x = core + size_t(lane) * 4u;
    float sum = multiply(widen(x[1]), widen(x[1]));
    sum = fmaf(widen(x[0]), widen(x[0]), sum);
    sum = fmaf(widen(x[2]), widen(x[2]), sum);
    return fmaf(widen(x[3]), widen(x[3]), sum);
}

QRT_Q2_GATED_INLINE float gated_inverse(float sum, const unsigned char* rsqrt) {
    using namespace qrt_sm121_q1;
    return qrt_sm121_rsqrt::evaluate(rsqrt, add(multiply(sum, 0x1p-7f), 1.0e-6f));
}

QRT_Q2_GATED_INLINE uint16_t gated_value(uint16_t core, uint16_t z,
    uint16_t weight, float inverse, const float* silu) {
    using namespace qrt_sm121_q1;
    return bf16(multiply(multiply(multiply(widen(core), inverse), widen(weight)), silu[z]));
}
} // namespace qrt_sm121_q2

#undef QRT_Q2_GATED_INLINE
