#pragma once
#include "sm121_q1_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_Q1_RESIDUAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_Q1_RESIDUAL_INLINE inline
#endif

namespace qrt_sm121_q1_residual {
// Original single-row GemmaRMSNorm PTX 5a5d4df7... assigns four adjacent
// values to each of 512 logical lanes, then reduces 16 warp totals. Its
// residual sum is unrounded FP32; only the normalized numerator is BF16.
// Multi-row prefill and Q2 use a different, eight-values-per-lane reduction.
// The actual SM121 cubin 440514e7... contracts the final square/add as well:
// FMUL square1, then FFMA square0, square2, square3. Preserve the SASS order.
QRT_Q1_RESIDUAL_INLINE float lane_sumsq(const float* values) {
    using namespace qrt_sm121_q1;
    float sum = multiply(values[1], values[1]);
    sum = fmaf(values[0], values[0], sum);
    sum = fmaf(values[2], values[2], sum);
    return fmaf(values[3], values[3], sum);
}

QRT_Q1_RESIDUAL_INLINE float sum_warps(const float* warps) {
    float partial[16];
    for (unsigned i = 0; i < 16u; ++i) partial[i] = warps[i];
    for (unsigned offset = 8u; offset; offset >>= 1u)
        for (unsigned i = 0; i < offset; ++i)
            partial[i] = qrt_sm121_q1::add(partial[i], partial[i + offset]);
    return partial[0];
}
} // namespace qrt_sm121_q1_residual
#undef QRT_Q1_RESIDUAL_INLINE
