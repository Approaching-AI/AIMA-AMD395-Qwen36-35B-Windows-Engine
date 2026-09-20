#pragma once
#include "sm121_mtp_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_RESIDUAL_INLINE __host__ __device__ __forceinline__
#else
#define QRT_MTP_RESIDUAL_INLINE inline
#endif

namespace qrt_sm121_mtp {
// Original FLLJ... PTX 7a0d727b... uses the unrounded FP32 residual sum
// for variance, then its BF16 endpoint for the normalized numerator.
// One row owns eight warps, with eight adjacent values in each lane.
QRT_MTP_RESIDUAL_INLINE float residual_lane_sumsq(
    const uint16_t* input, const uint16_t* residual, unsigned int lane) {
    using namespace qrt_sm121_q1;
    float values[8];
    for (unsigned int i = 0; i < 8u; ++i) {
        const unsigned int column = lane * 8u + i;
        values[i] = add(widen(input[column]), widen(residual[column]));
    }
    float sum = multiply(values[1], values[1]);
    sum = fmaf(values[0], values[0], sum);
    sum = fmaf(values[2], values[2], sum);
    sum = add(sum, multiply(values[3], values[3]));
    sum = fmaf(values[4], values[4], sum);
    sum = add(sum, multiply(values[5], values[5]));
    sum = fmaf(values[6], values[6], sum);
    return add(sum, multiply(values[7], values[7]));
}

QRT_MTP_RESIDUAL_INLINE float residual_sum_warps(const float* warps) {
    float partial[8];
    for (unsigned int i = 0; i < 8u; ++i) partial[i] = warps[i];
    for (unsigned int offset = 4u; offset; offset >>= 1)
        for (unsigned int i = 0; i < offset; ++i)
            partial[i] = qrt_sm121_q1::add(partial[i], partial[i + offset]);
    return partial[0];
}

QRT_MTP_RESIDUAL_INLINE uint16_t residual_endpoint(uint16_t input, uint16_t residual) {
    using namespace qrt_sm121_q1;
    return bf16(add(widen(input), widen(residual)));
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_RESIDUAL_INLINE
