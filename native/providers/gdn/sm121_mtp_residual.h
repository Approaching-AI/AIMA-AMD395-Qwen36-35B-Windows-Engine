#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_residual_math.h"

namespace qrt_sm121_mtp {
__global__ void residual_normalize_rows(
    const uint16_t* inputs, const uint16_t* residuals, const uint16_t* weights,
    const unsigned char* table, uint16_t* outputs, uint16_t* residual_outputs) {
    __shared__ float warp_sums[8], rstd;
    const size_t offset = static_cast<size_t>(blockIdx.x) * 2048u;
    const uint16_t* input = inputs + offset;
    const uint16_t* residual = residuals + offset;
    const unsigned int lane = threadIdx.x;
    float sum = residual_lane_sumsq(input, residual, lane);
    for (unsigned int mask = 16u; mask; mask >>= 1)
        sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, mask, 32));
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = sum;
    __syncthreads();
    if (lane == 0u) rstd = inverse(residual_sum_warps(warp_sums), table);
    __syncthreads();
    for (unsigned int i = 0; i < 8u; ++i) {
        const unsigned int column = lane * 8u + i;
        const uint16_t combined = residual_endpoint(input[column], residual[column]);
        outputs[offset + column] = normalized(qrt_sm121_q1::widen(combined), rstd, weights[column]);
        residual_outputs[offset + column] = combined;
    }
}

// Outputs may replace their corresponding input rows after the row fence;
// the normalized and residual output buffers must be distinct.
inline hipError_t launch_residual_normalize(
    const uint16_t* inputs, const uint16_t* residuals, const uint16_t* weights,
    const unsigned char* table, unsigned int rows, uint16_t* outputs,
    uint16_t* residual_outputs, hipStream_t stream = nullptr) {
    if (!inputs || !residuals || !weights || !table || !outputs || !residual_outputs ||
        outputs == residual_outputs || !rows || rows > 8192u) return hipErrorInvalidValue;
    hipLaunchKernelGGL(residual_normalize_rows, dim3(rows), dim3(256), 0, stream,
                      inputs, residuals, weights, table, outputs, residual_outputs);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
