#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_math.h"

namespace qrt_sm121_mtp {
// Launch with 512 threads. One CTA owns a row; separating the original
// two-row CTA preserves each row's lane and warp arithmetic.
__device__ __forceinline__ float row_inverse(const uint16_t* row,
                                            const unsigned char* table,
                                            float* warp_sums, float* inverse_shared) {
    const unsigned int lane = threadIdx.x;
    float sum = lane_sumsq(row, lane);
    for (unsigned int offset = 16; offset; offset >>= 1)
        sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, offset, 32));
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = sum;
    __syncthreads();
    if (lane == 0u) *inverse_shared = inverse(sum_warps(warp_sums), table);
    __syncthreads();
    return *inverse_shared;
}

__global__ void normalize_rows(const uint16_t* inputs, const uint16_t* weights,
                              const unsigned char* table, uint16_t* outputs) {
    __shared__ float warp_sums[16], inverse_shared;
    const uint16_t* row = inputs + static_cast<size_t>(blockIdx.x) * 2048u;
    uint16_t* output = outputs + static_cast<size_t>(blockIdx.x) * 2048u;
    const float rstd = row_inverse(row, table, warp_sums, &inverse_shared);
    for (unsigned int i = threadIdx.x; i < 2048u; i += 512u)
        output[i] = normalized(qrt_sm121_q1::widen(row[i]), rstd, weights[i]);
}

// shifted_ids are validated request IDs. They include the caller-selected
// original partial-prefill backup at discarded chunk boundaries. The second
// grid dimension selects [embedding, target hidden] in the original FC order.
// invalid_input must start at zero and be checked before projecting the output.
__global__ void gather_fusion_inputs(
    const uint16_t* embeddings, const uint16_t* target_hidden,
    const uint32_t* shifted_ids, const uint16_t* embedding_weights,
    const uint16_t* hidden_weights, const unsigned char* table,
    uint16_t* fusion_inputs, uint32_t* invalid_input) {
    __shared__ float warp_sums[16], inverse_shared;
    const unsigned int row_index = blockIdx.x, part = blockIdx.y;
    const uint32_t token = shifted_ids[row_index];
    if (token >= 248320u) {
        if (threadIdx.x == 0u) atomicExch(invalid_input, 1u);
        return;
    }
    const uint16_t* row = part == 0u
        ? embeddings + static_cast<size_t>(token) * 2048u
        : target_hidden + static_cast<size_t>(row_index) * 2048u;
    const uint16_t* weights = part == 0u ? embedding_weights : hidden_weights;
    uint16_t* output = fusion_inputs + static_cast<size_t>(row_index) * 4096u + part * 2048u;
    const float rstd = row_inverse(row, table, warp_sums, &inverse_shared);
    for (unsigned int i = threadIdx.x; i < 2048u; i += 512u)
        output[i] = normalized(qrt_sm121_q1::widen(row[i]), rstd, weights[i]);
}

inline hipError_t launch_normalize(const uint16_t* input, const uint16_t* weights,
    const unsigned char* table, unsigned int rows, uint16_t* output,
    hipStream_t stream = nullptr) {
    if (!input || !weights || !table || !output || !rows || rows > 8192u)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(normalize_rows, dim3(rows), dim3(512), 0, stream,
                      input, weights, table, output);
    return hipGetLastError();
}

inline hipError_t launch_fusion_inputs(const uint16_t* embeddings,
    const uint16_t* target_hidden, const uint32_t* shifted_ids,
    const uint16_t* embedding_weights, const uint16_t* hidden_weights,
    const unsigned char* table, unsigned int rows, uint16_t* output,
    uint32_t* invalid_input, hipStream_t stream = nullptr) {
    if (!embeddings || !target_hidden || !shifted_ids || !embedding_weights ||
        !hidden_weights || !table || !output || !invalid_input || !rows || rows > 8192u)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(gather_fusion_inputs, dim3(rows, 2), dim3(512), 0, stream,
                      embeddings, target_hidden, shifted_ids, embedding_weights,
                      hidden_weights, table, output, invalid_input);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
