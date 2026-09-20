#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_query_math.h"

namespace qrt_sm121_mtp {
// Q projection is [row,16 heads, Q256 + gate256]. Q normalization uses
// two stride-64 warps; K uses a different four-warp layout. Positions refer
// to the actual target hidden rows, before the proposer shifts token IDs.
__global__ void query_rows(const uint16_t* projected, const uint16_t* weights,
    const unsigned char* rsqrt_table, const uint16_t* rope_table,
    unsigned int first_position, uint16_t* query, uint16_t* gates, uint16_t* norm_observation) {
    __shared__ float values[256], warps[2], rstd;
    __shared__ uint16_t head_norm[256];
    const unsigned int row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const size_t source = size_t(row) * 8192u + head * 512u;
    const size_t target = size_t(row) * 4096u + head * 256u;
    for (unsigned int channel = lane; channel < 256u; channel += 64u)
        values[channel] = qrt_sm121_q1::widen(projected[source + channel]);
    __syncthreads();
    float sum = qrt_sm121_q1::head_norm_lane_sumsq(values, lane, false);
    for (unsigned int mask = 16u; mask; mask >>= 1)
        sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, mask, 32));
    if ((lane & 31u) == 0u) warps[lane / 32u] = sum;
    __syncthreads();
    if (lane == 0u) rstd = query_inverse(warps, rsqrt_table);
    __syncthreads();
    for (unsigned int channel = lane; channel < 256u; channel += 64u)
        head_norm[channel] = normalized(values[channel], rstd, weights[channel]);
    __syncthreads();
    const uint16_t* coefficients = rope_table + size_t(first_position + row) * 64u;
    for (unsigned int channel = lane; channel < 256u; channel += 64u) {
        query[target + channel] = key_rotated(head_norm, channel, coefficients);
        gates[target + channel] = projected[source + 256u + channel];
        if (norm_observation) norm_observation[target + channel] = head_norm[channel];
    }
}

inline hipError_t launch_queries(const uint16_t* projected, const uint16_t* weights,
    const unsigned char* rsqrt_table, const uint16_t* rope_table, unsigned int rope_rows,
    unsigned int first_position, unsigned int rows, uint16_t* queries, uint16_t* gates,
    uint16_t* norm_observation = nullptr, hipStream_t stream = nullptr) {
    if (!projected || !weights || !rsqrt_table || !rope_table || !queries || !gates ||
        queries == gates || !rows || rows > 8192u || first_position >= 262144u ||
        rows > 262144u - first_position || first_position >= rope_rows ||
        rows > rope_rows - first_position) return hipErrorInvalidValue;
    hipLaunchKernelGGL(query_rows, dim3(rows,16u), dim3(64u), 0, stream,
        projected, weights, rsqrt_table, rope_table, first_position, queries, gates, norm_observation);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
