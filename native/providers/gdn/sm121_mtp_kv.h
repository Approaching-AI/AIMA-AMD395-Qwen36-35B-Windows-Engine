#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_kv_math.h"

namespace qrt_sm121_mtp {
// Input is [row, K512 + V512]. Cache is token-major [K512 + V512], with
// absolute original target positions. Each block owns one row and K head.
__global__ void key_value_rows(const uint16_t* projected, const uint16_t* key_weights,
    const unsigned char* rsqrt_table, const uint16_t* rope_table,
    unsigned int first_position, uint16_t* cache, uint16_t* norm_observation) {
    __shared__ float values[256], warp_sums[4], inverse_shared;
    __shared__ uint16_t normalized_head[256];
    const unsigned int row = blockIdx.x, head = blockIdx.y, channel = threadIdx.x;
    const size_t source = size_t(row) * 1024u + head * 256u;
    values[channel] = qrt_sm121_q1::widen(projected[source + channel]);
    __syncthreads();
    if (channel < 128u) {
        float sum = qrt_sm121_q1::head_norm_lane_sumsq(values, channel, true);
        for (unsigned int offset = 16u; offset; offset >>= 1u)
            sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, offset, 32));
        if ((channel & 31u) == 0u) warp_sums[channel / 32u] = sum;
    }
    __syncthreads();
    if (channel == 0u) inverse_shared = key_inverse(warp_sums, rsqrt_table);
    __syncthreads();
    normalized_head[channel] = normalized(values[channel], inverse_shared, key_weights[channel]);
    __syncthreads();
    if (norm_observation)
        norm_observation[size_t(row) * 512u + head * 256u + channel] = normalized_head[channel];
    const unsigned int position = first_position + row;
    const size_t destination = size_t(position) * 1024u + head * 256u + channel;
    cache[destination] = key_rotated(normalized_head, channel, rope_table + size_t(position) * 64u);
    cache[destination + 512u] = projected[source + 512u + channel];
}

inline hipError_t launch_key_values(const uint16_t* projected, const uint16_t* key_weights,
    const unsigned char* rsqrt_table, const uint16_t* rope_table, unsigned int rope_rows,
    unsigned int first_position, unsigned int rows, unsigned int capacity,
    uint16_t* cache, uint16_t* norm_observation = nullptr, hipStream_t stream = nullptr) {
    if (!projected || !key_weights || !rsqrt_table || !rope_table || !cache ||
        !rows || rows > 8192u || !capacity || capacity > 262144u ||
        first_position >= capacity || rows > capacity - first_position ||
        first_position >= rope_rows || rows > rope_rows - first_position)
        return hipErrorInvalidValue;
    hipLaunchKernelGGL(key_value_rows, dim3(rows, 2u), dim3(256u), 0, stream,
                      projected, key_weights, rsqrt_table, rope_table, first_position,
                      cache, norm_observation);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
