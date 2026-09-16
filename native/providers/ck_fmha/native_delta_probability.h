#pragma once
#include "blackwell_attention.h"
#include "../gdn/sm121_exp2_native_delta.h"

// Isolated experiment: preserve the original one-wave recurrence and replace
// only its exp2 decoder. Every non-encodable SFU result uses the source table.
namespace qrt_native_delta_probability {
using namespace qrt_blackwell_attention;
__global__ void probabilities(
    const float* scores, uint16_t* probabilities, float* scales,
    unsigned int query_start, unsigned int score_stride,
    const unsigned char* exp2_table, bool vllm_sum, const unsigned char* packed) {
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned int tokens = query_start + blockIdx.y + 1u;
    const unsigned int tile_stride = (score_stride + kExactTileTokens - 1u) / kExactTileTokens;
    const unsigned int tile_count = (tokens + kExactTileTokens - 1u) / kExactTileTokens;
    float running_max = -INFINITY, running_sum = 1.0f;
    for (unsigned int tile = 0u; tile < tile_count; ++tile) {
        const unsigned int key = tile * kExactTileTokens + lane;
        const float score = key < tokens ? scores[static_cast<size_t>(row) * score_stride + key] : -INFINITY;
        float next_max = fmaxf(running_max, score);
        for (unsigned int mask = 16u; mask; mask >>= 1u)
            next_max = fmaxf(next_max, __shfl_xor(next_max, mask, 32u));
        const float alpha = qrt_sm121_exp2_native_delta::evaluate(exp2_table, packed, (running_max - next_max) * kExactLog2e);
        const float probability = key < tokens ? qrt_sm121_exp2_native_delta::evaluate(exp2_table, packed, (score - next_max) * kExactLog2e) : 0.0f;
        if (key < score_stride)
            probabilities[static_cast<size_t>(row) * score_stride + key] = f32_to_bf16(probability);
        float sum = probability;
        if (vllm_sum) {
            constexpr unsigned int order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
            for (unsigned int step = 0u; step < 5u; ++step)
                sum += __shfl_xor(sum, order[step], 32u);
        } else {
            for (unsigned int mask = 16u; mask; mask >>= 1u)
                sum += __shfl_xor(sum, mask, 32u);
        }
        running_sum = running_sum * alpha + sum;
        running_max = next_max;
        if (lane == 0u) scales[static_cast<size_t>(row) * (tile_stride + 1u) + tile] = alpha;
    }
    if (lane == 0u)
        scales[static_cast<size_t>(row) * (tile_stride + 1u) + tile_stride] = running_sum;
}

} // namespace qrt_native_delta_probability
