#pragma once
#include "prepared_decoded_qk.h"

// Component candidate: keep a query tile resident while each K32 score tile
// feeds the original ordered online softmax. No score matrix is materialized
// unless the diagnostic output is requested. The existing exact fallback,
// K16 carries, exponential lookup and K32 denominator order remain unchanged.
namespace qrt_fused_qk_probability {
namespace attention = qrt_blackwell_attention;
namespace decoded = qrt_sm121_decoded_bf16;

template<unsigned Rows = 8u>
__global__ void run(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags,
    uint16_t* probabilities, float* scales, float* diagnostic_scores,
    unsigned query_start, unsigned query_count, unsigned stride,
    unsigned key_stride, const unsigned char* exp2_table, bool vllm_sum,
    unsigned prepared_query_start = 0u) {
    static_assert(Rows * 32u == attention::kThreads);
    static_assert(attention::kHeadDim == 128u && attention::kExactTileTokens == 32u);
    __shared__ uint32_t qvalues[Rows][attention::kHeadDim];
    __shared__ uint32_t kvalues[attention::kHeadDim][32u];
    const unsigned local_row = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const unsigned row_tile = blockIdx.y * Rows, row = row_tile + local_row;
    const unsigned head = blockIdx.x;
    const unsigned kv_head = head / (attention::kQueryHeads / attention::kKvHeads);
    const unsigned prepared_start = query_start - prepared_query_start;
    const bool live_row = row < query_count;
    const unsigned tokens = query_start + row + 1u;
    const unsigned tile_stride = (stride + 31u) / 32u;
    const unsigned row_tiles = (tokens + 31u) / 32u;
    const unsigned last_query = query_start + min(row_tile + Rows, query_count) - 1u;
    const unsigned tiles = (last_query + 32u) / 32u;
    const size_t output_row = size_t(row) * attention::kQueryHeads + head;
    const bool query_eligible = live_row &&
        query_flags[(prepared_start + row) * attention::kQueryHeads + head];
    for (unsigned cell = threadIdx.x; cell < Rows * attention::kHeadDim; cell += attention::kThreads) {
        const unsigned r = cell / attention::kHeadDim, feature = cell % attention::kHeadDim;
        qvalues[r][feature] = row_tile + r < query_count
            ? packed_query[(size_t(prepared_start + row_tile + r) * attention::kQueryHeads + head) *
                           attention::kHeadDim + feature]
            : decoded::pack(0u);
    }
    __syncthreads();
    float running_max = -INFINITY, running_sum = 1.0f;
    for (unsigned tile = 0u; tile < tiles; ++tile) {
        const unsigned key_base = tile * 32u, key = key_base + lane;
        for (unsigned cell = threadIdx.x; cell < attention::kHeadDim * 32u; cell += attention::kThreads) {
            const unsigned feature = cell / 32u, column = cell % 32u;
            kvalues[feature][column] = key_base + column < stride
                ? packed_key[(size_t(kv_head) * attention::kHeadDim + feature) * key_stride + key_base + column]
                : decoded::pack(0u);
        }
        __syncthreads();
        const bool active = live_row && key < tokens;
        float score = -INFINITY;
        if (active) {
            bool fallback = !query_eligible || !key_flags[key * attention::kKvHeads + kv_head];
            float carry = 0.0f;
            if (!fallback) {
                for (unsigned base = 0u; base < attention::kHeadDim; base += 16u) {
                    qrt_sm121_float_alignment::Group group;
#pragma unroll
                    for (unsigned item = 0u; item < 16u; ++item)
                        decoded::set_packed(group, item, qvalues[local_row][base + item], kvalues[base + item][lane]);
                    float next;
                    if (!qrt_sm121_f32_carry::accumulate<0u>(carry, group, &next)) {
                        fallback = true;
                        break;
                    }
                    carry = next;
                }
            }
            score = fallback ? qrt_decoded_window_qk::raw_dot(
                query + (size_t(query_start + row) * attention::kQueryHeads + head) * attention::kHeadDim,
                transposed_key + size_t(kv_head) * attention::kHeadDim * key_stride + key, key_stride)
                : carry * attention::kExactScale;
        }
        if (diagnostic_scores && live_row && key < stride)
            diagnostic_scores[output_row * stride + key] = score;
        // One complete wave owns one query row. Inactive rows/tiles skip this
        // recurrence uniformly, but all threads reach both LDS barriers.
        if (live_row && tile < row_tiles) {
            float next_max = fmaxf(running_max, score);
            for (unsigned mask = 16u; mask; mask >>= 1u)
                next_max = fmaxf(next_max, __shfl_xor(next_max, mask, 32u));
            const float alpha = attention::blackwell_attention_exp(running_max - next_max, exp2_table);
            const float probability = key < tokens
                ? attention::blackwell_attention_exp(score - next_max, exp2_table) : 0.0f;
            if (key < stride) probabilities[output_row * stride + key] = attention::f32_to_bf16(probability);
            float sum = probability;
            if (vllm_sum) {
                constexpr unsigned order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
                for (unsigned step = 0u; step < 5u; ++step)
                    sum += __shfl_xor(sum, order[step], 32u);
            } else {
                for (unsigned mask = 16u; mask; mask >>= 1u)
                    sum += __shfl_xor(sum, mask, 32u);
            }
            running_sum = running_sum * alpha + sum;
            running_max = next_max;
            if (!lane) scales[output_row * (tile_stride + 1u) + tile] = alpha;
        }
        __syncthreads();
    }
    if (live_row && !lane) scales[output_row * (tile_stride + 1u) + tile_stride] = running_sum;
    if (diagnostic_scores && live_row)
        for (unsigned key = tiles * 32u + lane; key < stride; key += 32u)
            diagnostic_scores[output_row * stride + key] = -INFINITY;
}
} // namespace qrt_fused_qk_probability
