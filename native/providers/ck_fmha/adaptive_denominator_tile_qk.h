#pragma once
#include "adaptive_denominator_qk.h"

namespace qrt_adaptive_denominator_qk {
// The score collector remains unchanged. Group its selected cells into original
// 16-query/16-key tiles. Per-tile bitsets are cleared before each marking pass;
// exactly one thread publishes each nonempty tile. The next kernel observes all
// bitsets through ordinary stream ordering, without a host count read.
__global__ void mark_tiles(const unsigned* indices, const unsigned* count,
    unsigned stride, unsigned* masks, unsigned* owners, unsigned* tiles,
    unsigned* tile_count) {
    const unsigned key_tiles = (stride + 15u) / 16u;
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count;
         slot += gridDim.x * blockDim.x) {
        const unsigned cell = indices[slot], row = cell / stride, key = cell % stride;
        const unsigned head = row % kQueryHeads, query = row / kQueryHeads;
        const unsigned tile = (query / 16u * kQueryHeads + head) * key_tiles + key / 16u;
        const unsigned position = query % 16u * 16u + key % 16u;
        const unsigned previous = atomicOr(masks + size_t(tile) * 8u + position / 32u, 1u << (position % 32u));
        if (!previous && !atomicExch(owners + tile, 1u)) tiles[atomicAdd(tile_count, 1u)] = tile;
    }
}

// Compact selected dots within each tile before the arithmetic, so unused
// warps perform no K16 work. All selected dots reuse the same original packed
// Q/K operands through two 128-feature shared windows. The carried value and
// exceptional full-dot restart are identical to the retained prepared kernel.
template<bool UpdateProbability>
__global__ void repair_tiles(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* scores, float* errors,
    unsigned start, unsigned queries, unsigned stride, unsigned key_stride,
    const unsigned* masks, const unsigned* tiles, const unsigned* tile_count,
    const float* maxima, float* lower, float* upper, const unsigned char* exp2) {
    __shared__ uint32_t qvalues[16u][128u], kvalues[128u][16u];
    __shared__ unsigned positions[256u], selected_count;
    const unsigned lane = threadIdx.x % 32u, key_tiles = (stride + 15u) / 16u;
    const unsigned probability_tiles = (stride + 31u) / 32u;
    for (unsigned slot = blockIdx.x; slot < *tile_count; slot += gridDim.x) {
        const unsigned tile = tiles[slot], tile_row = tile / key_tiles;
        const unsigned head = tile_row % kQueryHeads, query_tile = tile_row / kQueryHeads * 16u;
        const unsigned key_tile = tile % key_tiles * 16u, kv_head = head / (kQueryHeads / kKvHeads);
        if (!threadIdx.x) selected_count = 0u;
        __syncthreads();
        const bool selected = (masks[size_t(tile) * 8u + threadIdx.x / 32u] >> lane) & 1u;
        const unsigned ballot = __ballot(selected);
        unsigned begin = !lane && ballot ? atomicAdd(&selected_count, unsigned(__popc(ballot))) : 0u;
        begin = __shfl(begin, 0u, 32u);
        if (selected) positions[begin + unsigned(__popc(ballot & ((uint32_t(1u) << lane) - 1u)))] = threadIdx.x;
        __syncthreads();
        const bool active = threadIdx.x < selected_count;
        const unsigned position = active ? positions[threadIdx.x] : 0u;
        const unsigned qr = position / 16u, kc = position % 16u;
        const unsigned row = query_tile + qr, key = key_tile + kc;
        bool fallback = active && (!query_flags[(start + row) * kQueryHeads + head] ||
            !key_flags[key * kKvHeads + kv_head]);
        float carry = 0.0f;
        for (unsigned window = 0u; window < kHeadDim; window += 128u) {
            for (unsigned cell = threadIdx.x; cell < 16u * 128u; cell += 256u) {
                const unsigned r = cell / 128u, c = cell % 128u;
                qvalues[r][c] = query_tile + r < queries
                    ? packed_query[(size_t(start + query_tile + r) * kQueryHeads + head) * kHeadDim + window + c]
                    : prepared::decoded::pack(0u);
            }
            for (unsigned cell = threadIdx.x; cell < 128u * 16u; cell += 256u) {
                const unsigned r = cell / 16u, c = cell % 16u;
                kvalues[r][c] = key_tile + c < stride
                    ? packed_key[(size_t(kv_head) * kHeadDim + window + r) * key_stride + key_tile + c]
                    : prepared::decoded::pack(0u);
            }
            __syncthreads();
            if (active && !fallback) {
                for (unsigned base = 0u; base < 128u; base += 16u) {
                    qrt_sm121_float_alignment::Group group;
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i)
                        prepared::decoded::set_packed(group, i, qvalues[qr][base + i], kvalues[base + i][kc]);
                    float next;
                    if (!qrt_sm121_f32_carry::accumulate<0u>(carry, group, &next)) { fallback = true; break; }
                    carry = next;
                }
            }
            __syncthreads();
        }
        if (active) {
            const unsigned output_row = row * kQueryHeads + head;
            const size_t cell = size_t(output_row) * stride + key;
            const float score = fallback ? qrt_decoded_window_qk::raw_dot(
                query + (size_t(start + row) * kQueryHeads + head) * kHeadDim,
                transposed_key + size_t(kv_head) * kHeadDim * key_stride + key, key_stride) : carry * kExactScale;
            scores[cell] = score; errors[cell] = 0.0f;
            if constexpr (UpdateProbability) {
                const float maximum = maxima[size_t(output_row) * probability_tiles + key / 32u];
                const float p = blackwell_attention_exp(fminf(score, maximum) - maximum, exp2);
                lower[cell] = p; upper[cell] = p;
            }
        }
        // No lane may retire the tile's shared operands or compact queue while
        // another lane still consumes them, including an exceptional fallback.
        __syncthreads();
    }
}
} // namespace qrt_adaptive_denominator_qk
