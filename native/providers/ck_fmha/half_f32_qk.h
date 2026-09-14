#pragma once
#include "scaled_half_qk.h"
#include "prepared_decoded_qk.h"
#include "../moe_accumulator/sm121_half_f32_carry.h"

// Component route: compact original operands and a single-register carry.
// The production dispatcher is unchanged. Query rows are local to this slab;
// original fallback addresses use its absolute query_start.
namespace qrt_half_f32_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_sm121_half_f32_carry::Row;

template<unsigned Window>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    const Row* prepared_query, const Row* prepared_key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Window == 64u || Window == 128u || Window == 256u);
    constexpr unsigned groups = Window / 16u, words = sizeof(Row) / 4u;
    __shared__ Row left[groups][16], right[groups][16];
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * 16u;
    const unsigned row = threadIdx.x / 16u, column = threadIdx.x % 16u;
    const unsigned output_row = query_tile + row, key_row = key_tile + column;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    const bool live = output_row < query_count && key_row < stride;
    const bool active = live && key_row <= query_start + output_row;
    float carry = 0.0f;
    bool fallback = false;
    if (key_tile <= last_query) for (unsigned start = 0u; start < 16u; start += groups) {
        for (unsigned item = threadIdx.x; item < groups * 32u * words; item += kThreads) {
            const unsigned index = item / words, word = item % words, group = index / 32u, local = index % 32u;
            const bool is_query = local < 16u;
            const unsigned input_row = is_query ? query_tile + local : key_tile + local - 16u;
            uint32_t value = 0u;
            if (input_row < (is_query ? query_count : stride)) {
                const Row* source = is_query
                    ? prepared_query + (size_t(input_row) * kQueryHeads + head) * 16u + start + group
                    : prepared_key + (size_t(kv_head) * 16u + start + group) * key_stride + input_row;
                __builtin_memcpy(&value, reinterpret_cast<const unsigned char*>(source) + word * 4u, 4u);
            }
            Row* target = is_query ? &left[group][local] : &right[group][local - 16u];
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target) + word * 4u, &value, 4u);
        }
        __syncthreads();
        if (active && !fallback) {
#pragma unroll 1
            for (unsigned group = 0u; group < groups; ++group) {
                float next;
                if (!qrt_sm121_half_f32_carry::accumulate(carry, left[group][row], right[group][column], &next)) {
                    fallback = true;
                    break;
                }
                carry = next;
            }
        }
        // Inactive and failed cells still own shared-memory barrier duties.
        __syncthreads();
    }
    if (live) {
        float result = -INFINITY;
        if (active) result = fallback ? qrt_decoded_window_qk::raw_dot(
            query + (size_t(query_start + output_row) * kQueryHeads + head) * kHeadDim,
            transposed_key + size_t(kv_head) * kHeadDim * key_stride + key_row, key_stride)
            : carry * kExactScale;
        output[(size_t(output_row) * kQueryHeads + head) * stride + key_row] = result;
    }
}
} // namespace qrt_half_f32_qk
