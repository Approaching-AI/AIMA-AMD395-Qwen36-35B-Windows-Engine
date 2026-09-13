#ifndef QRT_F32_CARRY_QK_H
#define QRT_F32_CARRY_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_f32_carry.h"

// Component experiment only. Product dispatch does not include this header.
namespace qrt_f32_carry_qk {
using namespace qrt_blackwell_attention;
namespace fast = qrt_sm121_f32_carry;
template<unsigned Method>
__device__ __forceinline__ bool dot(
    const uint32_t (&queries)[kTiledExactQueries][kHeadDim / 2u],
    const uint32_t (&keys)[kHeadDim / 2u][kTiledExactKeys],
    unsigned row, unsigned key, float* output) {
    float carry = 0.0f;
    for (unsigned base = 0u; base < kHeadDim; base += 16u) {
        qrt_sm121_float_alignment::Group group;
#pragma unroll
        for (unsigned i = 0u; i < 16u; i += 2u) {
            const uint32_t q = queries[row][(base + i) / 2u], k = keys[(base + i) / 2u][key];
            group.set(i, uint16_t(q), uint16_t(k));
            group.set(i + 1u, uint16_t(q >> 16u), uint16_t(k >> 16u));
        }
        float next;
        if (!fast::accumulate<Method>(carry, group, &next)) return false;
        carry = next;
    }
    *output = carry * kExactScale;
    return true;
}

template<unsigned Method>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    __shared__ uint32_t queries[kTiledExactQueries][kHeadDim / 2u];
    __shared__ uint32_t keys[kHeadDim / 2u][kTiledExactKeys];
    __shared__ unsigned fallback;
    const unsigned head = blockIdx.y, query_tile = blockIdx.z * kTiledExactQueries;
    const unsigned key_tile = blockIdx.x * kTiledExactKeys;
    const unsigned local_query = threadIdx.x / kTiledExactKeys, local_key = threadIdx.x % kTiledExactKeys;
    const unsigned row = query_tile + local_query, key = key_tile + local_key;
    const unsigned last_query = query_start + min(query_tile + kTiledExactQueries, query_count) - 1u;
    if (key_tile > last_query) {
        if (row < query_count && key < stride)
            output[(size_t(row) * kQueryHeads + head) * stride + key] = -INFINITY;
        return;
    }
    if (!threadIdx.x) fallback = 0u;
    __syncthreads();
    bool invalid = false;
    for (unsigned cell = threadIdx.x; cell < kTiledExactQueries * (kHeadDim / 2u); cell += kThreads) {
        const unsigned qrow = cell / (kHeadDim / 2u), pair = cell % (kHeadDim / 2u);
        uint16_t a = 0u, b = 0u;
        if (query_tile + qrow < query_count) {
            const size_t base = (size_t(query_start + query_tile + qrow) * kQueryHeads + head) * kHeadDim + pair * 2u;
            a = query[base]; b = query[base + 1u];
        }
        invalid = invalid || !qrt_sm121_float_alignment::eligible(a) || !qrt_sm121_float_alignment::eligible(b);
        queries[qrow][pair] = uint32_t(a) | (uint32_t(b) << 16u);
    }
    const unsigned kv_head = head / (kQueryHeads / kKvHeads);
    for (unsigned cell = threadIdx.x; cell < (kHeadDim / 2u) * kTiledExactKeys; cell += kThreads) {
        const unsigned pair = cell / kTiledExactKeys, column = cell % kTiledExactKeys;
        uint16_t a = 0u, b = 0u;
        if (key_tile + column < stride) {
            const size_t base = (size_t(kv_head) * kHeadDim + pair * 2u) * key_stride + key_tile + column;
            a = transposed_key[base]; b = transposed_key[base + key_stride];
        }
        invalid = invalid || !qrt_sm121_float_alignment::eligible(a) || !qrt_sm121_float_alignment::eligible(b);
        keys[pair][column] = uint32_t(a) | (uint32_t(b) << 16u);
    }
    if (invalid) atomicOr(&fallback, 1u);
    __syncthreads();
    if (row >= query_count || key >= stride) return;
    const size_t cell = (size_t(row) * kQueryHeads + head) * stride + key;
    if (key > query_start + row) { output[cell] = -INFINITY; return; }
    float value;
    if (fallback || !dot<Method>(queries, keys, local_query, local_key, &value))
        value = blackwell_tiled_qk_dot<false>(queries, keys, local_query, local_key);
    output[cell] = value;
}

} // namespace qrt_f32_carry_qk
#endif
