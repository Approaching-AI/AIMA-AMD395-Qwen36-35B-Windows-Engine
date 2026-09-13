#ifndef QRT_FLOAT_SUBGROUP_QK_H
#define QRT_FLOAT_SUBGROUP_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_float_subgroup.h"

// Component comparison only. Keep the scalar product dispatcher unchanged
// until this ownership/layout change has real-model numerical evidence.
namespace qrt_float_subgroup_qk {
using namespace qrt_blackwell_attention;

template<unsigned Lanes, unsigned Staging, unsigned QueryRows, unsigned KeyPitch,
    bool Eligible>
__device__ __forceinline__ float dot(
    const uint32_t (&queries)[QueryRows][kHeadDim / 2u],
    const uint32_t (&keys)[kHeadDim / 2u][KeyPitch], unsigned row, unsigned key) {
    static_assert(Lanes == 4u || Lanes == 8u || Lanes == 16u);
    static_assert(Staging == 1u || Staging == 4u);
    constexpr unsigned items = 16u / Lanes;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    qrt_q1_moe_hawkeye::Value carry{0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < kHeadDim; base += 16u * Staging) {
        qrt_sm121_float_subgroup::Product products[Staging][items];
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group) {
#pragma unroll
            for (unsigned i = 0u; i < items; ++i) {
                const unsigned feature = base + group * 16u + lane * items + i;
                const unsigned shift = (feature & 1u) * 16u;
                const uint16_t a = uint16_t(queries[row][feature / 2u] >> shift);
                const uint16_t b = uint16_t(keys[feature / 2u][key] >> shift);
                if constexpr (Eligible) {
                    const bool zero = !(a & 0x7fffu) || !(b & 0x7fffu);
                    products[group][i] = {
                        qrt_sm121_float_alignment::from_bits(uint32_t(a) << 16u) *
                            qrt_sm121_float_alignment::from_bits(uint32_t(b) << 16u),
                        uint32_t(a) | (uint32_t(b) << 16u),
                        zero ? -133 : int((a >> 7u) & 255u) + int((b >> 7u) & 255u) - 254};
                } else {
                    // Force the existing integer fallback for the complete
                    // tile; original BF16 inputs retain subnormal/edge bits.
                    products[group][i] = {0.0f, uint32_t(a) | (uint32_t(b) << 16u), 512};
                }
            }
        }
#pragma unroll
        for (unsigned group = 0u; group < Staging; ++group)
            carry = qrt_sm121_float_subgroup::accumulate<Lanes>(carry, products[group]);
    }
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(carry)) * kExactScale;
}

template<unsigned Lanes, unsigned Staging, unsigned QueryRows>
__global__ void scores(const uint16_t* query, const uint16_t* transposed_key,
    float* output, unsigned query_start, unsigned query_count, unsigned stride,
    unsigned key_stride) {
    constexpr unsigned KeyColumns = kThreads / (Lanes * QueryRows);
    constexpr unsigned KeyPitch = KeyColumns + 1u;
    static_assert(KeyColumns * QueryRows * Lanes == kThreads);
    __shared__ uint32_t queries[QueryRows][kHeadDim / 2u];
    __shared__ uint32_t keys[kHeadDim / 2u][KeyPitch];
    __shared__ unsigned fallback;
    const unsigned lane = threadIdx.x & (Lanes - 1u);
    const unsigned head = blockIdx.y, first_query = blockIdx.z * QueryRows;
    const unsigned first_key = blockIdx.x * KeyColumns;
    const unsigned local_query = (threadIdx.x / Lanes) / KeyColumns;
    const unsigned local_key = (threadIdx.x / Lanes) % KeyColumns;
    const unsigned row = first_query + local_query, key = first_key + local_key;
    const unsigned last_query = query_start + min(first_query + QueryRows, query_count) - 1u;
    if (first_key > last_query) {
        if (!lane && row < query_count && key < stride)
            output[(size_t(row) * kQueryHeads + head) * stride + key] = -INFINITY;
        return;
    }
    if (!threadIdx.x) fallback = 0u;
    __syncthreads();
    bool invalid = false;
    for (unsigned cell = threadIdx.x; cell < QueryRows * (kHeadDim / 2u); cell += kThreads) {
        const unsigned qrow = cell / (kHeadDim / 2u), pair = cell % (kHeadDim / 2u);
        uint16_t a = 0u, b = 0u;
        if (first_query + qrow < query_count) {
            const size_t base = (size_t(query_start + first_query + qrow) * kQueryHeads + head) * kHeadDim + pair * 2u;
            a = query[base]; b = query[base + 1u];
        }
        invalid |= !qrt_sm121_float_alignment::eligible(a) || !qrt_sm121_float_alignment::eligible(b);
        queries[qrow][pair] = uint32_t(a) | (uint32_t(b) << 16u);
    }
    const unsigned kv_head = head / (kQueryHeads / kKvHeads);
    for (unsigned cell = threadIdx.x; cell < (kHeadDim / 2u) * KeyColumns; cell += kThreads) {
        const unsigned pair = cell / KeyColumns, column = cell % KeyColumns;
        uint16_t a = 0u, b = 0u;
        if (first_key + column < stride) {
            const size_t base = (size_t(kv_head) * kHeadDim + pair * 2u) * key_stride + first_key + column;
            a = transposed_key[base]; b = transposed_key[base + key_stride];
        }
        invalid |= !qrt_sm121_float_alignment::eligible(a) || !qrt_sm121_float_alignment::eligible(b);
        keys[pair][column] = uint32_t(a) | (uint32_t(b) << 16u);
    }
    if (invalid) atomicOr(&fallback, 1u);
    __syncthreads();
    // Every lane of a dot shares these predicates; active subgroups remain
    // complete through each DPP/shuffle reduction, including causal tails.
    if (row >= query_count || key >= stride) return;
    const size_t cell = (size_t(row) * kQueryHeads + head) * stride + key;
    if (key > query_start + row) {
        if (!lane) output[cell] = -INFINITY;
        return;
    }
    const float value = fallback ? dot<Lanes, Staging, QueryRows, KeyPitch, false>(queries, keys, local_query, local_key)
                                : dot<Lanes, Staging, QueryRows, KeyPitch, true>(queries, keys, local_query, local_key);
    if (!lane) output[cell] = value;
}
} // namespace qrt_float_subgroup_qk
#endif
