#pragma once
#include "scaled_half_qk.h"
#include "deferred_qk_fallback.h"
#include "../moe_accumulator/sm121_half_f32_carry.h"

// Isolated four-score schedule using the established lossless scaled-half
// rows. Each K16 block keeps its original carry order. Full carries retain
// the original per-group fallback; FP32 carries mark complete-dot replay.
namespace qrt_microtile_half_qk {
using Row = qrt_sm121_scaled_half_products::Row;
using Value = qrt_q1_moe_hawkeye::Value;
template<bool FullCarry>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    constexpr unsigned groups = 8u, rows = 32u, columns = 32u;
    constexpr unsigned words = sizeof(Row) / 4u, threads = 256u;
    __shared__ Row left[groups][rows], right[groups][columns];
    const unsigned head = blockIdx.y, kv_head = head / 8u;
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * columns;
    const unsigned qr = threadIdx.x / 16u, kc = threadIdx.x % 16u;
    const unsigned last_query = query_start + min(query_tile + rows, query_count) - 1u;
    if (key_tile > last_query) {
#pragma unroll
        for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
            for (unsigned k = 0u; k < 2u; ++k) {
                const unsigned r = query_tile + qr + q * 16u, c = key_tile + kc + k * 16u;
                if (r < query_count && c < stride)
                    output[(size_t(r) * 16u + head) * stride + c] = -INFINITY;
            }
        }
        return;
    }
    Value integer[2][2];
    float floating[2][2]{};
    bool active[2][2], fallback[2][2]{};
#pragma unroll
    for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < 2u; ++k) {
            integer[q][k] = {0u, qrt_blackwell_attention::kBlackwellZeroExponent, false};
            const unsigned r = query_tile + qr + q * 16u, c = key_tile + kc + k * 16u;
            active[q][k] = r < query_count && c < stride && c <= query_start + r;
        }
    }
    for (unsigned start = 0u; start < 16u; start += groups) {
        for (unsigned item = threadIdx.x; item < groups * (rows + columns) * words; item += threads) {
            const unsigned index = item / words, word = item % words;
            const unsigned group = index / (rows + columns), local = index % (rows + columns);
            const bool is_query = local < rows;
            const unsigned r = is_query ? query_tile + local : key_tile + local - rows;
            uint32_t value = 0u;
            if (r < (is_query ? query_count : stride)) {
                const Row* source = is_query
                    ? query + (size_t(r) * 16u + head) * 16u + start + group
                    : key + (size_t(kv_head) * 16u + start + group) * key_stride + r;
                __builtin_memcpy(&value, reinterpret_cast<const unsigned char*>(source) + word * 4u, 4u);
            }
            Row* target = is_query ? &left[group][local] : &right[group][local - rows];
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target) + word * 4u, &value, 4u);
        }
        __syncthreads();
#pragma unroll 1
        for (unsigned group = 0u; group < groups; ++group) {
#pragma unroll
            for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
                for (unsigned k = 0u; k < 2u; ++k) if (active[q][k] && !fallback[q][k]) {
                    if constexpr (FullCarry) {
                        integer[q][k] = qrt_sm121_scaled_half_products::accumulate<true>(
                            integer[q][k], left[group][qr + q * 16u], right[group][kc + k * 16u]);
                    } else {
                        float next;
                        if (qrt_sm121_half_f32_carry::accumulate(floating[q][k],
                            left[group][qr + q * 16u], right[group][kc + k * 16u], &next))
                            floating[q][k] = next;
                        else fallback[q][k] = true;
                    }
                }
            }
        }
        // Failed and inactive cells retain their whole-CTA barrier duties.
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < 2u; ++k) {
            const unsigned r = query_tile + qr + q * 16u, c = key_tile + kc + k * 16u;
            if (r < query_count && c < stride) {
                float result = -INFINITY;
                if (active[q][k]) {
                    if constexpr (FullCarry)
                        result = qrt_q1_moe_hawkeye::value_to_float(
                            qrt_sm121_group16::finish_accumulator(integer[q][k])) * qrt_blackwell_attention::kExactScale;
                    else result = fallback[q][k]
                        ? qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits)
                        : floating[q][k] * qrt_blackwell_attention::kExactScale;
                }
                output[(size_t(r) * 16u + head) * stride + c] = result;
            }
        }
    }
}
} // namespace qrt_microtile_half_qk
