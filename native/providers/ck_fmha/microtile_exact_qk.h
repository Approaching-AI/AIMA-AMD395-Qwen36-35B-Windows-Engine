#pragma once
#include "deferred_qk_fallback.h"

// Isolated exact scalar matrix schedule. A thread owns a small rectangle of
// independent score cells, sharing decoded operands between its key columns.
// K16 products, exponent maxima, modulo sums and ordered carries are unchanged.
// All rejected cells use the existing complete original-dot scan afterwards.
namespace qrt_microtile_exact_qk {
namespace decoded = qrt_sm121_decoded_bf16;
template<unsigned QueryCells, unsigned KeyCells>
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags, float* output,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    static_assert((QueryCells == 1u || QueryCells == 2u) &&
        (KeyCells == 1u || KeyCells == 2u));
    constexpr unsigned window_width = 128u, rows = 16u * QueryCells;
    constexpr unsigned columns = 16u * KeyCells, threads = 256u;
    __shared__ uint32_t qvalues[rows][window_width], kvalues[window_width][columns];
    const unsigned head = blockIdx.y, kv_head = head / 8u;
    const unsigned query_tile = blockIdx.z * rows, key_tile = blockIdx.x * columns;
    const unsigned qr = threadIdx.x / 16u, kc = threadIdx.x % 16u;
    const unsigned last_query = start + min(query_tile + rows, count) - 1u;
    if (key_tile > last_query) {
#pragma unroll
        for (unsigned q = 0u; q < QueryCells; ++q) {
#pragma unroll
            for (unsigned k = 0u; k < KeyCells; ++k) {
                const unsigned row = query_tile + qr + q * 16u, key = key_tile + kc + k * 16u;
                if (row < count && key < stride)
                    output[(size_t(row) * 16u + head) * stride + key] = -INFINITY;
            }
        }
        return;
    }
    float carry[QueryCells][KeyCells]{};
    bool active[QueryCells][KeyCells], fallback[QueryCells][KeyCells];
#pragma unroll
    for (unsigned q = 0u; q < QueryCells; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < KeyCells; ++k) {
            const unsigned row = query_tile + qr + q * 16u, key = key_tile + kc + k * 16u;
            active[q][k] = row < count && key < stride && key <= start + row;
            fallback[q][k] = active[q][k] &&
                (!query_flags[(start + row) * 16u + head] || !key_flags[key * 2u + kv_head]);
        }
    }
    for (unsigned window = 0u; window < 256u; window += window_width) {
        for (unsigned cell = threadIdx.x; cell < rows * window_width; cell += threads) {
            const unsigned r = cell / window_width, c = cell % window_width;
            qvalues[r][c] = query_tile + r < count
                ? packed_query[(size_t(start + query_tile + r) * 16u + head) * 256u + window + c]
                : decoded::pack(0u);
        }
        for (unsigned cell = threadIdx.x; cell < window_width * columns; cell += threads) {
            const unsigned r = cell / columns, c = cell % columns;
            kvalues[r][c] = key_tile + c < stride
                ? packed_key[(size_t(kv_head) * 256u + window + r) * key_stride + key_tile + c]
                : decoded::pack(0u);
        }
        __syncthreads();
        for (unsigned base = 0u; base < window_width; base += 16u) {
#pragma unroll
            for (unsigned q = 0u; q < QueryCells; ++q) {
                bool any = false;
#pragma unroll
                for (unsigned k = 0u; k < KeyCells; ++k) any |= active[q][k] && !fallback[q][k];
                if (any) {
                    qrt_sm121_float_alignment::Group groups[KeyCells];
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i) {
                        const uint32_t left = qvalues[qr + q * 16u][base + i];
#pragma unroll
                        for (unsigned k = 0u; k < KeyCells; ++k)
                            decoded::set_packed(groups[k], i, left, kvalues[base + i][kc + k * 16u]);
                    }
#pragma unroll
                    for (unsigned k = 0u; k < KeyCells; ++k) if (active[q][k] && !fallback[q][k]) {
                        float next;
                        if (qrt_sm121_f32_carry::accumulate<0u>(carry[q][k], groups[k], &next))
                            carry[q][k] = next;
                        else fallback[q][k] = true;
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0u; q < QueryCells; ++q) {
#pragma unroll
        for (unsigned k = 0u; k < KeyCells; ++k) {
            const unsigned row = query_tile + qr + q * 16u, key = key_tile + kc + k * 16u;
            if (row < count && key < stride)
                output[(size_t(row) * 16u + head) * stride + key] = !active[q][k] ? -INFINITY
                    : fallback[q][k] ? qrt_sm121_float_alignment::from_bits(qrt_deferred_qk_fallback::deferred_bits)
                    : carry[q][k] * qrt_blackwell_attention::kExactScale;
        }
    }
}
} // namespace qrt_microtile_exact_qk
