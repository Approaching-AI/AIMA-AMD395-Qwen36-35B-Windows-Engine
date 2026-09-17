#pragma once
#include "deferred_qk_fallback.h"
#include "../moe_accumulator/sm121_certified_f32_group.h"

// Isolated complete-QK experiment. Reuse exact K16 operand-grid certificates
// across all score cells; rejected groups retain the paired-exponent scan.
// Original carry order, normalizer, causal mask and full-dot fallback remain.
namespace qrt_certified_group_qk {
namespace decoded = qrt_sm121_decoded_bf16;
namespace certificate = qrt_sm121_certified_f32_group;
template<bool Key>
__global__ void prepare_bounds(const uint16_t* input, uint32_t* output, unsigned tokens) {
    constexpr unsigned heads = Key ? 2u : 16u;
    const size_t cell = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (cell >= size_t(tokens) * heads * 16u) return;
    const unsigned row = unsigned(cell / 16u), group = unsigned(cell % 16u);
    const size_t target = Key ? (size_t(row % heads) * 16u + group) * tokens + row / heads : cell;
    output[target] = certificate::bounds::prepare(input + size_t(row) * 256u + group * 16u);
}
template<unsigned QueryCells, unsigned KeyCells>
__global__ void scores(const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags,
    const uint32_t* query_bounds, const uint32_t* key_bounds, float* output,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride) {
    static_assert((QueryCells == 1u || QueryCells == 2u) &&
        (KeyCells == 1u || KeyCells == 2u));
    constexpr unsigned window_width = 128u, rows = 16u * QueryCells;
    constexpr unsigned columns = 16u * KeyCells, threads = 256u;
    __shared__ uint32_t qvalues[rows][window_width], kvalues[window_width][columns];
    __shared__ uint32_t qbounds[rows][8], kbounds[8][columns];
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
        for (unsigned cell = threadIdx.x; cell < rows * 8u; cell += threads) {
            const unsigned r = cell / 8u, g = cell % 8u;
            qbounds[r][g] = query_tile + r < count
                ? query_bounds[(size_t(start + query_tile + r) * 16u + head) * 16u + window / 16u + g]
                : certificate::bounds::valid_bit | 0xff00u;
        }
        for (unsigned cell = threadIdx.x; cell < 8u * columns; cell += threads) {
            const unsigned g = cell / columns, c = cell % columns;
            kbounds[g][c] = key_tile + c < stride
                ? key_bounds[(size_t(kv_head) * 16u + window / 16u + g) * key_stride + key_tile + c]
                : certificate::bounds::valid_bit | 0xff00u;
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
                    int maximum[KeyCells]; bool certified[KeyCells];
#pragma unroll
                    for (unsigned k = 0u; k < KeyCells; ++k)
                        certified[k] = active[q][k] && !fallback[q][k] && certificate::exponent(
                            carry[q][k], qbounds[qr + q * 16u][base / 16u],
                            kbounds[base / 16u][kc + k * 16u], &maximum[k]);
#pragma unroll
                    for (unsigned k = 0u; k < KeyCells; ++k) {
                        if (certified[k]) {
#pragma unroll
                            for (unsigned i = 0u; i < 16u; ++i)
                                groups[k].products[i] = qrt_sm121_float_alignment::from_bits(qvalues[qr + q * 16u][base + i] & 0xffff0000u) *
                                    qrt_sm121_float_alignment::from_bits(kvalues[base + i][kc + k * 16u] & 0xffff0000u);
                            groups[k].first_negative = ((qvalues[qr + q * 16u][base] ^ kvalues[base][kc + k * 16u]) & 0x80000000u) != 0u;
                        } else {
#pragma unroll
                            for (unsigned i = 0u; i < 16u; ++i)
                                decoded::set_packed(groups[k], i, qvalues[qr + q * 16u][base + i], kvalues[base + i][kc + k * 16u]);
                        }
                    }
#pragma unroll
                    for (unsigned k = 0u; k < KeyCells; ++k) if (active[q][k] && !fallback[q][k]) {
                        float next;
                        const bool accepted = certified[k]
                            ? certificate::accumulate(carry[q][k], groups[k].products, groups[k].first_negative, maximum[k], &next)
                            : qrt_sm121_f32_carry::accumulate<0u>(carry[q][k], groups[k], &next);
                        if (accepted) carry[q][k] = next;
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
} // namespace qrt_certified_group_qk
