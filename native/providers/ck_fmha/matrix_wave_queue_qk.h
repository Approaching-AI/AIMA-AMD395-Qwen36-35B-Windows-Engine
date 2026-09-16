#pragma once
#include "positive_integer_qk.h"
#include "../moe_accumulator/sm121_float_subgroup.h"

// Component experiment only. Matrix-certified cells preserve the original
// carry. Rejected cells are compacted within each wave using ballot prefixes;
// scalar fallback consumers then process consecutive live jobs. Each wave owns
// a disjoint bounded LDS queue. No atomics, inter-wave queue or arithmetic
// approximation is introduced. The direct arm uses the same certificate and
// scalar fallback without compaction.
namespace qrt_matrix_wave_queue_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_positive_integer_qk::Row;

__device__ __forceinline__ qrt_q1_moe_hawkeye::Value fallback(
    qrt_q1_moe_hawkeye::Value carry, const Row& left, const Row& right) {
    qrt_sm121_group16::AlignedSum sum;
    bool accepted = false;
    if (left.core.original[17] && right.core.original[17] &&
        (!carry.significand || carry.exponent >= -126)) {
        qrt_sm121_float_alignment::Group group;
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i)
            group.set(i, left.core.original[i], right.core.original[i]);
        accepted = qrt_sm121_float_alignment::sum(carry, group, &sum);
    }
    if (!accepted) {
        uint32_t products[16];
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i)
            products[i] = qrt_sm121_group16::pack_product(
                qrt_q1_moe_hawkeye::multiply_bf16(left.core.original[i], right.core.original[i], -133));
        sum = qrt_sm121_group16::sum_packed(carry, products);
    }
    return qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
}

template<unsigned Columns, bool Compact>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Columns == 32u || Columns == 64u);
    constexpr unsigned cells = 16u * Columns, per_thread = cells / kThreads, words = sizeof(Row) / 4u;
    __shared__ Row left[16], right[Columns];
    __shared__ int32_t partials[4][cells];
    // Preserve the original internal exponent across every K16 boundary,
    // including generated carries beyond the finite FP32 exponent range.
    __shared__ qrt_q1_moe_hawkeye::Value carries[cells];
    __shared__ unsigned pending[cells];
    constexpr unsigned queue_capacity = cells / (kThreads / 32u);
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * Columns;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    for (unsigned cell = threadIdx.x; cell < cells; cell += kThreads)
        carries[cell] = {0u, kBlackwellZeroExponent, false};
    __syncthreads();
    if (key_tile <= last_query) for (unsigned group = 0u; group < 16u; ++group) {
        unsigned queued = 0u;
        for (unsigned item = threadIdx.x; item < (16u + Columns) * words; item += kThreads) {
            const unsigned row = item / words, word = item % words;
            const bool is_query = row < 16u;
            const unsigned input_row = is_query ? query_tile + row : key_tile + row - 16u;
            uint32_t value = 0u;
            if (input_row < (is_query ? query_count : stride)) {
                const Row* source = is_query ? query + (size_t(input_row) * kQueryHeads + head) * 16u + group
                    : key + (size_t(kv_head) * 16u + group) * key_stride + input_row;
                __builtin_memcpy(&value, reinterpret_cast<const unsigned char*>(source) + word * 4u, 4u);
            }
            Row* target = is_query ? left + row : right + row - 16u;
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target) + word * 4u, &value, 4u);
        }
        __syncthreads();
        if (wave < Columns / 16u) {
            const auto result = qrt_positive_integer_qk::products(left[lane % 16u], right[wave * 16u + lane % 16u]);
            #pragma unroll
            for (unsigned element = 0u; element < 8u; ++element) {
                const unsigned row = 2u * element + lane / 16u, column = wave * 16u + lane % 16u;
                const unsigned cell = row * Columns + column;
                const int total = int(left[row].core.original[16]) + int(right[column].core.original[16]);
                partials[0][cell] = result.high[element];
                partials[1][cell] = int(result.combined[element]) - result.high[element] - result.low[element] - total * 128 + 262144;
                partials[2][cell] = 0;
                partials[3][cell] = result.low[element];
            }
        }
        __syncthreads();
        #pragma unroll
        for (unsigned item = 0u; item < per_thread; ++item) {
            const unsigned cell = threadIdx.x + item * kThreads, row = cell / Columns, column = cell % Columns;
            const unsigned output_row = query_tile + row, key_row = key_tile + column;
            bool rejected = false;
            if (output_row < query_count && key_row < stride && key_row <= query_start + output_row) {
                const auto& a = left[row].core;
                const auto& b = right[column].core;
                const auto carry = carries[cell];
                const uint32_t exceptions = (a.exceptions | b.exceptions) & a.nonzero & b.nonzero;
                qrt_sm121_group16::AlignedSum sum;
                bool accepted = false;
                if (a.unit >= 0 && b.unit >= 0 && !exceptions) {
                    const int64_t product = int64_t(partials[0][cell]) * 65536 +
                        int64_t(partials[1][cell]) * 256 + partials[3][cell];
                    accepted = qrt_sm121_integer_parts::sum_exact_integer_product(carry, product,
                        a.unit, a.maximum, b.unit, b.maximum, &sum, a.trailing, b.trailing);
                }
                if (accepted) {
                    carries[cell] = qrt_sm121_wave16::normalize(
                        sum.value.magnitude, sum.value.negative, sum.max_exponent);
                } else if constexpr (Compact) {
                    rejected = true;
                } else {
                    carries[cell] = fallback(carry, left[row], right[column]);
                }
            }
            if constexpr (Compact) {
                // All32 lanes participate, including masked/accepted cells.
                // A wave owns queue_capacity == per_thread *32 slots; every
                // owned cell enters at most once in this original K16 group.
                const unsigned mask = unsigned(__ballot(rejected));
                if (rejected) pending[wave * queue_capacity + queued +
                    unsigned(__popc(mask & ((uint32_t(1u) << lane) - 1u)))] = cell;
                queued += unsigned(__popc(mask));
            }
        }
        __syncthreads();
        if constexpr (Compact) {
            // A consumer reads only its wave's published prefix. The last
            // partial batch uses ordinary scalar code, with no collectives.
            for (unsigned slot = lane; slot < queued; slot += 32u) {
                const unsigned cell = pending[wave * queue_capacity + slot];
                carries[cell] = fallback(carries[cell], left[cell / Columns], right[cell % Columns]);
            }
        }
        // Finish all queue consumers before replacing rows, carries or indices
        // for the next original K16 group.
        __syncthreads();
    }
    #pragma unroll
    for (unsigned item = 0u; item < per_thread; ++item) {
        const unsigned cell = threadIdx.x + item * kThreads, row = query_tile + cell / Columns, key_row = key_tile + cell % Columns;
        if (row < query_count && key_row < stride)
            output[(size_t(row) * kQueryHeads + head) * stride + key_row] =
                key_row <= query_start + row ? qrt_q1_moe_hawkeye::value_to_float(
                    qrt_sm121_group16::finish_accumulator(carries[cell])) * kExactScale : -INFINITY;
    }
}
} // namespace qrt_matrix_wave_queue_qk
