#pragma once
#include "matrix_wave_queue_qk.h"

// Isolated component experiment. Keep every full integer carry in its owning
// thread across K16 groups. The three published matrix partials are dead before
// their storage is reused for a bounded per-wave fallback queue. This admits
// 128 key columns without narrowing a carry or allocating a second LDS array.
namespace qrt_register_matrix_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_positive_integer_qk::Row;
using Value = qrt_q1_moe_hawkeye::Value;

template<unsigned Columns, bool Compact>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Columns == 64u || Columns == 128u);
    constexpr unsigned cells = 16u * Columns, per_thread = cells / kThreads;
    constexpr unsigned words = sizeof(Row) / 4u, queue_capacity = per_thread * 32u;
    static_assert(cells < 0x80000000u);
    __shared__ Row left[16], right[Columns];
    __shared__ int32_t scratch[3][cells];
    Value carries[per_thread];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * Columns;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
#pragma unroll
    for (unsigned item = 0u; item < per_thread; ++item)
        carries[item] = {0u, kBlackwellZeroExponent, false};
    if (key_tile <= last_query) for (unsigned group = 0u; group < 16u; ++group) {
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
                scratch[0][cell] = result.high[element];
                scratch[1][cell] = int(result.combined[element]) - result.high[element] - result.low[element] - total * 128 + 262144;
                scratch[2][cell] = result.low[element];
            }
        }
        __syncthreads();
        unsigned positions[per_thread], queued = 0u;
#pragma unroll
        for (unsigned item = 0u; item < per_thread; ++item) {
            const unsigned cell = threadIdx.x + item * kThreads, row = cell / Columns, column = cell % Columns;
            const unsigned output_row = query_tile + row, key_row = key_tile + column;
            bool rejected = false;
            if (output_row < query_count && key_row < stride && key_row <= query_start + output_row) {
                const auto& a = left[row].core;
                const auto& b = right[column].core;
                const auto carry = carries[item];
                const uint32_t exceptions = (a.exceptions | b.exceptions) & a.nonzero & b.nonzero;
                qrt_sm121_group16::AlignedSum sum;
                bool accepted = false;
                if (a.unit >= 0 && b.unit >= 0 && !exceptions) {
                    const int64_t product = int64_t(scratch[0][cell]) * 65536 +
                        int64_t(scratch[1][cell]) * 256 + scratch[2][cell];
                    accepted = qrt_sm121_integer_parts::sum_exact_integer_product(carry, product,
                        a.unit, a.maximum, b.unit, b.maximum, &sum, a.trailing, b.trailing);
                }
                if (accepted) carries[item] = qrt_sm121_wave16::normalize(
                    sum.value.magnitude, sum.value.negative, sum.max_exponent);
                else if constexpr (Compact) rejected = true;
                else carries[item] = qrt_matrix_wave_queue_qk::fallback(carry, left[row], right[column]);
            }
            if constexpr (Compact) {
                // All lanes enter the ballot, including causal/tail masks.
                // Keep publication private until every matrix partial is read.
                const unsigned mask = unsigned(__ballot(rejected));
                positions[item] = rejected ? wave * queue_capacity + queued +
                    unsigned(__popc(mask & ((uint32_t(1u) << lane) - 1u))) : cells;
                queued += unsigned(__popc(mask));
            }
        }
        if constexpr (Compact) {
            // Matrix-reader retirement precedes the scratch lifetime change.
            __syncthreads();
#pragma unroll
            for (unsigned item = 0u; item < per_thread; ++item) if (positions[item] != cells) {
                const unsigned slot = positions[item], cell = threadIdx.x + item * kThreads;
                // Cell IDs use fewer than31 bits. Store the sign with that ID,
                // preserving every original significand and exponent bit.
                scratch[0][slot] = int32_t(cell | (carries[item].negative ? 0x80000000u : 0u));
                scratch[1][slot] = int32_t(carries[item].significand);
                scratch[2][slot] = carries[item].exponent;
            }
            __syncthreads();
            for (unsigned offset = lane; offset < queued; offset += 32u) {
                const unsigned slot = wave * queue_capacity + offset;
                const uint32_t tagged = uint32_t(scratch[0][slot]), cell = tagged & 0x7fffffffu;
                const Value carry{uint32_t(scratch[1][slot]), int16_t(scratch[2][slot]), bool(tagged >> 31u)};
                const auto value = qrt_matrix_wave_queue_qk::fallback(carry, left[cell / Columns], right[cell % Columns]);
                scratch[0][slot] = int32_t(cell | (value.negative ? 0x80000000u : 0u));
                scratch[1][slot] = int32_t(value.significand);
                scratch[2][slot] = value.exponent;
            }
            __syncthreads();
#pragma unroll
            for (unsigned item = 0u; item < per_thread; ++item) if (positions[item] != cells) {
                const unsigned slot = positions[item];
                carries[item] = {uint32_t(scratch[1][slot]), int16_t(scratch[2][slot]),
                    bool(uint32_t(scratch[0][slot]) >> 31u)};
            }
        }
        // Finish original/fallback readers before replacing rows or scratch.
        __syncthreads();
    }
#pragma unroll
    for (unsigned item = 0u; item < per_thread; ++item) {
        const unsigned cell = threadIdx.x + item * kThreads, row = query_tile + cell / Columns, key_row = key_tile + cell % Columns;
        if (row < query_count && key_row < stride)
            output[(size_t(row) * kQueryHeads + head) * stride + key_row] =
                key_row <= query_start + row ? qrt_q1_moe_hawkeye::value_to_float(
                    qrt_sm121_group16::finish_accumulator(carries[item])) * kExactScale : -INFINITY;
    }
}
} // namespace qrt_register_matrix_qk
