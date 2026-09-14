#pragma once
#include "positive_integer_qk.h"
#include "../moe_accumulator/sm121_float_subgroup.h"

// Component experiment only. Matrix-certified cells keep the existing exact
// integer certificate. Other cells enter a bounded LDS queue and use the
// existing four-lane K16 alignment, instead of divergent scalar fallback in
// each matrix consumer. No global intermediate surfaces or extra launches.
namespace qrt_matrix_cooperative_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_positive_integer_qk::Row;

template<unsigned Columns>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Columns == 32u || Columns == 64u);
    constexpr unsigned cells = 16u * Columns, per_thread = cells / kThreads, words = sizeof(Row) / 4u;
    __shared__ Row left[16], right[Columns];
    __shared__ int32_t partials[4][cells];
    // Preserve the original internal exponent across every K16 boundary,
    // including generated carries beyond the finite FP32 exponent range.
    __shared__ qrt_q1_moe_hawkeye::Value carries[cells];
    __shared__ unsigned pending[cells], pending_count;
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * Columns;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    for (unsigned cell = threadIdx.x; cell < cells; cell += kThreads)
        carries[cell] = {0u, kBlackwellZeroExponent, false};
    __syncthreads();
    if (key_tile <= last_query) for (unsigned group = 0u; group < 16u; ++group) {
        if (!threadIdx.x) pending_count = 0u;
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
                } else {
                    // Every valid cell visits this branch at most once, so a
                    // completely uncertified tile still fits the owning queue.
                    pending[atomicAdd(&pending_count, 1u)] = cell;
                }
            }
        }
        __syncthreads();
        constexpr unsigned lanes = 4u, items = 4u, subgroups = kThreads / lanes;
        const unsigned subgroup = threadIdx.x / lanes, member = threadIdx.x % lanes;
        for (unsigned slot = subgroup; slot < pending_count; slot += subgroups) {
            const unsigned cell = pending[slot], row = cell / Columns, column = cell % Columns;
            const auto* a = left[row].core.original;
            const auto* b = right[column].core.original;
            auto carry = carries[cell];
            if (a[17] && b[17]) {
                qrt_sm121_float_subgroup::Product products[items];
                #pragma unroll
                for (unsigned item = 0u; item < items; ++item) {
                    const uint16_t x = a[member * items + item], y = b[member * items + item];
                    const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
                    products[item] = {
                        qrt_sm121_float_alignment::from_bits(uint32_t(x) << 16u) *
                            qrt_sm121_float_alignment::from_bits(uint32_t(y) << 16u),
                        uint32_t(x) | (uint32_t(y) << 16u),
                        zero ? -133 : int((x >> 7u) & 255u) + int((y >> 7u) & 255u) - 254};
                }
                carry = qrt_sm121_float_subgroup::accumulate<lanes>(carry, products);
            } else {
                carry = qrt_sm121_subgroup::accumulate<lanes>(carry, a, b);
            }
            if (!member) carries[cell] = carry;
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
} // namespace qrt_matrix_cooperative_qk
