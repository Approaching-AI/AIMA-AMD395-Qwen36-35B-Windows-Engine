#ifndef QRT_MODULAR_INTEGER_QK_H
#define QRT_MODULAR_INTEGER_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_modular_integer_core.h"
namespace qrt_modular_integer_qk {
using namespace qrt_blackwell_attention;
using Row = qrt_sm121_modular_core::Row;
using qrt_sm121_modular_core::prepare;
template<IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input, Row* output, unsigned tokens, unsigned start, unsigned count) {
    static_assert(Kind == IntegerRowKind::Query || Kind == IntegerRowKind::Key);
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= integer_row_count(Kind, tokens, count)) return;
    Row result{};
    for (unsigned i = 0u; i < 16u; ++i) result.original[i] = input[integer_row_input_index(Kind, row, i, tokens, start, count)];
    prepare(result); output[row] = result;
}
template<unsigned Columns>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    static_assert(Columns == 16u || Columns == 32u || Columns == 64u);
    constexpr unsigned cells = 16u * Columns, per_thread = cells / kThreads, words = sizeof(Row) / 4u;
    __shared__ Row left[16], right[Columns];
    __shared__ float partials[cells];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * Columns;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    qrt_q1_moe_hawkeye::Value accumulators[per_thread];
    for (unsigned i = 0u; i < per_thread; ++i) accumulators[i] = {0u, kBlackwellZeroExponent, false};
    if (key_tile <= last_query) for (unsigned group = 0u; group < 16u; ++group) {
        for (unsigned item = threadIdx.x; item < (16u + Columns) * words; item += kThreads) {
            const unsigned row = item / words, word = item % words;
            const bool is_query = row < 16u;
            const unsigned input_row = is_query ? query_tile + row : key_tile + row - 16u;
            uint32_t value = 0u;
            if (input_row < (is_query ? query_count : stride)) {
                const Row* source = is_query ? query + (size_t(input_row) * kQueryHeads + head) * 16u + group :
                    key + (size_t(kv_head) * 16u + group) * key_stride + input_row;
                __builtin_memcpy(&value, reinterpret_cast<const unsigned char*>(source) + word * 4u, 4u);
            }
            Row* target = is_query ? left + row : right + row - 16u;
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target) + word * 4u, &value, 4u);
        }
        __syncthreads();
        if (wave < Columns / 16u) {
            const auto result = qrt_sm121_modular_core::products(left[lane % 16u], right[wave * 16u + lane % 16u]);
#pragma unroll
            for (unsigned element = 0u; element < 8u; ++element)
                partials[(2u * element + lane / 16u) * Columns + wave * 16u + lane % 16u] = result[element];
        }
        __syncthreads();
#pragma unroll
        for (unsigned item = 0u; item < per_thread; ++item) {
            const unsigned cell = threadIdx.x + item * kThreads, row = cell / Columns, column = cell % Columns;
            const unsigned output_row = query_tile + row, key_row = key_tile + column;
            if (output_row < query_count && key_row < stride && key_row <= query_start + output_row) {
                qrt_sm121_group16::AlignedSum sum;
                if (!qrt_sm121_modular_core::sum(accumulators[item], left[row], right[column], partials[cell], &sum))
                    sum = qrt_sm121_modular_core::fallback(accumulators[item], left[row], right[column]);
                accumulators[item] = qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned item = 0u; item < per_thread; ++item) {
        const unsigned cell = threadIdx.x + item * kThreads, row = query_tile + cell / Columns, key_row = key_tile + cell % Columns;
        if (row < query_count && key_row < stride) output[(size_t(row) * kQueryHeads + head) * stride + key_row] =
            key_row <= query_start + row ? qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(accumulators[item])) * kExactScale : -INFINITY;
    }
}
}
#endif
