#ifndef QRT_WIDE_INTEGER_QK_H
#define QRT_WIDE_INTEGER_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_wide_integer_core.h"
#include "../moe_accumulator/sm121_compact_wide_integer_core.h"
#include <type_traits>

// Full-shape component experiment. No product dispatcher includes this file.
namespace qrt_wide_integer_qk {
namespace attention = qrt_blackwell_attention;
namespace wide = qrt_sm121_wide_core;
using Row = wide::Row;
using CompactRow = qrt_sm121_compact_wide_core::Row;

template<attention::IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input, Row* output,
    unsigned tokens, unsigned start, unsigned count) {
    static_assert(Kind == attention::IntegerRowKind::Query || Kind == attention::IntegerRowKind::Key);
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= attention::integer_row_count(Kind, tokens, count)) return;
    Row result{};
    for (unsigned i = 0u; i < 16u; ++i)
        result.original[i] = input[attention::integer_row_input_index(Kind, row, i, tokens, start, count)];
    wide::prepare(result); output[row] = result;
}

template<attention::IntegerRowKind Kind>
__global__ void prepare_compact_rows(const uint16_t* input, CompactRow* output,
    unsigned tokens, unsigned start, unsigned count) {
    static_assert(Kind == attention::IntegerRowKind::Query || Kind == attention::IntegerRowKind::Key);
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= attention::integer_row_count(Kind, tokens, count)) return;
    Row result{};
    for (unsigned i = 0u; i < 16u; ++i)
        result.original[i] = input[attention::integer_row_input_index(Kind, row, i, tokens, start, count)];
    wide::prepare(result);
    output[row] = qrt_sm121_compact_wide_core::pack(result);
}

__device__ __forceinline__ float accumulate(float accumulator,
    const Row& left, const Row& right, int64_t mathematical) {
    const auto carry = qrt_q1_moe_hawkeye::value_from_float(accumulator, attention::kBlackwellZeroExponent);
    qrt_sm121_group16::AlignedSum sum;
    if (!wide::sum_integer_product(carry, left, right, mathematical, &sum)) {
        uint32_t products[16];
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i)
            products[i] = qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(
                left.original[i], right.original[i], attention::kBlackwellZeroExponent));
        sum = qrt_sm121_group16::sum_packed(carry, products);
    }
    return qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(
        qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent)));
}

template<class InputRow>
__device__ __forceinline__ void scores_body(const InputRow* query, const InputRow* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride,
    Row* left, Row* right) {
    using namespace attention;
    constexpr bool compact = std::is_same<InputRow, CompactRow>::value;
    static_assert(compact || std::is_same<InputRow, Row>::value);
    constexpr unsigned words = sizeof(InputRow) / 4u;
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * kIntegerMatrixColumns;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    MantissaF32x8 accumulator{};
    if (key_tile <= last_query) for (unsigned group = 0u; group < 16u; ++group) {
        for (unsigned item = threadIdx.x; item < (16u + kIntegerMatrixColumns) * words; item += kThreads) {
            const unsigned row = item / words, word = item % words;
            const bool is_query = row < 16u;
            const unsigned input_row = is_query ? query_tile + row : key_tile + row - 16u;
            uint32_t value = 0u;
            if (input_row < (is_query ? query_count : stride)) {
                const InputRow* source = is_query ? query + (size_t(input_row) * kQueryHeads + head) * 16u + group :
                    key + (size_t(kv_head) * 16u + group) * key_stride + input_row;
                __builtin_memcpy(&value, reinterpret_cast<const unsigned char*>(source) + word * 4u, 4u);
            }
            Row* target = is_query ? left + row : right + row - 16u;
            const unsigned target_word = compact ? qrt_sm121_compact_wide_core::expanded_word(word) : word;
            __builtin_memcpy(reinterpret_cast<unsigned char*>(target) + target_word * 4u, &value, 4u);
        }
        __syncthreads();
        if constexpr (compact) {
            // Reconstruct each digit once in LDS, then reuse the same expanded
            // rows for all eight matrix waves and canonical carry consumers.
            for (unsigned item = threadIdx.x; item < (16u + kIntegerMatrixColumns) * 8u; item += kThreads) {
                const unsigned row = item / 8u, pair = item % 8u;
                Row& target = row < 16u ? left[row] : right[row - 16u];
                qrt_sm121_compact_wide_core::expand_pair(target, pair);
            }
            __syncthreads();
        }
        const Row& column = right[wave * 16u + lane % 16u];
        const auto matrix = wide::products(left[lane % 16u], column);
#pragma unroll
        for (unsigned element = 0u; element < 8u; ++element) {
            const unsigned local_row = 2u * element + lane / 16u, row = query_tile + local_row;
            const unsigned key_row = key_tile + wave * 16u + lane % 16u;
            if (row < query_count && key_row < stride && key_row <= query_start + row) {
                const int64_t mathematical = wide::reconstruct(int(matrix.high[element]), int(matrix.low[element]),
                    int(matrix.combined[element]), wide::unsigned_row_sum(left[local_row]), wide::unsigned_row_sum(column));
                accumulator[element] = accumulate(accumulator[element], left[local_row], column, mathematical);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) {
        const unsigned row = query_tile + 2u * element + lane / 16u, key_row = key_tile + wave * 16u + lane % 16u;
        if (row < query_count && key_row < stride)
            output[(size_t(row) * kQueryHeads + head) * stride + key_row] = key_row <= query_start + row ? accumulator[element] * kExactScale : -INFINITY;
    }
}

__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    __shared__ Row left[16], right[attention::kIntegerMatrixColumns];
    scores_body(query, key, output, query_start, query_count, stride, key_stride, left, right);
}

__global__ void compact_scores(const CompactRow* query, const CompactRow* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    __shared__ Row left[16], right[attention::kIntegerMatrixColumns];
    scores_body(query, key, output, query_start, query_count, stride, key_stride, left, right);
}
}  // namespace qrt_wide_integer_qk
#endif
