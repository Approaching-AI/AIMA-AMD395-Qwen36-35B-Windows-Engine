#ifndef QRT_POSITIVE_INTEGER_QK_H
#define QRT_POSITIVE_INTEGER_QK_H
#include "blackwell_attention.h"
#include "../moe_accumulator/sm121_positive_karatsuba.h"

// Full-shape component experiment. No product dispatcher includes this file.
namespace qrt_positive_integer_qk {
namespace attention = qrt_blackwell_attention;
namespace positive = qrt_sm121_positive_karatsuba;
struct Row {
    qrt_sm121_integer_core::Row core;
    uint16_t sum_digits[16];
};
static_assert(sizeof(Row) == 148u);  // Odd dword stride in shared memory.

__host__ __device__ __forceinline__ void prepare(Row& row) {
    qrt_sm121_integer_core::prepare(row.core);
    unsigned total = 0u;
    for (unsigned i = 0u; i < 16u; ++i) {
        const uint16_t encoded = qrt_sm121_integer_core::encode(row.core.original[i], row.core.unit);
        const unsigned sum = positive::sum_digit(encoded);
        row.sum_digits[i] = positive::positive_half_bits(sum);
        total += sum;
    }
    // Core arithmetic reads only original[0..15]. Its two padding halfwords
    // remain private to this extended row; one holds sum(h+l+128)<=8160.
    row.core.original[16] = uint16_t(total);
    row.core.original[17] = 0u;
}

template<attention::IntegerRowKind Kind>
__global__ void prepare_rows(const uint16_t* input, Row* output,
    unsigned tokens, unsigned start, unsigned count) {
    static_assert(Kind == attention::IntegerRowKind::Query || Kind == attention::IntegerRowKind::Key);
    const size_t row = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= attention::integer_row_count(Kind, tokens, count)) return;
    Row result{};
    for (unsigned i = 0u; i < 16u; ++i)
        result.core.original[i] = input[attention::integer_row_input_index(Kind, row, i, tokens, start, count)];
    prepare(result); output[row] = result;
}

__device__ __forceinline__ positive::Parts products(const Row& left, const Row& right) {
    positive::I32x4 ah{}, al{}, bh{}, bl{}; positive::F16x16 as{}, bs{};
#pragma unroll
    for (unsigned i = 0u; i < 4u; ++i) {
        ah[i] = left.core.high[i]; al[i] = left.core.low[i];
        bh[i] = right.core.high[i]; bl[i] = right.core.low[i];
    }
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        as[i] = __builtin_bit_cast(_Float16, left.sum_digits[i]);
        bs[i] = __builtin_bit_cast(_Float16, right.sum_digits[i]);
    }
    const positive::I32x8 zero{}; const positive::F32x8 fzero{};
    return {__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, ah, true, bh, zero, false),
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(false, al, false, bl, zero, false),
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(as, bs, fzero)};
}

template<bool Positive>
__global__ void scores(const Row* query, const Row* key, float* output,
    unsigned query_start, unsigned query_count, unsigned stride, unsigned key_stride) {
    using namespace attention;
    __shared__ Row left[16], right[kIntegerMatrixColumns];
    constexpr unsigned words = sizeof(Row) / 4u;
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.y, kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned query_tile = blockIdx.z * 16u, key_tile = blockIdx.x * kIntegerMatrixColumns;
    const unsigned last_query = query_start + min(query_tile + 16u, query_count) - 1u;
    MantissaF32x8 accumulator{};
    if (key_tile <= last_query) for (unsigned group = 0u; group < 16u; ++group) {
        // Cooperative word loads include the prepared FP16 digits. They are
        // reused by all eight matrix waves and all canonical carry consumers.
        for (unsigned item = threadIdx.x; item < (16u + kIntegerMatrixColumns) * words; item += kThreads) {
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
        IntegerMatrixParts matrix{};
        if constexpr (Positive) {
            const auto result = products(left[lane % 16u], right[wave * 16u + lane % 16u]);
#pragma unroll
            for (unsigned element = 0u; element < 8u; ++element) {
                const unsigned local_row = 2u * element + lane / 16u;
                const int total = int(left[local_row].core.original[16]) + int(right[wave * 16u + lane % 16u].core.original[16]);
                matrix.value[0][element] = result.high[element];
                matrix.value[1][element] = int(result.combined[element]) - result.high[element] - result.low[element] - total * 128 + 262144;
                matrix.value[3][element] = result.low[element];
            }
        } else matrix = blackwell_integer_prepared_products(left[lane % 16u].core, right[wave * 16u + lane % 16u].core);
#pragma unroll
        for (unsigned element = 0u; element < 8u; ++element) {
            const unsigned local_row = 2u * element + lane / 16u, row = query_tile + local_row;
            const unsigned key_row = key_tile + wave * 16u + lane % 16u;
            if (row < query_count && key_row < stride && key_row <= query_start + row) {
                const int32_t partials[4] = {matrix.value[0][element], matrix.value[1][element], matrix.value[2][element], matrix.value[3][element]};
                accumulator[element] = blackwell_integer_accumulate(accumulator[element], left[local_row].core, right[wave * 16u + lane % 16u].core, partials);
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
}  // namespace qrt_positive_integer_qk
#endif
