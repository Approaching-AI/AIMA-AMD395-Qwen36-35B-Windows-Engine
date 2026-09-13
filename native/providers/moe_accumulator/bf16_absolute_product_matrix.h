#ifndef QRT_BF16_ABSOLUTE_PRODUCT_MATRIX_H
#define QRT_BF16_ABSOLUTE_PRODUCT_MATRIX_H
#include <hip/hip_runtime.h>
#include "bf16_positive_sum_bound.h"

namespace qrt_bf16_absolute_product_matrix {
using Bf16x16 = uint16_t __attribute__((ext_vector_type(16)));
using F32x8 = float __attribute__((ext_vector_type(8)));
constexpr unsigned row_tile = 128u, token_tile = 64u, threads = 256u;

// Bound one flat output window without rebasing a pointer before its owning
// allocation. Original token-major projection cells and operands stay intact.
// All rows excluded by the lossless operand preparation are zero-filled for
// the matrix instruction, then assigned infinite bounds at their endpoints.
__global__ __launch_bounds__(threads) void window_kernel(
    const uint16_t* weights, const uint16_t* inputs,
    const unsigned* weight_eligible, const unsigned* input_eligible,
    float* bounds, unsigned rows, unsigned tokens, unsigned k,
    size_t first_element, unsigned window_elements) {
    const unsigned wave = threadIdx.x / 32u, lane = threadIdx.x % 32u;
    const unsigned source = lane % 16u, row_segment = lane / 16u;
    const unsigned row_base = blockIdx.x * row_tile + wave * 16u;
    const unsigned first_token = unsigned(first_element / rows);
    const unsigned token_base = first_token + blockIdx.y * token_tile;
    const unsigned row = row_base + source;
    const bool weight_valid = row < rows && weight_eligible[row];
    F32x8 accumulator[4]{};
#pragma unroll 1
    for (unsigned base = 0u; base < k; base += 16u) {
        Bf16x16 weight{};
#pragma unroll
        for (unsigned item = 0u; item < 16u; ++item)
            weight[item] = weight_valid ? weights[size_t(row) * k + base + item] & 0x7fffu : 0u;
#pragma unroll
        for (unsigned fragment = 0u; fragment < 4u; ++fragment) {
            const unsigned token = token_base + fragment * 16u + source;
            const bool input_valid = token < tokens && input_eligible[token];
            Bf16x16 input{};
#pragma unroll
            for (unsigned item = 0u; item < 16u; ++item)
                input[item] = input_valid ? inputs[size_t(token) * k + base + item] & 0x7fffu : 0u;
            accumulator[fragment] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(input, weight, accumulator[fragment]);
        }
    }
    if (row >= rows) return;
#pragma unroll
    for (unsigned fragment = 0u; fragment < 4u; ++fragment) {
#pragma unroll
        for (unsigned item = 0u; item < 8u; ++item) {
            const unsigned token = token_base + fragment * 16u + 2u * item + row_segment;
            const size_t index = size_t(token) * rows + row;
            if (token < tokens && index >= first_element && index - first_element < window_elements)
                bounds[index - first_element] = weight_valid && input_eligible[token]
                    ? qrt_bf16_positive_sum_bound::finish(accumulator[fragment][item], k)
                    : qrt_bf16_positive_sum_bound::value(0x7f800000u);
        }
    }
}
}
#endif
