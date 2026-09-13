#ifndef QRT_BF16_ABSOLUTE_PRODUCT_VIEWS_H
#define QRT_BF16_ABSOLUTE_PRODUCT_VIEWS_H
#include <hip/hip_runtime.h>
#include "bf16_positive_sum_bound.h"

namespace qrt_bf16_absolute_product_views {
// Read only original operands and completed whole-row flags. Excluded rows
// contain zeros in the matrix view and receive infinity after multiplication.
__global__ void prepare_rows_kernel(const uint16_t* source, const unsigned* eligible,
    uint16_t* magnitudes, unsigned rows, unsigned width) {
    const unsigned row = blockIdx.x;
    if (row >= rows) return;
    const bool valid = eligible[row] != 0u;
    for (unsigned k = threadIdx.x; k < width; k += blockDim.x) {
        const size_t index = size_t(row) * width + k;
        magnitudes[index] = valid ? uint16_t(source[index] & 0x7fffu) : 0u;
    }
}

// The backend writes whole token columns into its owning allocation. Partial
// flat windows take a pointer inside this allocation after outward finishing.
__global__ void finish_matrix_kernel(float* matrix, const unsigned* weight_eligible,
    const unsigned* input_eligible, unsigned rows, unsigned first_token,
    unsigned reduction_size, unsigned elements) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned row = index % rows, token = first_token + index / rows;
    matrix[index] = weight_eligible[row] && input_eligible[token]
        ? qrt_bf16_positive_sum_bound::finish(matrix[index], reduction_size)
        : qrt_bf16_positive_sum_bound::value(0x7f800000u);
}
}
#endif
