// FLA block-inverse decomposition; see LICENSE.flash-linear-attention.
#include "blackwell_inverse.h"
#include "blackwell_inverse_math.h"
#include "blackwell_accumulator.h"
namespace qrt_fla_blackwell_inverse {
namespace {
__device__ __forceinline__ void product(const float* a, const float* b, float* output, bool accumulate, bool negate) {
    const unsigned cell = threadIdx.x;
    const float result = dot16(a, b, cell / 16u, cell % 16u, accumulate ? output[cell] : 0.0f);
    output[cell] = negate ? 0.0f - result : result;
    __syncthreads();
}
__global__ void inverse_kernel(const float* input, uint16_t* output, unsigned tokens) {
    __shared__ float a[16][256];
    __shared__ float inverse[16][256];
    __shared__ float temporary[256];
    const unsigned cell = threadIdx.x, row = cell / 16u, col = cell % 16u;
    const unsigned first = blockIdx.x * 64u, head = blockIdx.y;
    for (unsigned br = 0; br < 4; ++br) for (unsigned bc = 0; bc < 4; ++bc) {
        const unsigned token = first + br * 16u + row, block = br * 4u + bc;
        const float value = token < tokens ? input[(token * 32u + head) * 64u + bc * 16u + col] : 0.0f;
        a[block][cell] = value;
        inverse[block][cell] = br == bc && row > col ? 0.0f - value : 0.0f;
    }
    __syncthreads();
    for (unsigned step = 2; step < 16; ++step) {
        // Each active thread owns one complete column of one diagonal block.
        // Previous rows are immutable, and the barrier publishes the new row.
        if (cell < 64u) {
            const unsigned block = (cell / 16u) * 5u, column = cell % 16u;
            inverse[block][step * 16u + column] = row_update(a[block], inverse[block], step, column);
        }
        __syncthreads();
    }
    for (unsigned block = 0; block < 16; block += 5u) inverse[block][cell] += row == col ? 1.0f : 0.0f;
    __syncthreads();
    product(inverse[5], a[4], temporary, false, false);
    product(temporary, inverse[0], inverse[4], false, true);
    product(inverse[10], a[9], temporary, false, false);
    product(temporary, inverse[5], inverse[9], false, true);
    product(inverse[15], a[14], temporary, false, false);
    product(temporary, inverse[10], inverse[14], false, true);
    // The second and third matrix products carry the preceding FP32 dot
    // accumulator. Rounding each product separately before adding is different.
    product(a[8], inverse[0], temporary, false, false);
    product(a[9], inverse[4], temporary, true, false);
    product(inverse[10], temporary, inverse[8], false, true);
    product(a[13], inverse[5], temporary, false, false);
    product(a[14], inverse[9], temporary, true, false);
    product(inverse[15], temporary, inverse[13], false, true);
    product(a[12], inverse[0], temporary, false, false);
    product(a[13], inverse[4], temporary, true, false);
    product(a[14], inverse[8], temporary, true, false);
    product(inverse[15], temporary, inverse[12], false, true);
    for (unsigned br = 0; br < 4; ++br) for (unsigned bc = 0; bc < 4; ++bc) {
        const unsigned token = first + br * 16u + row;
        if (token < tokens) output[(token * 32u + head) * 64u + bc * 16u + col] =
            qrt_fla_blackwell::to_bf16(inverse[br * 4u + bc][cell]);
    }
}
}
hipError_t solve(const float* a, uint16_t* inverse, unsigned tokens, hipStream_t stream) {
    if (!valid_solve(a, inverse, tokens)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(inverse_kernel, dim3((tokens + 63u) / 64u, 32u), dim3(256u), 0, stream, a, inverse, tokens);
    return hipGetLastError();
}
}
