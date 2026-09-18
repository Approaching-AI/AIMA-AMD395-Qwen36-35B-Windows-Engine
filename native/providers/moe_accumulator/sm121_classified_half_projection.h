#pragma once
#include "sm121_partitioned_half_projection.h"

// Prepare exactly the existing lossless K16 representation and classify the
// complete row in the same pass. A row occupies one or more whole wave32s and
// never crosses a 256-thread CTA. Tail CTAs still execute every barrier.
namespace qrt_sm121_classified_half_projection {
using Row = qrt_sm121_scaled_half_products::Row;
constexpr bool supported_width(unsigned width) {
    return width == 512u || width == 1024u || width == 2048u || width == 4096u;
}
__global__ __launch_bounds__(256) void prepare_rows(const uint16_t* input,
    Row* output, uint32_t* flags, unsigned rows, unsigned width) {
    __shared__ unsigned wave_classes[8];
    const unsigned groups_per_row = width / 16u;
    const size_t group = size_t(blockIdx.x) * 256u + threadIdx.x;
    const bool valid = group < size_t(rows) * groups_per_row;
    unsigned classification = 3u;
    if (valid) {
        uint16_t original[16];
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i) original[i] = input[group * 16u + i];
        const Row prepared = qrt_sm121_scaled_half_products::prepare(original);
        output[group] = prepared;
        classification = int16_t(prepared.control) == -32768 ? 0u :
            (prepared.control >> 16u) == 65535u ? 3u : 1u;
    }
#pragma unroll
    for (unsigned offset = 16u; offset; offset >>= 1u)
        classification &= __shfl_xor(classification, offset, 32u);
    if (!(threadIdx.x & 31u)) wave_classes[threadIdx.x / 32u] = classification;
    __syncthreads();
    if (valid && !(threadIdx.x % groups_per_row)) {
        unsigned complete = 3u;
        for (unsigned wave = 0u; wave < groups_per_row / 32u; ++wave)
            complete &= wave_classes[threadIdx.x / 32u + wave];
        flags[group / groups_per_row] = complete;
    }
}
inline hipError_t prepare(const uint16_t* input, Row* output, uint32_t* flags,
    unsigned rows, unsigned width, hipStream_t stream) {
    if (!input || !output || !flags || !rows || !supported_width(width))
        return hipErrorInvalidValue;
    const size_t groups = size_t(rows) * (width / 16u);
    hipLaunchKernelGGL(prepare_rows, dim3((groups + 255u) / 256u), dim3(256u), 0u,
        stream, input, output, flags, rows, width);
    return hipGetLastError();
}
} // namespace qrt_sm121_classified_half_projection
