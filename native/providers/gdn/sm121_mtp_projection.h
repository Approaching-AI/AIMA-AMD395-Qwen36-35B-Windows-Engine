#pragma once
#include <hip/hip_runtime.h>
#include "sm121_q1_math.h"
#include "../moe_accumulator/sm121_subgroup.h"

namespace qrt_sm121_mtp {
// Exact K16 baseline for the original FC and KV frontiers. This independent
// producer allocates no workspace and does not borrow a target-layer GEMM
// heuristic merely because its dimensions match. Complete MTP integration
// must qualify the original projection schedule for every admitted shape.
__global__ void projection_kernel(const uint16_t* weights, const uint16_t* input,
    uint16_t* output, unsigned int output_features, unsigned int input_features,
    unsigned int first_cell, unsigned int end_cell) {
    constexpr unsigned int lanes = 4u;
    const unsigned int cell = first_cell + (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (cell >= end_cell) return;
    const unsigned int token = cell / output_features;
    const unsigned int feature = cell % output_features;
    const float value = qrt_sm121_subgroup::dot<lanes, 1u, false>(
        input + size_t(token) * input_features,
        weights + size_t(feature) * input_features, input_features);
    if (!(threadIdx.x & (lanes - 1u))) output[cell] = qrt_sm121_q1::bf16(value);
}

inline hipError_t launch_projection(const uint16_t* weights, const uint16_t* input,
    uint16_t* output, unsigned int output_features, unsigned int input_features,
    unsigned int tokens, unsigned int maximum_blocks = 1024u, hipStream_t stream = nullptr) {
    const bool shape = (output_features == 2048u && input_features == 4096u) ||
                       (output_features == 1024u && input_features == 2048u);
    if (!weights || !input || !output || output == input || output == weights || !shape ||
        !tokens || tokens > 8192u || !maximum_blocks || maximum_blocks > 4096u)
        return hipErrorInvalidValue;
    const unsigned int cells = tokens * output_features;
    const unsigned int capacity = maximum_blocks * 64u;
    for (unsigned int first = 0; first < cells; first += capacity) {
        const unsigned int remaining = cells - first;
        const unsigned int count = remaining < capacity ? remaining : capacity;
        hipLaunchKernelGGL(projection_kernel, dim3((count + 63u) / 64u), dim3(256u), 0u,
            stream, weights, input, output, output_features, input_features, first, first + count);
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) return status;
    }
    return hipSuccess;
}
} // namespace qrt_sm121_mtp
