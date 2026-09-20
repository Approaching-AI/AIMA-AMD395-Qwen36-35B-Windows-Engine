#pragma once
#include <hip/hip_runtime.h>
#include "sm121_mtp_head_math.h"
#include "sm121_q1_math.h"
#include "sm121_mtp_moe_layout.h"

namespace qrt_sm121_mtp {
// The original one-row output head uses GEMV: 16 strided FMA partials,
// followed by offsets 8,4,2,1. It is not the K16 MMA projection schedule.
__global__ void head_projection(const uint16_t* weights, const uint16_t* input,
    uint16_t* logits, unsigned first_cell, unsigned end_cell) {
    const unsigned cell = first_cell + (blockIdx.x * blockDim.x + threadIdx.x) / 16u;
    if (cell >= end_cell) return;
    const unsigned lane = threadIdx.x & 15u;
    const unsigned row = cell / head_vocabulary, token = cell % head_vocabulary;
    float sum = qrt_sm121_shared_gate::lane_dot(input + size_t(row) * 2048u,
        weights + size_t(token) * 2048u, lane);
    for (unsigned offset = 8u; offset; offset >>= 1u)
        sum = qrt_sm121_q1::add(sum, __shfl_down(sum, offset, 16));
    if (!lane) logits[cell] = qrt_sm121_q1::bf16(sum);
}

__global__ void head_argmax(const uint16_t* logits, uint32_t* tokens,
                            float* values, uint32_t* invalid) {
    __shared__ uint32_t candidate_tokens[256];
    __shared__ float candidate_logits[256];
    const unsigned lane = threadIdx.x;
    HeadBest best;
    const uint16_t* row = logits + size_t(blockIdx.x) * head_vocabulary;
    for (unsigned token = lane; token < head_vocabulary; token += 256u)
        if (!head_candidate(row[token], token, &best)) atomicOr(invalid, 1u);
    candidate_tokens[lane] = best.token;
    candidate_logits[lane] = best.logit;
    __syncthreads();
    for (unsigned stride = 128u; stride; stride >>= 1u) {
        if (lane < stride) {
            const HeadBest current{candidate_tokens[lane], candidate_logits[lane]};
            const HeadBest other{candidate_tokens[lane + stride], candidate_logits[lane + stride]};
            if (head_better(other.logit, other.token, current)) {
                candidate_tokens[lane] = other.token;
                candidate_logits[lane] = other.logit;
            }
        }
        __syncthreads();
    }
    if (!lane) {
        tokens[blockIdx.x] = candidate_tokens[0];
        values[blockIdx.x] = candidate_logits[0];
    }
}

// Each requested row is an independent one-row GEMV, even in a two-row
// verification batch. No captured candidates, logits or acceptance decisions
// enter the calculation. The caller fences and checks invalid before publish.
inline hipError_t launch_head(const uint16_t* weights, const uint16_t* input,
    uint16_t* logits, uint32_t* tokens, float* values, uint32_t* invalid,
    unsigned rows, unsigned maximum_blocks = 1024u, hipStream_t stream = nullptr) {
    if (!rows || rows > 2u || !maximum_blocks || maximum_blocks > 4096u)
        return hipErrorInvalidValue;
    struct Range { const void* pointer; size_t bytes; };
    const Range read[] = {{weights, size_t(head_vocabulary) * 2048u * 2u},
                          {input, size_t(rows) * 2048u * 2u}};
    const Range write[] = {{logits, size_t(rows) * head_vocabulary * 2u},
        {tokens, size_t(rows) * sizeof(uint32_t)}, {values, size_t(rows) * sizeof(float)},
        {invalid, sizeof(uint32_t)}};
    for (unsigned i = 0; i < 4u; ++i) {
        for (const auto& source : read)
            if (!moe_disjoint(write[i].pointer, write[i].bytes, source.pointer, source.bytes))
                return hipErrorInvalidValue;
        for (unsigned j = 0; j < i; ++j)
            if (!moe_disjoint(write[i].pointer, write[i].bytes, write[j].pointer, write[j].bytes))
                return hipErrorInvalidValue;
    }
    hipError_t status = hipMemsetAsync(invalid, 0, sizeof(uint32_t), stream);
    if (status != hipSuccess) return status;
    const unsigned cells = rows * head_vocabulary, capacity = maximum_blocks * 16u;
    for (unsigned first = 0; first < cells; first += capacity) {
        const unsigned remaining = cells - first, count = remaining < capacity ? remaining : capacity;
        hipLaunchKernelGGL(head_projection, dim3((count + 15u) / 16u), dim3(256u), 0u, stream,
            weights, input, logits, first, first + count);
        status = hipGetLastError(); if (status != hipSuccess) return status;
    }
    hipLaunchKernelGGL(head_argmax, dim3(rows), dim3(256u), 0u, stream, logits, tokens, values, invalid);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
