#pragma once
#include <hip/hip_runtime.h>
#include "../ck_fmha/blackwell_attention.h"

namespace qrt_sm121_mtp {
// One or two selected contiguous Q rows read the request's complete retained
// K512/V512 cache. Each score owns one original K16 accumulator across 16 lanes.
__global__ void attention_scores(const uint16_t* query, const uint16_t* cache,
    float* scores, unsigned int first_position, unsigned int query_rows,
    unsigned int score_stride) {
    const unsigned int cell = blockIdx.x * 16u + threadIdx.x / 16u;
    const unsigned int cells = query_rows * 16u * score_stride;
    if (cell >= cells) return;
    const unsigned int lane = threadIdx.x & 15u, token = cell % score_stride;
    const unsigned int head = (cell / score_stride) % 16u;
    const unsigned int row = cell / (score_stride * 16u);
    if (token > first_position + row) { if (!lane) scores[cell] = -INFINITY; return; }
    const size_t query_base = size_t(row) * 4096u + head * 256u;
    const size_t key_base = size_t(token) * 1024u + (head / 8u) * 256u;
    qrt_q1_moe_hawkeye::Value sum{0u, -133, false};
    for (unsigned int base = 0; base < 256u; base += 16u)
        sum = qrt_sm121_wave16::accumulate(sum, query[query_base + base + lane],
            cache[key_base + base + lane], lane);
    if (!lane) scores[cell] = qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(sum)) * 0.0625f;
}

__global__ void publish_attention_context(const float* context, uint16_t* output,
                                         unsigned int elements) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < elements) output[i] = qrt_blackwell_attention::f32_to_bf16(context[i]);
}

inline size_t attention_workspace_bytes(unsigned int rows, unsigned int score_stride) {
    if (!rows || rows > 2u || !score_stride || score_stride > 262144u || score_stride % 32u)
        return 0u;
    return size_t(rows) * (size_t(16u) * score_stride + 4096u) * sizeof(float);
}

// Scratch and output are caller-owned, disjoint buffers. No complete-history
// conversion or allocation occurs here. The caller fences and quarantines all
// borrowed operands if completion fails, as with the prompt-cache append.
inline hipError_t launch_attention(const uint16_t* query, const uint16_t* cache,
    unsigned int cache_tokens, unsigned int first_position, unsigned int query_rows,
    const unsigned char* exp2_table, const unsigned char* reciprocal_table,
    float* scores, unsigned int score_stride, float* context, uint16_t* output,
    hipStream_t stream = nullptr) {
    if (!query || !cache || !exp2_table || !reciprocal_table || !scores || !context || !output ||
        !attention_workspace_bytes(query_rows, score_stride) || !cache_tokens || cache_tokens > 262144u ||
        first_position >= cache_tokens || query_rows > cache_tokens - first_position ||
        score_stride < first_position + query_rows || scores == context || output == query || output == cache)
        return hipErrorInvalidValue;
    const unsigned int cells = query_rows * 16u * score_stride;
    hipLaunchKernelGGL(attention_scores, dim3((cells + 15u) / 16u), dim3(256u), 0u, stream,
        query, cache, scores, first_position, query_rows, score_stride);
    hipError_t status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
        qrt_blackwell_attention::blackwell_exact_attention_kernel<true, true, false, false, false, false, false, true>),
        dim3(16u, query_rows), dim3(256u), 0u, stream,
        static_cast<const uint16_t*>(nullptr), static_cast<const uint16_t*>(nullptr), cache + 512u,
        context, first_position, 0u, exp2_table, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        true, reciprocal_table, scores, score_stride, static_cast<const uint16_t*>(nullptr), 0u);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(publish_attention_context, dim3(query_rows * 16u), dim3(256u), 0u, stream,
        context, output, query_rows * 4096u);
    return hipGetLastError();
}
} // namespace qrt_sm121_mtp
