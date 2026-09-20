#pragma once
#include "sm121_q2_attention_block_layout.h"
#include "sm121_mtp_query.h"
#include "sm121_mtp_attention.h"
#include "sm121_mtp_gate.h"
#include "../moe_accumulator/sm121_q1_moe.h"

namespace qrt_sm121_q2 {
namespace attention_block_detail {
__global__ void private_key_values(const uint16_t* projected, const uint16_t* weights,
    const unsigned char* rsqrt, const uint16_t* rope, unsigned first_position,
    uint16_t* staged, uint16_t* norm) {
    __shared__ float values[256], warp_sums[4], inverse;
    __shared__ uint16_t normalized[256];
    const unsigned row = blockIdx.x, head = blockIdx.y, lane = threadIdx.x;
    const size_t source = size_t(row)*1024u + head*256u;
    values[lane] = qrt_sm121_q1::widen(projected[source + lane]);
    __syncthreads();
    if (lane < 128u) {
        float sum = qrt_sm121_q1::head_norm_lane_sumsq(values, lane, true);
        for (unsigned mask = 16u; mask; mask >>= 1u)
            sum = qrt_sm121_q1::add(sum, __shfl_xor(sum, mask, 32));
        if (!(lane & 31u)) warp_sums[lane/32u] = sum;
    }
    __syncthreads();
    if (!lane) inverse = qrt_sm121_mtp::key_inverse(warp_sums, rsqrt);
    __syncthreads();
    normalized[lane] = qrt_sm121_mtp::normalized(values[lane], inverse, weights[lane]);
    __syncthreads();
    norm[size_t(row)*512u + head*256u + lane] = normalized[lane];
    staged[source + lane] = qrt_sm121_mtp::key_rotated(normalized, lane,
        rope + size_t(first_position + row)*64u);
    staged[source + 512u + lane] = projected[source + 512u + lane];
}

__global__ void private_scores(const uint16_t* queries, const uint16_t* history,
    const uint16_t* staged, float* scores, unsigned first_position, unsigned stride) {
    const unsigned cell = blockIdx.x*16u + threadIdx.x/16u;
    if (cell >= 2u*16u*stride) return;
    const unsigned lane = threadIdx.x & 15u, token = cell % stride;
    const unsigned head = (cell/stride) % 16u, row = cell/(stride*16u);
    if (token > first_position + row) { if (!lane) scores[cell] = -INFINITY; return; }
    const bool tail = token >= first_position;
    const auto* key = tail ? staged : history;
    const unsigned source_token = tail ? token - first_position : token;
    const size_t qb = size_t(row)*4096u + head*256u;
    const size_t kb = size_t(source_token)*1024u + (head/8u)*256u;
    qrt_q1_moe_hawkeye::Value sum{0u, -133, false};
    for (unsigned base = 0; base < 256u; base += 16u)
        sum = qrt_sm121_wave16::accumulate(sum, queries[qb + base + lane], key[kb + base + lane], lane);
    if (!lane) scores[cell] = qrt_sm121_q1::multiply(qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(sum)), 0.0625f);
}

inline hipError_t project_input(const AttentionBlockViews& v, hipStream_t stream) {
    const uint16_t* weights[] = {v.q_weights, v.k_weights, v.v_weights};
    const unsigned columns[] = {8192u, 512u, 512u};
    for (unsigned part = 0; part < 3u; ++part) for (unsigned row = 0; row < 2u; ++row) {
        auto* output = part ? v.kv_projected + size_t(row)*1024u + (part - 1u)*512u
                            : v.q_projected + size_t(row)*8192u;
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_sm121_q1_moe::projection<2048u>),
            dim3((columns[part] + 15u)/16u), dim3(256u), 0u, stream,
            v.normalized_input + size_t(row)*2048u, weights[part], output, columns[part]);
        const auto status = hipGetLastError(); if (status != hipSuccess) return status;
    }
    return hipSuccess;
}
} // namespace attention_block_detail

// Complete two-row attention with read-only committed history. All producers
// and candidate KV remain private. The transaction must fence or quarantine
// partial submissions and commit only the actual target's accepted rows.
inline hipError_t launch_attention_block(const AttentionBlockViews& v,
    const AttentionBlockTables& t, hipStream_t stream = nullptr) {
    if (!valid_attention_block(v, t)) return hipErrorInvalidValue;
    auto status = attention_block_detail::project_input(v, stream);
    if (status != hipSuccess) return status;
    hipLaunchKernelGGL(qrt_sm121_mtp::query_rows, dim3(2u,16u), dim3(64u), 0u, stream,
        v.q_projected, v.q_norm_weights, t.rsqrt, t.rope, v.first_position, v.queries, v.gates, v.q_norm);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(attention_block_detail::private_key_values, dim3(2u,2u), dim3(256u), 0u, stream,
        v.kv_projected, v.k_norm_weights, t.rsqrt, t.rope, v.first_position, v.staged_kv, v.k_norm);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(attention_block_detail::private_scores,
        dim3(2u*v.score_stride), dim3(256u), 0u, stream,
        v.queries, v.history, v.staged_kv, v.scores, v.first_position, v.score_stride);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
        qrt_blackwell_attention::blackwell_exact_attention_kernel<true, true, true, false, false, false, false, true>),
        dim3(16u,2u), dim3(256u), 0u, stream,
        static_cast<const uint16_t*>(nullptr), static_cast<const uint16_t*>(nullptr), v.history + 512u,
        v.float_context, v.first_position, 0u, t.exp2, static_cast<float*>(nullptr), static_cast<float*>(nullptr),
        true, t.reciprocal, v.scores, v.score_stride, v.staged_kv + 512u, v.first_position);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    hipLaunchKernelGGL(qrt_sm121_mtp::publish_attention_context, dim3(32u), dim3(256u), 0u, stream,
        v.float_context, v.context, 8192u);
    status = hipGetLastError(); if (status != hipSuccess) return status;
    status = qrt_sm121_mtp::launch_gate(v.context, v.gates, t.sigmoid, v.gated, 2u, stream);
    if (status != hipSuccess) return status;
    for (unsigned row = 0; row < 2u; ++row) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(qrt_sm121_q1_moe::projection<4096u>),
            dim3(128u), dim3(256u), 0u, stream,
            v.gated + size_t(row)*4096u, v.output_weights, v.output + size_t(row)*2048u, 2048u);
        status = hipGetLastError(); if (status != hipSuccess) return status;
    }
    return hipSuccess;
}
} // namespace qrt_sm121_q2
