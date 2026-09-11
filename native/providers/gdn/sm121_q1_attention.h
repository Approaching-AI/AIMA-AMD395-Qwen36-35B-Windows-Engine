#ifndef QRT_SM121_Q1_ATTENTION_H
#define QRT_SM121_Q1_ATTENTION_H
#include "../ck_fmha/blackwell_attention.h"

namespace qrt_sm121_q1_attention {
// Publish only the current K/V row into the transaction's reserved tail slot.
__global__ void append(const float *rope, uint16_t *key, uint16_t *value, unsigned tail_tokens) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 512u) return;
    key[static_cast<size_t>(tail_tokens) * 512u + i] = qrt_blackwell_attention::f32_to_bf16(rope[8192u + i]);
    value[static_cast<size_t>(tail_tokens) * 512u + i] = qrt_blackwell_attention::f32_to_bf16(rope[8704u + i]);
}

// The query is one local row. Prefix and tail remain separate owned caches;
// logical position selects their address without copying the full history.
__global__ void scores(const float *rope, const uint16_t *prefix_key,
                       const uint16_t *tail_key, float *output,
                       unsigned prefix_tokens, unsigned tokens, unsigned stride) {
    const unsigned cell = blockIdx.x * 16u + threadIdx.x / 16u;
    if (cell >= 16u * stride) return;
    const unsigned head = cell / stride, token = cell % stride;
    const unsigned lane = threadIdx.x & 15u;
    if (token >= tokens) { if (!lane) output[cell] = -INFINITY; return; }
    const bool tail = token >= prefix_tokens;
    const uint16_t *key = tail ? tail_key : prefix_key;
    const unsigned local_token = tail ? token - prefix_tokens : token;
    const size_t key_base = (static_cast<size_t>(local_token) * 2u + head / 8u) * 256u;
    const unsigned query_base = head * 256u;
    qrt_q1_moe_hawkeye::Value sum{0u, -133, false};
    for (unsigned base = 0u; base < 256u; base += 16u) {
        sum = qrt_sm121_wave16::accumulate(sum,
            qrt_blackwell_attention::f32_to_bf16(rope[query_base + base + lane]),
            key[key_base + base + lane], lane);
    }
    if (!lane) output[cell] = qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(sum)) * 0.0625f;
}
}  // namespace qrt_sm121_q1_attention
#endif
