#ifndef QRT_FLA_BLACKWELL_KKT_H
#define QRT_FLA_BLACKWELL_KKT_H
#include "blackwell_accumulator.h"
namespace qrt_fla_blackwell {

__global__ void dot_kernel(const uint16_t* k, const uint16_t* beta, float* a,
                           unsigned int chunk_index) {
    const unsigned int cell = blockIdx.x * (kThreads / kGroup) + threadIdx.x / kGroup;
    const unsigned int row = cell / kChunk, column = cell % kChunk;
    const unsigned int head = blockIdx.y, lane = threadIdx.x % kGroup;
    const unsigned int token = chunk_index * kChunk + row;
    const unsigned int index = (token * 32 + head) * kChunk + column;
    if (row >= kChunk) return;
    if (column >= row) { if (lane == 0) a[index] = 0.0f; return; }
    const unsigned int other = chunk_index * kChunk + column;
    const float beta_value = from_bf16(beta[token * 32 + head]);
    qrt_q1_moe_hawkeye::Value accumulator{0u, kZeroExponent, false};
    for (unsigned int base = 0; base < 128; base += kGroup) {
        const unsigned int dimension = base + lane;
        const uint16_t left = to_bf16(from_bf16(k[(token * 16 + head / 2) * 128 + dimension]) * beta_value);
        const uint16_t right = k[(other * 16 + head / 2) * 128 + dimension];
        accumulator = accumulate(accumulator, left, right, lane);
    }
    if (lane == 0) {
        accumulator = qrt_q1_moe_hawkeye::group_sum<26, kZeroExponent>(&accumulator, 1);
        a[index] = qrt_q1_moe_hawkeye::value_to_float(accumulator);
    }
}

__global__ void gate_kernel(float* a, const float* g, unsigned int tokens) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * 32 * kChunk) return;
    const unsigned int token = index / (32 * kChunk);
    const unsigned int head = index / kChunk % 32, column = index % kChunk;
    if (column >= token % kChunk) return;  // Upper triangle is already zero.
    const float difference = g[token * 32 + head] - g[(token / kChunk * kChunk + column) * 32 + head];
    a[index] *= exp2f(difference * 1.4426950408889634074f);
}
}  // namespace qrt_fla_blackwell
#endif
