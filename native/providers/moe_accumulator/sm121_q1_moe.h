#ifndef QRT_SM121_Q1_MOE_H
#define QRT_SM121_Q1_MOE_H
#include "sm121_wave16.h"
#include "sm121_shared_gate.h"
#include "sm121_router_exp.h"

namespace qrt_sm121_q1_moe {
__device__ inline float widen(uint16_t x) { return __uint_as_float(uint32_t(x) << 16u); }
__device__ inline uint16_t round(float x) {
    const uint32_t u = __float_as_uint(x);
    return uint16_t((u + 0x7fffu + ((u >> 16u) & 1u)) >> 16u);
}
__device__ inline float dot(const uint16_t *x, const uint16_t *w, unsigned count) {
    const unsigned lane = threadIdx.x & 15u;
    qrt_q1_moe_hawkeye::Value accumulator{0u, -133, false};
    for (unsigned base = 0; base < count; base += 16u)
        accumulator = qrt_sm121_wave16::accumulate(accumulator, x[base + lane], w[base + lane], lane);
    return lane == 0u ? qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(accumulator)) : 0.0f;
}
template<unsigned K>
__global__ void projection(const uint16_t *input, const uint16_t *weights,
                           uint16_t *output, unsigned rows) {
    const unsigned row = blockIdx.x * 16u + threadIdx.x / 16u;
    if (row >= rows) return;
    const float sum = dot(input, weights + size_t(row) * K, K);
    if ((threadIdx.x & 15u) == 0u) output[row] = round(sum);
}
__global__ void router(const uint16_t *logits, uint32_t *ids, float *weights,
                       const uint32_t *fraction) {
    const unsigned lane = threadIdx.x;
    if (lane >= 32u) return;
    float p[8], maximum = -INFINITY;
    for (unsigned j = 0; j < 8u; ++j) { p[j] = widen(logits[lane * 8u + j]); maximum = fmaxf(maximum, p[j]); }
    for (unsigned mask = 16u; mask; mask >>= 1u) maximum = fmaxf(maximum, __shfl_xor(maximum, mask, 32));
    float sum = 0.0f;
    for (unsigned j = 0; j < 8u; ++j) { p[j] = qrt_sm121_router::exp(p[j] - maximum, fraction); sum = __fadd_rn(sum, p[j]); }
    for (unsigned mask = 16u; mask; mask >>= 1u) sum = __fadd_rn(sum, __shfl_xor(sum, mask, 32));
    const float inv = 1.0f / sum;
    for (float &v : p) v = __fmul_rn(v, inv);
    float denominator = 0.0f;
    for (unsigned route = 0; route < 8u; ++route) {
        float best = p[0]; unsigned id = lane * 8u;
        for (unsigned j = 1; j < 8u; ++j) if (p[j] > best) { best = p[j]; id = lane * 8u + j; }
        for (unsigned mask = 16u; mask; mask >>= 1u) {
            const float other = __shfl_xor(best, mask, 32);
            const unsigned oi = __shfl_xor(id, mask, 32);
            if (other > best || (other == best && oi < id)) { best = other; id = oi; }
        }
        if (lane == 0u) { ids[route] = id; weights[route] = best; denominator = __fadd_rn(denominator, best); }
        if (lane == id / 8u) p[id % 8u] = -10000.0f;
    }
    if (lane == 0u) for (unsigned route = 0; route < 8u; ++route) weights[route] /= denominator;
}
__global__ void routed_activation(const uint16_t *input, const uint16_t *weights,
                                  const uint32_t *ids, const uint16_t *silu,
                                  uint16_t *gate_up, uint16_t *activated) {
    const unsigned index = blockIdx.x * 16u + threadIdx.x / 16u;
    if (index >= 8u * 512u) return;
    const unsigned route = index / 512u, row = index % 512u;
    const uint16_t *gate = weights + (size_t(ids[route]) * 1024u + row) * 2048u;
    const float g = dot(input, gate, 2048u), u = dot(input, gate + 512u * 2048u, 2048u);
    if ((threadIdx.x & 15u) == 0u) {
        const uint16_t gb = round(g), ub = round(u);
        gate_up[route * 1024u + row] = gb; gate_up[route * 1024u + 512u + row] = ub;
        activated[index] = round(__fmul_rn(widen(silu[gb]), widen(ub)));
    }
}
__global__ void routed_down(const uint16_t *activated, const uint16_t *weights,
                            const uint32_t *ids, const float *route_weights,
                            uint16_t *weighted) {
    const unsigned index = blockIdx.x * 16u + threadIdx.x / 16u;
    if (index >= 8u * 2048u) return;
    const unsigned route = index / 2048u, row = index % 2048u;
    const float sum = dot(activated + route * 512u,
        weights + (size_t(ids[route]) * 2048u + row) * 512u, 512u);
    // The original weighted W2 rounds after weighting its FP32 accumulator.
    if ((threadIdx.x & 15u) == 0u) weighted[index] = round(__fmul_rn(sum, route_weights[route]));
}
__global__ void route_sum(const uint16_t *weighted, float *output) {
    const unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= 2048u) return;
    float sum = __fadd_rn(widen(weighted[col]), widen(weighted[4u * 2048u + col]));
    for (unsigned j = 1u; j < 4u; ++j) sum = __fadd_rn(sum,
        __fadd_rn(widen(weighted[j * 2048u + col]), widen(weighted[(j + 4u) * 2048u + col])));
    output[col] = widen(round(sum));
}
__global__ void shared_activation(const uint16_t *input, const uint16_t *gate,
                                  const uint16_t *up, const uint16_t *silu,
                                  uint16_t *activated) {
    const unsigned row = blockIdx.x * 16u + threadIdx.x / 16u;
    if (row >= 512u) return;
    const float g = dot(input, gate + size_t(row) * 2048u, 2048u);
    const float u = dot(input, up + size_t(row) * 2048u, 2048u);
    if ((threadIdx.x & 15u) == 0u) activated[row] = round(__fmul_rn(widen(silu[round(g)]), widen(round(u))));
}
__global__ void shared_gate(const uint16_t *input, const uint16_t *weights, uint16_t *output) {
    const unsigned lane = threadIdx.x;
    if (lane >= 16u) return;
    float sum = qrt_sm121_shared_gate::lane_dot(input, weights, lane);
    for (unsigned offset = 8u; offset; offset >>= 1u) sum = __fadd_rn(sum, __shfl_down(sum, offset, 16));
    if (lane == 0u) output[0] = round(sum);
}
}  // namespace qrt_sm121_q1_moe
#endif
