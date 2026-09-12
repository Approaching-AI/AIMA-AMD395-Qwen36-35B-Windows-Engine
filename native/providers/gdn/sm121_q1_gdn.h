#ifndef QRT_SM121_Q1_GDN_H
#define QRT_SM121_Q1_GDN_H
#include <hip/hip_runtime.h>
#include "sm121_q1_math.h"
#include "sm121_silu_table.h"

namespace qrt_sm121_q1 {
__device__ inline float ring_value(float value) { return value; }
__device__ inline float ring_value(uint16_t value) { return widen(value); }
__device__ inline void ring_store(float *where, float value) { *where = value; }
__device__ inline void ring_store(uint16_t *where, float value) { *where = bf16(value); }

template <typename Element>
__global__ void convolution(const float *current, Element *ring,
                            const uint16_t *weights, float *output,
                            size_t position, const unsigned char *silu) {
    const unsigned int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= 8192) return;
    float sum = 0.0f;
    for (unsigned int tap = 0; tap < 4; ++tap) {
        if (position + tap < 3) continue;
        const size_t source = position + tap - 3;
        const float value = source == position ? current[feature]
            : ring_value(ring[(source % 4) * 8192 + feature]);
        sum = add(sum, widen(bf16(multiply(widen(bf16(value)), widen(weights[feature * 4 + tap])))));
    }
    output[feature] = widen(qrt_sm121_silu::evaluate(silu, sum));
    ring_store(ring + (position % 4) * 8192 + feature, current[feature]);
}

// Prepare raw prefill inputs without invoking the decode recurrence. The FLA
// provider owns Q/K normalization, BF16 chunk boundaries and FP32 state update.
__global__ void prepare_fla_suffix(const float* conv, const float* a, const float* b,
                                  float* raw_copy, float* gates,
                                  const float* g_table, const float* beta_table) {
    const unsigned feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature < 8192u) raw_copy[feature] = conv[feature];
    if (feature < 32u) {
        gates[feature] = g_table[feature * 65536u + bf16(a[feature])];
        gates[32u + feature] = beta_table[bf16(b[feature])];
    }
}

// Each thread owns one V row of the recurrent state, reads the entire old
// row before updating it, and emits the BF16 core endpoint in an F32 carrier.
__global__ void recurrent(const float *conv, const float *a, const float *b,
                          float *state, bool key_major, float *core,
                          float *postconv, float *gates, float *diagnostic,
                          const float *g_table, const float *beta_table,
                          const unsigned char *exp2, const unsigned char *rsqrt) {
    const unsigned int head = blockIdx.x, v = threadIdx.x, key_head = head / 2;
    __shared__ float q[128], k[128], q_inverse, k_inverse, decay, beta, g;
    q[v] = conv[key_head * 128 + v];
    k[v] = conv[2048 + key_head * 128 + v];
    __syncthreads();
    if (v == 0) {
        q_inverse = inverse_norm(q, rsqrt);
        k_inverse = inverse_norm(k, rsqrt);
        g = g_table[head * 65536u + bf16(a[head])];
        beta = beta_table[bf16(b[head])];
        decay = qrt_sm121_exp2::evaluate(exp2, multiply(g, 0x1.715476p+0f));
    }
    __syncthreads();
    q[v] = multiply(multiply(q[v], q_inverse), 0x1.6a09e6p-4f);
    k[v] = multiply(k[v], k_inverse);
    __syncthreads();
    const float value = conv[4096 + head * 128 + v];
    const unsigned int stride = key_major ? 128 : 1;
    float *row = state + head * 16384 + (key_major ? v : v * 128);
    const float projected = state_dot(row, stride, decay, k, v, true);
    const float residual = multiply(add(value, -projected), beta);
    for (unsigned int i = 0; i < 128; ++i)
        row[i * stride] = fmaf(residual, k[i], multiply(row[i * stride], decay));
    core[head * 128 + v] = widen(bf16(state_dot(row, stride, 1.0f, q, v, false)));
    if (postconv) {
        if (head % 2 == 0) {
            postconv[key_head * 128 + v] = q[v];
            postconv[2048 + key_head * 128 + v] = k[v];
        }
        postconv[4096 + head * 128 + v] = value;
    }
    if (gates && v == 0) { gates[head] = g; gates[32 + head] = beta; }
    if (diagnostic) {
        float *out = diagnostic + head * 515;
        out[v] = q[v]; out[128 + v] = k[v];
        out[256 + v] = projected; out[384 + v] = residual;
        if (v == 0) { out[512] = g; out[513] = decay; out[514] = beta; }
    }
}
} // namespace qrt_sm121_q1
#endif
