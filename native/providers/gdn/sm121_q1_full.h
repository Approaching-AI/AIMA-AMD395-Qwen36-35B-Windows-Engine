#ifndef QRT_SM121_Q1_FULL_H
#define QRT_SM121_Q1_FULL_H
#include <hip/hip_runtime.h>
#include "sm121_q1_math.h"

namespace qrt_sm121_q1_full {
// One CTA owns a Q or K head. The original compiled head-256 norm uses
// two stride-64 warps, followed by the original BF16 MRoPE instructions.
// Cache coefficients cover all positions and are independent of prompt IDs.
__global__ void prepare_qkv(
    const uint16_t *qkv, const uint16_t *q_weight, const uint16_t *k_weight,
    const uint16_t *rope_cache, const unsigned char *rsqrt_table,
    unsigned int position, float *rope, float *norm_observation) {
    using namespace qrt_sm121_q1;
    __shared__ float normalized[256];
    __shared__ float warp_sum[2];
    __shared__ float inverse;
    const unsigned int head = blockIdx.x, dim = threadIdx.x;
    const bool is_key = head >= 16u;
    const unsigned int local_head = is_key ? head - 16u : head;
    const unsigned int source = is_key ? 8192u + local_head * 256u : head * 512u;
    const uint16_t *weight = is_key ? k_weight : q_weight;
    normalized[dim] = widen(qkv[source + dim]);
    __syncthreads();
    if (dim < 64u) {
        float sum = 0.0f;
        for (unsigned int item = 0u; item < 4u; ++item) {
            const float x = normalized[dim + item * 64u];
            sum = add(sum, multiply(x, x));
        }
        for (unsigned int offset = 16u; offset; offset >>= 1u)
            sum = add(sum, __shfl_xor(sum, offset, 32));
        if ((dim & 31u) == 0u) warp_sum[dim / 32u] = sum;
    }
    __syncthreads();
    if (dim == 0u) {
        inverse = qrt_sm121_rsqrt::evaluate(rsqrt_table,
            add(multiply(add(warp_sum[0], warp_sum[1]), 1.0f / 256.0f), 1.0e-6f));
    }
    __syncthreads();
    normalized[dim] = widen(bf16(multiply(multiply(normalized[dim], inverse),
                                        add(1.0f, widen(weight[dim])))));
    __syncthreads();
    const unsigned int norm_index = is_key ? 4096u + local_head * 256u + dim
                                           : head * 256u + dim;
    if (norm_observation) norm_observation[norm_index] = normalized[dim];
    float result = normalized[dim];
    if (dim < 64u) {
        const unsigned int pair = dim & 31u;
        const float c = widen(rope_cache[static_cast<size_t>(position) * 64u + pair]);
        const float s = widen(rope_cache[static_cast<size_t>(position) * 64u + 32u + pair]);
        const float first = normalized[pair], second = normalized[pair + 32u];
        result = dim < 32u
            ? widen(bf16(fmaf(first, c, -widen(bf16(multiply(second, s))))))
            : widen(bf16(fmaf(second, c, widen(bf16(multiply(first, s))))));
    }
    if (is_key) {
        rope[8192u + local_head * 256u + dim] = result;
        rope[8704u + local_head * 256u + dim] = widen(qkv[8704u + local_head * 256u + dim]);
    } else {
        rope[head * 256u + dim] = result;
        rope[4096u + head * 256u + dim] = widen(qkv[source + 256u + dim]);
    }
}
}  // namespace qrt_sm121_q1_full
#endif
