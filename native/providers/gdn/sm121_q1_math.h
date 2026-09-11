#ifndef QRT_SM121_Q1_MATH_H
#define QRT_SM121_Q1_MATH_H
#include "sm121_rsqrt_table.h"
#include <cmath>

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_Q1_INLINE __host__ __device__ __forceinline__
#else
#define QRT_Q1_INLINE inline
#endif

namespace qrt_sm121_q1 {
QRT_Q1_INLINE float multiply(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    float result;
    asm("v_mul_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
#else
    volatile float result = a * b;
    return result;
#endif
}
QRT_Q1_INLINE float add(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__)
    float result;
    asm("v_add_f32 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
    return result;
#else
    volatile float result = a + b;
    return result;
#endif
}
QRT_Q1_INLINE uint16_t bf16(float value) {
    const uint32_t bits = qrt_sm121_exp2::bits(value);
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}
QRT_Q1_INLINE float widen(uint16_t value) {
    return qrt_sm121_exp2::value(static_cast<uint32_t>(value) << 16u);
}

// Original no-residual GemmaRMSNorm owns eight adjacent embedding values
// per lane. Its SM121 lowering starts with square 1, then FMA 0 and 2..7.
QRT_Q1_INLINE float embedding_lane_sumsq(const float values[8]) {
    float sum = multiply(values[1], values[1]);
    sum = fmaf(values[0], values[0], sum);
    for (unsigned int i = 2; i < 8; ++i)
        sum = fmaf(values[i], values[i], sum);
    return sum;
}

QRT_Q1_INLINE float embedding_norm_value(float value, float inverse,
                                         uint16_t weight) {
    return widen(bf16(multiply(multiply(value, inverse),
                                add(1.0f, widen(weight)))));
}

// The original Gemma head-256 kernels have different layouts: contiguous Q
// uses two stride-64 warps; the strided K view uses four stride-128 warps.
QRT_Q1_INLINE unsigned int head_norm_warps(bool is_key) {
    return is_key ? 4u : 2u;
}
QRT_Q1_INLINE float head_norm_lane_sumsq(const float *values,
                                        unsigned int lane, bool is_key) {
    const unsigned int stride = head_norm_warps(is_key) * 32u;
    float sum = multiply(values[lane], values[lane]);
    for (unsigned int item = stride; item < 256u; item += stride) {
        const float x = values[lane + item];
        sum = add(sum, multiply(x, x));
    }
    return sum;
}
QRT_Q1_INLINE float head_norm_warp_sum(const float *warp, bool is_key) {
    return is_key ? add(add(warp[0], warp[2]), add(warp[1], warp[3]))
                  : add(warp[0], warp[1]);
}

// Original SM121 PTX 24dc1d6e... uses one K element per lane for
// normalization, XOR 16/8/4/2/1 inside each warp, then XOR 2/1 across warps.
QRT_Q1_INLINE float inverse_norm(const float *values, const unsigned char *rsqrt) {
    float warp[4];
    for (unsigned int w = 0; w < 4; ++w) {
        float partial[16];
        for (unsigned int i = 0; i < 16; ++i) {
            const float low = values[w * 32 + i];
            const float high = values[w * 32 + i + 16];
            partial[i] = fmaf(low, low, multiply(high, high));
        }
        for (unsigned int step = 8; step; step >>= 1)
            for (unsigned int i = 0; i < step; ++i)
                partial[i] = add(partial[i], partial[i + step]);
        warp[w] = partial[0];
    }
    return qrt_sm121_rsqrt::evaluate(rsqrt,
        add(add(add(warp[0], warp[2]), add(warp[1], warp[3])), 1.0e-6f));
}

// The recurrent matrix layout gives four adjacent K values to each lane.
// In the decayed K projection, the last four V rows in each BV=32 block
// use the paired-FP32 lowering (product 0, then FMA 1/2/3); other rows use
// product 1, FMA 0/2/3. The updated-state Q output uses the latter order
// for every V row, as in original runtime PTX 52baed3b... lines 1146-1196.
// This is a fixed layout rule, independent of tensor values and token IDs.
QRT_Q1_INLINE float state_dot(const float *state, unsigned int stride,
                             float decay, const float *right,
                             unsigned int value_dim, bool apply_decay) {
    float partial[32];
    for (unsigned int lane = 0; lane < 32; ++lane) {
        const unsigned int base = lane * 4;
        const unsigned int first = (apply_decay && value_dim % 32 >= 28) ? 0 : 1;
        const unsigned int second = first ^ 1u;
        float x = state[(base + first) * stride];
        if (apply_decay) x = multiply(x, decay);
        float sum = multiply(x, right[base + first]);
        x = state[(base + second) * stride];
        if (apply_decay) x = multiply(x, decay);
        sum = fmaf(x, right[base + second], sum);
        for (unsigned int item = 2; item < 4; ++item) {
            x = state[(base + item) * stride];
            if (apply_decay) x = multiply(x, decay);
            sum = fmaf(x, right[base + item], sum);
        }
        partial[lane] = sum;
    }
    for (unsigned int step = 16; step; step >>= 1)
        for (unsigned int lane = 0; lane < step; ++lane)
            partial[lane] = add(partial[lane], partial[lane + step]);
    return partial[0];
}
} // namespace qrt_sm121_q1
#undef QRT_Q1_INLINE
#endif
