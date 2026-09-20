#pragma once
#include "sm121_q1_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_Q2_INLINE __host__ __device__ __forceinline__
#else
#define QRT_Q2_INLINE inline
#endif

namespace qrt_sm121_q2 {
constexpr size_t state_elements = 32u * 128u * 128u;
constexpr size_t core_elements = 32u * 128u;
constexpr unsigned scheduled_rows = 2u;

struct RecurrentTables {
    const float* g = nullptr;             // [32,65536], model layer specific.
    const float* beta = nullptr;          // [65536], FP32 sigmoid.
    const unsigned char* exp2 = nullptr;
    const unsigned char* rsqrt = nullptr;
};

struct RecurrentHead {
    float q[128], k[128];
    float decay, beta;
};

// One-draft fused_sigmoid_gating_delta_rule_update uses the same fixed
// normalization and per-V reduction order as the qualified non-packed q1
// recurrence. The long-context packed single-token route is a different op.
QRT_Q2_INLINE void prepare_head(RecurrentHead& out, const uint16_t* conv,
    uint16_t a, uint16_t b, unsigned head, const RecurrentTables& tables) {
    using namespace qrt_sm121_q1;
    for (unsigned i = 0; i < 128u; ++i) {
        out.q[i] = widen(conv[(head / 2u) * 128u + i]);
        out.k[i] = widen(conv[2048u + (head / 2u) * 128u + i]);
    }
    const float qi = inverse_norm(out.q, tables.rsqrt);
    const float ki = inverse_norm(out.k, tables.rsqrt);
    for (unsigned i = 0; i < 128u; ++i) {
        out.q[i] = multiply(multiply(out.q[i], qi), 0x1.6a09e6p-4f);
        out.k[i] = multiply(out.k[i], ki);
    }
    out.decay = qrt_sm121_exp2::evaluate(tables.exp2,
        multiply(tables.g[size_t(head) * 65536u + a], 0x1.715476p+0f));
    out.beta = tables.beta[b];
}

QRT_Q2_INLINE size_t state_offset(unsigned head, unsigned value, bool key_major) {
    return size_t(head) * 16384u + (key_major ? value : value * 128u);
}

// Input and output are distinct state rows. Row one must receive the computed
// row-zero state, including when row one is subsequently rejected by sampling.
QRT_Q2_INLINE uint16_t recurrent_value(const float* before, float* after,
    unsigned stride, const RecurrentHead& head, uint16_t value, unsigned value_dim) {
    using namespace qrt_sm121_q1;
    const float projected = state_dot(before, stride, head.decay, head.k, value_dim, true, false);
    const float residual = multiply(add(widen(value), -projected), head.beta);
    for (unsigned i = 0; i < 128u; ++i)
        after[i * stride] = fmaf(residual, head.k[i], multiply(before[i * stride], head.decay));
    return bf16(state_dot(after, stride, 1.0f, head.q, value_dim, false, false));
}
} // namespace qrt_sm121_q2
#undef QRT_Q2_INLINE
