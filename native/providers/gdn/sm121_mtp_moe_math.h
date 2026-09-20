#pragma once
#include "sm121_q1_math.h"
#include "../moe_accumulator/sm121_router_exp.h"
#include "../moe_accumulator/sm121_shared_gate.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_MTP_MOE_HD __host__ __device__
#else
#define QRT_MTP_MOE_HD
#endif

namespace qrt_sm121_mtp {
QRT_MTP_MOE_HD inline float moe_divide(float a, float b) {
    // Compile without unsafe/reciprocal math: the native router requires an
    // ordinary rounded FP32 division, not an approximate reciprocal intrinsic.
    volatile float value = a / b;
    return value;
}

QRT_MTP_MOE_HD inline bool moe_finite(float value) {
    return (qrt_sm121_exp2::bits(value) & 0x7f800000u) != 0x7f800000u;
}

// Original TP1/EP1 internal router: adjacent groups of eight probabilities,
// XOR32 sum, then eight greedy selections and ordered renormalization. The
// caller owns the output arrays and must reject a false result before publish.
QRT_MTP_MOE_HD inline bool moe_route(const uint16_t* logits,
    const uint32_t* exp_fraction, uint32_t* ids, float* weights) {
    using qrt_sm121_q1::add;
    using qrt_sm121_q1::multiply;
    using qrt_sm121_q1::widen;
    float probability[256], sum[32] = {}, maximum = -INFINITY;
    for (unsigned i = 0; i < 256u; ++i) {
        const float value = widen(logits[i]);
        if (!moe_finite(value)) return false;
        maximum = value > maximum ? value : maximum;
    }
    for (unsigned lane = 0; lane < 32u; ++lane)
        for (unsigned item = 0; item < 8u; ++item) {
            const unsigned index = lane * 8u + item;
            probability[index] = qrt_sm121_router::exp(add(widen(logits[index]), -maximum), exp_fraction);
            if (!moe_finite(probability[index]) || probability[index] < 0.0f) return false;
            sum[lane] = add(sum[lane], probability[index]);
        }
    for (unsigned mask = 16u; mask; mask >>= 1u) {
        float next[32];
        for (unsigned lane = 0; lane < 32u; ++lane) next[lane] = add(sum[lane], sum[lane ^ mask]);
        for (unsigned lane = 0; lane < 32u; ++lane) sum[lane] = next[lane];
    }
    if (!moe_finite(sum[0]) || !(sum[0] > 0.0f)) return false;
    const float inverse = moe_divide(1.0f, sum[0]);
    for (unsigned i = 0; i < 256u; ++i) probability[i] = multiply(probability[i], inverse);
    float selected[8], denominator = 0.0f;
    for (unsigned route = 0; route < 8u; ++route) {
        unsigned best = 0;
        for (unsigned i = 1; i < 256u; ++i)
            if (probability[i] > probability[best]) best = i;
        ids[route] = best;
        selected[route] = probability[best];
        denominator = add(denominator, selected[route]);
        probability[best] = -10000.0f;
    }
    if (!moe_finite(denominator) || !(denominator > 0.0f)) return false;
    for (unsigned route = 0; route < 8u; ++route) {
        weights[route] = moe_divide(selected[route], denominator);
        if (!moe_finite(weights[route]) || weights[route] < 0.0f || weights[route] > 1.0f) return false;
    }
    return true;
}

// silu points at the first BF16 domain entry, after the artifact's header.
QRT_MTP_MOE_HD inline uint16_t moe_activate(uint16_t gate, uint16_t up, const uint16_t* silu) {
    return qrt_sm121_q1::bf16(qrt_sm121_q1::multiply(
        qrt_sm121_q1::widen(silu[gate]), qrt_sm121_q1::widen(up)));
}

QRT_MTP_MOE_HD inline uint16_t moe_shared_product(uint16_t gate, uint16_t down,
                                                const uint16_t* sigmoid) {
    return qrt_sm121_q1::bf16(qrt_sm121_q1::multiply(
        qrt_sm121_q1::widen(sigmoid[gate]), qrt_sm121_q1::widen(down)));
}

QRT_MTP_MOE_HD inline uint16_t moe_routed_sum(const uint16_t* weighted, unsigned channel) {
    using qrt_sm121_q1::add;
    using qrt_sm121_q1::widen;
    float value = add(widen(weighted[channel]), widen(weighted[channel + 4u * 2048u]));
    for (unsigned route = 1; route < 4u; ++route)
        value = add(value, add(widen(weighted[channel + route * 2048u]),
                               widen(weighted[channel + (route + 4u) * 2048u])));
    return qrt_sm121_q1::bf16(value);
}

QRT_MTP_MOE_HD inline uint16_t moe_output(uint16_t shared, uint16_t routed) {
    return qrt_sm121_q1::bf16(qrt_sm121_q1::add(qrt_sm121_q1::widen(shared), qrt_sm121_q1::widen(routed)));
}
} // namespace qrt_sm121_mtp
#undef QRT_MTP_MOE_HD
