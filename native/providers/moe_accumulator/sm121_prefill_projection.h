#ifndef QRT_SM121_PREFILL_PROJECTION_H
#define QRT_SM121_PREFILL_PROJECTION_H

#include "q1_moe_hawkeye_bf16_accumulator.h"
#if defined(__HIPCC__) || defined(__CUDACC__)
#include "sm121_subgroup.h"
#define QRT_SHORT_PROJECTION_INLINE __host__ __device__ __forceinline__
#else
#define QRT_SHORT_PROJECTION_INLINE inline
#endif

namespace qrt_sm121_prefill_projection {
enum class Stage { Router, SharedGateUp, SharedDown };
struct Plan {
    unsigned splits = 1;
    unsigned accumulators = 1;
    unsigned warp_m = 0;
    unsigned warp_n = 0;
    bool bf16_partials = false;
};

// The pinned GB10 cuBLAS shapes were profiled for every M=1..4096. Loaded
// cubins establish the K64 traversal, K16 accumulator assignment and final
// FP32 merge. The fused shared gate/up reference has N=1024 even though the
// native provider stores its two N=512 projections separately.
QRT_SHORT_PROJECTION_INLINE Plan plan(Stage stage, unsigned tokens) {
    if (stage == Stage::Router) {
        if (tokens >= 4 && tokens <= 16) return {8, 1, 0, 0, true};
        if (tokens >= 17 && tokens <= 28) return {4, 1, 0, 0, false};
        if (tokens >= 29 && tokens <= 32) return {12, 3, 32, 16, false};
        if (tokens >= 33 && tokens <= 94) return {4, 3, 16, 32, false};
        if (tokens >= 95 && tokens <= 114) return {3, 3, 16, 32, false};
        if (tokens >= 115 && tokens <= 128) return {3, 3, 32, 16, false};
        if (tokens >= 129 && tokens <= 159) return {1, 3, 16, 32, false};
        if (tokens >= 160 && tokens <= 250) return {1, 3, 32, 16, false};
        if (tokens >= 251 && tokens <= 256) return {1, 3, 16, 32, false};
        if (tokens >= 257 && tokens <= 330) return {1, 3, 32, 16, false};
        if (tokens >= 331 && tokens <= 608) return {1, 3, 16, 32, false};
    } else if (stage == Stage::SharedGateUp) {
        if ((tokens >= 17 && tokens <= 32) || (tokens >= 65 && tokens <= 128))
            return {1, 3, 16, 32, false};
    } else if (stage == Stage::SharedDown) {
        if ((tokens >= 17 && tokens <= 32) || (tokens >= 47 && tokens <= 64))
            return {1, 3, 16, 32, false};
    }
    return {};
}

QRT_SHORT_PROJECTION_INLINE bool changes_dot(Plan p) {
    return p.splits != 1 || p.accumulators != 1;
}

QRT_SHORT_PROJECTION_INLINE float add(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#else
    volatile float result = a + b;
    return result;
#endif
}
QRT_SHORT_PROJECTION_INLINE float round_bf16(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16u) & 1u)) & 0xffff0000u;
    memcpy(&value, &bits, sizeof(value));
    return value;
}
QRT_SHORT_PROJECTION_INLINE qrt_q1_moe_hawkeye::Value scalar_group(
    qrt_q1_moe_hawkeye::Value carry, const uint16_t *a, const uint16_t *b) {
    using namespace qrt_q1_moe_hawkeye;
    Value values[17];
    values[0] = carry;
    for (unsigned k = 0; k < 16; ++k) values[k + 1] = multiply_bf16(a[k], b[k], -133);
    return group_sum<26, -133>(values, 17);
}

// GB10 cuBLAS nvjet split-K walks K64 tiles with stride=split_count. Its
// three-accumulator forms assign K16 groups 0/3, 1 and 2 independently in
// each tile. The output's warp subtile rotates the physical accumulator
// assignment; merge in physical register order with two FP32 additions.
template<unsigned Lanes>
QRT_SHORT_PROJECTION_INLINE float dot(
    const uint16_t *a, const uint16_t *b, unsigned count, Plan p,
    unsigned token, unsigned feature) {
    static_assert(Lanes == 1 || Lanes == 16, "unsupported projection subgroup");
    using namespace qrt_q1_moe_hawkeye;
    unsigned lane = 0;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    lane = threadIdx.x & (Lanes - 1u);
#endif
    const unsigned cell = p.accumulators == 3
        ? ((feature % p.warp_m) / 16) * (p.warp_n / 8) + (token % p.warp_n) / 8
        : 0;
    float sum = 0.0f;
    for (unsigned split = 0; split < p.splits; ++split) {
        Value accumulators[3]{{0, -133, false}, {0, -133, false}, {0, -133, false}};
        // The small CUTLASS path instead uses contiguous K256 BF16 partials.
        const unsigned begin = p.bf16_partials ? split * (count / p.splits) : split * 64;
        const unsigned end = p.bf16_partials ? begin + count / p.splits : count;
        const unsigned step = p.bf16_partials ? 64 : 64 * p.splits;
        for (unsigned base = begin; base < end; base += step) {
            for (unsigned group = 0; group < 4; ++group) {
                const unsigned k = base + group * 16;
                if (k >= end) break;
                const unsigned slot = p.accumulators == 3 ? (group + cell) % 3 : 0;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                if constexpr (Lanes == 16)
                    accumulators[slot] = qrt_sm121_subgroup::accumulate<16>(
                        accumulators[slot], a + k, b + k);
                else
#endif
                    accumulators[slot] = scalar_group(accumulators[slot], a + k, b + k);
            }
        }
        if (lane == 0) {
            float partial = value_to_float(group_sum<26, -133>(&accumulators[0], 1));
            for (unsigned slot = 1; slot < p.accumulators; ++slot)
                partial = add(partial, value_to_float(group_sum<26, -133>(&accumulators[slot], 1)));
            sum = add(sum, p.bf16_partials ? round_bf16(partial) : partial);
        }
    }
    return lane == 0 ? sum : 0.0f;
}
} // namespace qrt_sm121_prefill_projection
#endif
