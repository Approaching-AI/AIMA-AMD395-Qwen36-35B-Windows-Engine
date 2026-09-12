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
enum class Stage { Router, SharedGateUp, SharedDown, AttentionOutput, LinearBA };
struct Plan {
    unsigned splits = 1;
    unsigned accumulators = 1;
    unsigned warp_m = 0;
    unsigned warp_n = 0;
    bool bf16_partials = false;
    unsigned serial_bf16_tile = 0;
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
    } else if (stage == Stage::AttentionOutput) {
        if (tokens >= 17 && tokens <= 32) return {3, 1, 0, 0, false, 64};
        if (tokens >= 42 && tokens <= 49) return {8, 1, 0, 0, false, 64};
        if (tokens >= 50 && tokens <= 64) return {1, 3, 32, 16, false};
    } else if (stage == Stage::LinearBA) {
        // Reference B/A is one N=64 projection; native A and B have N=32.
        // The A feature offset is a multiple of each selected warp-M extent.
        if (tokens == 16) return {8, 1, 0, 0, true};
        if ((tokens >= 17 && tokens <= 22) || (tokens >= 25 && tokens <= 32) ||
            (tokens >= 41 && tokens <= 48) || (tokens >= 78 && tokens <= 80))
            return {16, 3, 32, 8, false};
        if ((tokens >= 23 && tokens <= 24) || (tokens >= 33 && tokens <= 40) || tokens == 72)
            return {4, 4, 16, 8, false};
        if ((tokens >= 49 && tokens <= 64) || (tokens >= 84 && tokens <= 96))
            return {16, 3, 32, 16, false};
        if ((tokens >= 65 && tokens <= 71) || (tokens >= 73 && tokens <= 77) ||
            (tokens >= 81 && tokens <= 83) || (tokens >= 97 && tokens <= 114))
            return {8, 3, 16, 32, false};
        if ((tokens >= 115 && tokens <= 128) || (tokens >= 146 && tokens <= 288))
            return {4, 3, 32, 16, false};
        if (tokens >= 129 && tokens <= 145) return {4, 3, 16, 16, false};
        if ((tokens >= 289 && tokens <= 320) || (tokens >= 353 && tokens <= 443))
            return {3, 3, 32, 16, false};
        if (tokens >= 321 && tokens <= 352) return {3, 3, 16, 32, false};
        if (tokens >= 444 && tokens <= 544) return {1, 3, 32, 8, false};
        if ((tokens >= 545 && tokens <= 1472) || (tokens >= 1493 && tokens <= 1664) ||
            (tokens >= 1673 && tokens <= 1728) || (tokens >= 1799 && tokens <= 1856) ||
            (tokens >= 1905 && tokens <= 1920) || (tokens >= 2025 && tokens <= 2048) ||
            (tokens >= 2173 && tokens <= 2176))
            return {1, 3, 16, 32, false};
        if ((tokens >= 2276 && tokens <= 2304) || (tokens >= 2308 && tokens <= 2560) ||
            (tokens >= 2568 && tokens <= 2816))
            return {3, 1, 0, 0, false, 64};
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
// assignment; merge in physical register order with FP32 additions. The
// warp16x8 form uses four independent K16 accumulators in ascending order.
template<unsigned Lanes>
QRT_SHORT_PROJECTION_INLINE float dot(
    const uint16_t *a, const uint16_t *b, unsigned count, Plan p,
    unsigned token, unsigned feature) {
    static_assert(Lanes == 1 || Lanes == 4 || Lanes == 8 || Lanes == 16,
                  "unsupported projection subgroup");
    using namespace qrt_q1_moe_hawkeye;
    unsigned lane = 0;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
    lane = threadIdx.x & (Lanes - 1u);
#endif
    const unsigned cell = p.accumulators > 1
        ? ((feature % p.warp_m) / 16) * (p.warp_n / 8) + (token % p.warp_n) / 8
        : 0;
    float sum = 0.0f;
    const unsigned serial_extent = p.serial_bf16_tile
        ? ((count + p.splits - 1) / p.splits + p.serial_bf16_tile - 1) /
            p.serial_bf16_tile * p.serial_bf16_tile : 0;
    for (unsigned split = 0; split < p.splits; ++split) {
        Value accumulators[4]{{0, -133, false}, {0, -133, false},
                              {0, -133, false}, {0, -133, false}};
        // CUTLASS uses contiguous partitions. Its serial output-type path
        // rounds the running output after each partition; the parallel path
        // rounds independent partials before the final FP32 reduction.
        const unsigned extent = serial_extent ? serial_extent : count / p.splits;
        const bool contiguous = p.bf16_partials || p.serial_bf16_tile;
        const unsigned begin = contiguous ? split * extent : split * 64;
        const unsigned stop = contiguous ? begin + extent : count;
        const unsigned end = stop < count ? stop : count;
        const unsigned step = contiguous ? 64 : 64 * p.splits;
        for (unsigned base = begin; base < end; base += step) {
            for (unsigned group = 0; group < 4; ++group) {
                const unsigned k = base + group * 16;
                if (k >= end) break;
                const unsigned slot = p.accumulators > 1 ? (group + cell) % p.accumulators : 0;
#if defined(__HIP_DEVICE_COMPILE__) || defined(__CUDA_ARCH__)
                if constexpr (Lanes != 1)
                    accumulators[slot] = qrt_sm121_subgroup::accumulate<Lanes>(
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
            if (p.serial_bf16_tile) sum = round_bf16(sum);
        }
    }
    return lane == 0 ? sum : 0.0f;
}
} // namespace qrt_sm121_prefill_projection
#endif
