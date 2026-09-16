#pragma once
#include "sm121_folded_half_products.h"
#include "sm121_staged_half_projection.h"

// Isolated replay candidate. No product dispatcher opts in. The owner supplies
// complete prepared spans, width divisible by16 and four lanes per dot.
namespace qrt_sm121_folded_half_projection {
namespace folded = qrt_sm121_folded_half_products;
namespace staged = qrt_sm121_staged_half_projection;
using Row = folded::Row;
using Value = folded::Value;
using LaneOperands = staged::LaneOperands;
struct Stats { unsigned folded = 0u, fallback = 0u; };
__global__ void prepare_rows(const uint16_t* input, Row* output, unsigned rows, unsigned width) {
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group >= size_t(rows) * (width / 16u)) return;
    uint16_t original[16];
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) original[i] = input[group * 16u + i];
    output[group] = folded::prepare(original);
}
__device__ __forceinline__ LaneOperands load(const Row& left, const Row& right) {
    const unsigned pair = (threadIdx.x & 3u) * 2u;
    LaneOperands operands;
    __builtin_memcpy(operands.left, left.pairs + pair, 8u);
    __builtin_memcpy(operands.right, right.pairs + pair, 8u);
    operands.left_control = left.control; operands.right_control = right.control;
    return operands;
}
__device__ __forceinline__ Value fallback(Value carry, LaneOperands operands) {
    operands.left_control = folded::legacy_control(operands.left_control);
    operands.right_control = folded::legacy_control(operands.right_control);
    return staged::accumulate(carry, operands);
}
__device__ __forceinline__ Value accumulate(Value carry, const LaneOperands& operands, bool* used = nullptr) {
    if (used) *used = false;
    if ((operands.left_control | operands.right_control) & 0x8000u ||
        ((operands.left_control & operands.right_control) >> 16u) != 65535u)
        return fallback(carry, operands);
    uint32_t paired = 0u;
#pragma unroll
    for (unsigned i = 0u; i < 2u; ++i) {
        const uint32_t exponents = ((operands.left[i] >> 10u) & 0x001f001fu) +
            ((operands.right[i] >> 10u) & 0x001f001fu);
        paired = qrt_sm121_prepared_integer_pairs::maximum_pair(paired, exponents);
    }
    int maximum = int((paired & 65535u) > (paired >> 16u) ? paired & 65535u : paired >> 16u);
    maximum = qrt_sm121_lane_reduce::maximum<4u>(maximum);
    const auto plan = folded::plan(carry, operands.left_control, operands.right_control, maximum);
    if (!plan.allowed) return fallback(carry, operands);
    if (used) *used = true;
    uint32_t total = 0u;
#pragma unroll
    for (unsigned i = 0u; i < 2u; ++i) {
        const uint32_t left = operands.left[i] + (plan.left ? plan.delta : 0u);
        const uint32_t right = operands.right[i] + (plan.left ? 0u : plan.delta);
        total += uint32_t(int32_t(folded::half::product<false>(left, right)));
        total += uint32_t(int32_t(folded::half::product<true>(left, right)));
    }
    total = qrt_sm121_lane_reduce::sum<4u>(total);
    const unsigned shift = unsigned(plan.maximum - carry.exponent);
    const uint32_t aligned = shift >= 32u ? 0u : (carry.significand << 2u) >> shift;
    total += carry.negative ? 0u - aligned : aligned;
    const auto sum = qrt_sm121_group16::decode_modulo_sum(total,
        ((operands.left[0] ^ operands.right[0]) & 0x8000u) != 0u);
    return qrt_sm121_canonical::normalize(sum.magnitude, sum.negative, plan.maximum);
}
template<unsigned StagingGroups, bool Audit = false>
__device__ __forceinline__ float dot(const Row* left, const Row* right, unsigned width,
    uint32_t* raw_trace = nullptr, Stats* stats = nullptr) {
    static_assert(StagingGroups == 1u || StagingGroups == 2u || StagingGroups == 4u || StagingGroups == 8u);
    const unsigned lane = threadIdx.x & 3u, groups = width / 16u;
    Value carry{0u, -133, false}; Stats counts;
#pragma unroll 1
    for (unsigned base = 0u; base < groups; base += StagingGroups) {
        LaneOperands operands[StagingGroups];
        const unsigned count = groups - base < StagingGroups ? groups - base : StagingGroups;
#pragma unroll
        for (unsigned i = 0u; i < StagingGroups; ++i)
            if (i < count) operands[i] = load(left[base + i], right[base + i]);
#pragma unroll
        for (unsigned i = 0u; i < StagingGroups; ++i) if (i < count) {
            bool used;
            carry = accumulate(carry, operands[i], Audit ? &used : nullptr);
            if constexpr (Audit) {
                counts.folded += used; counts.fallback += !used;
                if (!lane && raw_trace) {
                    raw_trace[3u * (base + i)] = carry.significand;
                    raw_trace[3u * (base + i) + 1u] = uint32_t(int32_t(carry.exponent));
                    raw_trace[3u * (base + i) + 2u] = unsigned(carry.negative);
                }
            }
        }
    }
    if constexpr (Audit) if (!lane && stats) *stats = counts;
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_folded_half_projection
