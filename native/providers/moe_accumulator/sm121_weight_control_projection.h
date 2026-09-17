#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_weight_control.h"

// Isolated exact replay component. Immutable weights remain in their original
// BF16 storage. Cache only the lossless K16 scale/nonzero control and reconstruct
// each lane's FP16 payload on demand. Current activations use the retained
// 36-byte prepared rows. No model/reference outputs participate in the cache.
namespace qrt_sm121_weight_control_projection {
namespace staged = qrt_sm121_staged_half_projection;
namespace half = qrt_sm121_scaled_half_products;
using Row = half::Row;
using Value = staged::Value;
using Stats = staged::Stats;

__global__ void prepare_controls(const uint16_t* weights, unsigned* output, size_t groups) {
    const size_t group = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (group < groups) output[group] = control(weights + group * 16u);
}

inline hipError_t prepare(const uint16_t* weights, size_t weight_words,
    unsigned* output, size_t output_words, unsigned rows, unsigned width, hipStream_t stream) {
    if (!weights || !output || !rows || rows > 524288u || !width || width > 8192u || width % 16u ||
        (uintptr_t(weights) & 1u) || (uintptr_t(output) & 3u)) return hipErrorInvalidValue;
    const size_t elements = size_t(rows) * width, groups = elements / 16u;
    if (weight_words < elements || output_words < groups) return hipErrorInvalidValue;
    const uintptr_t a = uintptr_t(weights), b = uintptr_t(output);
    const size_t a_bytes = elements * sizeof(uint16_t), b_bytes = groups * sizeof(unsigned);
    if (a > UINTPTR_MAX - a_bytes || b > UINTPTR_MAX - b_bytes ||
        (a < b + b_bytes && b < a + a_bytes)) return hipErrorInvalidValue;
    hipLaunchKernelGGL(prepare_controls, dim3((groups + 255u) / 256u), dim3(256u), 0u, stream,
        weights, output, groups);
    return hipGetLastError();
}

__device__ __forceinline__ staged::LaneOperands load(const Row& input,
    const uint16_t* weights, unsigned metadata) {
    const unsigned pair = (threadIdx.x & 3u) * 2u;
    staged::LaneOperands p;
    __builtin_memcpy(p.left, input.pairs + pair, 8u);
    __builtin_memcpy(p.right, weights + pair * 2u, 8u);
    p.left_control = input.control; p.right_control = metadata;
    p.right[0] = encode_pair(p.right[0], metadata, pair);
    p.right[1] = encode_pair(p.right[1], metadata, pair + 1u);
    return p;
}

template<unsigned StagingGroups = 2u, bool Audit = false>
__device__ __forceinline__ float dot(const Row* inputs, const uint16_t* weights,
    const unsigned* controls, unsigned width, uint32_t* trace = nullptr, Stats* stats = nullptr) {
    static_assert(StagingGroups == 1u || StagingGroups == 2u || StagingGroups == 4u);
    const unsigned lane = threadIdx.x & 3u, groups = width / 16u;
    Value carry{0u, -133, false}; Stats counts;
#pragma unroll 1
    for (unsigned base = 0u; base < groups; base += StagingGroups) {
        staged::LaneOperands operands[StagingGroups];
        const unsigned count = groups - base < StagingGroups ? groups - base : StagingGroups;
#pragma unroll
        for (unsigned i = 0u; i < StagingGroups; ++i) if (i < count)
            operands[i] = load(inputs[base + i], weights + size_t(base + i) * 16u, controls[base + i]);
#pragma unroll
        for (unsigned i = 0u; i < StagingGroups; ++i) if (i < count) {
            bool transformed;
            carry = staged::accumulate(carry, operands[i], Audit ? &transformed : nullptr);
            if constexpr (Audit) {
                counts.transformed += transformed; counts.original += !transformed;
                if (!lane && trace) {
                    trace[3u * (base + i)] = carry.significand;
                    trace[3u * (base + i) + 1u] = uint32_t(int32_t(carry.exponent));
                    trace[3u * (base + i) + 2u] = unsigned(carry.negative);
                }
            }
        }
    }
    if constexpr (Audit) if (!lane && stats) *stats = counts;
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_weight_control_projection
