#pragma once
#include "sm121_staged_half_projection.h"
#include "sm121_blocked_half_layout.h"

namespace qrt_sm121_blocked_half_projection {
namespace layout = qrt_sm121_blocked_half_layout;
namespace staged = qrt_sm121_staged_half_projection;
using Row = staged::Row;
using Stats = staged::Stats;

__global__ void prepare_weights(const uint16_t* values, Row* output, unsigned rows, unsigned width) {
    const size_t slot = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned groups = width / 16u;
    if (slot >= size_t(rows) * groups) return;
    const unsigned row = unsigned(slot / groups), group = unsigned(slot % groups);
    output[layout::offset(rows, width, row, group)] = staged::half::prepare(values + slot * 16u);
}

inline hipError_t prepare(const uint16_t* values, Row* output, size_t capacity_records,
    unsigned rows, unsigned width, hipStream_t stream) {
    const size_t required = layout::records(rows, width);
    if (!values || !output || !required || capacity_records < required) return hipErrorInvalidValue;
    const size_t groups = size_t(rows) * (width / 16u);
    hipLaunchKernelGGL(prepare_weights, dim3(unsigned((groups + 255u) / 256u)), dim3(256u),
        0u, stream, values, output, rows, width);
    return hipGetLastError();
}

// Inputs retain their row-major view. Only weight addressing changes. The
// original two-group prefetch, four-lane product primitive, ordered integer
// carries and unsupported-BF16 fallback are shared with the selected route.
template<bool Audit = false>
__device__ __forceinline__ float dot(const Row* input_row, const Row* weights,
    unsigned rows, unsigned weight_row, unsigned width,
    uint32_t* raw_trace = nullptr, Stats* statistics = nullptr) {
    const unsigned lane = threadIdx.x & 3u, groups = width / 16u;
    const Row* weight_block = weights + layout::offset(rows, width, weight_row, 0u);
    staged::Value carry{0u, -133, false};
    Stats counts;
#pragma unroll 1
    for (unsigned base = 0u; base < groups; base += 2u) {
        staged::LaneOperands operands[2];
        const unsigned count = groups - base < 2u ? groups - base : 2u;
#pragma unroll
        for (unsigned i = 0u; i < 2u; ++i) if (i < count)
            operands[i] = staged::load(input_row[base + i], weight_block[i]);
#pragma unroll
        for (unsigned i = 0u; i < 2u; ++i) if (i < count) {
            bool transformed;
            carry = staged::accumulate(carry, operands[i], Audit ? &transformed : nullptr);
            if constexpr (Audit) {
                counts.transformed += transformed; counts.original += !transformed;
                if (!lane && raw_trace) {
                    raw_trace[3u * (base + i)] = carry.significand;
                    raw_trace[3u * (base + i) + 1u] = uint32_t(int32_t(carry.exponent));
                    raw_trace[3u * (base + i) + 2u] = unsigned(carry.negative);
                }
            }
        }
        if (base + count < groups) weight_block += layout::tile_records;
    }
    if constexpr (Audit) if (!lane && statistics) *statistics = counts;
    return lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
}
} // namespace qrt_sm121_blocked_half_projection
