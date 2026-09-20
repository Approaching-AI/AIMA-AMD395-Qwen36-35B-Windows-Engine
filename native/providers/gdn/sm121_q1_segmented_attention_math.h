#pragma once
#include "sm121_q1_math.h"

#if defined(__HIPCC__) || defined(__CUDACC__)
#define QRT_SEGMENT_HD __host__ __device__ __forceinline__
#else
#define QRT_SEGMENT_HD inline
#endif

namespace qrt_sm121_q1_segmented_attention {
constexpr unsigned segments = 16u, tile_tokens = 16u, heads = 16u, dimension = 256u;
constexpr size_t output_elements = size_t(heads)*segments*dimension;
constexpr size_t scalar_elements = size_t(heads)*segments;

QRT_SEGMENT_HD float exponential(float value, const unsigned char* table) {
    return qrt_sm121_exp2::evaluate(table, qrt_sm121_q1::multiply(value, 1.4426950408889634074f));
}

// Original 16-column MMA layout: adjacent columns, lane XOR2/1, then
// the two column groups exchanged through shared memory.
QRT_SEGMENT_HD float tile_sum(float values[tile_tokens]) {
    constexpr unsigned order[] = {1u, 4u, 2u, 8u};
    unsigned consumed = 0u;
    for (unsigned stride : order) {
        consumed |= stride;
        for (unsigned i = 0u; i < tile_tokens; ++i)
            if (!(i & consumed)) values[i] = qrt_sm121_q1::add(values[i], values[i+stride]);
    }
    return values[0];
}

QRT_SEGMENT_HD unsigned tiles_per_segment(unsigned tokens) {
    return (tokens + segments*tile_tokens - 1u)/(segments*tile_tokens);
}
QRT_SEGMENT_HD unsigned active_segments(unsigned tokens) {
    const unsigned width = tiles_per_segment(tokens)*tile_tokens;
    return width ? (tokens + width - 1u)/width : 0u;
}

// The original merge compiler fuses each low-half product with its rounded
// high-half partner, then follows XOR4/2/1. A separately rounded generic
// reduction changes the final context at sensitive BF16 boundaries.
QRT_SEGMENT_HD float merge_denominator(const float* sums, const float* scales) {
    float partial[8];
    for (unsigned i = 0u; i < 8u; ++i)
        partial[i] = fmaf(sums[i], scales[i], qrt_sm121_q1::multiply(sums[i+8u], scales[i+8u]));
    for (unsigned stride = 4u; stride; stride >>= 1u)
        for (unsigned i = 0u; i < stride; ++i)
            partial[i] = qrt_sm121_q1::add(partial[i], partial[i+stride]);
    return partial[0];
}

// Each original lane advances either the even or odd segment chain. Its
// second product starts the chain; products0 and2..7 are ordered FP32 FMAs.
QRT_SEGMENT_HD float merge_numerator(const float* values, const float* scales) {
    float chain[2];
    for (unsigned parity = 0u; parity < 2u; ++parity) {
        float sum = qrt_sm121_q1::multiply(values[(parity+2u)*dimension], scales[parity+2u]);
        sum = fmaf(values[parity*dimension], scales[parity], sum);
        for (unsigned segment = parity+4u; segment < segments; segment += 2u)
            sum = fmaf(values[segment*dimension], scales[segment], sum);
        chain[parity] = sum;
    }
    return qrt_sm121_q1::add(chain[0], chain[1]);
}
} // namespace qrt_sm121_q1_segmented_attention
#undef QRT_SEGMENT_HD
