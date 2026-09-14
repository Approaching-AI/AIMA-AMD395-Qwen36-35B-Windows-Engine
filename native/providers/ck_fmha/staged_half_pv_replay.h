#pragma once
#include <hip/hip_runtime.h>
#include "../sm121_attention_capacity.h"
#include "../moe_accumulator/sm121_staged_half_projection.h"
#include "../gdn/sm121_attention_rcp.h"

// Component route. Prepare causal P rows and feature-major V columns once,
// sharing the lossless K16 representation across selected output cells.
namespace qrt_staged_half_pv {
namespace staged = qrt_sm121_staged_half_projection;
using Row = staged::Row;
constexpr unsigned heads = 16u, kv_heads = 2u, dimensions = 256u, threads = 256u;
constexpr unsigned maximum_tokens = qrt_sm121_attention_capacity::kTokens;

template<bool Probability>
__global__ void prepare_rows(const uint16_t* source, unsigned tokens,
    unsigned query_start, unsigned query_count, Row* output) {
    const unsigned groups = (tokens + 15u) / 16u;
    const unsigned rows = Probability ? query_count * heads : kv_heads * dimensions;
    const size_t slot = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (slot >= size_t(rows) * groups) return;
    const unsigned row = unsigned(slot / groups), group = unsigned(slot % groups);
    const unsigned extent = Probability ? query_start + row / heads + 1u : tokens;
    uint16_t values[16];
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        const unsigned key = group * 16u + i;
        values[i] = key < extent ? source[size_t(row) * tokens + key] : 0u;
    }
    output[slot] = staged::half::prepare(values);
}

template<bool Probability>
inline int prepare(const uint16_t* source, unsigned tokens, unsigned query_start,
    unsigned query_count, Row* output, size_t capacity_rows, hipStream_t stream) {
    if (!source || !output || !tokens || tokens > maximum_tokens) return int(hipErrorInvalidValue);
    if constexpr (Probability) {
        if (!query_count || query_count > 128u || query_start >= tokens ||
            query_count > tokens - query_start) return int(hipErrorInvalidValue);
    } else if (query_start || query_count) return int(hipErrorInvalidValue);
    const size_t records = size_t(Probability ? query_count * heads : kv_heads * dimensions) *
        ((tokens + 15u) / 16u);
    if (capacity_rows < records) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(prepare_rows<Probability>), dim3(unsigned((records + 255u) / 256u)),
        dim3(threads), 0u, stream, source, tokens, query_start, query_count, output);
    return int(hipGetLastError());
}

template<bool Audit = false>
__global__ void replay(const Row* probability, const Row* value, const float* scales,
    float* output, unsigned query_start, unsigned query_count, unsigned output_start,
    unsigned stride, unsigned value_stride, const unsigned char* rcp_table,
    float* raw_accumulator, float* raw_denominator, const unsigned* indices,
    const unsigned* count, unsigned long long* audit = nullptr) {
    constexpr unsigned lanes = 4u;
    const unsigned member = threadIdx.x & 3u, step = gridDim.x * blockDim.x / lanes;
    const unsigned probability_groups = (stride + 15u) / 16u, value_groups = (value_stride + 15u) / 16u;
    unsigned long long stats[5]{};
    Row zero{};
    zero.control = uint16_t(-15); // Canonical encoding of sixteen positive zeros.
    for (unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes; slot < *count; slot += step) {
        const unsigned cell = indices[slot], column = cell % dimensions, row = cell / dimensions;
        const unsigned head = row % heads, query = row / heads, kv = head / (heads / kv_heads);
        const unsigned tokens = query_start + query + 1u, tiles = (stride + 31u) / 32u;
        float accumulator = 0.0f;
        for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
            staged::LaneOperands operands[2];
#pragma unroll
            for (unsigned group = 0u; group < 2u; ++group) {
                const unsigned index = tile * 2u + group;
                const Row& p = index < probability_groups ? probability[size_t(row) * probability_groups + index] : zero;
                const Row& v = index < value_groups ? value[(size_t(kv) * dimensions + column) * value_groups + index] : zero;
                operands[group] = staged::load(p, v);
            }
            const float alpha = scales[size_t(row) * (tiles + 1u) + tile];
            volatile float rounded = accumulator * alpha;
            auto partial = qrt_q1_moe_hawkeye::value_from_float(rounded, -133);
#pragma unroll
            for (unsigned group = 0u; group < 2u; ++group) {
                bool transformed;
                partial = staged::accumulate(partial, operands[group], Audit ? &transformed : nullptr);
                if constexpr (Audit) { ++stats[0]; ++stats[transformed ? 3u : 4u]; }
                // PV has an FP32 endpoint after each K16, unlike dense/QK dots.
                partial = qrt_sm121_group16::finish_accumulator(partial);
                partial = qrt_q1_moe_hawkeye::value_from_float(qrt_q1_moe_hawkeye::value_to_float(partial), -133);
            }
            accumulator = qrt_q1_moe_hawkeye::value_to_float(partial);
        }
        if (!member) {
            const float denominator = scales[size_t(row) * (tiles + 1u) + tiles];
            const size_t index = size_t(output_start) * heads * dimensions + cell;
            output[index] = rcp_table ? accumulator * qrt_sm121_attention_rcp::evaluate(rcp_table, denominator) : accumulator / denominator;
            if (raw_accumulator) raw_accumulator[index] = accumulator;
            if (raw_denominator && !column) raw_denominator[size_t(output_start) * heads + row] = denominator;
        }
    }
    if constexpr (Audit) if (!member && audit)
        for (unsigned i = 0u; i < 5u; ++i) if (stats[i]) atomicAdd(audit + i, stats[i]);
    (void)query_count;
}

template<bool Audit = false>
inline int launch(const Row* probability, size_t probability_rows,
    const Row* value, size_t value_rows, const float* scales, float* output,
    unsigned query_start, unsigned query_count, unsigned output_start, unsigned stride,
    unsigned value_stride, const unsigned char* rcp_table, float* raw_accumulator,
    float* raw_denominator, const unsigned* indices, const unsigned* count,
    hipStream_t stream, unsigned long long* audit = nullptr) {
    if (!probability || !value || !scales || !output || !indices || !count ||
        !query_count || query_count > 128u || query_start >= stride ||
        query_count > stride - query_start || stride > maximum_tokens ||
        value_stride < stride || value_stride > maximum_tokens ||
        output_start >= maximum_tokens || query_count > maximum_tokens - output_start ||
        probability_rows < size_t(query_count) * heads * ((stride + 15u) / 16u) ||
        value_rows < size_t(kv_heads) * dimensions * ((value_stride + 15u) / 16u) ||
        (Audit && !audit)) return int(hipErrorInvalidValue);
    const unsigned cells = query_count * heads * dimensions;
    const unsigned requested = (cells + threads / 4u - 1u) / (threads / 4u);
    const unsigned blocks = requested < 1024u ? requested : 1024u;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(replay<Audit>), dim3(blocks), dim3(threads), 0u, stream,
        probability, value, scales, output, query_start, query_count, output_start, stride,
        value_stride, rcp_table, raw_accumulator, raw_denominator, indices, count, audit);
    return int(hipGetLastError());
}
} // namespace qrt_staged_half_pv
