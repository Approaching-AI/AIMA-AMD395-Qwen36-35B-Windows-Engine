#pragma once
#include "float_pv_replay.h"
#include "../moe_accumulator/sm121_pv_group_plan.h"

// Component-only replacement for the exact selected PV replay. Candidate
// identities, K32 alpha, K16 rounding and final reciprocal stay unchanged.
namespace qrt_pv_group_replay {
using namespace qrt_sm121_float_pv;
namespace plan = qrt_sm121_pv_group_plan;

template<bool Probability>
__global__ void prepare(const uint16_t* source, unsigned tokens, unsigned query_start,
    unsigned query_count, uint32_t* output) {
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
        // Value input is the original lossless feature-major transpose.
        values[i] = key < extent ? source[size_t(row) * tokens + key] : 0u;
    }
    output[slot] = plan::metadata(values, 16u);
}

template<bool Skip, bool Float, bool Audit = false>
__global__ void replay(const uint16_t* probability, const uint16_t* transposed_value,
    const uint32_t* probability_plan, const uint32_t* value_plan, const float* scales,
    float* output, unsigned query_start, unsigned query_count, unsigned output_start,
    unsigned stride, unsigned value_stride, const unsigned char* rcp_table,
    float* raw_accumulator, float* raw_denominator, const unsigned* indices,
    const unsigned* count, unsigned long long* audit = nullptr) {
    constexpr unsigned lanes = 4u, items = 4u;
    const unsigned member = threadIdx.x & 3u, step = gridDim.x * blockDim.x / lanes;
    const unsigned probability_groups = (stride + 15u) / 16u, value_groups = (value_stride + 15u) / 16u;
    unsigned long long stats[5]{};
    for (unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes; slot < *count; slot += step) {
        const unsigned cell = indices[slot], column = cell % dimensions, row = cell / dimensions;
        const unsigned head = row % heads, query = row / heads, kv = head / (heads / kv_heads);
        const unsigned tokens = query_start + query + 1u, tiles = (stride + 31u) / 32u;
        float accumulator = 0.0f;
        for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
            const float alpha = scales[size_t(row) * (tiles + 1u) + tile];
            volatile float rounded = accumulator * alpha;
            auto partial = qrt_q1_moe_hawkeye::value_from_float(rounded, -133);
            for (unsigned begin = 0u; begin < 32u; begin += 16u) {
                const unsigned group = tile * 2u + begin / 16u;
                // The second K16 group of the last K32 tile can be entirely
                // outside the allocation's key extent, while remaining part
                // of the original rounding recurrence.
                const uint32_t pm = group < probability_groups ? probability_plan[size_t(row) * probability_groups + group] : plan::float_eligible;
                const uint32_t vm = group < value_groups ? value_plan[(size_t(kv) * dimensions + column) * value_groups + group] : plan::float_eligible;
                const unsigned skip = Skip ? plan::skip_kind(partial.exponent, pm, vm) : 0u;
                if constexpr (Audit) ++stats[0];
                if (skip) {
                    if constexpr (Audit) ++stats[skip];
                } else {
                    uint16_t left[items], right[items];
#pragma unroll
                    for (unsigned item = 0u; item < items; ++item) {
                        const unsigned key = tile * 32u + begin + member * items + item;
                        left[item] = key < tokens ? probability[size_t(row) * stride + key] : 0u;
                        right[item] = key < tokens ? transposed_value[(size_t(kv) * dimensions + column) * value_stride + key] : 0u;
                    }
                    if (Float && plan::use_float(pm, vm)) {
                        qrt_sm121_float_subgroup::Product products[items];
#pragma unroll
                        for (unsigned item = 0u; item < items; ++item) {
                            const uint16_t a = left[item], b = right[item];
                            const bool zero = !(a & 0x7fffu) || !(b & 0x7fffu);
                            products[item] = {qrt_sm121_float_alignment::from_bits(uint32_t(a) << 16u) *
                                qrt_sm121_float_alignment::from_bits(uint32_t(b) << 16u),
                                uint32_t(a) | (uint32_t(b) << 16u),
                                zero ? -133 : int((a >> 7u) & 255u) + int((b >> 7u) & 255u) - 254};
                        }
                        if constexpr (Audit) {
                            bool used_float = false;
                            partial = qrt_sm121_float_subgroup::accumulate<lanes>(partial, products, &used_float);
                            ++stats[used_float ? 3u : 4u];
                        } else partial = qrt_sm121_float_subgroup::accumulate<lanes>(partial, products);
                    } else {
                        uint32_t products[items];
#pragma unroll
                        for (unsigned item = 0u; item < items; ++item)
                            products[item] = qrt_sm121_group16::pack_product(
                                qrt_q1_moe_hawkeye::multiply_bf16(left[item], right[item], -133));
                        partial = qrt_sm121_subgroup::accumulate_products<lanes>(partial, products);
                        if constexpr (Audit) ++stats[4];
                    }
                }
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
} // namespace qrt_pv_group_replay
