#pragma once
#include "sm121_scalar_projection.h"

namespace qrt_sm121_interleaved_projection {
using Value = qrt_q1_moe_hawkeye::Value;
struct Stats { unsigned floating = 0u, integer = 0u; };

// Independent outputs share four-lane ownership. Each output retains its
// original ascending K16 recurrence; loads and independent carries can overlap.
// Adjacent compacted indices commonly share a token, so reuse that input word
// directly without a new layout, shared-memory tile or persistent workspace.
template<unsigned Outputs, bool Trace = false>
__device__ __forceinline__ void dot(const uint16_t* const* left,
    const uint16_t* const* right, const bool* eligible, unsigned active,
    unsigned width, float* result, uint32_t* const* traces = nullptr,
    Stats* statistics = nullptr) {
    static_assert(Outputs == 2u || Outputs == 4u);
    constexpr unsigned lanes = 4u, items = 4u;
    const unsigned lane = threadIdx.x & (lanes - 1u);
    Value carry[Outputs]; Stats stats[Outputs];
#pragma unroll
    for (unsigned slot = 0u; slot < Outputs; ++slot) carry[slot] = {0u, -133, false};
#pragma unroll 1
    for (unsigned base = 0u; base < width; base += 16u) {
        qrt_sm121_float_subgroup::Product products[Outputs][items];
        uint64_t input_words[Outputs];
#pragma unroll
        for (unsigned slot = 0u; slot < Outputs; ++slot) if (slot < active) {
            if (slot && left[slot] == left[0]) input_words[slot] = input_words[0];
            else __builtin_memcpy(&input_words[slot], left[slot] + base + lane * items,
                sizeof(uint64_t));
            uint64_t weight;
            __builtin_memcpy(&weight, right[slot] + base + lane * items, sizeof(weight));
#pragma unroll
            for (unsigned item = 0u; item < items; ++item) {
                const uint16_t x = uint16_t(input_words[slot] >> (16u * item));
                const uint16_t y = uint16_t(weight >> (16u * item));
                const uint32_t original = uint32_t(x) | (uint32_t(y) << 16u);
                if (!eligible[slot]) products[slot][item] = {0.0f, original, 512};
                else {
                    const bool zero = !(x & 0x7fffu) || !(y & 0x7fffu);
                    products[slot][item] = {
                        qrt_sm121_float_alignment::from_bits(uint32_t(x) << 16u) *
                            qrt_sm121_float_alignment::from_bits(uint32_t(y) << 16u),
                        original, zero ? -133 : int((x >> 7u) & 255u) +
                            int((y >> 7u) & 255u) - 254};
                }
            }
        }
#pragma unroll
        for (unsigned slot = 0u; slot < Outputs; ++slot) if (slot < active) {
            bool floating;
            carry[slot] = qrt_sm121_float_subgroup::accumulate<lanes>(
                carry[slot], products[slot], statistics ? &floating : nullptr);
            if (statistics) {
                stats[slot].floating += unsigned(floating);
                stats[slot].integer += unsigned(!floating);
            }
            if constexpr (Trace) if (!lane) {
                const float value = qrt_q1_moe_hawkeye::value_to_float(
                    qrt_sm121_group16::finish_accumulator(carry[slot]));
                __builtin_memcpy(traces[slot] + base / 16u, &value, sizeof(value));
            }
        }
    }
#pragma unroll
    for (unsigned slot = 0u; slot < Outputs; ++slot) if (slot < active) {
        result[slot] = lane ? 0.0f : qrt_q1_moe_hawkeye::value_to_float(
            qrt_sm121_group16::finish_accumulator(carry[slot]));
        if (statistics) statistics[slot] = stats[slot];
    }
}
} // namespace qrt_sm121_interleaved_projection
