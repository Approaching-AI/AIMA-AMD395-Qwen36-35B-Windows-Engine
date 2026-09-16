#pragma once
#include "selective_qk.h"
#include "prepared_decoded_qk.h"

// Isolated component route. Original QK/softmax/PV comparisons and external
// GB10 contexts belong to the test caller and never enter these kernels.
namespace qrt_adaptive_denominator_qk {
using namespace qrt_blackwell_attention;
namespace original = qrt_selective_qk;
namespace prepared = qrt_prepared_decoded_qk;
constexpr unsigned rounds = 18u;

// One independent original dot per thread. Packed operands carry only the
// original BF16 values and decoded exponents. Any rejected row or intermediate
// carry restarts the entire original dot, as in the retained prepared kernel.
template<bool UpdateProbability>
__global__ void repair(const uint16_t* query, const uint16_t* transposed_key,
    const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags,
    float* scores, float* errors, unsigned start, unsigned stride, unsigned key_stride,
    const unsigned* indices, const unsigned* count, const float* maxima,
    float* lower, float* upper, const unsigned char* exp2) {
    const unsigned tiles = (stride + 31u) / 32u;
    for (unsigned slot = blockIdx.x * blockDim.x + threadIdx.x; slot < *count;
         slot += gridDim.x * blockDim.x) {
        const unsigned cell = indices[slot], row = cell / stride, key = cell % stride;
        const unsigned head = row % kQueryHeads, token = start + row / kQueryHeads;
        const unsigned kv_head = head / (kQueryHeads / kKvHeads);
        const size_t query_base = (size_t(token) * kQueryHeads + head) * kHeadDim;
        const size_t key_base = size_t(kv_head) * kHeadDim * key_stride + key;
        bool fallback = !query_flags[token * kQueryHeads + head] ||
            !key_flags[key * kKvHeads + kv_head];
        float carry = 0.0f;
        if (!fallback) {
            for (unsigned base = 0u; base < kHeadDim; base += 16u) {
                qrt_sm121_float_alignment::Group group;
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    prepared::decoded::set_packed(group, i,
                        packed_query[query_base + base + i],
                        packed_key[key_base + size_t(base + i) * key_stride]);
                float next;
                if (!qrt_sm121_f32_carry::accumulate<0u>(carry, group, &next)) {
                    fallback = true; break;
                }
                carry = next;
            }
        }
        const float score = fallback ? qrt_decoded_window_qk::raw_dot(
            query + query_base, transposed_key + key_base, key_stride) : carry * kExactScale;
        scores[cell] = score; errors[cell] = 0.0f;
        if constexpr (UpdateProbability) {
            const float maximum = maxima[size_t(row) * tiles + key / 32u];
            const float p = blackwell_attention_exp(fminf(score, maximum) - maximum, exp2);
            lower[cell] = p; upper[cell] = p;
        }
    }
}

// Prefix maxima and BF16 probabilities have already received the original
// repairs. Cache their unrounded probability intervals once. Subsequent score
// replay replaces only selected intervals with their exact scalar probability.
__global__ void initialize(const float* scores, const float* errors, const float* maxima,
    uint16_t* probabilities, float* scales, float* lower, float* upper,
    float* maximum_cost, unsigned* pending, unsigned start, unsigned stride,
    const unsigned char* exp2) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned tokens = start + blockIdx.y + 1u, tiles = (stride + 31u) / 32u;
    float maximum = -INFINITY, center = 1.0f, cost = 0.0f;
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        const float next = maxima[size_t(row) * tiles + tile];
        const float alpha = blackwell_attention_exp(maximum - next, exp2);
        const float value = key < tokens ? blackwell_attention_exp(fminf(scores[cell], next) - next, exp2) : 0.0f;
        const auto interval = key < tokens ? original::probability_interval(scores[cell], errors[cell], next, exp2)
                                          : original::Interval{0.0f, 0.0f};
        if (key < stride) {
            probabilities[cell] = f32_to_bf16(value);
            lower[cell] = interval.low; upper[cell] = interval.high;
        }
        const float width = interval.high - interval.low;
        cost = fmaxf(cost, isfinite(width) && width >= 0.0f ? width : INFINITY);
        center = center * alpha + original::wave_sum(value);
        if (!lane) scales[size_t(row) * (tiles + 1u) + tile] = alpha;
        maximum = next;
    }
    cost = original::wave_maximum(cost);
    if (!lane) {
        scales[size_t(row) * (tiles + 1u) + tiles] = center;
        maximum_cost[row] = cost; pending[row] = 1u;
    }
}

// One original 32-lane butterfly and ordered recurrence per row. Every
// normalized output must have a fixed BF16 endpoint before a row is admitted.
// Priority affects work only. Callers admit step1 or step4; the penultimate
// round selects all remaining scores and the final round must therefore see
// an exact denominator. No approximate denominator is an
// acceptance condition. A fixed denominator uses the exact reciprocal table.
__global__ void certify_and_collect(const float* errors, const float* lower,
    const float* upper, const float* accumulator, float* scales, float* denominators,
    float* output, const float* maximum_cost, unsigned* pending,
    unsigned start, unsigned stride, const unsigned char* reciprocal,
    unsigned round, unsigned step, unsigned* indices, unsigned* count, unsigned* stats) {
    const unsigned lane = threadIdx.x, row = blockIdx.y * kQueryHeads + blockIdx.x;
    if (!pending[row]) return;
    const unsigned last_round = 16u / step + 1u;
    const unsigned tokens = start + blockIdx.y + 1u, tiles = (stride + 31u) / 32u;
    float low = 1.0f, high = 1.0f;
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        const float alpha = scales[size_t(row) * (tiles + 1u) + tile];
        const float a = original::wave_sum(key < tokens ? lower[cell] : 0.0f);
        const float b = original::wave_sum(key < tokens ? upper[cell] : 0.0f);
        low = low * alpha + a; high = high * alpha + b;
    }
    const auto interval = original::reciprocal_interval({low, high}, reciprocal);
    const float center = qrt_sm121_attention_rcp::evaluate(reciprocal, low);
    bool stable = true;
    for (unsigned d = lane; d < kHeadDim; d += 32u) {
        const size_t cell = size_t(row) * kHeadDim + d;
        const float carry = accumulator[cell], a = carry * interval.low, b = carry * interval.high;
        stable = stable && isfinite(a) && isfinite(b) && f32_to_bf16(a) == f32_to_bf16(b);
        output[cell] = carry * center;
    }
    const bool all_stable = __ballot(!stable) == 0u;
    if (!lane) {
        denominators[size_t(row) * 2u] = low; denominators[size_t(row) * 2u + 1u] = high;
        scales[size_t(row) * (tiles + 1u) + tiles] = low;
        atomicAdd(stats + round, 1u);
        if (all_stable) { pending[row] = 0u; atomicAdd(stats + rounds, 1u); }
        else if (round == last_round - 1u) atomicAdd(stats + rounds + 1u, 1u);
    }
    if (all_stable || round == last_round) return;
    const float threshold = maximum_cost[row] * ldexpf(1.0f, -int((round + 1u) * step - 1u));
    for (unsigned tile = 0u; tile < (tokens + 31u) / 32u; ++tile) {
        const unsigned key = tile * 32u + lane, cell = row * stride + key;
        bool selected = key < tokens && errors[cell] != 0.0f;
        if (selected && round != last_round - 1u) {
            const float width = upper[cell] - lower[cell];
            selected = !isfinite(width) || (width > 0.0f && width >= threshold);
        }
        original::collect(cell, selected, indices, count);
    }
}
} // namespace qrt_adaptive_denominator_qk
