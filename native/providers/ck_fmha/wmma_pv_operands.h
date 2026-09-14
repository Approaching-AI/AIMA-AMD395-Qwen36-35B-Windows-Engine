#pragma once
#include "blackwell_attention.h"

// Component-only lossless operand views for native PV. The ordinary probability
// tensor remains available to the original exact replay. No product dispatch
// includes this header.
namespace qrt_wmma_pv_operands {
using namespace qrt_blackwell_attention;
struct alignas(32) Row { MantissaBf16x16 words; };
static_assert(sizeof(Row) == 32u && alignof(Row) == 32u);

// [K16 group][KV head][column][16 original BF16 words]. Both groups of
// the final K32 tile are allocated and the out-of-history words are zero.
__global__ void prepare_value(const uint16_t* source, Row* output, unsigned tokens) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= ((tokens + 31u) / 32u * 2u) * kKvHeads * kHeadDim) return;
    const unsigned group = row / (kKvHeads * kHeadDim), feature = row % (kKvHeads * kHeadDim);
    Row value{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        const unsigned key = group * 16u + i;
        value.words[i] = key < tokens ? source[size_t(key) * kKvHeads * kHeadDim + feature] : 0u;
    }
    output[row] = value;
}

// Original one-wave online recurrence, with a second lossless destination:
// [K16 group][head][query capacity][16 probability words]. There is no
// separate probability transpose, exponent encoding or changed reduction.
__global__ void probability(const float* scores, uint16_t* probabilities, float* scales,
    Row* packed, unsigned query_start, unsigned score_stride, unsigned query_capacity,
    const unsigned char* exp2_table, bool vllm_sum) {
    const unsigned lane = threadIdx.x, query = blockIdx.y, head = blockIdx.x;
    const unsigned row = query * kQueryHeads + head, tokens = query_start + query + 1u;
    const unsigned tile_stride = (score_stride + 31u) / 32u, tile_count = (tokens + 31u) / 32u;
    float running_max = -INFINITY, running_sum = 1.0f;
    for (unsigned tile = 0u; tile < tile_count; ++tile) {
        const unsigned key = tile * 32u + lane;
        const float score = key < tokens ? scores[size_t(row) * score_stride + key] : -INFINITY;
        float next_max = fmaxf(running_max, score);
        for (unsigned mask = 16u; mask; mask >>= 1u)
            next_max = fmaxf(next_max, __shfl_xor(next_max, mask, 32u));
        const float alpha = blackwell_attention_exp(running_max - next_max, exp2_table);
        const float p = key < tokens ? blackwell_attention_exp(score - next_max, exp2_table) : 0.0f;
        const uint16_t word = f32_to_bf16(p);
        if (key < score_stride) probabilities[size_t(row) * score_stride + key] = word;
        // Every lane owns one distinct BF16 word, including the causal zeros
        // and the padded final K32 group outside score_stride.
        auto* words = reinterpret_cast<uint16_t*>(packed);
        words[((size_t(tile * 2u + lane / 16u) * kQueryHeads + head) * query_capacity + query) * 16u + lane % 16u] = word;
        float sum = p;
        if (vllm_sum) {
            constexpr unsigned order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
            for (unsigned step = 0u; step < 5u; ++step) sum += __shfl_xor(sum, order[step], 32u);
        } else {
            for (unsigned mask = 16u; mask; mask >>= 1u) sum += __shfl_xor(sum, mask, 32u);
        }
        running_sum = running_sum * alpha + sum;
        running_max = next_max;
        if (!lane) scales[size_t(row) * (tile_stride + 1u) + tile] = alpha;
    }
    if (!lane) scales[size_t(row) * (tile_stride + 1u) + tile_stride] = running_sum;
}

template<bool Full>
__device__ __forceinline__ void rescale(MantissaF32x8& accumulator, MantissaF32x8& errors,
    const float* scales, unsigned base, unsigned query_tile, unsigned query_start,
    unsigned query_count, unsigned head, unsigned tile_stride, unsigned lane) {
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) {
        const unsigned row = query_tile + 2u * element + lane / 16u;
        const unsigned tokens = query_start + row + 1u;
        if (Full || (row < query_count && base / 32u < (tokens + 31u) / 32u)) {
            const float alpha = scales[(size_t(row) * kQueryHeads + head) * (tile_stride + 1u) + base / 32u];
            errors[element] = qrt_sm121_pv_final_bound::rescale(errors[element], accumulator[element], alpha);
            volatile float rounded = accumulator[element] * alpha;
            accumulator[element] = rounded;
        }
    }
}

template<bool PackedProbability, bool Full>
__device__ __forceinline__ void group(MantissaF32x8& accumulator, MantissaF32x8& errors,
    const uint16_t* probabilities, const Row* packed_probability, const Row* packed_value,
    unsigned base, unsigned query_tile, unsigned query_start, unsigned query_count,
    unsigned query_capacity, unsigned score_stride, unsigned head, unsigned column_tile,
    unsigned last_tokens, unsigned lane, unsigned wave) {
    const unsigned input_row = query_tile + lane % 16u;
    const unsigned input_tokens = query_start + input_row + 1u;
    const unsigned column = column_tile + wave * 16u + lane % 16u;
    const unsigned kv_head = head / (kQueryHeads / kKvHeads);
    MantissaBf16x16 left{}, right{};
    if constexpr (PackedProbability) {
        if (Full || (input_row < query_count && base / 32u < (input_tokens + 31u) / 32u))
            left = packed_probability[(size_t(base / 16u) * kQueryHeads + head) * query_capacity + input_row].words;
    } else {
#pragma unroll
        for (unsigned i = 0u; i < 16u; ++i) {
            const unsigned key = base + i;
            left[i] = input_row < query_count && key < input_tokens
                ? probabilities[(size_t(input_row) * kQueryHeads + head) * score_stride + key] : 0u;
        }
    }
    // All K32 groups are allocated, but a query tile may stop earlier than
    // the complete V history. Preserve original V masking even for NaNs.
    right = packed_value[(size_t(base / 16u) * kKvHeads + kv_head) * kHeadDim + column].words;
    if constexpr (!Full) {
        if (base + 16u > last_tokens) {
#pragma unroll
            for (unsigned i = 0u; i < 16u; ++i) if (base + i >= last_tokens) right[i] = 0u;
        }
    }
    const auto next = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(left, right, accumulator);
    MantissaBf16x16 absolute_left{}, absolute_right{};
#pragma unroll
    for (unsigned i = 0u; i < 16u; ++i) {
        absolute_left[i] = left[i] & 0x7fffu;
        absolute_right[i] = right[i] & 0x7fffu;
    }
    const auto magnitudes = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(absolute_left, absolute_right, MantissaF32x8{});
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) {
        const unsigned row = query_tile + 2u * element + lane / 16u, tokens = query_start + row + 1u;
        if (Full || (row < query_count && base / 32u < (tokens + 31u) / 32u)) {
            errors[element] = qrt_sm121_pv_final_bound::group(errors[element], accumulator[element], magnitudes[element]);
            accumulator[element] = next[element];
        }
    }
}

template<bool PackedProbability, bool PairedK32 = false>
__global__ void value(const uint16_t* probabilities, const float* scales,
    const Row* packed_probability, const Row* packed_value, float* output,
    unsigned query_start, unsigned query_count, unsigned output_start,
    unsigned score_stride, unsigned query_capacity, const unsigned char* rcp_table,
    float* raw_accumulator, float* raw_denominator, float* error_bounds) {
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u, head = blockIdx.y;
    const unsigned column_tile = blockIdx.x * kIntegerMatrixColumns, query_tile = blockIdx.z * 16u;
    const unsigned tile_stride = (score_stride + 31u) / 32u;
    const unsigned last_tokens = query_start + min(query_tile + 16u, query_count);
    const unsigned end = (last_tokens + 31u) / 32u * 32u;
    MantissaF32x8 accumulator{}, errors{};
    if constexpr (PairedK32) {
        // Complete query tiles have a common causal prefix. Dispatch that
        // prefix uniformly per CTA and keep the original masked final tiles.
        const unsigned complete_end = query_tile + 16u <= query_count
            ? (query_start + query_tile + 1u) / 32u * 32u : 0u;
        for (unsigned base = 0u; base < end; base += 32u) {
#define QRT_PACKED_PV_PAIR(full) \
            rescale<full>(accumulator,errors,scales,base,query_tile,query_start,query_count,head,tile_stride,lane); \
            group<PackedProbability,full>(accumulator,errors,probabilities,packed_probability,packed_value,base,query_tile,query_start,query_count,query_capacity,score_stride,head,column_tile,last_tokens,lane,wave); \
            group<PackedProbability,full>(accumulator,errors,probabilities,packed_probability,packed_value,base+16u,query_tile,query_start,query_count,query_capacity,score_stride,head,column_tile,last_tokens,lane,wave)
            if (base < complete_end) { QRT_PACKED_PV_PAIR(true); }
            else { QRT_PACKED_PV_PAIR(false); }
#undef QRT_PACKED_PV_PAIR
        }
    } else {
        for (unsigned base = 0u; base < end; base += 16u) {
            if (!(base % 32u)) rescale<false>(accumulator,errors,scales,base,query_tile,query_start,query_count,head,tile_stride,lane);
            group<PackedProbability,false>(accumulator,errors,probabilities,packed_probability,packed_value,base,query_tile,query_start,query_count,query_capacity,score_stride,head,column_tile,last_tokens,lane,wave);
        }
    }
#pragma unroll
    for (unsigned element = 0u; element < 8u; ++element) {
        const unsigned row = query_tile + 2u * element + lane / 16u;
        const unsigned column = column_tile + wave * 16u + lane % 16u;
        if (row < query_count) {
            const float denominator = scales[(size_t(row) * kQueryHeads + head) * (tile_stride + 1u) + tile_stride];
            const size_t index = (size_t(output_start + row) * kQueryHeads + head) * kHeadDim + column;
            output[index] = rcp_table ? accumulator[element] * qrt_sm121_attention_rcp::evaluate(rcp_table,denominator) : accumulator[element] / denominator;
            errors[element] = qrt_sm121_pv_final_bound::finalize(errors[element],((query_start + row + 32u) / 32u) * 2u);
            error_bounds[(size_t(row) * kQueryHeads + head) * kHeadDim + column] = qrt_sm121_pv_bound::finish(errors[element],accumulator[element],qrt_sm121_attention_rcp::evaluate(rcp_table,denominator));
            if (raw_accumulator) raw_accumulator[index] = accumulator[element];
            if (raw_denominator && !column) raw_denominator[size_t(output_start + row) * kQueryHeads + head] = denominator;
        }
    }
}
} // namespace qrt_wmma_pv_operands
