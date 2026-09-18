#pragma once
#include "streamed_exact_attention.h"

// Isolated ownership variants of the retained probability/native-PV consumer.
// QK remains independently parallel. Every query keeps the original K32
// softmax recurrence and ordered K16 PV updates, including its 16-row V mask.
// The runtime does not dispatch these variants.
namespace qrt_redistributed_probability_pv {
namespace attention = qrt_blackwell_attention;
namespace bound = qrt_sm121_pv_final_bound;

template<unsigned QueryRows, unsigned Threads>
__global__ __launch_bounds__(Threads) void produce(const float* source_scores, const uint16_t* value,
    uint16_t* probabilities, float* scales, float* output, float* errors,
    float* raw_accumulator, float* raw_denominator,
    unsigned start, unsigned count, unsigned stride,
    const unsigned char* exp2_table, const unsigned char* packed_exp,
    const unsigned char* rcp_table) {
    static_assert(QueryRows == 16u || QueryRows == 32u);
    static_assert(Threads == 256u || Threads == 512u);
    constexpr unsigned Waves = Threads / 32u;
    constexpr unsigned QueryFragments = QueryRows / 16u;
    constexpr unsigned ColumnFragments = 16u / Waves;
    constexpr unsigned SoftmaxRows = QueryRows / Waves;
    static_assert(attention::kHeadDim == 256u && attention::kQueryHeads == 16u && attention::kKvHeads == 2u);
    __shared__ float alpha[QueryRows], denominator[QueryRows];
    __shared__ uint16_t probability[QueryRows][32];
    const unsigned lane = threadIdx.x % 32u, wave = threadIdx.x / 32u;
    const unsigned head = blockIdx.x, kv_head = head / 8u, row_tile = blockIdx.y * QueryRows;
    const unsigned last_tokens = start + min(row_tile + QueryRows, count);
    const unsigned tiles = (last_tokens + 31u) / 32u, tile_stride = (stride + 31u) / 32u;
    float running_max[SoftmaxRows], running_sum[SoftmaxRows];
#pragma unroll
    for (unsigned r = 0u; r < SoftmaxRows; ++r) {
        running_max[r] = -INFINITY;
        running_sum[r] = 1.0f;
    }
    attention::MantissaF32x8 accumulator[QueryFragments][ColumnFragments]{};
    attention::MantissaF32x8 error[QueryFragments][ColumnFragments]{};
    constexpr bool vllm_sum = true;
    for (unsigned tile = 0u; tile < tiles; ++tile) {
        const unsigned key_base = tile * 32u;
#pragma unroll
        for (unsigned r = 0u; r < SoftmaxRows; ++r) {
            const unsigned local_row = wave + r * Waves, row = row_tile + local_row;
            const unsigned tokens = start + row + 1u, key = key_base + lane;
            const float score = row < count && key < tokens
                ? source_scores[(size_t(row) * 16u + head) * stride + key] : -INFINITY;
            float p = 0.0f;
            if (row < count && tile < (tokens + 31u) / 32u) {
                float next_max = fmaxf(running_max[r], score);
                for (unsigned mask = 16u; mask; mask >>= 1u)
                    next_max = fmaxf(next_max, __shfl_xor(next_max, mask, 32u));
                const float a = qrt_streamed_exact_attention::NativeExp::evaluate(exp2_table, packed_exp,
                    (running_max[r] - next_max) * attention::kExactLog2e);
                p = key < tokens ? qrt_streamed_exact_attention::NativeExp::evaluate(exp2_table, packed_exp,
                    (score - next_max) * attention::kExactLog2e) : 0.0f;
                if (key < stride) probabilities[(size_t(row) * 16u + head) * stride + key] = attention::f32_to_bf16(p);
                float sum = p;
                if (vllm_sum) {
                    constexpr unsigned order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
                    for (unsigned step = 0u; step < 5u; ++step) sum += __shfl_xor(sum, order[step], 32u);
                } else {
                    for (unsigned mask = 16u; mask; mask >>= 1u) sum += __shfl_xor(sum, mask, 32u);
                }
                running_sum[r] = running_sum[r] * a + sum;
                running_max[r] = next_max;
                if (!lane) {
                    alpha[local_row] = a;
                    denominator[local_row] = running_sum[r];
                    scales[(size_t(row) * 16u + head) * (tile_stride + 1u) + tile] = a;
                }
            }
            probability[local_row][lane] = attention::f32_to_bf16(p);
        }
        __syncthreads();
        for (unsigned part = 0u; part < 2u; ++part) {
#pragma unroll
            for (unsigned q = 0u; q < QueryFragments; ++q) {
                attention::NativeOperandRow left{};
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    left.original[i] = probability[q * 16u + lane % 16u][part * 16u + i];
#pragma unroll
                for (unsigned c = 0u; c < ColumnFragments; ++c) {
                    attention::NativeOperandRow right{};
                    const unsigned column = (wave + c * Waves) * 16u + lane % 16u;
                    // Match the separate 16-query PV tile's value mask exactly.
                    const unsigned pv_last = start + min(row_tile + q * 16u + 16u, count);
#pragma unroll
                    for (unsigned i = 0u; i < 16u; ++i) {
                        const unsigned key = key_base + part * 16u + i;
                        right.original[i] = key < pv_last
                            ? value[(size_t(key) * 2u + kv_head) * 256u + column] : 0u;
                    }
#pragma unroll
                    for (unsigned e = 0u; e < 8u; ++e) {
                        const unsigned local_row = q * 16u + 2u * e + lane / 16u, row = row_tile + local_row;
                        if (!part && row < count && tile < (start + row + 32u) / 32u) {
                            error[q][c][e] = bound::rescale(error[q][c][e], accumulator[q][c][e], alpha[local_row]);
                            accumulator[q][c][e] = bound::multiply(accumulator[q][c][e], alpha[local_row]);
                        }
                    }
                    const auto next = attention::blackwell_native_mma(left, right, accumulator[q][c]);
                    const auto magnitudes = attention::blackwell_native_mma<true>(left, right, attention::MantissaF32x8{});
#pragma unroll
                    for (unsigned e = 0u; e < 8u; ++e) {
                        const unsigned row = row_tile + q * 16u + 2u * e + lane / 16u;
                        if (row < count && tile < (start + row + 32u) / 32u) {
                            error[q][c][e] = bound::group(error[q][c][e], accumulator[q][c][e], magnitudes[e]);
                            accumulator[q][c][e] = next[e];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (unsigned q = 0u; q < QueryFragments; ++q) {
#pragma unroll
        for (unsigned c = 0u; c < ColumnFragments; ++c) {
#pragma unroll
            for (unsigned e = 0u; e < 8u; ++e) {
                const unsigned local_row = q * 16u + 2u * e + lane / 16u, row = row_tile + local_row;
                const unsigned column = (wave + c * Waves) * 16u + lane % 16u;
                if (row < count) {
                    const size_t cell = (size_t(row) * 16u + head) * 256u + column;
                    const float reciprocal = qrt_sm121_attention_rcp::evaluate(rcp_table, denominator[local_row]);
                    output[cell] = accumulator[q][c][e] * reciprocal;
                    const float final_error = bound::finalize(error[q][c][e], ((start + row + 32u) / 32u) * 2u);
                    errors[cell] = qrt_sm121_pv_bound::finish(final_error, accumulator[q][c][e], reciprocal);
                    if (raw_accumulator) raw_accumulator[cell] = accumulator[q][c][e];
                    if (!column) {
                        scales[(size_t(row) * 16u + head) * (tile_stride + 1u) + tile_stride] = denominator[local_row];
                        if (raw_denominator) raw_denominator[size_t(row) * 16u + head] = denominator[local_row];
                    }
                }
            }
        }
    }
}
} // namespace qrt_redistributed_probability_pv
