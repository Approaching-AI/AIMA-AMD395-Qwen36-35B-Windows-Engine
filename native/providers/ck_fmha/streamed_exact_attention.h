#pragma once
#include "microtile_exact_qk.h"
#include "../gdn/sm121_exp2_native_delta.h"

// Isolated complete producer. Keep 32 decoded query rows resident, consume
// each original K32 tile in order, and retain native PV accumulators and their
// original error bounds. P and alpha remain available for the unchanged exact
// PV replay. No runtime dispatch uses this header.
namespace qrt_streamed_exact_attention {
using namespace qrt_blackwell_attention;
namespace decoded = qrt_sm121_decoded_bf16;
namespace bound = qrt_sm121_pv_final_bound;

__global__ void produce(const uint16_t* query, const uint16_t* transposed_key,
    const uint16_t* value, const uint32_t* packed_query, const uint32_t* packed_key,
    const unsigned* query_flags, const unsigned* key_flags,
    uint16_t* probabilities, float* scales, float* output, float* errors,
    float* raw_accumulator, float* raw_denominator, float* diagnostic_scores,
    unsigned start, unsigned count, unsigned stride, unsigned key_stride,
    const unsigned char* exp2_table, const unsigned char* packed_exp,
    const unsigned char* rcp_table, bool vllm_sum) {
    static_assert(kHeadDim == 256u && kQueryHeads == 16u && kKvHeads == 2u);
    __shared__ uint32_t qvalues[32][256], kvalues[128][32];
    __shared__ float scores[32][32], alpha[32], denominator[32];
    __shared__ uint16_t probability[32][32];
    const unsigned thread = threadIdx.x, lane = thread % 32u, wave = thread / 32u;
    const unsigned qr = thread / 16u, kc = thread % 16u;
    const unsigned head = blockIdx.x, kv_head = head / 8u, row_tile = blockIdx.y * 32u;
    const unsigned last_tokens = start + min(row_tile + 32u, count);
    const unsigned tiles = (last_tokens + 31u) / 32u, tile_stride = (stride + 31u) / 32u;
    for (unsigned cell = thread; cell < 32u * 256u; cell += 256u) {
        const unsigned row = cell / 256u, feature = cell % 256u;
        qvalues[row][feature] = row_tile + row < count
            ? packed_query[(size_t(start + row_tile + row) * 16u + head) * 256u + feature]
            : decoded::pack(0u);
    }
    __syncthreads();
    // Four independent query rows per wave for the original one-wave softmax.
    float running_max[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    float running_sum[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    // Each wave owns two 16-column tiles for both 16-query halves.
    MantissaF32x8 accumulator[2][2]{}, error[2][2]{};
    for (unsigned tile = 0u; tile < tiles; ++tile) {
        const unsigned key_base = tile * 32u;
        float carry[2][2]{};
        bool active[2][2], fallback[2][2];
#pragma unroll
        for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
            for (unsigned k = 0u; k < 2u; ++k) {
                const unsigned row = row_tile + qr + q * 16u, key = key_base + kc + k * 16u;
                active[q][k] = row < count && key < stride && key <= start + row;
                fallback[q][k] = active[q][k] &&
                    (!query_flags[(start + row) * 16u + head] || !key_flags[key * 2u + kv_head]);
            }
        }
        for (unsigned window = 0u; window < 256u; window += 128u) {
            for (unsigned cell = thread; cell < 128u * 32u; cell += 256u) {
                const unsigned feature = cell / 32u, column = cell % 32u;
                kvalues[feature][column] = key_base + column < stride
                    ? packed_key[(size_t(kv_head) * 256u + window + feature) * key_stride + key_base + column]
                    : decoded::pack(0u);
            }
            __syncthreads();
            for (unsigned base = 0u; base < 128u; base += 16u) {
#pragma unroll
                for (unsigned q = 0u; q < 2u; ++q) {
                    if ((active[q][0] && !fallback[q][0]) || (active[q][1] && !fallback[q][1])) {
                        qrt_sm121_float_alignment::Group groups[2];
#pragma unroll
                        for (unsigned i = 0u; i < 16u; ++i) {
                            const uint32_t left = qvalues[qr + q * 16u][window + base + i];
#pragma unroll
                            for (unsigned k = 0u; k < 2u; ++k)
                                decoded::set_packed(groups[k], i, left, kvalues[base + i][kc + k * 16u]);
                        }
#pragma unroll
                        for (unsigned k = 0u; k < 2u; ++k) if (active[q][k] && !fallback[q][k]) {
                            float next;
                            if (qrt_sm121_f32_carry::accumulate<0u>(carry[q][k], groups[k], &next))
                                carry[q][k] = next;
                            else fallback[q][k] = true;
                        }
                    }
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
            for (unsigned k = 0u; k < 2u; ++k) {
                const unsigned local_row = qr + q * 16u, row = row_tile + local_row;
                const unsigned column = kc + k * 16u, key = key_base + column;
                const float score = !active[q][k] ? -INFINITY : fallback[q][k]
                    ? qrt_decoded_window_qk::raw_dot(query + (size_t(start + row) * 16u + head) * 256u,
                        transposed_key + size_t(kv_head) * 256u * key_stride + key, key_stride)
                    : carry[q][k] * kExactScale;
                scores[local_row][column] = score;
                if (diagnostic_scores && row < count && key < stride)
                    diagnostic_scores[(size_t(row) * 16u + head) * stride + key] = score;
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned r = 0u; r < 4u; ++r) {
            const unsigned local_row = wave + r * 8u, row = row_tile + local_row;
            const unsigned tokens = start + row + 1u, key = key_base + lane;
            float p = 0.0f;
            if (row < count && tile < (tokens + 31u) / 32u) {
                const float score = scores[local_row][lane];
                float next_max = fmaxf(running_max[r], score);
                for (unsigned mask = 16u; mask; mask >>= 1u)
                    next_max = fmaxf(next_max, __shfl_xor(next_max, mask, 32u));
                const float a = qrt_sm121_exp2_native_delta::evaluate(exp2_table, packed_exp,
                    (running_max[r] - next_max) * kExactLog2e);
                p = key < tokens ? qrt_sm121_exp2_native_delta::evaluate(exp2_table, packed_exp,
                    (score - next_max) * kExactLog2e) : 0.0f;
                if (key < stride) probabilities[(size_t(row) * 16u + head) * stride + key] = f32_to_bf16(p);
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
            probability[local_row][lane] = f32_to_bf16(p);
        }
        __syncthreads();
        for (unsigned part = 0u; part < 2u; ++part) {
#pragma unroll
            for (unsigned q = 0u; q < 2u; ++q) {
                NativeOperandRow left{};
#pragma unroll
                for (unsigned i = 0u; i < 16u; ++i)
                    left.original[i] = probability[q * 16u + lane % 16u][part * 16u + i];
#pragma unroll
                for (unsigned c = 0u; c < 2u; ++c) {
                    NativeOperandRow right{};
                    const unsigned column = (wave + c * 8u) * 16u + lane % 16u;
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
                    const auto next = blackwell_native_mma(left, right, accumulator[q][c]);
                    const auto magnitudes = blackwell_native_mma<true>(left, right, MantissaF32x8{});
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
    for (unsigned q = 0u; q < 2u; ++q) {
#pragma unroll
        for (unsigned c = 0u; c < 2u; ++c) {
#pragma unroll
            for (unsigned e = 0u; e < 8u; ++e) {
                const unsigned local_row = q * 16u + 2u * e + lane / 16u, row = row_tile + local_row;
                const unsigned column = (wave + c * 8u) * 16u + lane % 16u;
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
    if (diagnostic_scores) for (unsigned cell = thread; cell < 32u * stride; cell += 256u) {
        const unsigned row = row_tile + cell / stride, key = cell % stride;
        if (row < count && key >= tiles * 32u)
            diagnostic_scores[(size_t(row) * 16u + head) * stride + key] = -INFINITY;
    }
}
} // namespace qrt_streamed_exact_attention
