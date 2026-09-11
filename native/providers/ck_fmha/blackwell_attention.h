#ifndef QRT_BLACKWELL_ATTENTION_H
#define QRT_BLACKWELL_ATTENTION_H
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include "../moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include "../moe_accumulator/sm121_wave16.h"
#include "../gdn/sm121_exp2_table.h"
#include "../gdn/sm121_attention_rcp.h"
namespace qrt_blackwell_attention {
constexpr unsigned int kQueryHeads = 16u, kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u, kThreads = 256u;
__device__ uint16_t f32_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        if ((bits & 0x007fffffu) != 0u) {
            upper |= 0x0040u;
        }
        return upper;
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

constexpr unsigned int kExactTileTokens = 32u;
constexpr float kExactScale = 0.0625f;
constexpr float kExactLog2e = 1.4426950408889634074f;
constexpr unsigned int kBlackwellMmaGroup = 16u;
constexpr unsigned int kBlackwellSubgroups =
    kThreads / kBlackwellMmaGroup;
constexpr int16_t kBlackwellZeroExponent = -133;
static_assert(kExactTileTokens % kBlackwellSubgroups == 0u);
static_assert(kHeadDim % kBlackwellSubgroups == 0u);

__device__ __forceinline__ float blackwell_attention_exp(float value, const unsigned char* exp2_table) {
    const float argument = value * kExactLog2e;
    return exp2_table ? qrt_sm121_exp2::evaluate(exp2_table, argument)
                      : exp2f(argument);
}

// accumulate_bf16_impl ends with group_sum<26, -133> of the carried value.
// The shared wave16 normalizer has already produced a canonical FP32 value:
// a 24-bit normal significand, a subnormal at exponent -126, or zero.
// Its final one-value group only needs to clear any underflowed negative zero;
// it does not need to repeat integer normalization after QK or each PV K16.

// Materialize a bounded query slab of exact QK scores. Independent key cells
// expose parallelism across the whole causal history instead of waiting for
// each query CTA to finish its preceding PV tile. Online softmax/PV below still
// visits every tile in the original order and consumes the same FP32 scores.
__global__ void blackwell_exact_scores_kernel(
    const uint16_t* __restrict__ query, const uint16_t* __restrict__ key,
    float* __restrict__ scores, unsigned int query_start,
    unsigned int query_count, unsigned int score_stride) {
    const unsigned int cell = blockIdx.x * kBlackwellSubgroups +
        threadIdx.x / kBlackwellMmaGroup;
    const unsigned int cells = query_count * kQueryHeads * score_stride;
    if (cell >= cells) return;
    const unsigned int key_token = cell % score_stride;
    const unsigned int query_head = (cell / score_stride) % kQueryHeads;
    const unsigned int token = query_start + cell / (score_stride * kQueryHeads);
    const unsigned int lane = threadIdx.x % kBlackwellMmaGroup;
    if (key_token > token) {
        if (lane == 0u) scores[cell] = -INFINITY;
        return;
    }
    const size_t query_base = (static_cast<size_t>(token) * kQueryHeads + query_head) * kHeadDim;
    const unsigned int kv_head = query_head / (kQueryHeads / kKvHeads);
    const size_t key_base = (static_cast<size_t>(key_token) * kKvHeads + kv_head) * kHeadDim;
    qrt_q1_moe_hawkeye::Value dot{0u, kBlackwellZeroExponent, false};
    for (unsigned int base = 0u; base < kHeadDim; base += kBlackwellMmaGroup) {
        dot = qrt_sm121_wave16::accumulate(dot,
            query[query_base + base + lane], key[key_base + base + lane], lane);
    }
    if (lane == 0u) {
        scores[cell] = qrt_q1_moe_hawkeye::value_to_float(
            qrt_sm121_group16::finish_accumulator(dot)) * kExactScale;
    }
}

// A single bounded transpose makes adjacent independent QK lanes read adjacent
// keys. Each lane keeps its own K-continuous carry and sixteen packed products;
// no wave reduction is needed for the dot or its maximum exponent.
__global__ void blackwell_transpose_keys_kernel(
    const uint16_t* key, uint16_t* transposed, unsigned int tokens) {
    __shared__ uint16_t tile[32][33];
    const unsigned int column = blockIdx.x * 32u + threadIdx.x;
    const unsigned int row = blockIdx.y * 32u + threadIdx.y;
#pragma unroll
    for (unsigned int part = 0u; part < 32u; part += 8u) {
        if (row + part < tokens)
            tile[threadIdx.y + part][threadIdx.x] =
                key[static_cast<size_t>(row + part) * (kKvHeads * kHeadDim) + column];
    }
    __syncthreads();
    const unsigned int key_token = blockIdx.y * 32u + threadIdx.x;
    const unsigned int key_column = blockIdx.x * 32u + threadIdx.y;
#pragma unroll
    for (unsigned int part = 0u; part < 32u; part += 8u) {
        if (key_token < tokens)
            transposed[static_cast<size_t>(key_column + part) * tokens + key_token] =
                tile[threadIdx.x][threadIdx.y + part];
    }
}

__global__ void blackwell_transposed_scores_kernel(
    const uint16_t* __restrict__ query, const uint16_t* __restrict__ transposed_key,
    float* __restrict__ scores, unsigned int query_start,
    unsigned int query_count, unsigned int score_stride, unsigned int key_stride) {
    const unsigned int cell = blockIdx.x * kThreads + threadIdx.x;
    const unsigned int cells = query_count * kQueryHeads * score_stride;
    if (cell >= cells) return;
    const unsigned int key_token = cell % score_stride;
    const unsigned int head = (cell / score_stride) % kQueryHeads;
    const unsigned int token = query_start + cell / (score_stride * kQueryHeads);
    if (key_token > token) { scores[cell] = -INFINITY; return; }
    const size_t query_base = (static_cast<size_t>(token) * kQueryHeads + head) * kHeadDim;
    const unsigned int kv_head = head / (kQueryHeads / kKvHeads);
    qrt_q1_moe_hawkeye::Value dot{0u, kBlackwellZeroExponent, false};
    for (unsigned int base = 0u; base < kHeadDim; base += kBlackwellMmaGroup) {
        uint32_t products[kBlackwellMmaGroup];
#pragma unroll
        for (unsigned int item = 0u; item < kBlackwellMmaGroup; ++item) {
            products[item] = qrt_sm121_group16::pack_product(
                qrt_q1_moe_hawkeye::multiply_bf16(
                    query[query_base + base + item],
                    transposed_key[(static_cast<size_t>(kv_head) * kHeadDim + base + item) *
                                       key_stride + key_token], kBlackwellZeroExponent));
        }
        const auto sum = qrt_sm121_group16::sum_packed(dot, products);
        dot = qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
    }
    scores[cell] = qrt_q1_moe_hawkeye::value_to_float(
        qrt_sm121_group16::finish_accumulator(dot)) * kExactScale;
}

// One CTA owns one causal query/head. Both terminal replacement and bounded
// prefix replay share the same Blackwell QK and fused-C PV arithmetic.
template <bool SerialValue, bool PrecomputedScores = false, bool SplitDecodeValue = false>
__global__ void blackwell_exact_attention_kernel(
    const uint16_t *__restrict__ query,
    const uint16_t *__restrict__ key,
    const uint16_t *__restrict__ value,
    float *__restrict__ output,
    unsigned int query_start,
    unsigned int output_start,
    const unsigned char* exp2_table,
    float* raw_accumulator,
    float* raw_denominator,
    bool vllm_sum,
    const unsigned char* rcp_table,
    const float* precomputed_scores,
    unsigned int score_stride,
    const uint16_t* decode_tail_value,
    unsigned int decode_prefix_tokens) {
    static_assert(!SplitDecodeValue || (SerialValue && PrecomputedScores),
                  "split decode V requires precomputed QK and the serial-value layout");
    __shared__ float score[kExactTileTokens];
    __shared__ float probability[kExactTileTokens];
    __shared__ float sum_scratch[kExactTileTokens];
    __shared__ uint16_t probability_bf16[kExactTileTokens];
    __shared__ float output_accumulator[kHeadDim];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float tile_max;
    __shared__ float alpha;

    const unsigned int query_head = blockIdx.x;
    const unsigned int kv_head = query_head / (kQueryHeads / kKvHeads);
    const unsigned int token = query_start + blockIdx.y;
    const unsigned int tokens = token + 1u;
    const unsigned int output_token = output_start + blockIdx.y;
    const size_t query_base =
        (static_cast<size_t>(token) * kQueryHeads + query_head) * kHeadDim;
    const size_t output_base =
        (static_cast<size_t>(output_token) * kQueryHeads + query_head) *
        kHeadDim;
    const unsigned int thread = threadIdx.x;
    const unsigned int subgroup_lane = thread % kBlackwellMmaGroup;
    const unsigned int subgroup = thread / kBlackwellMmaGroup;
    output_accumulator[thread] = 0.0f;
    if (thread == 0u) {
        running_max = -INFINITY;
        running_sum = 1.0f;
    }
    __syncthreads();

    const unsigned int tile_count =
        (tokens + kExactTileTokens - 1u) / kExactTileTokens;
    for (unsigned int tile = 0u; tile < tile_count; ++tile) {
        if constexpr (PrecomputedScores) {
            if (thread < kExactTileTokens) {
                const unsigned int key_token = tile * kExactTileTokens + thread;
                score[thread] = key_token < tokens
                    ? precomputed_scores[(blockIdx.y * kQueryHeads + query_head) *
                                             score_stride + key_token]
                    : -INFINITY;
            }
        } else {
            for (unsigned int key_batch = 0u;
                 key_batch < kExactTileTokens / kBlackwellSubgroups;
                 ++key_batch) {
                const unsigned int key_item =
                    key_batch * kBlackwellSubgroups + subgroup;
                const unsigned int key_token =
                    tile * kExactTileTokens + key_item;
                const bool key_valid = key_token < tokens;
                qrt_q1_moe_hawkeye::Value dot{
                    0u,
                    kBlackwellZeroExponent,
                    false
                };
                for (unsigned int group = 0u;
                     group < kHeadDim / kBlackwellMmaGroup;
                     ++group) {
                    const unsigned int element =
                        group * kBlackwellMmaGroup + subgroup_lane;
                    uint16_t query_value = 0u;
                    uint16_t key_value = 0u;
                    if (key_valid) {
                        const size_t key_base =
                            (static_cast<size_t>(key_token) * kKvHeads + kv_head) *
                            kHeadDim;
                        query_value = query[query_base + element];
                        key_value = key[key_base + element];
                    }
                    dot = qrt_sm121_wave16::accumulate(
                        dot,
                        query_value,
                        key_value,
                        subgroup_lane
                    );
                }
                dot = qrt_sm121_group16::finish_accumulator(dot);
                if (subgroup_lane == 0u) {
                    score[key_item] = key_valid
                        ? qrt_q1_moe_hawkeye::value_to_float(dot) * kExactScale
                        : -INFINITY;
                }
            }
        }
        __syncthreads();

        if (thread == 0u) {
            float next_max = running_max;
#pragma unroll
            for (unsigned int item = 0u; item < kExactTileTokens; ++item) {
                next_max = fmaxf(next_max, score[item]);
            }
            tile_max = next_max;
            alpha = blackwell_attention_exp(running_max - next_max, exp2_table);
        }
        __syncthreads();

        if (thread < kExactTileTokens) {
            const unsigned int key_token = tile * kExactTileTokens + thread;
            const float p = key_token < tokens
                ? blackwell_attention_exp(score[thread] - tile_max, exp2_table)
                : 0.0f;
            probability[thread] = p;
            probability_bf16[thread] = f32_to_bf16(p);
            sum_scratch[thread] = p;
        }
        __syncthreads();

        if (vllm_sum) {
            // The original SM121 16x32 MMA layout first combines adjacent
            // columns within a thread, then lanes (4,2) and warps (16,8).
            // A generic 16,8,4,2,1 sum changes normalization at BF16 midpoints.
            constexpr unsigned int order[] = {1u, 4u, 2u, 16u, 8u};
            unsigned int reduced_bits = 0u;
#pragma unroll
            for (unsigned int step = 0u; step < 5u; ++step) {
                const unsigned int stride = order[step];
                reduced_bits |= stride;
                if (thread < kExactTileTokens && (thread & reduced_bits) == 0u)
                    sum_scratch[thread] += sum_scratch[thread + stride];
                __syncthreads();
            }
        } else {
            for (unsigned int stride = kExactTileTokens / 2u; stride > 0u;
                 stride >>= 1u) {
                if (thread < stride)
                    sum_scratch[thread] += sum_scratch[thread + stride];
                __syncthreads();
            }
        }

        if constexpr (SerialValue) {
            // A lane owns one output dimension. Adjacent lanes load adjacent
            // V values, and all 256 independent accumulators advance together.
            // Packed K16 products retain the exact max-exponent alignment and
            // truncation while avoiding subgroup shuffles for the PV product.
            volatile float rounded = output_accumulator[thread] * alpha;
            auto partial = qrt_q1_moe_hawkeye::value_from_float(
                rounded, kBlackwellZeroExponent);
            for (unsigned int begin = 0u; begin < kExactTileTokens;
                 begin += kBlackwellMmaGroup) {
                uint32_t products[kBlackwellMmaGroup];
#pragma unroll
                for (unsigned int item = 0u; item < kBlackwellMmaGroup; ++item) {
                    const unsigned int key_token = tile * kExactTileTokens + begin + item;
                    uint16_t v = 0u;
                    if (key_token < tokens) {
                        const bool in_tail = SplitDecodeValue && key_token >= decode_prefix_tokens;
                        const uint16_t *source = in_tail ? decode_tail_value : value;
                        const unsigned int source_token = in_tail ? key_token - decode_prefix_tokens : key_token;
                        v = source[(static_cast<size_t>(source_token) * kKvHeads + kv_head) * kHeadDim + thread];
                    }
                    products[item] = qrt_sm121_group16::pack_product(
                        qrt_q1_moe_hawkeye::multiply_bf16(
                            probability_bf16[begin + item], v, kBlackwellZeroExponent));
                }
                const auto sum = qrt_sm121_group16::sum_packed(partial, products);
                partial = qrt_sm121_wave16::normalize(
                    sum.value.magnitude, sum.value.negative, sum.max_exponent);
                partial = qrt_sm121_group16::finish_accumulator(partial);
                partial = qrt_q1_moe_hawkeye::value_from_float(
                    qrt_q1_moe_hawkeye::value_to_float(partial), kBlackwellZeroExponent);
            }
            output_accumulator[thread] = qrt_q1_moe_hawkeye::value_to_float(partial);
        } else {
            for (unsigned int output_batch = 0u;
                 output_batch < kHeadDim / kBlackwellSubgroups;
                 ++output_batch) {
                const unsigned int output_dimension =
                    output_batch * kBlackwellSubgroups + subgroup;
                float rescaled = 0.0f;
                if (subgroup_lane == 0u) {
                    volatile float rounded =
                        output_accumulator[output_dimension] * alpha;
                    rescaled = rounded;
                }
                rescaled = __shfl(
                    rescaled,
                    0,
                    kBlackwellMmaGroup
                );
                qrt_q1_moe_hawkeye::Value partial =
                    qrt_q1_moe_hawkeye::value_from_float(
                        rescaled,
                        kBlackwellZeroExponent
                    );
                for (unsigned int begin = 0u;
                     begin < kExactTileTokens;
                     begin += kBlackwellMmaGroup) {
                    const unsigned int key_token =
                        tile * kExactTileTokens + begin + subgroup_lane;
                    const uint16_t value_bf16 = key_token < tokens
                        ? value[
                              (static_cast<size_t>(key_token) * kKvHeads +
                               kv_head) *
                                  kHeadDim +
                              output_dimension]
                        : static_cast<uint16_t>(0u);
                    partial = qrt_sm121_wave16::accumulate(
                        partial,
                        probability_bf16[begin + subgroup_lane],
                        value_bf16,
                        subgroup_lane
                    );
                    partial = qrt_sm121_group16::finish_accumulator(partial);
                    partial = qrt_q1_moe_hawkeye::value_from_float(
                        qrt_q1_moe_hawkeye::value_to_float(partial),
                        kBlackwellZeroExponent
                    );
                }
                if (subgroup_lane == 0u) {
                    output_accumulator[output_dimension] =
                        qrt_q1_moe_hawkeye::value_to_float(partial);
                }
            }
        }
        if (thread == 0u) {
            running_sum = running_sum * alpha + sum_scratch[0];
            running_max = tile_max;
        }
        __syncthreads();
    }

    if (thread < kHeadDim) {
        output[output_base + thread] = rcp_table
            ? output_accumulator[thread] * qrt_sm121_attention_rcp::evaluate(rcp_table, running_sum)
            : output_accumulator[thread] / running_sum;
        if (raw_accumulator) raw_accumulator[output_base + thread] = output_accumulator[thread];
    }
    if (raw_denominator && thread == 0u)
        raw_denominator[static_cast<size_t>(output_token) * kQueryHeads + query_head] = running_sum;
}

// One wave materializes the same online probabilities, rescaling factors and
// denominator. The PV lanes can then advance without CTA-wide softmax barriers.
__global__ void blackwell_online_probability_kernel(
    const float* scores, uint16_t* probabilities, float* scales,
    unsigned int query_start, unsigned int score_stride,
    const unsigned char* exp2_table, bool vllm_sum) {
    const unsigned int lane = threadIdx.x;
    const unsigned int row = blockIdx.y * kQueryHeads + blockIdx.x;
    const unsigned int tokens = query_start + blockIdx.y + 1u;
    const unsigned int tile_stride = (score_stride + kExactTileTokens - 1u) / kExactTileTokens;
    const unsigned int tile_count = (tokens + kExactTileTokens - 1u) / kExactTileTokens;
    float running_max = -INFINITY, running_sum = 1.0f;
    for (unsigned int tile = 0u; tile < tile_count; ++tile) {
        const unsigned int key = tile * kExactTileTokens + lane;
        const float score = key < tokens ? scores[static_cast<size_t>(row) * score_stride + key] : -INFINITY;
        float next_max = fmaxf(running_max, score);
        for (unsigned int mask = 16u; mask; mask >>= 1u)
            next_max = fmaxf(next_max, __shfl_xor(next_max, mask, 32u));
        const float alpha = blackwell_attention_exp(running_max - next_max, exp2_table);
        const float probability = key < tokens ? blackwell_attention_exp(score - next_max, exp2_table) : 0.0f;
        if (key < score_stride)
            probabilities[static_cast<size_t>(row) * score_stride + key] = f32_to_bf16(probability);
        float sum = probability;
        if (vllm_sum) {
            constexpr unsigned int order[] = {1u, 4u, 2u, 16u, 8u};
#pragma unroll
            for (unsigned int step = 0u; step < 5u; ++step)
                sum += __shfl_xor(sum, order[step], 32u);
        } else {
            for (unsigned int mask = 16u; mask; mask >>= 1u)
                sum += __shfl_xor(sum, mask, 32u);
        }
        running_sum = running_sum * alpha + sum;
        running_max = next_max;
        if (lane == 0u) scales[static_cast<size_t>(row) * (tile_stride + 1u) + tile] = alpha;
    }
    if (lane == 0u)
        scales[static_cast<size_t>(row) * (tile_stride + 1u) + tile_stride] = running_sum;
}

__global__ void blackwell_probability_value_kernel(
    const uint16_t* value, const uint16_t* probabilities, const float* scales,
    float* output, unsigned int query_start, unsigned int output_start,
    unsigned int score_stride, const unsigned char* rcp_table,
    float* raw_accumulator, float* raw_denominator) {
    const unsigned int head = blockIdx.x, column = threadIdx.x;
    const unsigned int kv_head = head / (kQueryHeads / kKvHeads);
    const unsigned int row = blockIdx.y * kQueryHeads + head;
    const unsigned int tokens = query_start + blockIdx.y + 1u;
    const unsigned int tile_stride = (score_stride + kExactTileTokens - 1u) / kExactTileTokens;
    const unsigned int tile_count = (tokens + kExactTileTokens - 1u) / kExactTileTokens;
    float accumulator = 0.0f;
    for (unsigned int tile = 0u; tile < tile_count; ++tile) {
        const float alpha = scales[static_cast<size_t>(row) * (tile_stride + 1u) + tile];
        volatile float rounded = accumulator * alpha;
        auto partial = qrt_q1_moe_hawkeye::value_from_float(rounded, kBlackwellZeroExponent);
        for (unsigned int begin = 0u; begin < kExactTileTokens; begin += kBlackwellMmaGroup) {
            uint32_t products[kBlackwellMmaGroup];
#pragma unroll
            for (unsigned int item = 0u; item < kBlackwellMmaGroup; ++item) {
                const unsigned int key = tile * kExactTileTokens + begin + item;
                const uint16_t p = key < tokens
                    ? probabilities[static_cast<size_t>(row) * score_stride + key] : uint16_t(0u);
                const uint16_t v = key < tokens
                    ? value[(static_cast<size_t>(key) * kKvHeads + kv_head) * kHeadDim + column] : uint16_t(0u);
                products[item] = qrt_sm121_group16::pack_product(
                    qrt_q1_moe_hawkeye::multiply_bf16(p, v, kBlackwellZeroExponent));
            }
            const auto sum = qrt_sm121_group16::sum_packed(partial, products);
            partial = qrt_sm121_wave16::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
            partial = qrt_sm121_group16::finish_accumulator(partial);
            partial = qrt_q1_moe_hawkeye::value_from_float(
                qrt_q1_moe_hawkeye::value_to_float(partial), kBlackwellZeroExponent);
        }
        accumulator = qrt_q1_moe_hawkeye::value_to_float(partial);
    }
    const float denominator = scales[static_cast<size_t>(row) * (tile_stride + 1u) + tile_stride];
    const size_t output_index =
        (static_cast<size_t>(output_start + blockIdx.y) * kQueryHeads + head) * kHeadDim + column;
    output[output_index] = rcp_table
        ? accumulator * qrt_sm121_attention_rcp::evaluate(rcp_table, denominator) : accumulator / denominator;
    if (raw_accumulator) raw_accumulator[output_index] = accumulator;
    if (raw_denominator && column == 0u)
        raw_denominator[static_cast<size_t>(output_start + blockIdx.y) * kQueryHeads + head] = denominator;
}

// Bounded captured-prefix replay includes continuation beyond an 8192-token
// prompt. This diagnostic workspace bound does not change product contracts.
constexpr unsigned int kSplitMaxTokens = 16384u;

inline int transpose_keys(const uint16_t* key, uint16_t* transposed,
                          size_t elements, unsigned int tokens, hipStream_t stream) {
    if (!key || !transposed || !tokens || tokens > kSplitMaxTokens ||
        elements < static_cast<size_t>(tokens) * kKvHeads * kHeadDim)
        return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(blackwell_transpose_keys_kernel,
        dim3(kKvHeads * kHeadDim / 32u, (tokens + 31u) / 32u),
        dim3(32u, 8u), 0u, stream, key, transposed, tokens);
    return int(hipGetLastError());
}

inline size_t split_scratch_elements(unsigned int queries, unsigned int stride,
                                     unsigned int memory_layout) {
    if (!queries || queries > 32u || stride < queries || stride > kSplitMaxTokens ||
        memory_layout < 2u || memory_layout > 4u) return 0u;
    const size_t rows = static_cast<size_t>(queries) * kQueryHeads;
    const size_t cells = rows * stride;
    // Every row count is a multiple of sixteen, so the BF16 slab ends on a
    // float boundary. The final float of each scale row stores its denominator.
    return memory_layout != 3u ? cells : cells + cells / 2u +
        rows * ((stride + kExactTileTokens - 1u) / kExactTileTokens + 1u);
}

inline int launch_queries(const uint16_t* q, const uint16_t* k,
    const uint16_t* v, float* output, hipStream_t stream,
    unsigned int query_start, unsigned int query_count,
    unsigned int output_start, const unsigned char* exp2_table = nullptr,
    float* raw_accumulator = nullptr, float* raw_denominator = nullptr,
    bool vllm_sum = false, const unsigned char* rcp_table = nullptr,
    unsigned int memory_layout = 1u,
    float* score_scratch = nullptr, size_t score_scratch_elements = 0u,
    hipEvent_t scores_done = nullptr, hipEvent_t probabilities_done = nullptr,
    const uint16_t* transposed_key = nullptr, unsigned int key_stride = 0u) {
    if (!q || !k || !v || !output || query_count == 0u ||
        query_count > 8192u || query_start >= 262144u ||
        query_count > 262144u - query_start || output_start >= 262144u ||
        query_count > 262144u - output_start) return int(hipErrorInvalidValue);
    if (memory_layout > 4u) return int(hipErrorInvalidValue);
    if (memory_layout >= 2u) {
        // The split replay includes a bounded continuation of the captured prefix.
        // Long-context terminal calls retain their existing allocation-free path.
        if (!score_scratch || query_count > 32u || query_start + query_count > kSplitMaxTokens)
            return int(hipErrorInvalidValue);
        const unsigned int stride = query_start + query_count;
        if (memory_layout == 4u && (!transposed_key || key_stride < stride || key_stride > kSplitMaxTokens))
            return int(hipErrorInvalidValue);
        const size_t cells = static_cast<size_t>(query_count) * kQueryHeads * stride;
        if (score_scratch_elements < split_scratch_elements(query_count, stride, memory_layout))
            return int(hipErrorInvalidValue);
        if (memory_layout == 4u) {
            hipLaunchKernelGGL(blackwell_transposed_scores_kernel,
                dim3((cells + kThreads - 1u) / kThreads), dim3(kThreads), 0u, stream,
                q, transposed_key, score_scratch, query_start, query_count, stride, key_stride);
        } else {
            hipLaunchKernelGGL(blackwell_exact_scores_kernel,
                dim3((cells + kBlackwellSubgroups - 1u) / kBlackwellSubgroups),
                dim3(kThreads), 0u, stream, q, k, score_scratch,
                query_start, query_count, stride);
        }
        const auto status = hipGetLastError();
        if (status != hipSuccess) return int(status);
        if (scores_done) {
            const auto event_status = hipEventRecord(scores_done, stream);
            if (event_status != hipSuccess) return int(event_status);
        }
        if (memory_layout == 3u) {
            auto* probabilities = reinterpret_cast<uint16_t*>(score_scratch + cells);
            auto* scales = reinterpret_cast<float*>(probabilities + cells);
            hipLaunchKernelGGL(blackwell_online_probability_kernel,
                dim3(kQueryHeads, query_count), dim3(32u), 0u, stream,
                score_scratch, probabilities, scales, query_start, stride, exp2_table, vllm_sum);
            const auto probability_status = hipGetLastError();
            if (probability_status != hipSuccess) return int(probability_status);
            if (probabilities_done) {
                const auto event_status = hipEventRecord(probabilities_done, stream);
                if (event_status != hipSuccess) return int(event_status);
            }
            hipLaunchKernelGGL(blackwell_probability_value_kernel,
                dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
                v, probabilities, scales, output, query_start, output_start, stride,
                rcp_table, raw_accumulator, raw_denominator);
            return int(hipGetLastError());
        }
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_exact_attention_kernel<true, true>),
            dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
            q, k, v, output, query_start, output_start, exp2_table,
            raw_accumulator, raw_denominator, vllm_sum, rcp_table,
            score_scratch, stride, nullptr, 0u);
    } else if (memory_layout == 1u) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_exact_attention_kernel<true>),
            dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
            q, k, v, output, query_start, output_start, exp2_table,
            raw_accumulator, raw_denominator, vllm_sum, rcp_table, nullptr, 0u, nullptr, 0u);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_exact_attention_kernel<false>),
            dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
            q, k, v, output, query_start, output_start, exp2_table,
            raw_accumulator, raw_denominator, vllm_sum, rcp_table, nullptr, 0u, nullptr, 0u);
    }
    return int(hipGetLastError());
}
} // namespace qrt_blackwell_attention
#endif
