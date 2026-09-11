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
constexpr unsigned int kStagedKeyStride = kHeadDim + 16u;
constexpr unsigned int kStagedValueStride = kExactTileTokens + 17u;
constexpr unsigned int kStagedTileElements = kHeadDim * kStagedValueStride;
static_assert(kExactTileTokens * kStagedKeyStride <= kStagedTileElements);
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

// One CTA owns one causal query/head. Both terminal replacement and bounded
// prefix replay share the same Blackwell QK and fused-C PV arithmetic.
// Layout 0 retains direct loads. Layout 1 stages Q/V; layout 2 stages Q/K/V.
// K and transposed V share storage across the existing QK-completion barrier.
// Their padding separates subgroup reads while avoiding a power-of-two stride
// for the coalesced global-to-LDS transpose. Arithmetic and causal order match.
template <unsigned int MemoryLayout>
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
    const unsigned char* rcp_table) {
    __shared__ float score[kExactTileTokens];
    __shared__ float probability[kExactTileTokens];
    __shared__ float sum_scratch[kExactTileTokens];
    __shared__ uint16_t probability_bf16[kExactTileTokens];
    __shared__ float output_accumulator[kHeadDim];
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float tile_max;
    __shared__ float alpha;
    __shared__ uint16_t staged_query[MemoryLayout == 0u ? 1u : kHeadDim];
    __shared__ uint16_t staged_tile[MemoryLayout == 0u ? 1u : kStagedTileElements];

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
    if constexpr (MemoryLayout != 0u) {
        staged_query[thread] = query[query_base + thread];
    }
    if (thread == 0u) {
        running_max = -INFINITY;
        running_sum = 1.0f;
    }
    __syncthreads();

    const unsigned int tile_count =
        (tokens + kExactTileTokens - 1u) / kExactTileTokens;
    for (unsigned int tile = 0u; tile < tile_count; ++tile) {
        if constexpr (MemoryLayout == 2u) {
            for (unsigned int item = 0u; item < kExactTileTokens; ++item) {
                const unsigned int key_token = tile * kExactTileTokens + item;
                staged_tile[item * kStagedKeyStride + thread] = key_token < tokens
                    ? key[(static_cast<size_t>(key_token) * kKvHeads + kv_head) *
                              kHeadDim + thread]
                    : static_cast<uint16_t>(0u);
            }
            __syncthreads();
        }
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
                    if constexpr (MemoryLayout != 0u) {
                        query_value = staged_query[element];
                    } else {
                        query_value = query[query_base + element];
                    }
                    if constexpr (MemoryLayout == 2u) {
                        key_value = staged_tile[key_item * kStagedKeyStride + element];
                    } else {
                        key_value = key[key_base + element];
                    }
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
        __syncthreads();

        if constexpr (MemoryLayout != 0u) {
            for (unsigned int item = 0u; item < kExactTileTokens; ++item) {
                const unsigned int key_token = tile * kExactTileTokens + item;
                staged_tile[thread * kStagedValueStride + item] = key_token < tokens
                    ? value[(static_cast<size_t>(key_token) * kKvHeads + kv_head) *
                                kHeadDim + thread]
                    : static_cast<uint16_t>(0u);
            }
        }
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
                uint16_t value_bf16;
                if constexpr (MemoryLayout != 0u) {
                    value_bf16 = staged_tile[output_dimension * kStagedValueStride +
                                             begin + subgroup_lane];
                } else {
                    value_bf16 = key_token < tokens ? value[
                          (static_cast<size_t>(key_token) * kKvHeads +
                           kv_head) *
                              kHeadDim +
                          output_dimension]
                        : static_cast<uint16_t>(0u);
                }
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

inline int launch_queries(const uint16_t* q, const uint16_t* k,
    const uint16_t* v, float* output, hipStream_t stream,
    unsigned int query_start, unsigned int query_count,
    unsigned int output_start, const unsigned char* exp2_table = nullptr,
    float* raw_accumulator = nullptr, float* raw_denominator = nullptr,
    bool vllm_sum = false, const unsigned char* rcp_table = nullptr,
    unsigned int memory_layout = 0u) {
    if (!q || !k || !v || !output || query_count == 0u ||
        query_count > 8192u || query_start >= 262144u ||
        query_count > 262144u - query_start || output_start >= 262144u ||
        query_count > 262144u - output_start || memory_layout > 2u)
        return int(hipErrorInvalidValue);
    if (memory_layout == 2u) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_exact_attention_kernel<2u>),
            dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
            q, k, v, output, query_start, output_start, exp2_table,
            raw_accumulator, raw_denominator, vllm_sum, rcp_table);
    } else if (memory_layout == 1u) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_exact_attention_kernel<1u>),
            dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
            q, k, v, output, query_start, output_start, exp2_table,
            raw_accumulator, raw_denominator, vllm_sum, rcp_table);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(blackwell_exact_attention_kernel<0u>),
            dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
            q, k, v, output, query_start, output_start, exp2_table,
            raw_accumulator, raw_denominator, vllm_sum, rcp_table);
    }
    return int(hipGetLastError());
}
} // namespace qrt_blackwell_attention
#endif
