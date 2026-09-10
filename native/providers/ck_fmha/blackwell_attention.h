#ifndef QRT_BLACKWELL_ATTENTION_H
#define QRT_BLACKWELL_ATTENTION_H
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include "../moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
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

__device__ __forceinline__ qrt_q1_moe_hawkeye::Value
blackwell_normalize_group(int64_t signed_significand, int max_exponent) {
    constexpr int kInternalSignificandWidth = 26;
    constexpr int kInternalToFp32Shift =
        kInternalSignificandWidth - 24;
    constexpr int16_t kFp32MinNonzeroExponent = -126;

    const bool negative = signed_significand < 0;
    const uint64_t magnitude = negative
        ? static_cast<uint64_t>(-signed_significand)
        : static_cast<uint64_t>(signed_significand);
    const unsigned int width =
        qrt_q1_moe_hawkeye::bit_width_u64(magnitude);
    if (width == 0u) {
        return qrt_q1_moe_hawkeye::Value{
            0u,
            kBlackwellZeroExponent,
            negative
        };
    }

    int exponent = max_exponent + static_cast<int>(width) -
        kInternalSignificandWidth;
    uint64_t normalized = magnitude;
    if (width > static_cast<unsigned int>(kInternalSignificandWidth)) {
        normalized >>= width -
            static_cast<unsigned int>(kInternalSignificandWidth);
    } else {
        normalized <<= static_cast<unsigned int>(kInternalSignificandWidth) -
            width;
    }
    if (exponent < kFp32MinNonzeroExponent) {
        const unsigned int underflow_shift = static_cast<unsigned int>(
            kFp32MinNonzeroExponent - exponent
        );
        normalized = underflow_shift >= 64u
            ? 0u
            : normalized >> underflow_shift;
        exponent = kFp32MinNonzeroExponent;
    }
    normalized >>= kInternalToFp32Shift;
    if (normalized == 0u) {
        return qrt_q1_moe_hawkeye::Value{
            0u,
            kBlackwellZeroExponent,
            negative
        };
    }
    return qrt_q1_moe_hawkeye::Value{
        static_cast<uint32_t>(normalized),
        static_cast<int16_t>(exponent),
        negative
    };
}

/*
 * One wave64 owns four independent Blackwell K16 groups.  A 16-lane subgroup
 * computes the same max-exponent alignment and signed integer sum as
 * group_sum<26, -133>, but distributes the 16 BF16 products across its lanes.
 * Integer addition is exact here (the reachable sum is far below int64_t), so
 * the tree reduction cannot change the serialized Blackwell endpoint.
 */
__device__ __forceinline__ qrt_q1_moe_hawkeye::Value
blackwell_group16_wave(
    qrt_q1_moe_hawkeye::Value accumulator,
    uint16_t left,
    uint16_t right,
    unsigned int subgroup_lane
) {
    constexpr int kInternalToFp32Shift = 2;

    const qrt_q1_moe_hawkeye::Value product =
        qrt_q1_moe_hawkeye::multiply_bf16(
            left,
            right,
            kBlackwellZeroExponent
        );
    const uint32_t accumulator_significand = __shfl(
        accumulator.significand,
        0,
        kBlackwellMmaGroup
    );
    const int accumulator_exponent = __shfl(
        static_cast<int>(accumulator.exponent),
        0,
        kBlackwellMmaGroup
    );
    const int accumulator_negative = __shfl(
        static_cast<int>(accumulator.negative),
        0,
        kBlackwellMmaGroup
    );
    int max_exponent = static_cast<int>(product.exponent) >
            accumulator_exponent
        ? static_cast<int>(product.exponent)
        : accumulator_exponent;
    for (unsigned int lane_mask = kBlackwellMmaGroup / 2u;
         lane_mask != 0u;
         lane_mask >>= 1u) {
        const int other_exponent = __shfl_xor(
            max_exponent,
            lane_mask,
            kBlackwellMmaGroup
        );
        if (other_exponent > max_exponent) {
            max_exponent = other_exponent;
        }
    }

    const int product_shift =
        max_exponent - static_cast<int>(product.exponent);
    const uint64_t product_aligned = product_shift >= 32
        ? 0u
        : (static_cast<uint64_t>(product.significand)
               << kInternalToFp32Shift) >>
              static_cast<unsigned int>(product_shift);
    int64_t signed_significand = product.negative
        ? -static_cast<int64_t>(product_aligned)
        : static_cast<int64_t>(product_aligned);
    if (subgroup_lane == 0u) {
        const int accumulator_shift =
            max_exponent - accumulator_exponent;
        const uint64_t accumulator_aligned = accumulator_shift >= 32
            ? 0u
            : (static_cast<uint64_t>(accumulator_significand)
                   << kInternalToFp32Shift) >>
                  static_cast<unsigned int>(accumulator_shift);
        signed_significand += accumulator_negative != 0
            ? -static_cast<int64_t>(accumulator_aligned)
            : static_cast<int64_t>(accumulator_aligned);
    }
    for (unsigned int lane_mask = kBlackwellMmaGroup / 2u;
         lane_mask != 0u;
         lane_mask >>= 1u) {
        signed_significand += __shfl_down(
            signed_significand,
            lane_mask,
            kBlackwellMmaGroup
        );
    }
    if (subgroup_lane == 0u) {
        accumulator = blackwell_normalize_group(
            signed_significand,
            max_exponent
        );
    }
    return accumulator;
}

__device__ __forceinline__ qrt_q1_moe_hawkeye::Value
blackwell_finish_group_wave(
    qrt_q1_moe_hawkeye::Value accumulator,
    unsigned int subgroup_lane
) {
    // accumulate_bf16_impl performs one final one-value group_sum.  It is only
    // one value, so lane zero can preserve that endpoint without another wave
    // reduction; the next K16 group broadcasts lane zero's accumulator.
    if (subgroup_lane == 0u) {
        accumulator = qrt_q1_moe_hawkeye::group_sum<
            26,
            kBlackwellZeroExponent
        >(&accumulator, 1u);
    }
    return accumulator;
}

// One CTA owns one causal query/head. Both terminal replacement and bounded
// prefix replay share the same Blackwell QK and fused-C PV arithmetic.
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
                dot = blackwell_group16_wave(
                    dot,
                    query_value,
                    key_value,
                    subgroup_lane
                );
            }
            dot = blackwell_finish_group_wave(dot, subgroup_lane);
            if (subgroup_lane == 0u) {
                score[key_item] = key_valid
                    ? qrt_q1_moe_hawkeye::value_to_float(dot) * kExactScale
                    : -INFINITY;
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
                partial = blackwell_group16_wave(
                    partial,
                    probability_bf16[begin + subgroup_lane],
                    value_bf16,
                    subgroup_lane
                );
                partial = blackwell_finish_group_wave(
                    partial,
                    subgroup_lane
                );
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
    bool vllm_sum = false, const unsigned char* rcp_table = nullptr) {
    if (!q || !k || !v || !output || query_count == 0u ||
        query_count > 8192u || query_start >= 262144u ||
        query_count > 262144u - query_start || output_start >= 262144u ||
        query_count > 262144u - output_start) return int(hipErrorInvalidValue);
    hipLaunchKernelGGL(blackwell_exact_attention_kernel,
        dim3(kQueryHeads, query_count), dim3(kHeadDim), 0u, stream,
        q, k, v, output, query_start, output_start, exp2_table,
        raw_accumulator, raw_denominator, vllm_sum, rcp_table);
    return int(hipGetLastError());
}
} // namespace qrt_blackwell_attention
#endif
