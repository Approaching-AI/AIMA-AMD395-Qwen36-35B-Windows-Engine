// Exact-shape Windows wrapper around the generated CK-Tile FMHA instance.
// The generated kernel consumes runtime query/key-value lengths; only this
// wrapper's allocation and exported contracts are shape-specific.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <type_traits>
#include <utility>

#include "fmha_fwd.hpp"
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
#include "../moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#endif

#if defined(_WIN32)
#define QRT_CK_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_CK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

template <typename Args, typename = void>
struct HasHeadPartitionFields : std::false_type {};

template <typename Args>
struct HasHeadPartitionFields<
    Args,
    std::void_t<
        decltype(std::declval<Args &>().num_head_q_total),
        decltype(std::declval<Args &>().head_start)
    >
> : std::true_type {};

template <typename Args>
void set_head_partition_fields(Args &args, unsigned int query_heads) {
    if constexpr (HasHeadPartitionFields<Args>::value) {
        args.num_head_q_total = query_heads;
        args.head_start = 0;
    }
}

constexpr unsigned int kQ8192Tokens = 8192u;
constexpr unsigned int kQ16384Tokens = 16384u;
constexpr unsigned int kQ17408Tokens = 17408u;
constexpr unsigned int kQ32768Tokens = 32768u;
constexpr unsigned int kQ65536Tokens = 65536u;
constexpr unsigned int kQ129536Tokens = 129536u;
constexpr unsigned int kQ130560Tokens = 130560u;
constexpr unsigned int kQ131071Tokens = 131071u;
constexpr unsigned int kQ131072Tokens = 131072u;
constexpr unsigned int kQ131073Tokens = 131073u;
constexpr unsigned int kQ262143Tokens = 262143u;
constexpr unsigned int kQ262144Tokens = 262144u;
constexpr unsigned int kPrefillChunkTokens = 8192u;
constexpr unsigned int kSuffixTokens = 1024u;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kPackedRows = 2u * kQueryFeatures + 2u * kKvFeatures;
constexpr unsigned int kThreads = 256u;

struct ProviderState {
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
    unsigned int capacity_tokens = 0u;
};

ProviderState g_state;
std::mutex g_state_mutex;

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

__global__ void pack_qkv_kernel(
    const float *__restrict__ packed,
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    uint16_t *__restrict__ v,
    unsigned int tokens) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_elements =
        static_cast<size_t>(tokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(tokens) * kKvFeatures;
    if (index < q_elements) {
        const size_t token = index / kQueryFeatures;
        const size_t feature = index - token * kQueryFeatures;
        q[index] = f32_to_bf16(
            packed[token * kPackedRows + feature]);
        return;
    }
    const size_t kv_index = index - q_elements;
    if (kv_index >= 2u * kv_elements) {
        return;
    }
    const size_t token = (kv_index % kv_elements) / kKvFeatures;
    const size_t feature = (kv_index % kv_elements) - token * kKvFeatures;
    if (kv_index < kv_elements) {
        k[kv_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + feature]);
    } else {
        const size_t value_index = kv_index - kv_elements;
        v[value_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + kKvFeatures + feature]);
    }
}

// The terminal-Q1 product path consumes only the final query row.  Packing
// every earlier query would read intentionally unwritten dense-Q slots and
// launch a full-prefix CK kernel whose output is discarded.  Keep K/V dense,
// but populate only the terminal query row required by the exact correction.
__global__ void pack_terminal_q_kv_kernel(
    const float *__restrict__ packed,
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    uint16_t *__restrict__ v,
    unsigned int tokens) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_elements = kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(tokens) * kKvFeatures;
    if (index < q_elements) {
        const size_t terminal = static_cast<size_t>(tokens - 1u);
        q[terminal * kQueryFeatures + index] = f32_to_bf16(
            packed[terminal * kPackedRows + index]);
        return;
    }
    const size_t kv_index = index - q_elements;
    if (kv_index >= 2u * kv_elements) {
        return;
    }
    const size_t token = (kv_index % kv_elements) / kKvFeatures;
    const size_t feature = (kv_index % kv_elements) - token * kKvFeatures;
    if (kv_index < kv_elements) {
        k[kv_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + feature]);
    } else {
        const size_t value_index = kv_index - kv_elements;
        v[value_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + kKvFeatures + feature]);
    }
}

#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
constexpr unsigned int kExactTileTokens = 32u;
constexpr float kExactScale = 0.0625f;
constexpr float kExactLog2e = 1.4426950408889634074f;
constexpr unsigned int kBlackwellMmaGroup = 16u;
constexpr unsigned int kBlackwellSubgroups =
    kThreads / kBlackwellMmaGroup;
constexpr int16_t kBlackwellZeroExponent = -133;
static_assert(kExactTileTokens % kBlackwellSubgroups == 0u);
static_assert(kHeadDim % kBlackwellSubgroups == 0u);

__device__ __forceinline__ float blackwell_attention_exp(float value) {
    return exp2f(value * kExactLog2e);
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

/*
 * CK supplies the full fast attention surface.  This kernel replaces only the
 * final query token, which is the boundary consumed by first-token sampling.
 * It emulates the GB10 Blackwell BF16 MMA dataflow exactly: forward K16 QK
 * groups and a PV dot whose first MMA consumes the rescaled online accumulator
 * as C.  The latter fusion was verified against 80/80 self-checked GB10 q2560
 * accumulator boundaries; a standalone PV plus FP32 add is observably wrong.
 */
__global__ void blackwell_exact_terminal_attention_kernel(
    const uint16_t *__restrict__ query,
    const uint16_t *__restrict__ key,
    const uint16_t *__restrict__ value,
    float *__restrict__ output,
    unsigned int tokens,
    unsigned int output_token) {
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
    const unsigned int token = tokens - 1u;
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
            alpha = blackwell_attention_exp(running_max - next_max);
        }
        __syncthreads();

        if (thread < kExactTileTokens) {
            const unsigned int key_token = tile * kExactTileTokens + thread;
            const float p = key_token < tokens
                ? blackwell_attention_exp(score[thread] - tile_max)
                : 0.0f;
            probability[thread] = p;
            probability_bf16[thread] = f32_to_bf16(p);
            sum_scratch[thread] = p;
        }
        __syncthreads();

        for (unsigned int stride = kExactTileTokens / 2u; stride > 0u;
             stride >>= 1u) {
            if (thread < stride) {
                sum_scratch[thread] += sum_scratch[thread + stride];
            }
            __syncthreads();
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
        output[output_base + thread] =
            output_accumulator[thread] / running_sum;
    }
}

int launch_blackwell_exact_terminal(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    hipStream_t stream,
    unsigned int tokens,
    unsigned int output_token) {
    hipLaunchKernelGGL(
        blackwell_exact_terminal_attention_kernel,
        dim3(kQueryHeads),
        dim3(kHeadDim),
        0u,
        stream,
        q,
        k,
        v,
        output,
        tokens,
        output_token);
    return static_cast<int>(hipGetLastError());
}
#endif

bool supported_tokens(unsigned int tokens) {
    return tokens == kQ8192Tokens || tokens == kQ16384Tokens ||
        tokens == kQ32768Tokens || tokens == kQ65536Tokens ||
        tokens == kQ129536Tokens || tokens == kQ130560Tokens ||
        tokens == kQ131071Tokens ||
        tokens == kQ131072Tokens || tokens == kQ131073Tokens ||
        tokens == kQ262143Tokens || tokens == kQ262144Tokens;
}

int prepare_locked(unsigned int tokens) {
    // Full-prefix exports remain exact-shape surfaces, but the terminal-Q1
    // export consumes a runtime KV length.  Its storage has the same bounded
    // Q/K/V layout, so allow any product context while keeping callers of the
    // full-prefix helpers guarded by supported_tokens().
    if (tokens == 0u || tokens > kQ262144Tokens) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (g_state.q != nullptr && g_state.k != nullptr &&
        g_state.v != nullptr && g_state.capacity_tokens >= tokens) {
        return static_cast<int>(hipSuccess);
    }
    const size_t q_bytes =
        static_cast<size_t>(tokens) * kQueryFeatures * sizeof(uint16_t);
    const size_t kv_bytes =
        static_cast<size_t>(tokens) * kKvFeatures * sizeof(uint16_t);
    ProviderState next{};
    hipError_t status = hipMalloc(
        reinterpret_cast<void **>(&next.q), q_bytes);
    if (status == hipSuccess) {
        status = hipMalloc(reinterpret_cast<void **>(&next.k), kv_bytes);
    }
    if (status == hipSuccess) {
        status = hipMalloc(reinterpret_cast<void **>(&next.v), kv_bytes);
    }
    if (status != hipSuccess) {
        (void)hipFree(next.v);
        (void)hipFree(next.k);
        (void)hipFree(next.q);
        return static_cast<int>(status);
    }
    next.capacity_tokens = tokens;
    (void)hipFree(g_state.v);
    (void)hipFree(g_state.k);
    (void)hipFree(g_state.q);
    g_state = next;
    return static_cast<int>(hipSuccess);
}

int launch_bf16_attention(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    hipStream_t stream,
    unsigned int query_tokens,
    unsigned int kv_tokens,
    mask_enum mask_type) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        query_tokens == 0u || kv_tokens == 0u ||
        (mask_type != mask_enum::mask_top_left &&
         mask_type != mask_enum::mask_bottom_right)) {
        return static_cast<int>(hipErrorInvalidValue);
    }

    fmha_fwd_traits traits{};
    traits.hdim_q = kHeadDim;
    traits.hdim_v = kHeadDim;
    traits.data_type = "bf16";
    traits.is_group_mode = false;
    traits.is_v_rowmajor = true;
    traits.has_logits_soft_cap = false;
    traits.mask_type = mask_type;
    traits.bias_type = bias_enum::no_bias;
    traits.has_lse = false;
    traits.has_dropout = false;
    traits.qscale_type = quant_scale_enum::no_scale;
    traits.skip_min_seqlen_q = false;
    traits.has_sink = false;

    fmha_fwd_args args{};
    args.q_ptr = const_cast<uint16_t *>(q);
    args.k_ptr = const_cast<uint16_t *>(k);
    args.v_ptr = const_cast<uint16_t *>(v);
    args.o_ptr = output;
    args.seqlen_q = query_tokens;
    args.seqlen_k = kv_tokens;
    args.batch = 1;
    args.max_seqlen_q = query_tokens;
    args.hdim_q = kHeadDim;
    args.hdim_v = kHeadDim;
    args.nhead_q = kQueryHeads;
    args.nhead_k = kKvHeads;
    set_head_partition_fields(args, kQueryHeads);
    args.scale_s = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
    args.logits_soft_cap = 0.0f;
    args.stride_q = kQueryFeatures;
    args.stride_k = kKvFeatures;
    args.stride_v = kKvFeatures;
    args.stride_o = kQueryFeatures;
    args.nhead_stride_q = kHeadDim;
    args.nhead_stride_k = kHeadDim;
    args.nhead_stride_v = kHeadDim;
    args.nhead_stride_o = kHeadDim;
    args.batch_stride_q = query_tokens * kQueryFeatures;
    args.batch_stride_k = kv_tokens * kKvFeatures;
    args.batch_stride_v = kv_tokens * kKvFeatures;
    args.batch_stride_o = query_tokens * kQueryFeatures;
    args.window_size_left = -1;
    args.window_size_right = 0;
    args.sink_size = 0;
    args.mask_type = static_cast<int>(mask_type);
    args.min_seqlen_q = query_tokens;
    args.p_drop = 0.0f;
    args.s_randval = false;
    args.drop_seed_offset = std::make_pair(uint64_t{0}, uint64_t{0});

    const ck_tile::stream_config config{stream};
    const float launch_result = fmha_fwd(traits, args, config);
    if (launch_result < 0.0f) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    return static_cast<int>(hipGetLastError());
}

int launch_bf16_qkv(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    hipStream_t stream,
    unsigned int tokens) {
    if (!supported_tokens(tokens)) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const int status = launch_bf16_attention(
        q,
        k,
        v,
        output,
        stream,
        tokens,
        tokens,
        mask_enum::mask_top_left);
    if (status != static_cast<int>(hipSuccess)) {
        return status;
    }
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    return launch_blackwell_exact_terminal(
        q,
        k,
        v,
        output,
        stream,
        tokens,
        tokens - 1u);
#else
    return static_cast<int>(hipSuccess);
#endif
}

int launch_bf16_long_chunk8192(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    hipStream_t stream,
    unsigned int tokens) {
    if (q == nullptr || k == nullptr || v == nullptr || output == nullptr ||
        (tokens != kQ65536Tokens && tokens != kQ129536Tokens &&
         tokens != kQ130560Tokens &&
         tokens != kQ131071Tokens && tokens != kQ131072Tokens &&
         tokens != kQ131073Tokens && tokens != kQ262143Tokens &&
         tokens != kQ262144Tokens)) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    for (unsigned int query_start = 0u;
         query_start < tokens;
         query_start += kPrefillChunkTokens) {
        const unsigned int query_tokens =
            (tokens - query_start) < kPrefillChunkTokens
                ? tokens - query_start
                : kPrefillChunkTokens;
        const size_t query_offset =
            static_cast<size_t>(query_start) * kQueryFeatures;
        const unsigned int kv_tokens = query_start + query_tokens;
        const int status = launch_bf16_attention(
            q + query_offset,
            k,
            v,
            output + query_offset,
            stream,
            query_tokens,
            kv_tokens,
            mask_enum::mask_bottom_right);
        if (status != static_cast<int>(hipSuccess)) {
            return status;
        }
    }
    return static_cast<int>(hipSuccess);
}

int launch_bf16_long_tile8192(
    const uint16_t *q_tile,
    const uint16_t *k,
    const uint16_t *v,
    float *output_tile,
    unsigned int query_start,
    hipStream_t stream,
    unsigned int total_tokens) {
    if (q_tile == nullptr || k == nullptr || v == nullptr ||
        output_tile == nullptr ||
        (total_tokens != kQ129536Tokens &&
         total_tokens != kQ131072Tokens &&
         total_tokens != kQ131073Tokens &&
         total_tokens != kQ262143Tokens &&
         total_tokens != kQ262144Tokens) ||
        query_start >= total_tokens ||
        query_start % kPrefillChunkTokens != 0u) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    const unsigned int remaining_tokens = total_tokens - query_start;
    const unsigned int query_tokens =
        remaining_tokens < kPrefillChunkTokens
            ? remaining_tokens
            : kPrefillChunkTokens;
    return launch_bf16_attention(
        q_tile,
        k,
        v,
        output_tile,
        stream,
        query_tokens,
        query_start + query_tokens,
        mask_enum::mask_bottom_right);
}

QRT_CK_EXPORT int qrt_ck_fmha_q129536_tile8192_bf16_launch(
    const uint16_t *q_tile,
    const uint16_t *k,
    const uint16_t *v,
    float *output_tile,
    unsigned int query_start,
    void *stream_handle) {
    return launch_bf16_long_tile8192(
        q_tile,
        k,
        v,
        output_tile,
        query_start,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ129536Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q131072_tile8192_bf16_launch(
    const uint16_t *q_tile,
    const uint16_t *k,
    const uint16_t *v,
    float *output_tile,
    unsigned int query_start,
    void *stream_handle) {
    return launch_bf16_long_tile8192(
        q_tile,
        k,
        v,
        output_tile,
        query_start,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ131072Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q131073_tile8192_bf16_launch(
    const uint16_t *q_tile,
    const uint16_t *k,
    const uint16_t *v,
    float *output_tile,
    unsigned int query_start,
    void *stream_handle) {
    return launch_bf16_long_tile8192(
        q_tile,
        k,
        v,
        output_tile,
        query_start,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ131073Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q262143_tile8192_bf16_launch(
    const uint16_t *q_tile,
    const uint16_t *k,
    const uint16_t *v,
    float *output_tile,
    unsigned int query_start,
    void *stream_handle) {
    return launch_bf16_long_tile8192(
        q_tile,
        k,
        v,
        output_tile,
        query_start,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ262143Tokens);
}

int launch_f32_packed(
    const float *packed_qkv,
    float *output,
    hipStream_t stream,
    unsigned int tokens) {
    if (packed_qkv == nullptr || output == nullptr ||
        !supported_tokens(tokens)) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        const int status = prepare_locked(tokens);
        if (status != static_cast<int>(hipSuccess)) {
            return status;
        }
    }
    const size_t q_elements =
        static_cast<size_t>(tokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(tokens) * kKvFeatures;
    const size_t packed_elements = q_elements + 2u * kv_elements;
    hipLaunchKernelGGL(
        pack_qkv_kernel,
        dim3(static_cast<unsigned int>(
            (packed_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads),
        0u,
        stream,
        packed_qkv,
        g_state.q,
        g_state.k,
        g_state.v,
        tokens);
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        return static_cast<int>(status);
    }
    return launch_bf16_qkv(
        g_state.q,
        g_state.k,
        g_state.v,
        output,
        stream,
        tokens);
}

int launch_f32_terminal_q_kv_packed(
    const float *packed_qkv,
    float *compact_output,
    hipStream_t stream,
    unsigned int tokens) {
    if (packed_qkv == nullptr || compact_output == nullptr ||
        tokens == 0u || tokens > kQ262144Tokens) {
        return static_cast<int>(hipErrorInvalidValue);
    }
#if !defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    (void)stream;
    return static_cast<int>(hipErrorInvalidValue);
#else
    {
        std::lock_guard<std::mutex> lock(g_state_mutex);
        const int status = prepare_locked(tokens);
        if (status != static_cast<int>(hipSuccess)) {
            return status;
        }
    }
    const size_t kv_elements =
        static_cast<size_t>(tokens) * kKvFeatures;
    const size_t packed_elements = kQueryFeatures + 2u * kv_elements;
    hipLaunchKernelGGL(
        pack_terminal_q_kv_kernel,
        dim3(static_cast<unsigned int>(
            (packed_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads),
        0u,
        stream,
        packed_qkv,
        g_state.q,
        g_state.k,
        g_state.v,
        tokens);
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        return static_cast<int>(status);
    }
    return launch_blackwell_exact_terminal(
        g_state.q,
        g_state.k,
        g_state.v,
        compact_output,
        stream,
        tokens,
        0u);
#endif
}

}  // namespace

QRT_CK_EXPORT int qrt_ck_fmha_q8192_prepare() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    return prepare_locked(kQ8192Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_f32_launch(
    const float *packed_qkv,
    float *output,
    void *stream_handle) {
    return launch_f32_packed(
        packed_qkv,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ8192Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q1_kv8192_f32_launch(
    const float *packed_qkv,
    float *compact_output,
    void *stream_handle) {
    return launch_f32_terminal_q_kv_packed(
        packed_qkv,
        compact_output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ8192Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q1_dynamic_f32_launch(
    const float *packed_qkv,
    float *compact_output,
    void *stream_handle,
    unsigned int tokens) {
    return launch_f32_terminal_q_kv_packed(
        packed_qkv,
        compact_output,
        reinterpret_cast<hipStream_t>(stream_handle),
        tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_qkv(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ8192Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_dynamic_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle,
    unsigned int tokens) {
    if (tokens == 0u || tokens > kQ262144Tokens) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_handle);
    const int status = launch_bf16_attention(
        q,
        k,
        v,
        output,
        stream,
        tokens,
        tokens,
        mask_enum::mask_top_left);
    if (status != static_cast<int>(hipSuccess)) {
        return status;
    }
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    return launch_blackwell_exact_terminal(
        q,
        k,
        v,
        output,
        stream,
        tokens,
        tokens - 1u);
#else
    return static_cast<int>(hipSuccess);
#endif
}

QRT_CK_EXPORT int qrt_ck_fmha_q16384_f32_launch(
    const float *packed_qkv,
    float *output,
    void *stream_handle) {
    return launch_f32_packed(
        packed_qkv,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ16384Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q16384_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_qkv(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ16384Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q17408_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_attention(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ17408Tokens,
        kQ17408Tokens,
        mask_enum::mask_top_left);
}

QRT_CK_EXPORT int qrt_ck_fmha_q1024_kv17408_suffix_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_attention(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kSuffixTokens,
        kQ17408Tokens,
        mask_enum::mask_bottom_right);
}

QRT_CK_EXPORT int qrt_ck_fmha_q32768_f32_launch(
    const float *packed_qkv,
    float *output,
    void *stream_handle) {
    return launch_f32_packed(
        packed_qkv,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ32768Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q32768_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_qkv(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ32768Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q65536_f32_launch(
    const float *packed_qkv,
    float *output,
    void *stream_handle) {
    return launch_f32_packed(
        packed_qkv,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ65536Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q65536_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_qkv(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ65536Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q65536_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ65536Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q130560_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ130560Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q129536_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream
) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        static_cast<hipStream_t>(stream),
        kQ129536Tokens
    );
}

QRT_CK_EXPORT int qrt_ck_fmha_q131071_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ131071Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q131072_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ131072Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q131073_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ131073Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q262143_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ262143Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q262144_chunk8192_bf16_launch(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    void *stream_handle) {
    return launch_bf16_long_chunk8192(
        q,
        k,
        v,
        output,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ262144Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q262144_tile8192_bf16_launch(
    const uint16_t *q_tile,
    const uint16_t *k,
    const uint16_t *v,
    float *output_tile,
    unsigned int query_start,
    void *stream_handle) {
    return launch_bf16_long_tile8192(
        q_tile,
        k,
        v,
        output_tile,
        query_start,
        reinterpret_cast<hipStream_t>(stream_handle),
        kQ262144Tokens);
}

QRT_CK_EXPORT int qrt_ck_fmha_q8192_release() {
    std::lock_guard<std::mutex> lock(g_state_mutex);
    (void)hipFree(g_state.v);
    (void)hipFree(g_state.k);
    (void)hipFree(g_state.q);
    g_state = ProviderState{};
    return static_cast<int>(hipSuccess);
}
