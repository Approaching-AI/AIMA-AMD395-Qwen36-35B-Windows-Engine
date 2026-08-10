// Exact-shape Windows wrapper around the generated CK-Tile FMHA instance.
// The generated kernel consumes runtime query/key-value lengths; only this
// wrapper's allocation and exported contracts are shape-specific.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>

#include "fmha_fwd.hpp"

#if defined(_WIN32)
#define QRT_CK_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_CK_EXPORT extern "C" __attribute__((visibility("default")))
#endif

namespace {

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

bool supported_tokens(unsigned int tokens) {
    return tokens == kQ8192Tokens || tokens == kQ16384Tokens ||
        tokens == kQ32768Tokens || tokens == kQ65536Tokens ||
        tokens == kQ129536Tokens || tokens == kQ130560Tokens ||
        tokens == kQ131071Tokens ||
        tokens == kQ131072Tokens || tokens == kQ131073Tokens ||
        tokens == kQ262143Tokens || tokens == kQ262144Tokens;
}

int prepare_locked(unsigned int tokens) {
    if (!supported_tokens(tokens)) {
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
    args.num_head_q_total = kQueryHeads;
    args.head_start = 0;
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
    return launch_bf16_attention(
        q,
        k,
        v,
        output,
        stream,
        tokens,
        tokens,
        mask_enum::mask_top_left);
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
