// Exact-shape Windows wrapper around the generated CK-Tile FMHA instance.
// The generated kernel consumes runtime query/key-value lengths; only this
// wrapper's allocation and exported contracts are shape-specific.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>
#if defined(_WIN32)
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

#include "fmha_fwd.hpp"
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
#include "blackwell_attention.h"
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
std::mutex g_sm121_mutex;
unsigned char* g_sm121_exp2 = nullptr;
unsigned char* g_sm121_rcp = nullptr;
float* g_sm121_scores = nullptr;
uint16_t* g_sm121_transposed_keys = nullptr;
constexpr unsigned int kSm121QueryBatch = 8u;
constexpr size_t kSm121ScoreElements =
    static_cast<size_t>(kSm121QueryBatch) * kQueryHeads * kQ8192Tokens;
constexpr size_t kSm121KeyElements =
    static_cast<size_t>(kQ8192Tokens) * kKvHeads * kHeadDim;

bool sm121_attention_enabled(unsigned int tokens) {
    const char* flag = std::getenv("QRT_CK_FMHA_SM121_FULL_PREFIX");
    return tokens > 0u && tokens <= kQ8192Tokens && flag && std::strcmp(flag, "1") == 0;
}

template<class Validate>
hipError_t load_sm121_table(const char* environment, size_t bytes,
    const unsigned char* expected_sha, Validate validate, unsigned char** device) {
#if !defined(_WIN32)
    (void)environment; (void)bytes; (void)expected_sha; (void)validate; (void)device;
    return hipErrorNotSupported;
#else
    const char* path = std::getenv(environment);
    if (!path || !*path) return hipErrorInvalidValue;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(bytes)) return hipErrorInvalidValue;
    std::vector<unsigned char> data;
    try { data.resize(bytes); } catch (const std::bad_alloc&) { return hipErrorOutOfMemory; }
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(data.data()), bytes) || !validate(data.data(), bytes))
        return hipErrorInvalidValue;
    BCRYPT_ALG_HANDLE algorithm = nullptr; unsigned char digest[32]{};
    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        return hipErrorInvalidValue;
    const auto hashed = BCryptHash(algorithm, nullptr, 0, data.data(), ULONG(bytes), digest, sizeof(digest));
    BCryptCloseAlgorithmProvider(algorithm, 0);
    if (hashed < 0 || std::memcmp(digest, expected_sha, sizeof(digest))) return hipErrorInvalidValue;
    auto status = hipMalloc(reinterpret_cast<void**>(device), bytes);
    if (status == hipSuccess) status = hipMemcpy(*device, data.data(), bytes, hipMemcpyHostToDevice);
    if (status != hipSuccess) { (void)hipFree(*device); *device = nullptr; }
    return status;
#endif
}

int prepare_sm121_attention_locked() {
    if (g_sm121_exp2 && g_sm121_rcp && g_sm121_scores && g_sm121_transposed_keys)
        return int(hipSuccess);
    unsigned char* exp2 = nullptr; unsigned char* rcp = nullptr;
    float* scores = nullptr;
    uint16_t* transposed_keys = nullptr;
    auto status = load_sm121_table("QRT_CK_FMHA_SM121_EXP2_TABLE", qrt_sm121_exp2::table_bytes,
        qrt_sm121_exp2::sha256, qrt_sm121_exp2::valid_layout, &exp2);
    if (status == hipSuccess)
        status = load_sm121_table("QRT_CK_FMHA_SM121_RCP_TABLE", qrt_sm121_attention_rcp::table_bytes,
            qrt_sm121_attention_rcp::sha256, qrt_sm121_attention_rcp::valid_layout, &rcp);
    if (status == hipSuccess)
        status = hipMalloc(reinterpret_cast<void**>(&scores), kSm121ScoreElements * sizeof(float));
    if (status == hipSuccess)
        status = hipMalloc(reinterpret_cast<void**>(&transposed_keys), kSm121KeyElements * sizeof(uint16_t));
    if (status != hipSuccess) {
        (void)hipFree(transposed_keys); (void)hipFree(scores); (void)hipFree(rcp); (void)hipFree(exp2);
        return int(status);
    }
    g_sm121_exp2 = exp2; g_sm121_rcp = rcp; g_sm121_scores = scores;
    g_sm121_transposed_keys = transposed_keys;
    std::fprintf(stderr, "SM121_FULL_ATTENTION_TABLES exp2_bytes=%zu rcp_bytes=%zu score_scratch_bytes=%zu key_scratch_bytes=%zu model_independent=1\n",
        size_t(qrt_sm121_exp2::table_bytes), qrt_sm121_attention_rcp::table_bytes,
        kSm121ScoreElements * sizeof(float), kSm121KeyElements * sizeof(uint16_t));
    return int(hipSuccess);
}

int prepare_sm121_attention() {
    std::lock_guard<std::mutex> lock(g_sm121_mutex);
    return prepare_sm121_attention_locked();
}

int launch_sm121_attention(const uint16_t* q, const uint16_t* k,
    const uint16_t* v, float* output, hipStream_t stream,
    unsigned int query_start, unsigned int query_count, unsigned int output_start) {
    if (!q || !k || !v || !output || query_count == 0u || query_start >= kQ8192Tokens ||
        query_count > kQ8192Tokens - query_start) return int(hipErrorInvalidValue);
    // Own tables, the 4 MiB score slab and 8 MiB transposed-key slab until all
    // submitted work completes. No request or release can reuse them early.
    std::lock_guard<std::mutex> lock(g_sm121_mutex);
    int status = prepare_sm121_attention_locked();
    if (status != int(hipSuccess)) return status;
    const auto begin = std::chrono::steady_clock::now();
    const bool independent_dots = query_count > 1u;
    const unsigned int key_stride = query_start + query_count;
    if (independent_dots) {
        status = qrt_blackwell_attention::transpose_keys(
            k, g_sm121_transposed_keys, kSm121KeyElements, key_stride, stream);
        if (status != int(hipSuccess)) {
            (void)hipStreamSynchronize(stream);
            return status;
        }
    }
    for (unsigned int offset = 0; offset < query_count; offset += kSm121QueryBatch) {
        status = qrt_blackwell_attention::launch_queries(q, k, v, output, stream,
            query_start + offset, std::min(kSm121QueryBatch, query_count - offset), output_start + offset,
            g_sm121_exp2, nullptr, nullptr, true, g_sm121_rcp, independent_dots ? 4u : 2u,
            g_sm121_scores, kSm121ScoreElements, nullptr, nullptr,
            independent_dots ? g_sm121_transposed_keys : nullptr, key_stride);
        if (status != int(hipSuccess)) {
            // QK can already be queued if submitting its PV consumer failed.
            (void)hipStreamSynchronize(stream);
            return status;
        }
        status = int(hipStreamSynchronize(stream));
        if (status != int(hipSuccess)) return status;
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count() > 20.0)
            return int(hipErrorLaunchTimeOut);
    }
    std::fprintf(stderr, "SM121_FULL_ATTENTION query_start=%u query_count=%u maximum_queries_per_dispatch=%u split_qk_pv=1 transposed_keys=%u diagnostic_only=1\n",
        query_start, query_count, kSm121QueryBatch, unsigned(independent_dots));
    return int(hipSuccess);
}

int launch_blackwell_exact_terminal(
    const uint16_t *q, const uint16_t *k, const uint16_t *v,
    float *output, hipStream_t stream, unsigned int tokens,
    unsigned int output_token) {
    if (tokens == 0u) return static_cast<int>(hipErrorInvalidValue);
    if (sm121_attention_enabled(tokens))
        return launch_sm121_attention(q, k, v, output, stream, tokens - 1u, 1u, output_token);
    return qrt_blackwell_attention::launch_queries(
        q, k, v, output, stream, tokens - 1u, 1u, output_token);
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
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    if (sm121_attention_enabled(kQ8192Tokens)) {
        const int status = prepare_sm121_attention();
        if (status != int(hipSuccess)) return status;
    }
#endif
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

#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    if (query_tokens == kv_tokens && sm121_attention_enabled(query_tokens))
        return launch_sm121_attention(q, k, v, output, stream, 0u, query_tokens, 0u);
#endif

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
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    {
        std::lock_guard<std::mutex> tables_lock(g_sm121_mutex);
        (void)hipFree(g_sm121_exp2);
        (void)hipFree(g_sm121_rcp);
        (void)hipFree(g_sm121_scores);
        (void)hipFree(g_sm121_transposed_keys);
        g_sm121_exp2 = nullptr;
        g_sm121_rcp = nullptr;
        g_sm121_scores = nullptr;
        g_sm121_transposed_keys = nullptr;
    }
#endif
    return static_cast<int>(hipSuccess);
}
