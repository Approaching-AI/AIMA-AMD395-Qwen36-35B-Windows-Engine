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
#include "attention_deadline.h"
#include "selective_qk.h"
#include "selective_qk_tail_policy.h"
#include "register_pv_policy.h"
#include "exact_attention_policy.h"
#include "fused_probability_pv_policy.h"
#include "fused_probability_pv.h"
#include "microtile_exact_qk.h"
#include "narrow_domain_qk_policy.h"
#include "narrow_domain_qk_workspace.h"
#include "rz_tree_qk.h"
#include "native_exp2_workspace.h"
#include "prepared_decoded_qk.h"
#include "exponent_mask_qk.h"
#include "prepared_decoded_qk_range.h"
#include "long_attention_pipeline.h"
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
qrt_native_exp2_workspace::Workspace g_sm121_native_exp2;
unsigned char* g_sm121_rcp = nullptr;
float* g_sm121_scores = nullptr;
float* g_sm121_mantissa_scores = nullptr;
float* g_sm121_selective_qk = nullptr;
uint16_t* g_sm121_transposed_keys = nullptr;
uint16_t* g_sm121_transposed_values = nullptr;
uint32_t* g_sm121_prepared_values = nullptr;
uint32_t* g_sm121_prepared_decoded_qk = nullptr;
unsigned* g_sm121_narrow_qk_domain = nullptr;
struct Sm121LongDecodedQkWorkspace {
    uint32_t* words = nullptr;
    unsigned capacity_tokens = 0u;
};
Sm121LongDecodedQkWorkspace g_sm121_long_decoded_qk;
struct Sm121LongPipelineWorkspace {
    float* scratch = nullptr;
    size_t scratch_elements = 0u;
    unsigned* domain = nullptr;
    unsigned domain_capacity = 0u;
};
Sm121LongPipelineWorkspace g_sm121_long_pipeline;
constexpr unsigned int kSm121QueryBatch = 8u;
constexpr unsigned int kSm121MatrixQueryBatch = 32u;
// Match the exact kernel's checked extent, including the first token beyond
// the historical q8192 bucket. Dispatch selection, score storage, key storage
// and launch validation must use the same capacity.
constexpr unsigned int kSm121MaxTokens = qrt_blackwell_attention::kSplitMaxTokens;
constexpr unsigned int kSm121InitialTokens = qrt_sm121_attention_capacity::kInitialTokens;
constexpr size_t kSm121ScoreElements =
    static_cast<size_t>(kSm121QueryBatch) * kQueryHeads * kSm121InitialTokens;
constexpr size_t kSm121MatrixScoreElements =
    static_cast<size_t>(kSm121MatrixQueryBatch) * kQueryHeads * kSm121InitialTokens;
constexpr size_t kSm121MantissaElements = kSm121MatrixScoreElements + kSm121MatrixScoreElements / 2u +
    static_cast<size_t>(kSm121MatrixQueryBatch) * kQueryHeads *
        (kSm121InitialTokens / 32u + 1u + 2u * kHeadDim) + 1u;
constexpr size_t kSm121KeyElements =
    static_cast<size_t>(kSm121InitialTokens) * kKvHeads * kHeadDim;
constexpr size_t kSm121ExtendedScoreElements = size_t(kSm121QueryBatch) * kQueryHeads * kSm121MaxTokens;
constexpr size_t kSm121ExtendedMatrixElements = size_t(kSm121MatrixQueryBatch) * kQueryHeads * kSm121MaxTokens;
constexpr size_t kSm121ExtendedMantissaElements = kSm121ExtendedMatrixElements + kSm121ExtendedMatrixElements / 2u +
    size_t(kSm121MatrixQueryBatch) * kQueryHeads * (kSm121MaxTokens / 32u + 1u + 2u * kHeadDim) + 1u;
constexpr size_t kSm121ExtendedKeyElements = size_t(kSm121MaxTokens) * kKvHeads * kHeadDim;
struct Sm121ExtendedWorkspace {
    float *scores = nullptr, *mantissa_scores = nullptr;
    uint16_t *transposed_keys = nullptr;
    uint32_t *prepared_values = nullptr;
};
Sm121ExtendedWorkspace g_sm121_extended;
constexpr size_t kSm121TransposedValueElements = size_t(8192u) * kKvHeads * kHeadDim;
struct Sm121LongValueWorkspace {
    uint16_t* cells = nullptr;
    unsigned capacity_tokens = 0u;
};
Sm121LongValueWorkspace g_sm121_long_values;
constexpr size_t kSm121SelectiveQkElements = size_t(128u) * kQueryHeads * (2u * 8192u + 256u) + 1u;

struct Sm121SuffixWorkspace {
    uint16_t* cells = nullptr;
    unsigned int capacity_tokens = 0u;
};
Sm121SuffixWorkspace g_sm121_suffix;
std::mutex g_sm121_suffix_mutex;

bool sm121_attention_enabled(unsigned int tokens) {
    const char* flag = std::getenv("QRT_CK_FMHA_SM121_FULL_PREFIX");
    return tokens > 0u && tokens <= kSm121MaxTokens && flag && std::strcmp(flag, "1") == 0;
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
    auto status = load_sm121_table("QRT_CK_FMHA_SM121_EXP2_TABLE", qrt_blackwell_attention::exp2_backend::table_bytes,
        qrt_blackwell_attention::exp2_backend::sha256, qrt_blackwell_attention::exp2_backend::valid_layout, &exp2);
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
        size_t(qrt_blackwell_attention::exp2_backend::table_bytes), qrt_sm121_attention_rcp::table_bytes,
        kSm121ScoreElements * sizeof(float), kSm121KeyElements * sizeof(uint16_t));
    return int(hipSuccess);
}

int prepare_sm121_attention() {
    std::lock_guard<std::mutex> lock(g_sm121_mutex);
    return prepare_sm121_attention_locked();
}

int prepare_sm121_extended_attention_locked() {
    if(g_sm121_extended.scores && g_sm121_extended.transposed_keys) return int(hipSuccess);
    // Publish an independent workspace only after both allocations succeed.
    // The existing short-context buffers remain owned until provider release.
    Sm121ExtendedWorkspace next;
    auto status = hipMalloc(reinterpret_cast<void**>(&next.scores),kSm121ExtendedScoreElements*sizeof(float));
    if(status == hipSuccess)
        status = hipMalloc(reinterpret_cast<void**>(&next.transposed_keys),kSm121ExtendedKeyElements*sizeof(uint16_t));
    if(status != hipSuccess) {
        (void)hipFree(next.transposed_keys);(void)hipFree(next.scores);return int(status);
    }
    g_sm121_extended = next;
    std::fprintf(stderr,"SM121_EXTENDED_ATTENTION_WORKSPACE initial_tokens=%u maximum_tokens=%u score_bytes=%zu key_bytes=%zu lazy_allocation=1\n",
        kSm121InitialTokens,kSm121MaxTokens,kSm121ExtendedScoreElements*sizeof(float),kSm121ExtendedKeyElements*sizeof(uint16_t));
    return int(hipSuccess);
}

int launch_sm121_attention(const uint16_t* q, const uint16_t* k,
    const uint16_t* v, float* output, hipStream_t stream,
    unsigned int query_start, unsigned int query_count, unsigned int output_start) {
    if (!q || !k || !v || !output || query_count == 0u || query_start >= kSm121MaxTokens ||
        query_count > kSm121MaxTokens - query_start || output_start >= kSm121MaxTokens ||
        query_count > kSm121MaxTokens - output_start) return int(hipErrorInvalidValue);
    // Expose the already isolated native MMA candidates to real-model gates.
    // 1 changes PV only; 2 changes QK and PV; 3 adds selective exact PV replay.
    // 4 changes QK only and retains the exact complete PV accumulator.
    // Online softmax, tile order and
    // reference SFU tables are shared with the exact route. Q1 stays exact.
    const char* matrix_option = std::getenv("QRT_CK_SM121_NATIVE_BF16_MATRIX");
    if (matrix_option && matrix_option[0] != '\0' &&
        std::strcmp(matrix_option, "0") != 0 && std::strcmp(matrix_option, "1") != 0 &&
        std::strcmp(matrix_option, "2") != 0 && std::strcmp(matrix_option, "3") != 0 &&
        std::strcmp(matrix_option, "4") != 0)
        return int(hipErrorInvalidValue);
    const unsigned matrix_mode = query_count > 1u && matrix_option &&
        (matrix_option[0] >= '1' && matrix_option[0] <= '4')
        ? static_cast<unsigned>(matrix_option[0] - '0') : 0u;
    const char* tiled_option = std::getenv("QRT_CK_SM121_TILED_EXACT_QK");
    if (tiled_option && *tiled_option && std::strcmp(tiled_option, "0") != 0 &&
        std::strcmp(tiled_option, "1") != 0) return int(hipErrorInvalidValue);
    const bool tiled_qk = query_count > 1u && tiled_option && std::strcmp(tiled_option, "1") == 0;
    const char* warp_option = std::getenv("QRT_CK_SM121_WARP_SOFTMAX");
    if (warp_option && *warp_option && std::strcmp(warp_option, "0") != 0 &&
        std::strcmp(warp_option, "1") != 0) return int(hipErrorInvalidValue);
    const bool warp_softmax = query_count > 1u && warp_option && std::strcmp(warp_option, "1") == 0;
    if (warp_softmax && !tiled_qk) return int(hipErrorInvalidValue);
    const char* prepared_option = std::getenv("QRT_CK_SM121_PREPARED_VALUE");
    if (prepared_option && *prepared_option && std::strcmp(prepared_option, "0") != 0 &&
        std::strcmp(prepared_option, "1") != 0) return int(hipErrorInvalidValue);
    const bool prepared_value = query_count > 1u && prepared_option && std::strcmp(prepared_option, "1") == 0;
    if (prepared_value && (!tiled_qk || warp_softmax)) return int(hipErrorInvalidValue);
    const char* compact_pv_option = std::getenv("QRT_CK_SM121_COMPACT_PV_REPLAY");
    if (compact_pv_option && *compact_pv_option && std::strcmp(compact_pv_option, "0") != 0 &&
        std::strcmp(compact_pv_option, "1") != 0 && std::strcmp(compact_pv_option, "2") != 0 && std::strcmp(compact_pv_option, "3") != 0)
        return int(hipErrorInvalidValue);
    // Mode2 keeps per-query replay as a control; mode3 parallelizes probability
    // generation while retaining global PV replay and ordered denominators.
    const unsigned compact_pv_mode = query_count > 1u && compact_pv_option &&
        (*compact_pv_option >= '1' && *compact_pv_option <= '3') ? unsigned(*compact_pv_option - '0') : 0u;
    if (compact_pv_mode && (!tiled_qk || warp_softmax || prepared_value)) return int(hipErrorInvalidValue);
    const char* batch_option = std::getenv("QRT_CK_SM121_PREFILL_QUERY_BATCH");
    unsigned requested_batch = kSm121MatrixQueryBatch;
    if (batch_option && *batch_option) {
        if (std::strcmp(batch_option, "32") == 0) requested_batch = 32u;
        else if (std::strcmp(batch_option, "64") == 0) requested_batch = 64u;
        else if (std::strcmp(batch_option, "128") == 0) requested_batch = 128u;
        else return int(hipErrorInvalidValue);
    }
    const char* value_option = std::getenv("QRT_CK_SM121_COMPACT_PV_TRANSPOSE_VALUE");
    if (value_option && *value_option && std::strcmp(value_option,"0") && std::strcmp(value_option,"1"))
        return int(hipErrorInvalidValue);
    const char* long_value_option = std::getenv("QRT_CK_SM121_LONG_TRANSPOSE_VALUE");
    if (long_value_option && *long_value_option && std::strcmp(long_value_option,"0") &&
        std::strcmp(long_value_option,"1")) return int(hipErrorInvalidValue);
    const bool long_transpose_value = long_value_option && std::strcmp(long_value_option,"1")==0 &&
        query_start+query_count>8192u;
    const bool transpose_value = value_option && std::strcmp(value_option,"1")==0 &&
        (compact_pv_mode==1u || compact_pv_mode==3u) &&
        (query_start+query_count<=8192u || long_transpose_value);
    const char* final_bound_option = std::getenv("QRT_CK_SM121_FINAL_PV_BOUND");
    if (final_bound_option && *final_bound_option && std::strcmp(final_bound_option,"0") &&
        std::strcmp(final_bound_option,"1")) return int(hipErrorInvalidValue);
    const bool final_pv_bound = final_bound_option && std::strcmp(final_bound_option,"1")==0 &&
        (compact_pv_mode==1u || compact_pv_mode==3u) && query_start+query_count<=8192u;
    const char* direct_pv_option = std::getenv("QRT_CK_SM121_DIRECT_PV_OPERANDS");
    if (direct_pv_option && *direct_pv_option && std::strcmp(direct_pv_option,"0") &&
        std::strcmp(direct_pv_option,"1")) return int(hipErrorInvalidValue);
    const char* long_direct_pv_option = std::getenv("QRT_CK_SM121_LONG_DIRECT_PV_OPERANDS");
    if (long_direct_pv_option && *long_direct_pv_option && std::strcmp(long_direct_pv_option,"0") &&
        std::strcmp(long_direct_pv_option,"1")) return int(hipErrorInvalidValue);
    const bool long_direct_pv = long_direct_pv_option && std::strcmp(long_direct_pv_option,"1")==0;
    const char* all_pv_option = std::getenv("QRT_CK_SM121_LONG_ALL_PV_REPLAY");
    if (all_pv_option && *all_pv_option && std::strcmp(all_pv_option,"0") &&
        std::strcmp(all_pv_option,"1")) return int(hipErrorInvalidValue);
    const bool all_pv_replay = all_pv_option && std::strcmp(all_pv_option,"1")==0 &&
        (compact_pv_mode==1u || compact_pv_mode==3u) && query_start >= 8192u;
    // Extend only operand loading to an explicitly selected long-history run.
    // Long calls retain the original per-group error envelope and exact replay;
    // the final-envelope proof above remains limited to its512 K16 groups.
    const bool direct_pv_operands = direct_pv_option && std::strcmp(direct_pv_option,"1")==0 &&
        (compact_pv_mode==1u || compact_pv_mode==3u) && (query_start+query_count<=8192u || long_direct_pv);
    const char* selective_option = std::getenv("QRT_CK_SM121_SELECTIVE_QK_PROBABILITY");
    if (selective_option && *selective_option && std::strcmp(selective_option,"0") &&
        std::strcmp(selective_option,"1")) return int(hipErrorInvalidValue);
    const bool selective_qk = selective_option && std::strcmp(selective_option,"1")==0 &&
        (compact_pv_mode==1u || compact_pv_mode==3u) && query_start+query_count<=8192u;
    unsigned exact_tail = 0u;
    if (!qrt_selective_qk_tail::parse(
            std::getenv("QRT_CK_SM121_SELECTIVE_QK_EXACT_TAIL"), exact_tail) ||
        (exact_tail && selective_option && std::strcmp(selective_option,"1")==0))
        return int(hipErrorInvalidValue);
    const unsigned selective_prefix_queries = qrt_selective_qk_tail::prefix_queries(
        exact_tail, query_start, query_count, requested_batch);
    if (selective_prefix_queries &&
        (!tiled_qk || (compact_pv_mode != 1u && compact_pv_mode != 3u)))
        return int(hipErrorInvalidValue);
    const bool uses_selective_qk = selective_qk || selective_prefix_queries != 0u;
    bool register_pv_rescale = false;
    if (!qrt_register_pv_policy::select(std::getenv("QRT_CK_SM121_REGISTER_PV_RESCALE"),
            query_start, query_count, final_pv_bound, direct_pv_operands,
            transpose_value, uses_selective_qk, register_pv_rescale))
        return int(hipErrorInvalidValue);
    const char* float_alignment_option = std::getenv("QRT_CK_SM121_FLOAT_ALIGNMENT_QK");
    if (float_alignment_option && *float_alignment_option && std::strcmp(float_alignment_option,"0") &&
        std::strcmp(float_alignment_option,"1")) return int(hipErrorInvalidValue);
    const bool float_alignment_qk = query_count > 1u && float_alignment_option &&
        std::strcmp(float_alignment_option,"1") == 0;
    if (float_alignment_qk && (!tiled_qk || selective_qk)) return int(hipErrorInvalidValue);
    const char* decoded_qk_option = std::getenv("QRT_CK_SM121_PREPARED_DECODED_QK");
    if (decoded_qk_option && *decoded_qk_option && std::strcmp(decoded_qk_option,"0") &&
        std::strcmp(decoded_qk_option,"1")) return int(hipErrorInvalidValue);
    const char* long_decoded_option = std::getenv("QRT_CK_SM121_LONG_PREPARED_DECODED_QK");
    if (long_decoded_option && *long_decoded_option && std::strcmp(long_decoded_option,"0") &&
        std::strcmp(long_decoded_option,"1")) return int(hipErrorInvalidValue);
    // The separate range owner encodes only this call's Q interval. The
    // fixed cold owner and single-query decode retain their established path.
    const bool prepared_decoded_qk = decoded_qk_option && std::strcmp(decoded_qk_option,"1")==0 &&
        query_start == 0u && query_count > 1u && query_count <= 8192u;
    const bool long_prepared_decoded_qk = decoded_qk_option && std::strcmp(decoded_qk_option,"1")==0 &&
        long_decoded_option && std::strcmp(long_decoded_option,"1")==0 &&
        query_count > 1u && query_count <= qrt_prepared_decoded_qk_range::maximum_queries &&
        query_start + query_count > 8192u;
    if ((prepared_decoded_qk || long_prepared_decoded_qk) && (!float_alignment_qk || selective_qk ||
        (compact_pv_mode != 1u && compact_pv_mode != 3u))) return int(hipErrorInvalidValue);
    const char* exponent_mask_option = std::getenv("QRT_CK_SM121_EXPONENT_MASK_QK");
    if (exponent_mask_option && *exponent_mask_option && std::strcmp(exponent_mask_option,"0") &&
        std::strcmp(exponent_mask_option,"1")) return int(hipErrorInvalidValue);
    const bool exponent_mask_requested = exponent_mask_option && std::strcmp(exponent_mask_option,"1") == 0;
    if (exponent_mask_requested && query_start == 0u && query_count > 1u && query_count <= 8192u &&
        !prepared_decoded_qk) return int(hipErrorInvalidValue);
    const bool exponent_mask_qk = exponent_mask_requested && prepared_decoded_qk;
#if defined(QRT_CK_SM121_INTERPOLATED_EXP2) && QRT_CK_SM121_INTERPOLATED_EXP2
    constexpr bool interpolated_exp2 = true;
#else
    constexpr bool interpolated_exp2 = false;
#endif
    bool exact_attention = false;
    if (!qrt_exact_attention_policy::select(std::getenv("QRT_CK_SM121_EXACT_ATTENTION_PIPELINE"),
            query_start, query_count, prepared_decoded_qk, exponent_mask_qk,
            register_pv_rescale, compact_pv_mode, interpolated_exp2, exact_attention))
        return int(hipErrorInvalidValue);
    bool fused_probability_pv = false;
    if (!qrt_fused_probability_pv_policy::select(std::getenv("QRT_CK_SM121_FUSED_PROBABILITY_PV"),
            query_start,query_count,exact_attention,fused_probability_pv))
        return int(hipErrorInvalidValue);
    const char* rz_tree_option = std::getenv("QRT_CK_SM121_QK_RZ_TREE");
    if (rz_tree_option && *rz_tree_option && std::strcmp(rz_tree_option, "0") &&
        std::strcmp(rz_tree_option, "1")) return int(hipErrorInvalidValue);
    const bool rz_tree_qk = rz_tree_option && std::strcmp(rz_tree_option, "1") == 0 &&
        query_start == 0u && query_count == 8192u;
    if (rz_tree_qk && (!exact_attention || !prepared_decoded_qk || exponent_mask_qk || matrix_mode))
        return int(hipErrorInvalidValue);
    bool narrow_domain_qk = false;
    if (!qrt_narrow_domain_qk_policy::select(std::getenv("QRT_CK_SM121_NARROW_DOMAIN_QK"),
            query_start,query_count,exact_attention,prepared_decoded_qk,exponent_mask_qk,
            rz_tree_qk,matrix_mode,narrow_domain_qk)) return int(hipErrorInvalidValue);
    bool long_pipeline = false;
    if (!qrt_long_attention_layout::select(std::getenv("QRT_CK_SM121_LONG_ATTENTION_PIPELINE"),
            query_start,query_count,long_prepared_decoded_qk,transpose_value,
            direct_pv_operands,compact_pv_mode,all_pv_replay,matrix_mode!=0u,
            uses_selective_qk,long_pipeline) || (long_pipeline && !interpolated_exp2))
        return int(hipErrorInvalidValue);
    const char* profile_option = std::getenv("QRT_CK_SM121_PROFILE_COMPLETED_STAGES");
    if (profile_option && *profile_option && std::strcmp(profile_option,"0") &&
        std::strcmp(profile_option,"1")) return int(hipErrorInvalidValue);
    const bool profile_stages = query_count > 1u && profile_option &&
        std::strcmp(profile_option,"1") == 0;
    if (profile_stages && (uses_selective_qk || (compact_pv_mode != 1u && compact_pv_mode != 3u)))
        return int(hipErrorInvalidValue);
    const char* submission_option = std::getenv("QRT_CK_SM121_SUBMIT_SLABS");
    unsigned requested_submission_slabs = 1u;
    if (submission_option && *submission_option) {
        if (std::strcmp(submission_option, "1") == 0) requested_submission_slabs = 1u;
        else if (std::strcmp(submission_option, "8") == 0) requested_submission_slabs = 8u;
        else if (std::strcmp(submission_option, "64") == 0) requested_submission_slabs = 64u;
        else return int(hipErrorInvalidValue);
    }
    const unsigned submission_slabs = query_start == 0u && query_count > 1u && query_count <= 8192u
        ? requested_submission_slabs : 1u;
    if (submission_slabs > 1u && (profile_stages || uses_selective_qk ||
        (compact_pv_mode != 1u && compact_pv_mode != 3u))) return int(hipErrorInvalidValue);
    // Own tables, score/probability slabs and the transposed-key slab until all
    // submitted work completes. No request or release can reuse them early.
    std::lock_guard<std::mutex> lock(g_sm121_mutex);
    int status = prepare_sm121_attention_locked();
    if (status != int(hipSuccess)) return status;
    if (exact_attention || long_pipeline) {
        const bool creating = !g_sm121_native_exp2.packed;
        status = qrt_native_exp2_workspace::prepare(g_sm121_native_exp2, g_sm121_exp2, stream);
        if (status != int(hipSuccess)) return status;
        if (creating) std::fprintf(stderr,
            "SM121_NATIVE_EXP2_STORAGE bytes=%zu verified_inputs=%u source_table_unchanged=1\n",
            qrt_sm121_exp2_native_delta::packed_bytes, qrt_sm121_exp2_native_delta::cells);
    }
    const auto begin = std::chrono::steady_clock::now();
    // Completed host clocks avoid driver event intervals that can be negative.
    // This observer adds synchronization only when explicitly requested. All
    // kernels, admission bounds, workspace ownership and the call deadline stay
    // unchanged. The instrumented wall is diagnostic, not retained performance.
    struct CompletedProfile {
        std::chrono::steady_clock::time_point last;
        uint64_t stage_ns[5]{};
        unsigned stage_calls[5]{};
        unsigned next_stage = 0u;
        bool all_pv_replay = false;
    } profile{begin};
    profile.all_pv_replay = all_pv_replay;
    const auto elapsed_ns = [](auto start, auto end) {
        return static_cast<uint64_t>(std::chrono::duration_cast<
            std::chrono::nanoseconds>(end - start).count());
    };
    uint64_t entry_wait_ns = 0u, preparation_ns = 0u;
    qrt_blackwell_attention::SplitCompletionObserver observer{&profile,
        [](void* state, unsigned stage, hipStream_t completed_stream) -> int {
            auto& p = *static_cast<CompletedProfile*>(state);
            if (stage >= 5u || stage != p.next_stage) return int(hipErrorInvalidValue);
            const int completion = int(hipStreamSynchronize(completed_stream));
            if (completion != int(hipSuccess)) return completion;
            const auto now = std::chrono::steady_clock::now();
            p.stage_ns[stage] += static_cast<uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(now - p.last).count());
            ++p.stage_calls[stage];
            p.next_stage = p.all_pv_replay && stage == 1u ? 4u : (stage + 1u) % 5u;
            p.last = now;
            return int(hipSuccess);
        }};
    if (profile_stages) {
        status = int(hipStreamSynchronize(stream));
        if (status != int(hipSuccess)) return status;
        profile.last = std::chrono::steady_clock::now();
        entry_wait_ns = elapsed_ns(begin, profile.last);
    }
    // Retain the original q8192 deadline while accounting for the additional
    // key history consumed by long-prefix query windows. Completion groups
    // drain before the progress check, and the outer process has its own bound.
    const qrt_sm121_attention_deadline::Budget deadline{
        query_start, query_count, kSm121MaxTokens};
    constexpr unsigned deadline_window_queries = decltype(deadline)::window_queries;
    const double call_deadline_seconds = deadline.call_limit_seconds();
    auto window_begin = begin;
    const bool independent_dots = query_count > 1u;
    const char* mantissa_option = std::getenv("QRT_CK_SM121_MANTISSA_WMMA");
    const bool mantissa_wmma = independent_dots && mantissa_option &&
        mantissa_option[0] != '\0' && std::strcmp(mantissa_option, "0") != 0;
    const char* native_product_option = std::getenv("QRT_CK_SM121_NATIVE_PRODUCTS");
    const bool native_products = independent_dots && !mantissa_wmma && native_product_option &&
        native_product_option[0] != '\0' && std::strcmp(native_product_option, "0") != 0;
    if ((matrix_mode && (mantissa_wmma || native_products || tiled_qk)) ||
        (tiled_qk && (mantissa_wmma || native_products))) return int(hipErrorInvalidValue);
    const bool expanded_scratch = mantissa_wmma || matrix_mode != 0u || tiled_qk;
    const unsigned int memory_layout = compact_pv_mode ? 21u + compact_pv_mode : prepared_value ? 17u : warp_softmax ? 16u : tiled_qk ? 15u : matrix_mode >= 3u ? 10u + matrix_mode : matrix_mode ? 5u + matrix_mode
        : (mantissa_wmma ? 5u : (independent_dots ? 4u : 2u));
    const bool wider_slab = (compact_pv_mode == 1u || compact_pv_mode == 3u) &&
        (query_start + query_count <= 8192u || long_pipeline);
    const unsigned int query_batch = wider_slab ? requested_batch :
        matrix_mode || tiled_qk ? kSm121MatrixQueryBatch : kSm121QueryBatch;
    // The existing long-context scratch already covers a 128-query q8192 slab.
    // Keep the same allocation size and ownership lock. Queued slabs reuse it
    // only after the preceding PV replay on the same stream has consumed it.
    static_assert(kSm121MantissaElements >=
        size_t(128u) * kQueryHeads * (8192u + 8192u / 2u + 256u + 1u + 2u * kHeadDim) + 1u);
    const unsigned int key_stride = query_start + query_count;
    const bool extended = key_stride > kSm121InitialTokens;
    if(extended) {
        status = prepare_sm121_extended_attention_locked();
        if(status != int(hipSuccess)) return status;
    }
    float* scores = extended ? g_sm121_extended.scores : g_sm121_scores;
    float*& mantissa_scores = extended ? g_sm121_extended.mantissa_scores : g_sm121_mantissa_scores;
    uint16_t* transposed_keys = extended ? g_sm121_extended.transposed_keys : g_sm121_transposed_keys;
    uint32_t*& prepared_values = extended ? g_sm121_extended.prepared_values : g_sm121_prepared_values;
    const size_t score_elements = extended ? kSm121ExtendedScoreElements : kSm121ScoreElements;
    const size_t mantissa_elements = extended ? kSm121ExtendedMantissaElements : kSm121MantissaElements;
    const size_t key_elements = extended ? kSm121ExtendedKeyElements : kSm121KeyElements;
    if (expanded_scratch && !mantissa_scores) {
        status = int(hipMalloc(reinterpret_cast<void**>(&mantissa_scores),
            mantissa_elements * sizeof(float)));
        if (status != int(hipSuccess)) { mantissa_scores = nullptr; return status; }
    }
    // Reuse the existing slab while it fits. Larger histories get a separate
    // checked owner, grown before any producer is submitted under this lease.
    float* long_scratch = mantissa_scores;
    size_t long_scratch_elements = mantissa_elements;
    if (long_pipeline) {
        const size_t required = qrt_long_attention_layout::layout(query_batch,key_stride).elements;
        if (!required) return int(hipErrorInvalidValue);
        if (required > mantissa_elements) {
            if (g_sm121_long_pipeline.scratch_elements < required) {
                const unsigned capacity = std::min(kSm121MaxTokens,(key_stride+8191u)/8192u*8192u);
                const size_t elements = qrt_long_attention_layout::layout(query_batch,capacity).elements;
                float* next = nullptr;
                status = int(hipMalloc(reinterpret_cast<void**>(&next),elements*sizeof(float)));
                if (status != int(hipSuccess)) return status;
                (void)hipFree(g_sm121_long_pipeline.scratch);
                g_sm121_long_pipeline.scratch = next;
                g_sm121_long_pipeline.scratch_elements = elements;
            }
            long_scratch = g_sm121_long_pipeline.scratch;
            long_scratch_elements = g_sm121_long_pipeline.scratch_elements;
        }
    }
    if (uses_selective_qk && !g_sm121_selective_qk) {
        status = int(hipMalloc(reinterpret_cast<void**>(&g_sm121_selective_qk),
            kSm121SelectiveQkElements * sizeof(float)));
        if (status != int(hipSuccess)) { g_sm121_selective_qk = nullptr; return status; }
    }
    if (transpose_value && !long_transpose_value && !g_sm121_transposed_values) {
        status = int(hipMalloc(reinterpret_cast<void**>(&g_sm121_transposed_values),
            kSm121TransposedValueElements * sizeof(uint16_t)));
        if (status != int(hipSuccess)) { g_sm121_transposed_values = nullptr; return status; }
    }
    if (transpose_value && long_transpose_value && g_sm121_long_values.capacity_tokens < key_stride) {
        // Keep the short owner fixed. Publish growth only after allocation
        // succeeds; a failed growth preserves the previous reusable owner.
        const unsigned capacity = std::min(kSm121MaxTokens, (key_stride + 8191u) / 8192u * 8192u);
        const size_t bytes = size_t(capacity) * kKvHeads * kHeadDim * sizeof(uint16_t);
        uint16_t* next = nullptr;
        status = int(hipMalloc(reinterpret_cast<void**>(&next), bytes));
        if (status != int(hipSuccess)) return status;
        const unsigned previous = g_sm121_long_values.capacity_tokens;
        (void)hipFree(g_sm121_long_values.cells);
        g_sm121_long_values = Sm121LongValueWorkspace{next, capacity};
        std::fprintf(stderr, "SM121_LONG_PV_VALUE_WORKSPACE previous_tokens=%u capacity_tokens=%u workspace_bytes=%zu independent_short_owner=1\n",
            previous, capacity, bytes);
    }
    uint16_t* transposed_value = long_transpose_value ? g_sm121_long_values.cells : g_sm121_transposed_values;
    const unsigned transposed_value_capacity = long_transpose_value ? g_sm121_long_values.capacity_tokens : 8192u;
    const size_t transposed_value_elements = size_t(transposed_value_capacity) * kKvHeads * kHeadDim;
    qrt_prepared_decoded_qk::Workspace decoded_workspace;
    qrt_narrow_domain_qk::Workspace narrow_workspace;
    qrt_blackwell_attention::SplitQkProducer decoded_producer{&decoded_workspace,
        rz_tree_qk ? qrt_rz_tree_qk::launch_workspace :
        exact_attention ? qrt_microtile_exact_qk::launch_workspace :
        exponent_mask_qk ? qrt_exponent_mask_qk::launch_workspace : qrt_prepared_decoded_qk::launch_workspace};
    qrt_blackwell_attention::SplitProbabilityProducer exact_probability{
        &g_sm121_native_exp2, qrt_native_exp2_workspace::launch};
    qrt_blackwell_attention::SplitProbabilityValueProducer fused_probability_value{
        &g_sm121_native_exp2, qrt_fused_probability_pv::launch};
    qrt_prepared_decoded_qk_range::Workspace long_decoded_workspace;
    qrt_long_narrow_qk::Workspace long_narrow_workspace;
    qrt_blackwell_attention::SplitQkProducer long_decoded_producer{&long_decoded_workspace,
        qrt_prepared_decoded_qk_range::launch_workspace};
    if (prepared_decoded_qk) {
        if (!g_sm121_prepared_decoded_qk) {
            status = int(hipMalloc(reinterpret_cast<void**>(&g_sm121_prepared_decoded_qk),
                qrt_prepared_decoded_qk::workspace_words * sizeof(uint32_t)));
            if (status != int(hipSuccess)) { g_sm121_prepared_decoded_qk = nullptr; return status; }
        }
        decoded_workspace = {g_sm121_prepared_decoded_qk, key_stride};
        // Reuse the arena but refresh its representation and matching consumer
        // together. Nothing from a previous call or layer is reused implicitly.
        status = rz_tree_qk ? qrt_rz_tree_qk::prepare_workspace(q, k, transposed_keys, decoded_workspace, stream)
            : exponent_mask_qk
            ? qrt_exponent_mask_qk::prepare_workspace(q, k, transposed_keys, decoded_workspace, stream)
            : qrt_prepared_decoded_qk::prepare_workspace(q, k, transposed_keys, decoded_workspace, stream);
        if (status != int(hipSuccess)) { (void)hipStreamSynchronize(stream); return status; }
    }
    if (narrow_domain_qk) {
        if (!g_sm121_narrow_qk_domain) {
            status = int(hipMalloc(reinterpret_cast<void**>(&g_sm121_narrow_qk_domain),
                qrt_narrow_domain_qk::domain_words * sizeof(unsigned)));
            if (status != int(hipSuccess)) {
                g_sm121_narrow_qk_domain = nullptr;
                (void)hipStreamSynchronize(stream);
                return status;
            }
        }
        narrow_workspace = qrt_narrow_domain_qk::attach(decoded_workspace,g_sm121_narrow_qk_domain);
        status = qrt_narrow_domain_qk::prepare_domain(q,k,narrow_workspace,stream);
        if (status != int(hipSuccess)) { (void)hipStreamSynchronize(stream); return status; }
        decoded_producer = {&narrow_workspace,qrt_narrow_domain_qk::launch_workspace};
    }
    if (long_prepared_decoded_qk) {
        if (g_sm121_long_decoded_qk.capacity_tokens < key_stride) {
            const unsigned capacity = std::min(kSm121MaxTokens, (key_stride + 8191u) / 8192u * 8192u);
            const size_t bytes = qrt_prepared_decoded_qk_range::workspace_words(capacity) * sizeof(uint32_t);
            uint32_t* next = nullptr;
            status = int(hipMalloc(reinterpret_cast<void**>(&next), bytes));
            if (status != int(hipSuccess)) return status;
            const unsigned previous = g_sm121_long_decoded_qk.capacity_tokens;
            (void)hipFree(g_sm121_long_decoded_qk.words);
            g_sm121_long_decoded_qk = Sm121LongDecodedQkWorkspace{next, capacity};
            std::fprintf(stderr, "SM121_LONG_DECODED_QK_WORKSPACE previous_tokens=%u capacity_tokens=%u workspace_bytes=%zu independent_short_owner=1\n",
                previous, capacity, bytes);
        }
        const unsigned capacity = g_sm121_long_decoded_qk.capacity_tokens;
        long_decoded_workspace = {g_sm121_long_decoded_qk.words,
            qrt_prepared_decoded_qk_range::workspace_words(capacity), capacity,
            query_start, query_count, key_stride};
        status = qrt_prepared_decoded_qk_range::prepare_workspace(q, k, transposed_keys,
            long_decoded_workspace, stream);
        if (status != int(hipSuccess)) { (void)hipStreamSynchronize(stream); return status; }
        if (long_pipeline) {
            constexpr size_t query_words = size_t(8192u)*16u;
            if (g_sm121_long_pipeline.domain_capacity < capacity) {
                const size_t words = query_words+size_t(capacity)*2u+2u;
                unsigned* next = nullptr;
                status = int(hipMalloc(reinterpret_cast<void**>(&next),words*sizeof(unsigned)));
                if (status != int(hipSuccess)) { (void)hipStreamSynchronize(stream); return status; }
                (void)hipFree(g_sm121_long_pipeline.domain);
                g_sm121_long_pipeline.domain = next;
                g_sm121_long_pipeline.domain_capacity = capacity;
            }
            auto* domain = g_sm121_long_pipeline.domain;
            long_narrow_workspace = {long_decoded_workspace,domain,domain+query_words,
                domain+query_words+size_t(g_sm121_long_pipeline.domain_capacity)*2u};
            status = qrt_long_narrow_qk::prepare_domain(q,k,long_narrow_workspace,stream);
            if (status != int(hipSuccess)) { (void)hipStreamSynchronize(stream); return status; }
        }
    }
    if (prepared_value) {
        if (!prepared_values) {
            status = int(hipMalloc(reinterpret_cast<void**>(&prepared_values),
                key_elements * sizeof(uint32_t)));
            if (status != int(hipSuccess)) { prepared_values = nullptr; return status; }
        }
        // Re-encode this call's V before any consumer; the allocation is reused,
        // but no model or layer identity is inferred from the source address.
        status = qrt_blackwell_attention::prepare_value_encoding(
            v, prepared_values, key_elements, key_stride, stream);
        if (status != int(hipSuccess)) {
            (void)hipStreamSynchronize(stream);
            return status;
        }
    }
    if (independent_dots && !prepared_decoded_qk && !long_prepared_decoded_qk) {
        status = qrt_blackwell_attention::transpose_keys(
            k, transposed_keys, key_elements, key_stride, stream);
        if (status != int(hipSuccess)) {
            (void)hipStreamSynchronize(stream);
            return status;
        }
    }
    if (transpose_value) {
        // Refresh each layer/call under the same workspace lease. The original
        // token-major V continues to feed the approximate matrix producer.
        status = qrt_blackwell_attention::transpose_keys(v, transposed_value,
            transposed_value_elements, key_stride, stream);
        if (status != int(hipSuccess)) { (void)hipStreamSynchronize(stream); return status; }
    }
    if (profile_stages) {
        status = int(hipStreamSynchronize(stream));
        if (status != int(hipSuccess)) return status;
        preparation_ns = elapsed_ns(profile.last, std::chrono::steady_clock::now());
    }
    unsigned pending_slabs = 0u, completion_groups = 0u;
    for (unsigned int offset = 0; offset < query_count; offset += query_batch) {
        if (profile_stages) profile.last = std::chrono::steady_clock::now();
        if (long_pipeline) {
            status = qrt_long_attention_pipeline::launch(long_narrow_workspace,g_sm121_native_exp2,
                q,transposed_keys,v,transposed_value,output,query_start+offset,
                std::min(query_batch,query_count-offset),output_start+offset,key_stride,
                g_sm121_exp2,g_sm121_rcp,long_scratch,long_scratch_elements,stream,
                profile_stages?&observer:nullptr);
        } else if (selective_qk || offset < selective_prefix_queries) {
            status = qrt_selective_qk::launch_probability_attention(q, k, v, output, stream,
                query_start + offset, std::min(query_batch, query_count - offset), output_start + offset,
                g_sm121_exp2, g_sm121_rcp, memory_layout, mantissa_scores, mantissa_elements,
                g_sm121_selective_qk, kSm121SelectiveQkElements, transposed_keys, key_stride,
                transpose_value ? transposed_value : nullptr, transpose_value ? key_stride : 0u,
                final_pv_bound, direct_pv_operands);
        } else {
        status = qrt_blackwell_attention::launch_queries(q, k, v, output, stream,
            query_start + offset, std::min(query_batch, query_count - offset), output_start + offset,
            g_sm121_exp2, nullptr, nullptr, true, g_sm121_rcp, memory_layout,
            expanded_scratch ? mantissa_scores : scores,
            expanded_scratch ? mantissa_elements : score_elements, nullptr, nullptr,
            independent_dots ? transposed_keys : nullptr, key_stride, native_products, nullptr,
            prepared_value ? prepared_values : nullptr, prepared_value ? key_stride : 0u,
            nullptr, profile_stages ? &observer : nullptr,
            transpose_value ? transposed_value : nullptr, transpose_value ? key_stride : 0u,
            1u, 1u, final_pv_bound, direct_pv_operands, float_alignment_qk, 0u, false,
            prepared_decoded_qk ? &decoded_producer : long_prepared_decoded_qk ? &long_decoded_producer : nullptr,
            all_pv_replay, register_pv_rescale, exact_attention && !fused_probability_pv ? &exact_probability : nullptr,
            fused_probability_pv ? &fused_probability_value : nullptr);
        }
        if (status != int(hipSuccess)) {
            // Earlier slabs and this QK may be queued when a consumer fails.
            (void)hipStreamSynchronize(stream);
            return status;
        }
        ++pending_slabs;
        const unsigned submitted_queries = offset + std::min(query_batch, query_count - offset);
        if (pending_slabs < submission_slabs && submitted_queries < query_count &&
            submitted_queries % deadline_window_queries != 0u) {
            // Bound host submission as well as completed GPU work. A slow or
            // blocking submission forces a drain before reporting a timeout.
            const auto submitted_at = std::chrono::steady_clock::now();
            const double submitted_call_seconds = std::chrono::duration<double>(submitted_at - begin).count();
            const double submitted_window_seconds = std::chrono::duration<double>(submitted_at - window_begin).count();
            if (std::isfinite(submitted_call_seconds) && std::isfinite(submitted_window_seconds) &&
                submitted_call_seconds <= call_deadline_seconds &&
                submitted_window_seconds <= deadline.window_limit_seconds(submitted_queries)) continue;
        }
        status = int(hipStreamSynchronize(stream));
        if (status != int(hipSuccess)) return status;
        pending_slabs = 0u;
        ++completion_groups;
        const auto now = std::chrono::steady_clock::now();
        const double call_seconds = std::chrono::duration<double>(now - begin).count();
        const double window_seconds = std::chrono::duration<double>(now - window_begin).count();
        const unsigned completed_queries = submitted_queries;
        const double window_limit_seconds = deadline.window_limit_seconds(completed_queries);
        if (!std::isfinite(call_seconds) || !std::isfinite(window_seconds) ||
            call_seconds > call_deadline_seconds || window_seconds > window_limit_seconds) {
            std::fprintf(stderr, "SM121_FULL_ATTENTION_DEADLINE query_start=%u query_count=%u completed_queries=%u call_seconds=%.6f call_limit_seconds=%.1f window_seconds=%.6f window_limit_seconds=%.1f history_work_budget=1 application_deadline=1 stream_drained=1\n",
                query_start, query_count, completed_queries, call_seconds, call_deadline_seconds,
                window_seconds, window_limit_seconds);
            return int(hipErrorLaunchTimeOut);
        }
        if (completed_queries % deadline_window_queries == 0u) window_begin = now;
    }
    if (submission_slabs > 1u)
        std::fprintf(stderr,"SM121_SUBMIT_SLABS query_start=%u query_count=%u query_batch=%u maximum_slabs_per_completion=%u completion_groups=%u same_stream=1 workspace_reused=1 stream_drained=1\n",
            query_start,query_count,query_batch,submission_slabs,completion_groups);
    if (profile_stages) {
        const uint64_t total_ns = elapsed_ns(begin, std::chrono::steady_clock::now());
        uint64_t accounted_ns = entry_wait_ns + preparation_ns;
        const unsigned batches = (query_count + query_batch - 1u) / query_batch;
        for (unsigned stage = 0u; stage < 5u; ++stage) {
            const bool active = !all_pv_replay || stage == 0u || stage == 1u || stage == 4u;
            if (profile.stage_calls[stage] != (active ? batches : 0u) ||
                (!active && profile.stage_ns[stage])) return int(hipErrorInvalidValue);
            accounted_ns += profile.stage_ns[stage];
        }
        if (profile.next_stage || accounted_ns > total_ns) return int(hipErrorInvalidValue);
        std::fprintf(stderr, "SM121_COMPLETED_STAGE_PROFILE query_start=%u query_count=%u query_batch=%u batches=%u "
            "entry_wait_ms=%.6f preparation_ms=%.6f qk_ms=%.6f probability_ms=%.6f "
            "approximate_pv_ms=%.6f collect_pv_ms=%.6f exact_pv_ms=%.6f "
            "dispatch_remainder_ms=%.6f total_ms=%.6f completed_stages=%u all_pv_replay=%u "
            "clock=steady_host stream_drained=1 additional_device_bytes=0 instrumented=1\n",
            query_start, query_count, query_batch, batches,
            double(entry_wait_ns) / 1e6, double(preparation_ns) / 1e6,
            double(profile.stage_ns[0]) / 1e6, double(profile.stage_ns[1]) / 1e6,
            double(profile.stage_ns[2]) / 1e6, double(profile.stage_ns[3]) / 1e6,
            double(profile.stage_ns[4]) / 1e6, double(total_ns - accounted_ns) / 1e6,
            double(total_ns) / 1e6, batches * (all_pv_replay ? 3u : 5u), unsigned(all_pv_replay));
    }
    if (transpose_value)
        std::fprintf(stderr,"SM121_TRANSPOSED_PV_VALUE query_start=%u query_count=%u value_tokens=%u workspace_bytes=%zu refreshed=1 capacity_tokens=%u long_workspace=%u\n",
            query_start,query_count,key_stride,transposed_value_elements*sizeof(uint16_t),
            transposed_value_capacity,unsigned(long_transpose_value));
    if (long_pipeline) {
        unsigned counts[2]{};
        status = int(hipMemcpy(counts,long_narrow_workspace.tile_counts,sizeof(counts),hipMemcpyDeviceToHost));
        if (status != int(hipSuccess)) return status;
        const size_t domain_bytes = (size_t(8192u)*16u+size_t(g_sm121_long_pipeline.domain_capacity)*2u+2u)*sizeof(unsigned);
        std::fprintf(stderr,"SM121_LONG_ATTENTION_PIPELINE query_start=%u query_count=%u query_batch=%u key_tokens=%u narrow_tiles=%u original_tiles=%u original_k16=1 original_per_group_bounds=1 fused_probability_pv=1 register_exact_rescale=1 refreshed=1 scratch_bytes=%zu domain_bytes=%zu additional_scratch_bytes=%zu stream_drained=1\n",
            query_start,query_count,query_batch,key_stride,counts[1],counts[0],
            long_scratch_elements*sizeof(float),domain_bytes,g_sm121_long_pipeline.scratch_elements*sizeof(float));
    }
    if (final_pv_bound)
        std::fprintf(stderr,"SM121_FINAL_PV_BOUND query_start=%u query_count=%u maximum_k16_groups=512 enlarged_envelope=1\n",
            query_start,query_count);
    if (direct_pv_operands && !all_pv_replay)
        std::fprintf(stderr,"SM121_DIRECT_PV_OPERANDS query_start=%u query_count=%u token_major_values=1 additional_workspace_bytes=0\n",
            query_start,query_count);
    if (register_pv_rescale)
        std::fprintf(stderr,"SM121_REGISTER_PV_RESCALE query_start=%u query_count=%u native_and_exact=1 original_k16=1 additional_workspace_bytes=0\n",
            query_start,query_count);
    if (exact_attention && !rz_tree_qk)
        std::fprintf(stderr,"SM121_EXACT_ATTENTION_PIPELINE query_start=%u query_count=%u query_rows=32 key_columns=%u exact_native_exp=1 verified_inputs=%u workspace_bytes=%zu\n",
            query_start,query_count,narrow_domain_qk?64u:32u,qrt_sm121_exp2_native_delta::cells,qrt_sm121_exp2_native_delta::packed_bytes);
    if (narrow_domain_qk) {
        unsigned counts[2]{};
        status = int(hipMemcpy(counts,narrow_workspace.tile_counts,sizeof(counts),hipMemcpyDeviceToHost));
        if (status != int(hipSuccess)) return status;
        std::fprintf(stderr,"SM121_NARROW_DOMAIN_QK query_start=%u query_count=%u query_rows=32 key_columns=64 window=64 bf16_exponents=95:159 classified_rows=%u narrow_tiles=%u original_tiles=%u original_k16=1 complete_fallback=1 refreshed=1 workspace_bytes=%zu stream_drained=1\n",
            query_start,query_count,key_stride*18u,counts[1],counts[0],qrt_narrow_domain_qk::domain_words*sizeof(unsigned));
    }
    if (rz_tree_qk)
        std::fprintf(stderr,"SM121_QK_RZ_TREE query_start=%u query_count=%u query_rows=32 key_columns=32 group_size=16 original_qk=0 narrow_bf16_exponents=96:158 original_exceptional_dot_replay=1 wave_mode_restored=1 exact_native_exp=1 original_pv=1 additional_workspace_bytes=0 gb10_product_gate_required=1\n",
            query_start,query_count);
    if (all_pv_replay)
        std::fprintf(stderr,"SM121_ALL_PV_REPLAY query_start=%u query_count=%u original_k16=1 approximate_pv=0 compaction=0 additional_workspace_bytes=0\n",
            query_start,query_count);
    if (fused_probability_pv)
        std::fprintf(stderr,"SM121_FUSED_PROBABILITY_PV query_start=%u query_count=%u independent_exact_qk=%u original_probability_recurrence=1 original_final_bound=1 original_exact_replay=1 additional_workspace_bytes=0 profile_stage1_includes_native_pv=1\n",
            query_start,query_count,unsigned(!rz_tree_qk));
    if (float_alignment_qk)
        std::fprintf(stderr,"SM121_FLOAT_ALIGNMENT_QK query_start=%u query_count=%u canonical_k16=%u original_fallback=1 additional_workspace_bytes=0 selective_prefix_queries=%u\n",
            query_start,query_count,unsigned(!rz_tree_qk),selective_prefix_queries);
    if (prepared_decoded_qk)
        std::fprintf(stderr,"SM121_PREPARED_DECODED_QK query_start=%u query_count=%u window=%u query_rows=%u key_columns=%u workspace_bytes=%zu refreshed=1 original_fallback=1\n",
            query_start,query_count,narrow_domain_qk?64u:128u,exact_attention?32u:16u,narrow_domain_qk?64u:exact_attention?32u:16u,qrt_prepared_decoded_qk::workspace_words*sizeof(uint32_t));
    if (exponent_mask_qk)
        std::fprintf(stderr,"SM121_EXPONENT_MASK_QK query_start=%u query_count=%u exact_maximum=1 carry_bound=1 original_fallback=1 refreshed=1 additional_workspace_bytes=0\n",
            query_start,query_count);
    if (long_prepared_decoded_qk && !long_pipeline)
        std::fprintf(stderr,"SM121_LONG_PREPARED_DECODED_QK query_start=%u query_count=%u key_tokens=%u capacity_tokens=%u window=128 query_rows=16 key_columns=16 workspace_bytes=%zu refreshed=1 original_fallback=1 prepared_query_history=0\n",
            query_start,query_count,key_stride,long_decoded_workspace.key_capacity,
            long_decoded_workspace.word_count*sizeof(uint32_t));
    if (selective_qk)
        std::fprintf(stderr,"SM121_SELECTIVE_QK_PROBABILITY query_start=%u query_count=%u probability_endpoint_repair=1 approximate_denominator=1 workspace_bytes=%zu gb10_product_gate_required=1\n",
            query_start,query_count,kSm121SelectiveQkElements*sizeof(float));
    if (exact_tail && query_start == 0u && query_count > 1u && query_count <= 8192u)
        std::fprintf(stderr,"SM121_SELECTIVE_QK_EXACT_TAIL query_start=%u query_count=%u requested_tail=%u selective_queries=%u exact_queries=%u query_batch=%u probability_endpoint_repair=1 approximate_prefix_denominator=%u original_suffix_accumulator=1 stream_drained=1 gb10_product_gate_required=1\n",
            query_start,query_count,exact_tail,selective_prefix_queries,
            query_count-selective_prefix_queries,query_batch,unsigned(selective_prefix_queries != 0u));
    std::fprintf(stderr, "SM121_FULL_ATTENTION query_start=%u query_count=%u maximum_queries_per_dispatch=%u split_qk_pv=1 transposed_keys=%u native_products=%u mantissa_wmma=%u native_bf16_matrix=%u tiled_exact_qk=%u warp_softmax=%u prepared_value=%u diagnostic_only=1\n",
        query_start, query_count, query_batch, unsigned(independent_dots), unsigned(native_products), unsigned(mantissa_wmma), matrix_mode, unsigned(tiled_qk && !rz_tree_qk), unsigned(warp_softmax), unsigned(prepared_value));
    return int(hipSuccess);
}

int launch_sm121_suffix_attention(
    const uint16_t* q, const uint16_t* prefix_k, const uint16_t* prefix_v,
    const uint16_t* suffix_k, const uint16_t* suffix_v, float* output,
    hipStream_t stream, unsigned int prefix_tokens, unsigned int suffix_tokens) {
    if (!prefix_tokens || prefix_tokens >= kSm121MaxTokens || !suffix_tokens ||
        suffix_tokens > kPrefillChunkTokens || suffix_tokens > kSm121MaxTokens - prefix_tokens)
        return int(hipErrorInvalidValue);
    const unsigned int total = prefix_tokens + suffix_tokens;
    const void* inputs[] = {q, prefix_k, prefix_v, suffix_k, suffix_v};
    const size_t prefix_bytes = size_t(prefix_tokens) * kKvFeatures * sizeof(uint16_t);
    const size_t suffix_bytes = size_t(suffix_tokens) * kKvFeatures * sizeof(uint16_t);
    const size_t query_bytes = size_t(suffix_tokens) * kQueryFeatures * sizeof(uint16_t);
    const size_t sizes[] = {query_bytes, prefix_bytes, prefix_bytes, suffix_bytes, suffix_bytes};
    const size_t output_bytes = size_t(suffix_tokens) * kQueryFeatures * sizeof(float);
    const uintptr_t destination = reinterpret_cast<uintptr_t>(output);
    if (!destination || destination % alignof(float) || output_bytes > UINTPTR_MAX - destination)
        return int(hipErrorInvalidValue);
    for (unsigned int i = 0u; i < 5u; ++i) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(inputs[i]);
        if (!address || address % alignof(uint16_t) || sizes[i] > UINTPTR_MAX - address ||
            (destination < address + sizes[i] && address < destination + output_bytes))
            return int(hipErrorInvalidValue);
    }
    if (!sm121_attention_enabled(total)) return int(hipErrorNotSupported);
    // The exact kernels index Q by absolute position. Stage the compact suffix
    // at that position and assemble the complete KV history without recomputing
    // prefix projections. Prefix Q cells are never read by this query span.
    // Every call refreshes all consumed cells, even when the addresses repeat.
    std::lock_guard<std::mutex> lock(g_sm121_suffix_mutex);
    if (g_sm121_suffix.capacity_tokens < total) {
        uint16_t* next = nullptr;
        const auto status = hipMalloc(reinterpret_cast<void**>(&next),
            size_t(total) * (kQueryFeatures + 2u * kKvFeatures) * sizeof(uint16_t));
        if (status != hipSuccess) return int(status);
        (void)hipFree(g_sm121_suffix.cells);
        g_sm121_suffix = Sm121SuffixWorkspace{next, total};
    }
    auto* staged_q = g_sm121_suffix.cells;
    auto* staged_k = staged_q + size_t(g_sm121_suffix.capacity_tokens) * kQueryFeatures;
    auto* staged_v = staged_k + size_t(g_sm121_suffix.capacity_tokens) * kKvFeatures;
    void* destinations[] = {staged_q + size_t(prefix_tokens) * kQueryFeatures,
        staged_k, staged_v, staged_k + size_t(prefix_tokens) * kKvFeatures,
        staged_v + size_t(prefix_tokens) * kKvFeatures};
    for (unsigned int i = 0u; i < 5u; ++i) {
        const auto status = hipMemcpyAsync(destinations[i], inputs[i], sizes[i], hipMemcpyDeviceToDevice, stream);
        if (status != hipSuccess) {
            (void)hipStreamSynchronize(stream);
            return int(status);
        }
    }
    const int status = launch_sm121_attention(staged_q, staged_k, staged_v, output,
        stream, prefix_tokens, suffix_tokens, 0u);
    // Drain submitted copies too when a later launcher validation fails. The
    // caller can immediately discard its transaction after any failed call.
    if (status != int(hipSuccess)) (void)hipStreamSynchronize(stream);
    return status;
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

bool supported_dynamic_tokens(unsigned int tokens) {
    if(tokens && tokens <= kQ262144Tokens) return true;
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    return sm121_attention_enabled(tokens);
#else
    return false;
#endif
}

int prepare_locked(unsigned int tokens) {
    // Full-prefix exports remain exact-shape surfaces, but the terminal-Q1
    // export consumes a runtime KV length.  Its storage has the same bounded
    // Q/K/V layout, so allow any product context while keeping callers of the
    // full-prefix helpers guarded by supported_tokens().
    if (!supported_dynamic_tokens(tokens)) {
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
        !supported_dynamic_tokens(tokens)) {
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
    if (!supported_dynamic_tokens(tokens)) {
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

// Optional exact continuation ABI. Q and suffix K/V are compact token-major
// BF16; prefix K/V are immutable token-major BF16. Output is suffix-local F32.
QRT_CK_EXPORT int qrt_ck_fmha_sm121_suffix_bf16_v1(
    const uint16_t* q, const uint16_t* prefix_k, const uint16_t* prefix_v,
    const uint16_t* suffix_k, const uint16_t* suffix_v, float* output,
    void* stream, unsigned int prefix_tokens, unsigned int suffix_tokens) {
#if defined(QRT_CK_FMHA_BLACKWELL_EXACT_TERMINAL)
    return launch_sm121_suffix_attention(q, prefix_k, prefix_v, suffix_k, suffix_v,
        output, reinterpret_cast<hipStream_t>(stream), prefix_tokens, suffix_tokens);
#else
    return int(hipErrorNotSupported);
#endif
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
    std::lock_guard<std::mutex> suffix_lock(g_sm121_suffix_mutex);
    (void)hipFree(g_sm121_suffix.cells);
    g_sm121_suffix = Sm121SuffixWorkspace{};
    {
        std::lock_guard<std::mutex> tables_lock(g_sm121_mutex);
        qrt_native_exp2_workspace::release(g_sm121_native_exp2);
        (void)hipFree(g_sm121_exp2);
        (void)hipFree(g_sm121_rcp);
        (void)hipFree(g_sm121_scores);
        (void)hipFree(g_sm121_mantissa_scores);
        (void)hipFree(g_sm121_selective_qk);
        (void)hipFree(g_sm121_transposed_keys);
        (void)hipFree(g_sm121_transposed_values);
        (void)hipFree(g_sm121_long_values.cells);
        g_sm121_long_values = Sm121LongValueWorkspace{};
        (void)hipFree(g_sm121_prepared_values);
        (void)hipFree(g_sm121_prepared_decoded_qk);
        (void)hipFree(g_sm121_narrow_qk_domain);
        (void)hipFree(g_sm121_long_decoded_qk.words);
        g_sm121_long_decoded_qk = Sm121LongDecodedQkWorkspace{};
        (void)hipFree(g_sm121_long_pipeline.scratch);
        (void)hipFree(g_sm121_long_pipeline.domain);
        g_sm121_long_pipeline = Sm121LongPipelineWorkspace{};
        (void)hipFree(g_sm121_extended.prepared_values);
        (void)hipFree(g_sm121_extended.transposed_keys);
        (void)hipFree(g_sm121_extended.mantissa_scores);
        (void)hipFree(g_sm121_extended.scores);
        g_sm121_extended = Sm121ExtendedWorkspace{};
        g_sm121_exp2 = nullptr;
        g_sm121_rcp = nullptr;
        g_sm121_scores = nullptr;
        g_sm121_mantissa_scores = nullptr;
        g_sm121_selective_qk = nullptr;
        g_sm121_transposed_keys = nullptr;
        g_sm121_transposed_values = nullptr;
        g_sm121_prepared_values = nullptr;
        g_sm121_prepared_decoded_qk = nullptr;
        g_sm121_narrow_qk_domain = nullptr;
    }
#endif
    return static_cast<int>(hipSuccess);
}
