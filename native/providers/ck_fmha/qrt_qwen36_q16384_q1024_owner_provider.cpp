#define QRT_D232_DISABLE_MAIN
#include "q16384_suffix1024_ordered_forty_layer_owner_smoke.cpp"
#undef QRT_D232_DISABLE_MAIN

#include "../../src/c/qrt_qwen36_q1024_owner.h"

#include <chrono>
#include <memory>
#include <mutex>

#ifndef QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS
#define QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS 1
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_WIDE_PREFIX_PV_SOLUTION
#define QRT_QWEN36_Q1024_OWNER_WIDE_PREFIX_PV_SOLUTION 0
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_FUSED_SOFTMAX_PROBABILITY
#define QRT_QWEN36_Q1024_OWNER_FUSED_SOFTMAX_PROBABILITY 0
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_CONV
#define QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_CONV 0
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN
#define QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN 0
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN_SHARED_QK
#define QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN_SHARED_QK 0
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_SOFTMAX_REGISTER_DENOMINATOR
#define QRT_QWEN36_Q1024_OWNER_SOFTMAX_REGISTER_DENOMINATOR 0
#endif

#ifndef QRT_QWEN36_Q1024_OWNER_Q1_ATTENTION_QUERY_CHUNK
#define QRT_QWEN36_Q1024_OWNER_Q1_ATTENTION_QUERY_CHUNK 128
#endif

#ifdef _WIN32
#define QRT_Q1024_OWNER_EXPORT extern "C" __declspec(dllexport)
#define QRT_Q1024_OWNER_CALL __cdecl
#else
#define QRT_Q1024_OWNER_EXPORT extern "C"
#define QRT_Q1024_OWNER_CALL
#endif

namespace {

constexpr unsigned int kD233Vocab = 248320u;
constexpr size_t kD233LogitElements =
    static_cast<size_t>(kSuffixTokens) * kD233Vocab;
constexpr size_t kD233FullQProjectionElements =
    static_cast<size_t>(kSuffixTokens) * kD231QProjectionRows;
constexpr size_t kD233FullKProjectionElements =
    static_cast<size_t>(kSuffixTokens) * kD231KProjectionRows;
constexpr size_t kD233FullVProjectionElements =
    static_cast<size_t>(kSuffixTokens) * kD231VProjectionRows;
constexpr unsigned int kD233FullQkvRows =
    kD231QProjectionRows + kD231KProjectionRows + kD231VProjectionRows;
constexpr size_t kD233FullQkvProjectionElements =
    static_cast<size_t>(kSuffixTokens) * kD233FullQkvRows;
constexpr unsigned int kD233LayerDigestCount = kD232Layers + 1u;
constexpr unsigned int kD233Layer0StageDigestCount = 14u;
constexpr unsigned int kD233Q1AttentionMaxTailTokens = 1536u;
constexpr unsigned int kD233Q1AttentionScoreStride =
    kPrefixTokens + kD233Q1AttentionMaxTailTokens + 1u;
constexpr unsigned int kD233Q1AttentionQueryChunk =
    QRT_QWEN36_Q1024_OWNER_Q1_ATTENTION_QUERY_CHUNK;
constexpr unsigned int kD233Q1AttentionChunkCount =
    kSuffixTokens / kD233Q1AttentionQueryChunk;
constexpr unsigned int kD233Q1AttentionProfileBoundaryCount = 8u;
constexpr unsigned int kD233HeadsPerKv =
    kD231QueryHeads / kD231KvHeads;
constexpr unsigned int kD233ExactTailStreams =
    QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS;
constexpr rocblas_int kD233WidePrefixPvSolution =
    QRT_QWEN36_Q1024_OWNER_WIDE_PREFIX_PV_SOLUTION;
constexpr bool kD233FusedSoftmaxProbability =
    QRT_QWEN36_Q1024_OWNER_FUSED_SOFTMAX_PROBABILITY != 0;
constexpr bool kD233PersistentExactQ1Conv =
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_CONV != 0;
constexpr bool kD233PersistentExactQ1Gdn =
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN != 0;
constexpr bool kD233PersistentExactQ1GdnSharedQk =
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN_SHARED_QK != 0;
constexpr bool kD233SoftmaxRegisterDenominator =
    QRT_QWEN36_Q1024_OWNER_SOFTMAX_REGISTER_DENOMINATOR != 0;
constexpr size_t kD233Q1AttentionScoreElements =
    static_cast<size_t>(kD233Q1AttentionQueryChunk) *
    kD231QueryHeads * kD233Q1AttentionScoreStride;

static_assert(kD233Vocab == 248320u);
static_assert(kD232Layers == QRT_QWEN36_Q1024_OWNER_LAYER_COUNT);
static_assert(kPrefixTokens == QRT_QWEN36_Q1024_OWNER_PREFIX_TOKENS);
static_assert(kSuffixTokens == QRT_QWEN36_Q1024_OWNER_SUFFIX_TOKENS);
static_assert(kD231QueryHeads == 16u);
static_assert(kD231KvHeads == 2u);
static_assert(kD231HeadDim == 256u);
static_assert(kD233HeadsPerKv == 8u);
static_assert(
    kD233ExactTailStreams == 1u ||
    kD233ExactTailStreams == 2u ||
    kD233ExactTailStreams == 4u ||
    kD233ExactTailStreams == 8u
);
static_assert(kD233WidePrefixPvSolution <= 0);
static_assert(
    QRT_QWEN36_Q1024_OWNER_FUSED_SOFTMAX_PROBABILITY == 0 ||
    QRT_QWEN36_Q1024_OWNER_FUSED_SOFTMAX_PROBABILITY == 1
);
static_assert(
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_CONV == 0 ||
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_CONV == 1
);
static_assert(
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN == 0 ||
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN == 1
);
static_assert(
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN_SHARED_QK == 0 ||
    QRT_QWEN36_Q1024_OWNER_PERSISTENT_EXACT_Q1_GDN_SHARED_QK == 1
);
static_assert(
    QRT_QWEN36_Q1024_OWNER_SOFTMAX_REGISTER_DENOMINATOR == 0 ||
    QRT_QWEN36_Q1024_OWNER_SOFTMAX_REGISTER_DENOMINATOR == 1
);
static_assert(
    kD233Q1AttentionQueryChunk == 128u ||
    kD233Q1AttentionQueryChunk == 256u ||
    kD233Q1AttentionQueryChunk == 512u ||
    kD233Q1AttentionQueryChunk == 1024u
);
static_assert(kKeyDim == kValueDim);
static_assert(kD233Q1AttentionMaxTailTokens >= kSuffixTokens);
static_assert(kD233Q1AttentionQueryChunk <= kSuffixTokens);
static_assert(
    (kSuffixTokens % kD233Q1AttentionQueryChunk) == 0u
);
static_assert(
    kD233Q1AttentionChunkCount == 8u ||
    kD233Q1AttentionChunkCount == 4u ||
    kD233Q1AttentionChunkCount == 2u ||
    kD233Q1AttentionChunkCount == 1u
);

struct D233Token0RouteMetadata {
    uint32_t expert_ids[kD230TopK];
    float weights[kD230TopK];
};

static_assert(
    sizeof(D233Token0RouteMetadata) ==
        kD230TopK * (sizeof(uint32_t) + sizeof(float))
);

uint64_t d233_elapsed_ns(
    const std::chrono::steady_clock::time_point &start
) {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start
        ).count()
    );
}

void d233_zero_device(void *pointer, size_t bytes, const char *stage) {
    check_hip(hipMemset(pointer, 0, bytes), stage);
}

bool d233_layer_digests_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_LAYER_DIGESTS",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

unsigned int d233_stage_digest_layer() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_STAGE_LAYER",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return 0u;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed < kD232Layers
        ? static_cast<unsigned int>(parsed)
        : 0u;
}

unsigned int d233_stage_digest_token() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_STAGE_TOKEN",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return 0u;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed < kSuffixTokens
        ? static_cast<unsigned int>(parsed)
        : 0u;
}

bool d233_batched_q1_attention_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_BATCHED_Q1_ATTENTION",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

unsigned int d233_batched_q1_attention_full_layer_mask() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_BATCHED_Q1_ATTENTION_FULL_LAYER_MASK",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return UINT32_C(0x3ff);
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 0);
    return end != value && *end == '\0'
        ? static_cast<unsigned int>(parsed) & UINT32_C(0x3ff)
        : UINT32_C(0x3ff);
}

bool d233_triton_q1024_attention_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_TRITON_Q1024_ATTENTION",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

bool d233_exact_q1_gdn_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_EXACT_Q1_GDN",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

uint64_t d233_exact_q1_gdn_layer_mask() {
    constexpr uint64_t kAllLayers =
        (UINT64_C(1) << kD232Layers) - UINT64_C(1);
    char value[32]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_EXACT_Q1_GDN_LAYER_MASK",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return kAllLayers;
    }
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 0);
    return end != value && *end == '\0'
        ? static_cast<uint64_t>(parsed) & kAllLayers
        : kAllLayers;
}

bool d233_grouped_q1_attention_queries_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_GROUPED_Q1_ATTENTION_QUERIES",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

bool d233_grouped_q1_attention_exact_pv_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_GROUPED_Q1_ATTENTION_EXACT_PV",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

rocblas_int d233_grouped_q1_attention_exact_pv_prefix_solution() {
    char value[32]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_GROUPED_Q1_ATTENTION_EXACT_PV_PREFIX_SOLUTION",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    return end != value && *end == '\0' &&
            parsed >= static_cast<long>((std::numeric_limits<rocblas_int>::min)()) &&
            parsed <= static_cast<long>((std::numeric_limits<rocblas_int>::max)())
        ? static_cast<rocblas_int>(parsed)
        : 0;
}

bool d233_profile_layers_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_PROFILE_LAYERS",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

bool d233_rocblas_projections_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_ROCBLAS_PROJECTIONS",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

unsigned int d233_exact_projection_layer_count() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_EXACT_PROJECTION_LAYER_COUNT",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return 0u;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0'
        ? static_cast<unsigned int>(
              (std::min)(parsed, static_cast<unsigned long>(kD232Layers))
          )
        : 0u;
}

unsigned int d233_full_projection_token_pack() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_FULL_PROJECTION_TOKEN_PACK",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return 1u;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' &&
            (parsed == 1u || parsed == 2u ||
             parsed == 4u || parsed == 8u)
        ? static_cast<unsigned int>(parsed)
        : 1u;
}

unsigned int d233_linear_projection_token_pack() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_LINEAR_PROJECTION_TOKEN_PACK",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return 1u;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' &&
            (parsed == 1u || parsed == 2u ||
             parsed == 4u || parsed == 8u)
        ? static_cast<unsigned int>(parsed)
        : 1u;
}

std::string d233_full_projection_module_directory() {
    char value[32768]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_FULL_PROJECTION_MODULE_DIR",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value)
        ? std::string(value, length)
        : std::string();
}

std::string d233_linear_projection_module_directory() {
    char value[32768]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_LINEAR_PROJECTION_MODULE_DIR",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value)
        ? std::string(value, length)
        : std::string();
}

unsigned int d233_fast_moe_min_layer() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_FAST_MOE_MIN_LAYER",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return kD232Layers;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed <= kD232Layers
        ? static_cast<unsigned int>(parsed)
        : kD232Layers;
}

unsigned int d233_fast_moe_max_layer() {
    char value[16]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_FAST_MOE_MAX_LAYER",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length == 0u || length >= sizeof(value)) {
        return kD232Layers;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    return end != value && *end == '\0' && parsed <= kD232Layers
        ? static_cast<unsigned int>(parsed)
        : kD232Layers;
}

uint64_t d233_fast_moe_layer_mask(
    unsigned int fast_moe_min_layer,
    unsigned int fast_moe_max_layer
) {
    char value[32]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_FAST_MOE_LAYER_MASK",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    if (length > 0u && length < sizeof(value)) {
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 0);
        if (end != value && *end == '\0') {
            return static_cast<uint64_t>(parsed) &
                ((UINT64_C(1) << kD232Layers) - UINT64_C(1));
        }
    }
    uint64_t mask = UINT64_C(0);
    for (unsigned int layer = fast_moe_min_layer;
         layer < fast_moe_max_layer && layer < kD232Layers;
         ++layer) {
        mask |= UINT64_C(1) << layer;
    }
    return mask;
}

std::string d233_fast_moe_dll_path() {
    char value[32768]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_FAST_MOE_DLL",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value)
        ? std::string(value, length)
        : std::string();
}

bool d233_exact_q1_attention_tail_enabled() {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        "QRT_QWEN36_Q1024_OWNER_EXACT_Q1_ATTENTION_TAIL",
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length > 0u && length < sizeof(value) &&
        value[0] != '0';
}

bool d233_exact_q1_attention_tail_component_enabled(
    const char *environment_name
) {
    char value[8]{};
    const DWORD length = GetEnvironmentVariableA(
        environment_name,
        value,
        static_cast<DWORD>(sizeof(value))
    );
    return length == 0u
        ? d233_exact_q1_attention_tail_enabled()
        : length < sizeof(value) && value[0] != '0';
}

__global__ void d233_layer_digest_kernel(
    const float *values,
    size_t count,
    uint64_t *digest
) {
    uint64_t local = UINT64_C(0);
    const size_t stride =
        static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count;
         index += stride) {
        const uint64_t bits = static_cast<uint64_t>(
            __float_as_uint(values[index])
        );
        uint64_t mixed =
            bits ^
            (
                static_cast<uint64_t>(index) +
                UINT64_C(0x9e3779b97f4a7c15)
            );
        mixed ^= mixed >> 30u;
        mixed *= UINT64_C(0xbf58476d1ce4e5b9);
        mixed ^= mixed >> 27u;
        mixed *= UINT64_C(0x94d049bb133111eb);
        mixed ^= mixed >> 31u;
        local ^= mixed;
    }
    __shared__ uint64_t partial[kThreads];
    partial[threadIdx.x] = local;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] ^= partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0u) {
        atomicXor(
            reinterpret_cast<unsigned long long *>(digest),
            static_cast<unsigned long long>(partial[0])
        );
    }
}

void d233_launch_layer_digest(
    const float *values,
    size_t count,
    uint64_t *digest
) {
    hipLaunchKernelGGL(
        d233_layer_digest_kernel,
        dim3(256u),
        dim3(kThreads),
        0u,
        nullptr,
        values,
        count,
        digest
    );
}

__global__ void d233_fnv1a64_f32_digest_kernel(
    const float *values,
    size_t count,
    uint64_t *digest
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < count; ++index) {
        const uint32_t bits = __float_as_uint(values[index]);
        for (unsigned int byte = 0u; byte < 4u; ++byte) {
            hash ^= static_cast<uint8_t>(
                (bits >> (byte * 8u)) & UINT32_C(0xff)
            );
            hash *= UINT64_C(1099511628211);
        }
    }
    *digest = hash;
}

void d233_launch_fnv1a64_f32_digest(
    const float *values,
    size_t count,
    uint64_t *digest
) {
    hipLaunchKernelGGL(
        d233_fnv1a64_f32_digest_kernel,
        dim3(1u),
        dim3(1u),
        0u,
        nullptr,
        values,
        count,
        digest
    );
}

__global__ void d233_fnv1a64_bytes_digest_kernel(
    const uint8_t *values,
    size_t count,
    uint64_t *digest
) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < count; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    *digest = hash;
}

void d233_launch_fnv1a64_bytes_digest(
    const void *values,
    size_t bytes,
    uint64_t *digest
) {
    hipLaunchKernelGGL(
        d233_fnv1a64_bytes_digest_kernel,
        dim3(1u),
        dim3(1u),
        0u,
        nullptr,
        static_cast<const uint8_t *>(values),
        bytes,
        digest
    );
}

__global__ void d233_gather_embeddings_kernel(
    const uint32_t *token_ids,
    const uint16_t *embedding_weights,
    float *hidden
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kD232HiddenElements) {
        return;
    }
    const size_t token = index / kLayerHidden;
    const size_t column = index % kLayerHidden;
    hidden[index] = device_bf16_to_float(
        embedding_weights[
            static_cast<size_t>(token_ids[token]) * kLayerHidden + column
        ]
    );
}

constexpr unsigned int kD233EarlyProjectionRowsPerBlock = 4u;
constexpr unsigned int kD233EarlyProjectionTokensPerBlock = 16u;

__global__ void d233_early_f32_projection_exact_kernel(
    const uint16_t *weights,
    const float *inputs,
    float *outputs,
    unsigned int rows
) {
    __shared__ float partial[
        kD233EarlyProjectionTokensPerBlock
    ][kThreads];
    const unsigned int lane = threadIdx.x;
    const unsigned int row_begin =
        blockIdx.x * kD233EarlyProjectionRowsPerBlock;
    const unsigned int token_begin =
        blockIdx.y * kD233EarlyProjectionTokensPerBlock;
    #pragma unroll
    for (unsigned int local_row = 0u;
         local_row < kD233EarlyProjectionRowsPerBlock;
         ++local_row) {
        const unsigned int row = row_begin + local_row;
        float sums[kD233EarlyProjectionTokensPerBlock] = {};
        if (row < rows) {
            const uint16_t *row_weights =
                weights + static_cast<size_t>(row) * kLayerHidden;
            for (unsigned int column = lane;
                 column < kLayerHidden;
                 column += blockDim.x) {
                const float weight =
                    device_bf16_to_float(row_weights[column]);
                #pragma unroll
                for (unsigned int local_token = 0u;
                     local_token < kD233EarlyProjectionTokensPerBlock;
                     ++local_token) {
                    const unsigned int token = token_begin + local_token;
                    if (token < kSuffixTokens) {
                        sums[local_token] +=
                            weight *
                            inputs[
                                static_cast<size_t>(token) * kLayerHidden +
                                column
                            ];
                    }
                }
            }
        }
        #pragma unroll
        for (unsigned int local_token = 0u;
             local_token < kD233EarlyProjectionTokensPerBlock;
             ++local_token) {
            partial[local_token][lane] = sums[local_token];
        }
        __syncthreads();
        for (unsigned int stride = blockDim.x / 2u;
             stride > 0u;
             stride >>= 1u) {
            if (lane < stride) {
                #pragma unroll
                for (unsigned int local_token = 0u;
                     local_token < kD233EarlyProjectionTokensPerBlock;
                     ++local_token) {
                    partial[local_token][lane] +=
                        partial[local_token][lane + stride];
                }
            }
            __syncthreads();
        }
        if (lane == 0u && row < rows) {
            #pragma unroll
            for (unsigned int local_token = 0u;
                 local_token < kD233EarlyProjectionTokensPerBlock;
                 ++local_token) {
                const unsigned int token = token_begin + local_token;
                if (token < kSuffixTokens) {
                    outputs[
                        static_cast<size_t>(token) * rows + row
                    ] = device_bf16_round_to_float(
                        partial[local_token][0u]
                    );
                }
            }
        }
        __syncthreads();
    }
}

void d233_launch_early_f32_projection_exact(
    const uint16_t *weights,
    const float *inputs,
    float *outputs,
    unsigned int rows
) {
    hipLaunchKernelGGL(
        d233_early_f32_projection_exact_kernel,
        dim3(
            (rows + kD233EarlyProjectionRowsPerBlock - 1u) /
                kD233EarlyProjectionRowsPerBlock,
            (kSuffixTokens + kD233EarlyProjectionTokensPerBlock - 1u) /
                kD233EarlyProjectionTokensPerBlock
        ),
        dim3(kThreads),
        0u,
        nullptr,
        weights,
        inputs,
        outputs,
        rows
    );
}

using D233Bf16Vector8 = uint32_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float d233_dot2_f32_bf16(
    uint32_t left,
    uint32_t right,
    float accumulator
) {
    float result;
    asm("v_dot2_f32_bf16 %0, %1, %2, %3"
        : "=v"(result)
        : "v"(left), "v"(right), "v"(accumulator));
    return result;
}

constexpr unsigned int kD233Bf16ProjectionThreadRows = 16u;
constexpr unsigned int kD233Bf16ProjectionThreadTokens = 16u;
constexpr unsigned int kD233Bf16ProjectionTileRows = 64u;
constexpr unsigned int kD233Bf16ProjectionTileTokens = 64u;
constexpr unsigned int kD233Bf16ProjectionTileK = 16u;
constexpr unsigned int kD233Bf16ProjectionTileKPairs =
    kD233Bf16ProjectionTileK / 2u;
constexpr unsigned int kD233Bf16ProjectionSharedKPairs =
    kD233Bf16ProjectionTileKPairs + 1u;
constexpr unsigned int kD233Bf16ProjectionRowGroups =
    kD233Bf16ProjectionTileRows / kD233Bf16ProjectionThreadRows;
constexpr unsigned int kD233Bf16ProjectionTokenGroups =
    kD233Bf16ProjectionTileTokens / kD233Bf16ProjectionThreadTokens;

__device__ __forceinline__ uint32_t d233_pack_bf16_pair(
    uint16_t low,
    uint16_t high
) {
    return static_cast<uint32_t>(low) |
        (static_cast<uint32_t>(high) << 16u);
}

// Preserve the retained q1 projection's exact dot2 traversal while owning
// sixty-four independent suffix tokens per tile.  Publishing BF16 directly
// is equivalent to q1's BF16-rounded F32 endpoint and keeps the established
// q1024 convolution/GDN storage contract intact.
__global__ void d233_bf16_projection_exact_kernel(
    const uint16_t *weights,
    const uint16_t *inputs,
    uint16_t *outputs,
    unsigned int rows
) {
    __shared__ uint32_t input_tile
        [kD233Bf16ProjectionTileTokens]
        [kD233Bf16ProjectionSharedKPairs];
    __shared__ uint32_t weight_tile
        [kD233Bf16ProjectionTileRows]
        [kD233Bf16ProjectionSharedKPairs];

    const unsigned int local_row = threadIdx.x;
    const unsigned int local_token = threadIdx.y;
    const unsigned int row_base =
        blockIdx.x * kD233Bf16ProjectionTileRows + local_row;
    const unsigned int token_base =
        blockIdx.y * kD233Bf16ProjectionTileTokens + local_token;
    float sums
        [kD233Bf16ProjectionTokenGroups]
        [kD233Bf16ProjectionRowGroups] = {};

    for (unsigned int k0 = 0u;
         k0 < kLayerHidden;
         k0 += kD233Bf16ProjectionTileK) {
        const unsigned int pair_column = local_row;
        const unsigned int column = k0 + pair_column * 2u;
        #pragma unroll
        for (unsigned int token_group = 0u;
             token_group < kD233Bf16ProjectionTokenGroups;
             ++token_group) {
            const unsigned int token =
                token_base +
                token_group * kD233Bf16ProjectionThreadTokens;
            const unsigned int token_offset =
                local_token +
                token_group * kD233Bf16ProjectionThreadTokens;
            if (pair_column < kD233Bf16ProjectionTileKPairs) {
                if (token < kSuffixTokens) {
                    const size_t input_base =
                        static_cast<size_t>(token) * kLayerHidden +
                        column;
                    input_tile[token_offset][pair_column] =
                        d233_pack_bf16_pair(
                            inputs[input_base],
                            inputs[input_base + 1u]
                        );
                } else {
                    input_tile[token_offset][pair_column] = 0u;
                }
            }
        }
        #pragma unroll
        for (unsigned int row_group = 0u;
             row_group < kD233Bf16ProjectionRowGroups;
             ++row_group) {
            const unsigned int row =
                blockIdx.x * kD233Bf16ProjectionTileRows +
                local_token +
                row_group * kD233Bf16ProjectionThreadRows;
            const unsigned int row_offset =
                local_token +
                row_group * kD233Bf16ProjectionThreadRows;
            if (pair_column < kD233Bf16ProjectionTileKPairs) {
                if (row < rows) {
                    const size_t weight_base =
                        static_cast<size_t>(row) * kLayerHidden +
                        column;
                    weight_tile[row_offset][pair_column] =
                        d233_pack_bf16_pair(
                            weights[weight_base],
                            weights[weight_base + 1u]
                        );
                } else {
                    weight_tile[row_offset][pair_column] = 0u;
                }
            }
        }
        __syncthreads();

        for (unsigned int pair = 0u;
             pair < kD233Bf16ProjectionTileKPairs;
             ++pair) {
            uint32_t input_values[kD233Bf16ProjectionTokenGroups];
            uint32_t weight_values[kD233Bf16ProjectionRowGroups];
            #pragma unroll
            for (unsigned int token_group = 0u;
                 token_group < kD233Bf16ProjectionTokenGroups;
                 ++token_group) {
                input_values[token_group] =
                    input_tile[
                        local_token +
                        token_group * kD233Bf16ProjectionThreadTokens
                    ][pair];
            }
            #pragma unroll
            for (unsigned int row_group = 0u;
                 row_group < kD233Bf16ProjectionRowGroups;
                 ++row_group) {
                weight_values[row_group] =
                    weight_tile[
                        local_row +
                        row_group * kD233Bf16ProjectionThreadRows
                    ][pair];
            }
            #pragma unroll
            for (unsigned int token_group = 0u;
                 token_group < kD233Bf16ProjectionTokenGroups;
                 ++token_group) {
                #pragma unroll
                for (unsigned int row_group = 0u;
                     row_group < kD233Bf16ProjectionRowGroups;
                     ++row_group) {
                    sums[token_group][row_group] =
                        d233_dot2_f32_bf16(
                            input_values[token_group],
                            weight_values[row_group],
                            sums[token_group][row_group]
                        );
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned int token_group = 0u;
         token_group < kD233Bf16ProjectionTokenGroups;
         ++token_group) {
        const unsigned int token =
            token_base +
            token_group * kD233Bf16ProjectionThreadTokens;
        #pragma unroll
        for (unsigned int row_group = 0u;
             row_group < kD233Bf16ProjectionRowGroups;
             ++row_group) {
            const unsigned int row =
                row_base +
                row_group * kD233Bf16ProjectionThreadRows;
            if (token < kSuffixTokens && row < rows) {
                outputs[static_cast<size_t>(token) * rows + row] =
                    device_float_to_bf16(
                        sums[token_group][row_group]
                    );
            }
        }
    }
}

void d233_launch_bf16_projection_exact(
    const uint16_t *weights,
    const uint16_t *inputs,
    uint16_t *outputs,
    unsigned int rows
) {
    hipLaunchKernelGGL(
        d233_bf16_projection_exact_kernel,
        dim3(
            (rows + kD233Bf16ProjectionTileRows - 1u) /
                kD233Bf16ProjectionTileRows,
            (kSuffixTokens + kD233Bf16ProjectionTileTokens - 1u) /
                kD233Bf16ProjectionTileTokens
        ),
        dim3(
            kD233Bf16ProjectionThreadRows,
            kD233Bf16ProjectionThreadTokens
        ),
        0u,
        nullptr,
        weights,
        inputs,
        outputs,
        rows
    );
}

__device__ __forceinline__ float d233_dot8_f32_bf16(
    const D233Bf16Vector8 &input,
    const D233Bf16Vector8 &weight,
    float accumulator
) {
    accumulator =
        d233_dot2_f32_bf16(input[0], weight[0], accumulator);
    accumulator =
        d233_dot2_f32_bf16(input[1], weight[1], accumulator);
    accumulator =
        d233_dot2_f32_bf16(input[2], weight[2], accumulator);
    accumulator =
        d233_dot2_f32_bf16(input[3], weight[3], accumulator);
    return accumulator;
}

__device__ __forceinline__ float d233_wave32_tree_sum(float value) {
    #pragma unroll
    for (unsigned int offset = 16u; offset > 0u; offset >>= 1u) {
        value += __shfl_down(value, offset, 32u);
    }
    return value;
}

constexpr unsigned int kD233EarlyOutputRowsPerBlock = 2u;
constexpr unsigned int kD233EarlyOutputTokensPerBlock = 4u;

__global__ __launch_bounds__(kThreads)
void d233_early_output_projection_exact_kernel(
    const uint16_t *weights,
    const uint16_t *inputs,
    float *outputs
) {
    constexpr unsigned int kVectors = kValueFeatures / 8u;
    static_assert(kValueFeatures % 8u == 0u);
    __shared__ D233Bf16Vector8
        weight_tile[kD233EarlyOutputRowsPerBlock * kVectors];

    const unsigned int thread = threadIdx.x;
    const unsigned int lane = thread & 31u;
    const unsigned int wave = thread >> 5u;
    const unsigned int first_row =
        blockIdx.x * kD233EarlyOutputRowsPerBlock;
    const D233Bf16Vector8 *weight_vectors =
        reinterpret_cast<const D233Bf16Vector8 *>(
            weights + static_cast<size_t>(first_row) * kValueFeatures
        );
    for (unsigned int vector = thread;
         vector < kD233EarlyOutputRowsPerBlock * kVectors;
         vector += blockDim.x) {
        weight_tile[vector] = weight_vectors[vector];
    }
    __syncthreads();

    const unsigned int local_token =
        wave / kD233EarlyOutputRowsPerBlock;
    const unsigned int local_row =
        wave % kD233EarlyOutputRowsPerBlock;
    const unsigned int token =
        blockIdx.y * kD233EarlyOutputTokensPerBlock + local_token;
    const unsigned int row = first_row + local_row;
    float sum = 0.0f;
    if (token < kSuffixTokens && row < kLayerHidden) {
        const D233Bf16Vector8 *input_vectors =
            reinterpret_cast<const D233Bf16Vector8 *>(
                inputs + static_cast<size_t>(token) * kValueFeatures
            );
        const D233Bf16Vector8 *row_weights =
            weight_tile + local_row * kVectors;
        #pragma unroll 1
        for (unsigned int vector = lane;
             vector < kVectors;
             vector += 32u) {
            sum = d233_dot8_f32_bf16(
                input_vectors[vector],
                row_weights[vector],
                sum
            );
        }
        sum = d233_wave32_tree_sum(sum);
    }
    if (lane == 0u && token < kSuffixTokens && row < kLayerHidden) {
        outputs[
            static_cast<size_t>(token) * kLayerHidden + row
        ] = device_bf16_round_to_float(sum);
    }
}

void d233_launch_early_output_projection_exact(
    const uint16_t *weights,
    const uint16_t *inputs,
    float *outputs
) {
    hipLaunchKernelGGL(
        d233_early_output_projection_exact_kernel,
        dim3(
            (kLayerHidden + kD233EarlyOutputRowsPerBlock - 1u) /
                kD233EarlyOutputRowsPerBlock,
            (kSuffixTokens + kD233EarlyOutputTokensPerBlock - 1u) /
                kD233EarlyOutputTokensPerBlock
        ),
        dim3(kThreads),
        0u,
        nullptr,
        weights,
        inputs,
        outputs
    );
}

__global__ void d233_prepare_q_gate_kernel(
    const uint16_t *q_projection,
    const uint16_t *q_norm_weight,
    uint16_t *q,
    uint16_t *gate
) {
    __shared__ float partial[kD231HeadDim];
    __shared__ float inv_rms;
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (head >= kD231QueryHeads ||
        token >= kSuffixTokens ||
        dim >= kD231HeadDim) {
        return;
    }
    const size_t projection_base =
        static_cast<size_t>(token) * kD231QProjectionRows +
        static_cast<size_t>(head) * 2u * kD231HeadDim;
    const float value =
        device_bf16_to_float(q_projection[projection_base + dim]);
    partial[dim] = value * value;
    __syncthreads();
    for (unsigned int stride = kD231HeadDim / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (dim < stride) {
            partial[dim] += partial[dim + stride];
        }
        __syncthreads();
    }
    if (dim == 0u) {
        inv_rms = 1.0f / sqrtf(
            partial[0] / static_cast<float>(kD231HeadDim) +
            kRmsNormEpsilon
        );
    }
    __syncthreads();
    const size_t output_index =
        static_cast<size_t>(token) * kD231QFeatures +
        static_cast<size_t>(head) * kD231HeadDim + dim;
    q[output_index] = device_float_to_bf16(
        value * inv_rms *
        (1.0f + device_bf16_to_float(q_norm_weight[dim]))
    );
    gate[output_index] =
        q_projection[projection_base + kD231HeadDim + dim];
}

__global__ void d233_prepare_k_kernel(
    const uint16_t *k_projection,
    const uint16_t *k_norm_weight,
    uint16_t *k
) {
    __shared__ float partial[kD231HeadDim];
    __shared__ float inv_rms;
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (head >= kD231KvHeads ||
        token >= kSuffixTokens ||
        dim >= kD231HeadDim) {
        return;
    }
    const size_t index =
        static_cast<size_t>(token) * kD231KvFeatures +
        static_cast<size_t>(head) * kD231HeadDim + dim;
    const float value = device_bf16_to_float(k_projection[index]);
    partial[dim] = value * value;
    __syncthreads();
    for (unsigned int stride = kD231HeadDim / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (dim < stride) {
            partial[dim] += partial[dim + stride];
        }
        __syncthreads();
    }
    if (dim == 0u) {
        inv_rms = 1.0f / sqrtf(
            partial[0] / static_cast<float>(kD231HeadDim) +
            kRmsNormEpsilon
        );
    }
    __syncthreads();
    k[index] = device_float_to_bf16(
        value * inv_rms *
        (1.0f + device_bf16_to_float(k_norm_weight[dim]))
    );
}

__global__ void d233_rope_bf16_kernel(
    uint16_t *values,
    unsigned int heads,
    unsigned int features,
    unsigned int absolute_position_base
) {
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int pair = threadIdx.x;
    constexpr unsigned int kHalfRotary = kD231RotaryDim / 2u;
    if (head >= heads ||
        token >= kSuffixTokens ||
        pair >= kHalfRotary) {
        return;
    }
    const size_t base =
        static_cast<size_t>(token) * features +
        static_cast<size_t>(head) * kD231HeadDim;
    const unsigned int second_dim = pair + kHalfRotary;
    const float first =
        device_bf16_to_float(values[base + pair]);
    const float second =
        device_bf16_to_float(values[base + second_dim]);
    const double inv_freq =
        1.0 /
        pow(
            kD231RopeTheta,
            (2.0 * static_cast<double>(pair)) /
                static_cast<double>(kD231RotaryDim)
        );
    const double angle =
        static_cast<double>(absolute_position_base + token) * inv_freq;
    const float cosine = static_cast<float>(cos(angle));
    const float sine = static_cast<float>(sin(angle));
    values[base + pair] =
        device_float_to_bf16(first * cosine - second * sine);
    values[base + second_dim] =
        device_float_to_bf16(second * cosine + first * sine);
}

template <bool GroupedQueryLayout, bool PublishBf16 = false>
__global__ void d233_q1_attention_softmax_kernel(
    float *score_scratch,
    unsigned int query_base,
    unsigned int query_count
) {
    __shared__ float max_score_shared;
    __shared__ float denom_shared;
    __shared__ float reduction[256u];
    const unsigned int head = blockIdx.x;
    const unsigned int local_query = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (blockDim.x != 256u || head >= kD231QueryHeads ||
        local_query >= query_count || score_scratch == nullptr) {
        return;
    }
    const unsigned int total_tokens =
        kPrefixTokens + query_base + local_query + 1u;
    const size_t score_column = GroupedQueryLayout
        ? (
              static_cast<size_t>(head / kD233HeadsPerKv) *
                  query_count * kD233HeadsPerKv +
              static_cast<size_t>(local_query) * kD233HeadsPerKv +
              head % kD233HeadsPerKv
          )
        : static_cast<size_t>(local_query) * kD231QueryHeads + head;
    float *scores = score_scratch +
        score_column * kD233Q1AttentionScoreStride;

    float max_score = -3.4028234663852886e38f;
    for (unsigned int token = lane;
         token < total_tokens;
         token += blockDim.x) {
        max_score = fmaxf(max_score, scores[token]);
    }
    reduction[lane] = max_score;
    __syncthreads();
    if constexpr (kD233SoftmaxRegisterDenominator) {
        for (unsigned int stride = blockDim.x / 2u;
             stride > 32u;
             stride /= 2u) {
            if (lane < stride) {
                reduction[lane] = fmaxf(
                    reduction[lane],
                    reduction[lane + stride]
                );
            }
            __syncthreads();
        }
        if (lane < 32u) {
            float reduced = fmaxf(
                reduction[lane],
                reduction[lane + 32u]
            );
            for (unsigned int offset = 16u;
                 offset != 0u;
                 offset /= 2u) {
                reduced = fmaxf(
                    reduced,
                    __shfl_down(reduced, offset, 32)
                );
            }
            if (lane == 0u) {
                max_score_shared = reduced;
            }
        }
        __syncthreads();
    } else {
        for (unsigned int stride = blockDim.x / 2u;
             stride != 0u;
             stride /= 2u) {
            if (lane < stride) {
                reduction[lane] = fmaxf(
                    reduction[lane],
                    reduction[lane + stride]
                );
            }
            __syncthreads();
        }
        if (lane == 0u) {
            max_score_shared = reduction[0u];
        }
        __syncthreads();
    }

    float denominator = 0.0f;
    if constexpr (kD233SoftmaxRegisterDenominator) {
        for (unsigned int token = lane;
             token < total_tokens;
             token += blockDim.x) {
            const float exponential =
                expf(scores[token] - max_score_shared);
            scores[token] = exponential;
            denominator += exponential;
        }
    } else {
        for (unsigned int token = lane;
             token < total_tokens;
             token += blockDim.x) {
            scores[token] = expf(
                scores[token] - max_score_shared
            );
        }
        __syncthreads();
        for (unsigned int token = lane;
             token < total_tokens;
             token += blockDim.x) {
            denominator += scores[token];
        }
    }
    reduction[lane] = denominator;
    __syncthreads();
    if constexpr (kD233SoftmaxRegisterDenominator) {
        for (unsigned int stride = blockDim.x / 2u;
             stride > 32u;
             stride /= 2u) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            __syncthreads();
        }
        if (lane < 32u) {
            float reduced =
                reduction[lane] + reduction[lane + 32u];
            for (unsigned int offset = 16u;
                 offset != 0u;
                 offset /= 2u) {
                reduced += __shfl_down(reduced, offset, 32);
            }
            if (lane == 0u) {
                denom_shared = reduced;
            }
        }
        __syncthreads();
    } else {
        for (unsigned int stride = blockDim.x / 2u;
             stride != 0u;
             stride /= 2u) {
            if (lane < stride) {
                reduction[lane] += reduction[lane + stride];
            }
            __syncthreads();
        }
        if (lane == 0u) {
            denom_shared = reduction[0u];
        }
        __syncthreads();
    }

    if constexpr (PublishBf16) {
        uint16_t *probabilities = reinterpret_cast<uint16_t *>(scores);
        for (unsigned int token_base = 0u;
             token_base < kPrefixTokens + kSuffixTokens;
             token_base += blockDim.x) {
            const unsigned int token = token_base + lane;
            const bool in_allocation =
                token < kPrefixTokens + kSuffixTokens;
            const float probability =
                in_allocation && token < total_tokens &&
                    denom_shared > 0.0f
                ? scores[token] / denom_shared
                : 0.0f;
            // The BF16 output aliases the lower half of the F32 score row.
            // The first output tile overlaps F32 inputs from that same tile,
            // so every lane must finish its read before publication starts.
            // For every later tile, output token t aliases only F32 input
            // floor(t / 2), which an earlier iteration already consumed.
            // No later read/write or cross-iteration dependency remains.
            if (token_base == 0u) {
                __syncthreads();
            }
            if (in_allocation) {
                probabilities[token] =
                    device_float_to_bf16(probability);
            }
        }
    } else {
        for (unsigned int token = lane;
             token < total_tokens;
             token += blockDim.x) {
            scores[token] = denom_shared > 0.0f
                ? scores[token] / denom_shared
                : 0.0f;
        }
    }
}

template <bool GroupedQueryLayout>
__global__ void d233_q1_attention_probability_bf16_kernel(
    float *score_scratch,
    unsigned int query_base,
    unsigned int query_count
) {
    const unsigned int head = blockIdx.x;
    const unsigned int local_query = blockIdx.y;
    const unsigned int lane = threadIdx.x;
    if (head >= kD231QueryHeads || local_query >= query_count ||
        score_scratch == nullptr) {
        return;
    }
    const unsigned int total_tokens =
        kPrefixTokens + query_base + local_query + 1u;
    const size_t score_column = GroupedQueryLayout
        ? (
              static_cast<size_t>(head / kD233HeadsPerKv) *
                  query_count * kD233HeadsPerKv +
              static_cast<size_t>(local_query) * kD233HeadsPerKv +
              head % kD233HeadsPerKv
          )
        : static_cast<size_t>(local_query) * kD231QueryHeads + head;
    float *input = score_scratch +
        score_column * kD233Q1AttentionScoreStride;
    uint16_t *output = reinterpret_cast<uint16_t *>(input);
    for (unsigned int token_base = 0u;
         token_base < kPrefixTokens + kSuffixTokens;
         token_base += blockDim.x) {
        const unsigned int token = token_base + lane;
        const bool in_allocation =
            token < kPrefixTokens + kSuffixTokens;
        const float value =
            in_allocation && token < total_tokens
                ? input[token]
                : 0.0f;
        __syncthreads();
        if (in_allocation) {
            output[token] = device_float_to_bf16(value);
        }
        __syncthreads();
    }
}

__global__ void d233_pack_grouped_q1_queries_kernel(
    const uint16_t *input,
    uint16_t *output,
    unsigned int query_count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements =
        static_cast<size_t>(query_count) * kD231QFeatures;
    if (index >= elements) {
        return;
    }
    const size_t query = index / kD231QFeatures;
    const unsigned int feature =
        static_cast<unsigned int>(index - query * kD231QFeatures);
    const unsigned int head = feature / kD231HeadDim;
    const unsigned int dim = feature - head * kD231HeadDim;
    const unsigned int kv_head = head / kD233HeadsPerKv;
    const unsigned int local_head = head % kD233HeadsPerKv;
    const size_t grouped_column =
        static_cast<size_t>(kv_head) * query_count * kD233HeadsPerKv +
        query * kD233HeadsPerKv + local_head;
    output[grouped_column * kD231HeadDim + dim] = input[index];
}

__global__ void d233_scatter_grouped_q1_context_kernel(
    const float *input,
    float *output,
    unsigned int query_count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements =
        static_cast<size_t>(query_count) * kD231QFeatures;
    if (index >= elements) {
        return;
    }
    const size_t query = index / kD231QFeatures;
    const unsigned int feature =
        static_cast<unsigned int>(index - query * kD231QFeatures);
    const unsigned int head = feature / kD231HeadDim;
    const unsigned int dim = feature - head * kD231HeadDim;
    const unsigned int kv_head = head / kD233HeadsPerKv;
    const unsigned int local_head = head % kD233HeadsPerKv;
    const size_t grouped_column =
        static_cast<size_t>(kv_head) * query_count * kD233HeadsPerKv +
        query * kD233HeadsPerKv + local_head;
    output[index] = input[grouped_column * kD231HeadDim + dim];
}

__global__ void d233_top1_kernel(
    const uint16_t *logits,
    uint32_t *top1_ids,
    float *top1_logits
) {
    __shared__ float best_values[kThreads];
    __shared__ uint32_t best_ids[kThreads];
    const unsigned int token = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    float best_value = -INFINITY;
    uint32_t best_id = 0u;
    const size_t token_base =
        static_cast<size_t>(token) * kD233Vocab;
    for (unsigned int id = lane; id < kD233Vocab; id += blockDim.x) {
        const float value =
            device_bf16_to_float(logits[token_base + id]);
        if (isfinite(value) &&
            (value > best_value ||
             (value == best_value && id < best_id))) {
            best_value = value;
            best_id = id;
        }
    }
    best_values[lane] = best_value;
    best_ids[lane] = best_id;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (lane < stride) {
            const float other_value = best_values[lane + stride];
            const uint32_t other_id = best_ids[lane + stride];
            if (other_value > best_values[lane] ||
                (other_value == best_values[lane] &&
                 other_id < best_ids[lane])) {
                best_values[lane] = other_value;
                best_ids[lane] = other_id;
            }
        }
        __syncthreads();
    }
    if (lane == 0u) {
        top1_ids[token] = best_ids[0u];
        top1_logits[token] = best_values[0u];
    }
}

constexpr unsigned int kD233RetainedProjectionRows =
    kQkvRows + kZRows + 2u * kAbRows;
constexpr unsigned int kD233RetainedProjectionAuxRows =
    kZRows + 2u * kAbRows;
constexpr size_t kD233RetainedProjectionAuxElements =
    static_cast<size_t>(kSuffixTokens) *
    kD233RetainedProjectionAuxRows;

class D233LinearProjectionAot {
public:
    explicit D233LinearProjectionAot(const char *module_directory) {
        if (module_directory == nullptr || module_directory[0] == '\0') {
            throw std::runtime_error(
                "D233 linear projection module directory is empty"
            );
        }
        token_pack_ = d233_linear_projection_token_pack();
        const std::string override_directory =
            d233_linear_projection_module_directory();
        const std::string directory = override_directory.empty()
            ? std::string(module_directory)
            : override_directory;
        const std::string suffix = token_pack_ == 1u
            ? std::string()
            : std::string("_tokenpack") + std::to_string(token_pack_);
        const std::string path =
            directory +
            "\\q1024_triton_0626_linear_qkvz_ab_direct_f32" +
            suffix + "_w8.hsaco";
        hipError_t status = hipModuleLoad(&module_, path.c_str());
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string("D233 linear projection hipModuleLoad failed: ") +
                hipGetErrorString(status)
            );
        }
        status = hipModuleGetFunction(
            &function_,
            module_,
            "_linear_qkvz_ab_q1024_direct_f32_kernel"
        );
        if (status != hipSuccess) {
            (void)hipModuleUnload(module_);
            module_ = nullptr;
            throw std::runtime_error(
                std::string(
                    "D233 linear projection hipModuleGetFunction failed: "
                ) + hipGetErrorString(status)
            );
        }
    }

    ~D233LinearProjectionAot() {
        if (module_ != nullptr) {
            (void)hipModuleUnload(module_);
        }
    }

    D233LinearProjectionAot(const D233LinearProjectionAot &) = delete;
    D233LinearProjectionAot &operator=(
        const D233LinearProjectionAot &
    ) = delete;

    unsigned int token_pack() const {
        return token_pack_;
    }

    void launch(
        const uint16_t *weights,
        const uint16_t *inputs,
        float *qkv_output,
        float *aux_output
    ) const {
        void *global_scratch = nullptr;
        void *profile_scratch = nullptr;
        void *arguments[] = {
            &weights,
            &inputs,
            &qkv_output,
            &aux_output,
            &global_scratch,
            &profile_scratch,
        };
        check_hip(
            hipModuleLaunchKernel(
                function_,
                kD233RetainedProjectionRows,
                (kSuffixTokens + token_pack_ - 1u) / token_pack_,
                1u,
                256u,
                1u,
                1u,
                32u,
                nullptr,
                arguments,
                nullptr
            ),
            "D233 q1024 Triton-0626 linear QKVZ+A/B projection"
        );
    }

private:
    hipModule_t module_ = nullptr;
    hipFunction_t function_ = nullptr;
    unsigned int token_pack_ = 1u;
};

class D233FullProjectionAot {
public:
    explicit D233FullProjectionAot(const char *module_directory) {
        if (module_directory == nullptr || module_directory[0] == '\0') {
            throw std::runtime_error(
                "D233 full projection module directory is empty"
            );
        }
        token_pack_ = d233_full_projection_token_pack();
        const std::string override_directory =
            d233_full_projection_module_directory();
        const std::string directory = override_directory.empty()
            ? std::string(module_directory)
            : override_directory;
        const std::string suffix = token_pack_ == 1u
            ? std::string()
            : std::string("_tokenpack") + std::to_string(token_pack_);
        const std::string path =
            directory +
            "\\q1024_triton_0626_full_attention_qkv_exact" +
            suffix + "_w8.hsaco";
        hipError_t status = hipModuleLoad(&module_, path.c_str());
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string(
                    "D233 full projection hipModuleLoad failed: "
                ) + hipGetErrorString(status)
            );
        }
        status = hipModuleGetFunction(
            &function_,
            module_,
            "_full_attention_qkv_q1024_exact_kernel"
        );
        if (status != hipSuccess) {
            (void)hipModuleUnload(module_);
            module_ = nullptr;
            throw std::runtime_error(
                std::string(
                    "D233 full projection hipModuleGetFunction failed: "
                ) + hipGetErrorString(status)
            );
        }
    }

    ~D233FullProjectionAot() {
        if (module_ != nullptr) {
            (void)hipModuleUnload(module_);
        }
    }

    D233FullProjectionAot(const D233FullProjectionAot &) = delete;
    D233FullProjectionAot &operator=(
        const D233FullProjectionAot &
    ) = delete;

    unsigned int token_pack() const {
        return token_pack_;
    }

    void launch(
        const uint16_t *q_weights,
        const uint16_t *k_weights,
        const uint16_t *v_weights,
        const uint16_t *inputs,
        uint16_t *outputs
    ) const {
        void *global_scratch = nullptr;
        void *profile_scratch = nullptr;
        void *arguments[] = {
            &q_weights,
            &k_weights,
            &v_weights,
            &inputs,
            &outputs,
            &global_scratch,
            &profile_scratch,
        };
        check_hip(
            hipModuleLaunchKernel(
                function_,
                kD233FullQkvRows,
                (kSuffixTokens + token_pack_ - 1u) / token_pack_,
                1u,
                256u,
                1u,
                1u,
                32u,
                nullptr,
                arguments,
                nullptr
            ),
            "D233 q1024 Triton-0626 full-attention QKV projection"
        );
    }

private:
    hipModule_t module_ = nullptr;
    hipFunction_t function_ = nullptr;
    unsigned int token_pack_ = 1u;
};

class D233Q1024AttentionAot {
public:
    D233Q1024AttentionAot(
        const char *module_directory,
        bool enabled
    ) {
        if (!enabled) {
            return;
        }
        if (module_directory == nullptr || module_directory[0] == '\0') {
            throw std::runtime_error(
                "D233 q1024 attention module directory is empty"
            );
        }
        const std::string directory(module_directory);
        const std::string qk_path =
            directory +
            "\\q1024_triton_0626_full_attention_qk_n32_w8_"
            "q16384_tail1536.hsaco";
        const std::string pv_path =
            directory +
            "\\q1024_triton_0626_full_attention_pv_n64_d16_w4_"
            "q16384_tail1536.hsaco";
        hipError_t status = hipModuleLoad(&qk_module_, qk_path.c_str());
        if (status != hipSuccess) {
            throw std::runtime_error(
                std::string(
                    "D233 q1024 attention QK hipModuleLoad failed: "
                ) + hipGetErrorString(status)
            );
        }
        status = hipModuleGetFunction(
            &qk_function_,
            qk_module_,
            "_q1024_qk_kernel"
        );
        if (status != hipSuccess) {
            release();
            throw std::runtime_error(
                std::string(
                    "D233 q1024 attention QK hipModuleGetFunction failed: "
                ) + hipGetErrorString(status)
            );
        }
        status = hipModuleLoad(&pv_module_, pv_path.c_str());
        if (status != hipSuccess) {
            release();
            throw std::runtime_error(
                std::string(
                    "D233 q1024 attention PV hipModuleLoad failed: "
                ) + hipGetErrorString(status)
            );
        }
        status = hipModuleGetFunction(
            &pv_function_,
            pv_module_,
            "_q1024_pv_kernel"
        );
        if (status != hipSuccess) {
            release();
            throw std::runtime_error(
                std::string(
                    "D233 q1024 attention PV hipModuleGetFunction failed: "
                ) + hipGetErrorString(status)
            );
        }
        enabled_ = true;
    }

    ~D233Q1024AttentionAot() {
        release();
    }

    D233Q1024AttentionAot(const D233Q1024AttentionAot &) = delete;
    D233Q1024AttentionAot &operator=(
        const D233Q1024AttentionAot &
    ) = delete;

    bool enabled() const {
        return enabled_;
    }

    void launch_qk(
        const uint16_t *query,
        const uint16_t *prefix_key,
        const uint16_t *tail_key,
        float *scores,
        unsigned int query_base,
        unsigned int query_count
    ) const {
        if (!enabled_ || qk_function_ == nullptr) {
            throw std::runtime_error(
                "D233 q1024 Triton QK module is unavailable"
            );
        }
        int query_base_i32 = static_cast<int>(query_base);
        int query_count_i32 = static_cast<int>(query_count);
        void *global_scratch = nullptr;
        void *profile_scratch = nullptr;
        void *arguments[] = {
            &query,
            &prefix_key,
            &tail_key,
            &scores,
            &query_base_i32,
            &query_count_i32,
            &global_scratch,
            &profile_scratch,
        };
        check_hip(
            hipModuleLaunchKernel(
                qk_function_,
                2u,
                560u,
                query_count,
                256u,
                1u,
                1u,
                16384u,
                nullptr,
                arguments,
                nullptr
            ),
            "D233 q1024 Triton-0626 full-attention QK"
        );
    }

    void launch_pv(
        const uint16_t *probabilities,
        const uint16_t *prefix_value,
        const uint16_t *tail_value,
        float *output,
        unsigned int query_base,
        unsigned int query_count
    ) const {
        if (!enabled_ || pv_function_ == nullptr) {
            throw std::runtime_error(
                "D233 q1024 Triton PV module is unavailable"
            );
        }
        int query_base_i32 = static_cast<int>(query_base);
        int query_count_i32 = static_cast<int>(query_count);
        void *global_scratch = nullptr;
        void *profile_scratch = nullptr;
        void *arguments[] = {
            &probabilities,
            &prefix_value,
            &tail_value,
            &output,
            &query_base_i32,
            &query_count_i32,
            &global_scratch,
            &profile_scratch,
        };
        check_hip(
            hipModuleLaunchKernel(
                pv_function_,
                2u,
                16u,
                query_count,
                128u,
                1u,
                1u,
                2048u,
                nullptr,
                arguments,
                nullptr
            ),
            "D233 q1024 Triton-0626 full-attention PV"
        );
    }

private:
    void release() {
        enabled_ = false;
        qk_function_ = nullptr;
        pv_function_ = nullptr;
        if (pv_module_ != nullptr) {
            (void)hipModuleUnload(pv_module_);
            pv_module_ = nullptr;
        }
        if (qk_module_ != nullptr) {
            (void)hipModuleUnload(qk_module_);
            qk_module_ = nullptr;
        }
    }

    bool enabled_ = false;
    hipModule_t qk_module_ = nullptr;
    hipFunction_t qk_function_ = nullptr;
    hipModule_t pv_module_ = nullptr;
    hipFunction_t pv_function_ = nullptr;
};

__global__ void d233_publish_retained_projection_kernel(
    const float *qkv_f32,
    const float *aux_f32,
    uint16_t *qkv_bf16,
    float *z_f32,
    float *a_f32,
    float *b_f32
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < kD232LinearQkvElements) {
        qkv_bf16[index] = device_float_to_bf16(qkv_f32[index]);
    }
    if (index >= kD233RetainedProjectionAuxElements) {
        return;
    }
    const size_t token = index / kD233RetainedProjectionAuxRows;
    const unsigned int row =
        static_cast<unsigned int>(
            index % kD233RetainedProjectionAuxRows
        );
    if (row < kZRows) {
        z_f32[token * kZRows + row] = aux_f32[index];
    } else if (row < kZRows + kAbRows) {
        a_f32[
            token * kAbRows + row - kZRows
        ] = aux_f32[index];
    } else {
        b_f32[
            token * kAbRows + row - kZRows - kAbRows
        ] = aux_f32[index];
    }
}

__global__ void d233_publish_rocblas_retained_projection_kernel(
    const uint16_t *combined_bf16,
    uint16_t *qkv_bf16,
    float *qkv_f32,
    float *z_f32,
    float *a_f32,
    float *b_f32
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total =
        static_cast<size_t>(kSuffixTokens) *
        kD233RetainedProjectionRows;
    if (index >= total) {
        return;
    }
    const size_t token = index / kD233RetainedProjectionRows;
    const unsigned int row =
        static_cast<unsigned int>(
            index % kD233RetainedProjectionRows
        );
    const uint16_t value = combined_bf16[index];
    const float value_f32 = device_bf16_to_float(value);
    if (row < kQkvRows) {
        const size_t output =
            token * static_cast<size_t>(kQkvRows) + row;
        qkv_bf16[output] = value;
        qkv_f32[output] = value_f32;
    } else if (row < kQkvRows + kZRows) {
        z_f32[
            token * static_cast<size_t>(kZRows) + row - kQkvRows
        ] = value_f32;
    } else if (row < kQkvRows + kZRows + kAbRows) {
        a_f32[
            token * static_cast<size_t>(kAbRows) +
            row - kQkvRows - kZRows
        ] = value_f32;
    } else {
        b_f32[
            token * static_cast<size_t>(kAbRows) +
            row - kQkvRows - kZRows - kAbRows
        ] = value_f32;
    }
}

__global__ void d233_publish_full_projection_kernel(
    const uint16_t *qkv,
    uint16_t *q,
    uint16_t *k,
    uint16_t *v
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kD233FullQkvProjectionElements) {
        return;
    }
    const size_t token = index / kD233FullQkvRows;
    const unsigned int row =
        static_cast<unsigned int>(index % kD233FullQkvRows);
    if (row < kD231QProjectionRows) {
        q[token * kD231QProjectionRows + row] = qkv[index];
    } else if (row < kD231QProjectionRows + kD231KProjectionRows) {
        k[
            token * kD231KProjectionRows +
            row - kD231QProjectionRows
        ] = qkv[index];
    } else {
        v[
            token * kD231VProjectionRows +
            row - kD231QProjectionRows - kD231KProjectionRows
        ] = qkv[index];
    }
}

__global__ void d233_pack_rocblas_full_projection_kernel(
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    uint16_t *qkv
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kD233FullQkvProjectionElements) {
        return;
    }
    const size_t token = index / kD233FullQkvRows;
    const unsigned int row =
        static_cast<unsigned int>(index % kD233FullQkvRows);
    if (row < kD231QProjectionRows) {
        qkv[index] =
            q[token * kD231QProjectionRows + row];
    } else if (row < kD231QProjectionRows + kD231KProjectionRows) {
        qkv[index] = k[
            token * kD231KProjectionRows +
            row - kD231QProjectionRows
        ];
    } else {
        qkv[index] = v[
            token * kD231VProjectionRows +
            row - kD231QProjectionRows - kD231KProjectionRows
        ];
    }
}

template <bool GateValuesAreDecay>
__global__ void d233_exact_q1_gdn_token_step_kernel(
    const float *postconv_values,
    const float *gate_values,
    float *state,
    float *outputs
) {
    const unsigned int value_dim = threadIdx.x;
    const unsigned int value_head = blockIdx.x;
    if (value_dim >= kValueDim || value_head >= kValueHeads ||
        postconv_values == nullptr || gate_values == nullptr ||
        state == nullptr || outputs == nullptr) {
        return;
    }

    __shared__ float shared_decay;
    __shared__ float shared_beta;
    if (value_dim == 0u) {
        const float raw_gate = gate_values[value_head];
        shared_decay = GateValuesAreDecay
            ? raw_gate
            : expf(raw_gate);
        shared_beta = gate_values[kGateRows + value_head];
    }
    __syncthreads();

    const unsigned int value_index =
        value_head * kValueDim + value_dim;
    const unsigned int key_head =
        value_head / (kValueHeads / kKeyHeads);
    const unsigned int key_base = key_head * kKeyDim;
    const size_t state_head_base =
        static_cast<size_t>(value_head) * kKeyDim * kValueDim;
    float *const state_lane =
        state + state_head_base + value_dim;
    const float value =
        postconv_values[2u * kKeyFeatures + value_index];
    float projected = 0.0f;
    float output_accumulator = 0.0f;

#pragma unroll 4
    for (unsigned int key_index = 0u;
         key_index < kKeyDim;
         ++key_index) {
        const size_t state_index =
            static_cast<size_t>(key_index) * kValueDim;
        const float decayed =
            state_lane[state_index] * shared_decay;
        const float k_value =
            postconv_values[kKeyFeatures + key_base + key_index];
        projected += decayed * k_value;
    }

    const float residual_update =
        (value - projected) * shared_beta;
#pragma unroll 4
    for (unsigned int key_index = 0u;
         key_index < kKeyDim;
         ++key_index) {
        const size_t state_index =
            static_cast<size_t>(key_index) * kValueDim;
        const float k_value =
            postconv_values[kKeyFeatures + key_base + key_index];
        const float q_value =
            postconv_values[key_base + key_index];
        const float decayed =
            state_lane[state_index] * shared_decay;
        const float updated =
            decayed + residual_update * k_value;
        state_lane[state_index] = updated;
        output_accumulator += updated * q_value;
    }
    outputs[value_index] =
        device_bf16_round_to_float(output_accumulator);
}

// Each value lane owns an independent recurrent state vector.  Keep the q1
// token order, per-token shared gate publication, arithmetic order, and BF16
// output boundary, while collapsing 1024 host launches into one grid.
template <bool GateValuesAreDecay, bool SharedQk>
__global__ void d233_exact_q1_gdn_suffix_kernel(
    const float *postconv_values,
    const float *gate_values,
    float *state,
    float *outputs
) {
    const unsigned int value_dim = threadIdx.x;
    const unsigned int value_head = blockIdx.x;
    if (value_dim >= kValueDim || value_head >= kValueHeads ||
        postconv_values == nullptr || gate_values == nullptr ||
        state == nullptr || outputs == nullptr) {
        return;
    }

    __shared__ float shared_decay;
    __shared__ float shared_beta;
    __shared__ float shared_q[SharedQk ? kKeyDim : 1u];
    __shared__ float shared_k[SharedQk ? kKeyDim : 1u];
    const unsigned int value_index =
        value_head * kValueDim + value_dim;
    const unsigned int key_head =
        value_head / (kValueHeads / kKeyHeads);
    const unsigned int key_base = key_head * kKeyDim;
    const size_t state_head_base =
        static_cast<size_t>(value_head) * kKeyDim * kValueDim;
    float *const state_lane =
        state + state_head_base + value_dim;

    for (unsigned int token = 0u; token < kSuffixTokens; ++token) {
        const float *const token_postconv =
            postconv_values + static_cast<size_t>(token) * kQkvRows;
        const float *const token_gate =
            gate_values +
            static_cast<size_t>(token) * kGateOutputRows;
        if (value_dim == 0u) {
            const float raw_gate = token_gate[value_head];
            shared_decay = GateValuesAreDecay
                ? raw_gate
                : expf(raw_gate);
            shared_beta = token_gate[kGateRows + value_head];
        }
        if constexpr (SharedQk) {
            shared_q[value_dim] =
                token_postconv[key_base + value_dim];
            shared_k[value_dim] =
                token_postconv[
                    kKeyFeatures + key_base + value_dim
                ];
        }
        __syncthreads();

        const float value =
            token_postconv[2u * kKeyFeatures + value_index];
        float projected = 0.0f;
        float output_accumulator = 0.0f;

#pragma unroll 4
        for (unsigned int key_index = 0u;
             key_index < kKeyDim;
             ++key_index) {
            const size_t state_index =
                static_cast<size_t>(key_index) * kValueDim;
            const float decayed =
                state_lane[state_index] * shared_decay;
            const float k_value = SharedQk
                ? shared_k[key_index]
                : token_postconv[
                      kKeyFeatures + key_base + key_index
                  ];
            projected += decayed * k_value;
        }

        const float residual_update =
            (value - projected) * shared_beta;
#pragma unroll 4
        for (unsigned int key_index = 0u;
             key_index < kKeyDim;
             ++key_index) {
            const size_t state_index =
                static_cast<size_t>(key_index) * kValueDim;
            const float k_value = SharedQk
                ? shared_k[key_index]
                : token_postconv[
                      kKeyFeatures + key_base + key_index
                  ];
            const float q_value = SharedQk
                ? shared_q[key_index]
                : token_postconv[key_base + key_index];
            const float decayed =
                state_lane[state_index] * shared_decay;
            const float updated =
                decayed + residual_update * k_value;
            state_lane[state_index] = updated;
            output_accumulator += updated * q_value;
        }
        outputs[
            static_cast<size_t>(token) * kValueFeatures + value_index
        ] = device_bf16_round_to_float(output_accumulator);

        // A kernel boundary provided this barrier in the q1 launch sequence.
        // Preserve it before lane zero publishes the next token's gate.
        __syncthreads();
    }
}

// The q1 reference deliberately uses one lane to accumulate every Q/K norm in
// ascending dimension order.  The older q1024 owner used a tree reduction,
// which often rounded to the same BF16 endpoint but first diverged at suffix
// token 1 on the real layer-0 cache.  Batch tokens in the grid while preserving
// the q1 per-head arithmetic and publication boundary exactly.
__global__ void d233_postconv_qk_q1_serial_exact_kernel(float *values) {
    __shared__ float q_inv_shared;
    __shared__ float k_inv_shared;

    const unsigned int token = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (token >= kSuffixTokens || head >= kKeyHeads || dim >= kKeyDim ||
        values == nullptr) {
        return;
    }

    const size_t token_base =
        static_cast<size_t>(token) * kQkvRows;
    const unsigned int index = head * kKeyDim + dim;
    if (dim == 0u) {
        float q_sumsq = 0.0f;
        float k_sumsq = 0.0f;
        for (unsigned int reduction_dim = 0u;
             reduction_dim < kKeyDim;
             ++reduction_dim) {
            const unsigned int reduction_index =
                head * kKeyDim + reduction_dim;
            const float q_value =
                values[token_base + reduction_index];
            const float k_value =
                values[
                    token_base + kKeyFeatures + reduction_index
                ];
            q_sumsq +=
                device_bf16_round_to_float(q_value * q_value);
            k_sumsq +=
                device_bf16_round_to_float(k_value * k_value);
        }
        const float q_sumsq_bf16 =
            device_bf16_round_to_float(q_sumsq);
        const float k_sumsq_bf16 =
            device_bf16_round_to_float(k_sumsq);
        q_inv_shared = device_bf16_round_to_float(
            1.0f /
            sqrtf(
                device_bf16_round_to_float(
                    q_sumsq_bf16 + 1.0e-6f
                )
            )
        );
        k_inv_shared = device_bf16_round_to_float(
            1.0f /
            sqrtf(
                device_bf16_round_to_float(
                    k_sumsq_bf16 + 1.0e-6f
                )
            )
        );
    }
    __syncthreads();

    const float q_value = values[token_base + index];
    const float k_value =
        values[token_base + kKeyFeatures + index];
    const float q_norm =
        device_bf16_round_to_float(q_value * q_inv_shared);
    values[token_base + index] = q_norm * kQScale;
    values[token_base + kKeyFeatures + index] =
        device_bf16_round_to_float(k_value * k_inv_shared);
}

template <typename T>
__global__ void d233_exact_q1_conv_token_step_kernel(
    const T *current_qkv,
    T *qkv_ring,
    const uint16_t *weights,
    float *outputs,
    size_t absolute_position
) {
    const unsigned int feature =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (feature >= kQkvRows || current_qkv == nullptr ||
        qkv_ring == nullptr || weights == nullptr || outputs == nullptr) {
        return;
    }

    float accumulator = 0.0f;
    for (unsigned int tap = 0u; tap < kConvTaps; ++tap) {
        if (absolute_position + tap >=
            static_cast<size_t>(kConvTaps - 1u)) {
            const size_t source_token =
                absolute_position + tap -
                static_cast<size_t>(kConvTaps - 1u);
            float qkv_value = 0.0f;
            if (source_token == absolute_position) {
                qkv_value =
                    device_native_to_float(current_qkv[feature]);
            } else {
                const size_t source_slot =
                    source_token % static_cast<size_t>(kConvTaps);
                qkv_value = device_native_to_float(
                    qkv_ring[
                        source_slot * static_cast<size_t>(kQkvRows) +
                        feature
                    ]
                );
            }
            qkv_value = device_bf16_round_to_float(qkv_value);
            const float weight = device_bf16_to_float(
                weights[
                    static_cast<size_t>(feature) * kConvTaps + tap
                ]
            );
            accumulator += qkv_value * weight;
        }
    }
    const float rounded_accumulator =
        device_bf16_round_to_float(accumulator);
    const float silu =
        rounded_accumulator /
        (1.0f + expf(-rounded_accumulator));
    outputs[feature] = device_bf16_round_to_float(silu);

    const size_t destination_slot =
        absolute_position % static_cast<size_t>(kConvTaps);
    qkv_ring[
        destination_slot * static_cast<size_t>(kQkvRows) + feature
    ] = current_qkv[feature];
}

// Convolution ring state is independent by feature.  A persistent feature
// lane can therefore retain the exact q1 token sequence while eliminating the
// per-token launch boundary.
template <typename T>
__global__ void d233_exact_q1_conv_suffix_kernel(
    const T *qkv,
    T *qkv_ring,
    const uint16_t *weights,
    float *outputs,
    size_t first_absolute_position
) {
    const unsigned int feature =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (feature >= kQkvRows || qkv == nullptr || qkv_ring == nullptr ||
        weights == nullptr || outputs == nullptr) {
        return;
    }

    for (unsigned int token = 0u; token < kSuffixTokens; ++token) {
        const T *const current_qkv =
            qkv + static_cast<size_t>(token) * kQkvRows;
        const size_t absolute_position =
            first_absolute_position + token;
        float accumulator = 0.0f;
        for (unsigned int tap = 0u; tap < kConvTaps; ++tap) {
            if (absolute_position + tap >=
                static_cast<size_t>(kConvTaps - 1u)) {
                const size_t source_token =
                    absolute_position + tap -
                    static_cast<size_t>(kConvTaps - 1u);
                float qkv_value = 0.0f;
                if (source_token == absolute_position) {
                    qkv_value =
                        device_native_to_float(current_qkv[feature]);
                } else {
                    const size_t source_slot =
                        source_token %
                        static_cast<size_t>(kConvTaps);
                    qkv_value = device_native_to_float(
                        qkv_ring[
                            source_slot *
                                static_cast<size_t>(kQkvRows) +
                            feature
                        ]
                    );
                }
                qkv_value = device_bf16_round_to_float(qkv_value);
                const float weight = device_bf16_to_float(
                    weights[
                        static_cast<size_t>(feature) * kConvTaps + tap
                    ]
                );
                accumulator += qkv_value * weight;
            }
        }
        const float rounded_accumulator =
            device_bf16_round_to_float(accumulator);
        const float silu =
            rounded_accumulator /
            (1.0f + expf(-rounded_accumulator));
        outputs[
            static_cast<size_t>(token) * kQkvRows + feature
        ] = device_bf16_round_to_float(silu);

        const size_t destination_slot =
            absolute_position % static_cast<size_t>(kConvTaps);
        qkv_ring[
            destination_slot * static_cast<size_t>(kQkvRows) + feature
        ] = current_qkv[feature];
    }
}

class D233LinearOwner {
public:
    D233LinearOwner(
        const ProviderApi &gdn,
        const MoeApi &moe,
        const MoeApi &fast_moe,
        unsigned int fast_moe_min_layer,
        unsigned int fast_moe_max_layer,
        const RocblasContext &blas,
        const D233LinearProjectionAot &projection_aot
    )
        : gdn_(gdn),
          moe_(moe),
          fast_moe_(fast_moe),
          fast_moe_min_layer_(fast_moe_min_layer),
          fast_moe_max_layer_(fast_moe_max_layer),
          fast_moe_layer_mask_(d233_fast_moe_layer_mask(
              fast_moe_min_layer,
              fast_moe_max_layer
          )),
          blas_(blas),
          projection_aot_(projection_aot),
          exact_q1_gdn_(d233_exact_q1_gdn_enabled()),
          exact_q1_gdn_layer_mask_(d233_exact_q1_gdn_layer_mask()),
          rocblas_projections_(d233_rocblas_projections_enabled()),
          exact_projection_layer_count_(
              d233_exact_projection_layer_count()
          ),
          input_norm_f32_(kD232HiddenElements),
          input_norm_bf16_(kD232HiddenElements),
          early_qkv_(kD232LinearQkvElements),
          early_z_(kD232LinearZElements),
          early_a_(kD232LinearAbElements),
          early_b_(kD232LinearAbElements),
          early_out_(kD232HiddenElements),
          retained_qkv_(kD232LinearQkvElements),
          retained_z_(kD232LinearZElements),
          retained_a_(kD232LinearAbElements),
          retained_b_(kD232LinearAbElements),
          retained_out_(kD232HiddenElements),
          retained_aux_f32_(kD233RetainedProjectionAuxElements),
          retained_projection_bf16_(
              static_cast<size_t>(kSuffixTokens) *
              kD233RetainedProjectionRows
          ),
          z_f32_(kD232LinearZElements),
          a_f32_(kD232LinearAbElements),
          b_f32_(kD232LinearAbElements),
          gate_(kD232LinearGateElements),
          postconv_(kD232LinearQkvElements),
          core_(kD232LinearCoreElements),
          gated_(kD232LinearCoreElements),
          residual_(kD232HiddenElements),
          postnorm_(kD232HiddenElements),
          final_state_(
              static_cast<size_t>(kD232LinearLayers) *
              kD232LinearStateElements
          ),
          early_final_ring_(
              static_cast<size_t>(kD232EarlyLinearLayers) *
              kD232LinearRingElements
          ),
          retained_final_ring_(
              static_cast<size_t>(kD232RetainedLinearLayers) *
              kD232LinearRingElements
          ) {}

    bool uses_exact_q1_gdn() const {
        return exact_q1_gdn_;
    }

    uint64_t exact_q1_gdn_layer_mask() const {
        return exact_q1_gdn_layer_mask_;
    }

    bool uses_rocblas_projections() const {
        return rocblas_projections_;
    }

    unsigned int exact_projection_layer_count() const {
        return exact_projection_layer_count_;
    }

    void reset() {
        d233_zero_device(
            input_norm_f32_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset linear input norm F32"
        );
        d233_zero_device(
            input_norm_bf16_.get(),
            kD232HiddenElements * sizeof(uint16_t),
            "D233 reset linear input norm BF16"
        );
        d233_zero_device(
            early_qkv_.get(),
            kD232LinearQkvElements * sizeof(float),
            "D233 reset early linear QKV"
        );
        d233_zero_device(
            early_z_.get(),
            kD232LinearZElements * sizeof(float),
            "D233 reset early linear Z"
        );
        d233_zero_device(
            early_a_.get(),
            kD232LinearAbElements * sizeof(float),
            "D233 reset early linear A"
        );
        d233_zero_device(
            early_b_.get(),
            kD232LinearAbElements * sizeof(float),
            "D233 reset early linear B"
        );
        d233_zero_device(
            early_out_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset early linear output"
        );
        d233_zero_device(
            retained_qkv_.get(),
            kD232LinearQkvElements * sizeof(uint16_t),
            "D233 reset retained linear QKV"
        );
        d233_zero_device(
            retained_z_.get(),
            kD232LinearZElements * sizeof(uint16_t),
            "D233 reset retained linear Z"
        );
        d233_zero_device(
            retained_a_.get(),
            kD232LinearAbElements * sizeof(uint16_t),
            "D233 reset retained linear A"
        );
        d233_zero_device(
            retained_b_.get(),
            kD232LinearAbElements * sizeof(uint16_t),
            "D233 reset retained linear B"
        );
        d233_zero_device(
            retained_out_.get(),
            kD232HiddenElements * sizeof(uint16_t),
            "D233 reset retained linear output"
        );
        d233_zero_device(
            retained_aux_f32_.get(),
            kD233RetainedProjectionAuxElements * sizeof(float),
            "D233 reset retained linear projection auxiliary output"
        );
        d233_zero_device(
            retained_projection_bf16_.get(),
            static_cast<size_t>(kSuffixTokens) *
                kD233RetainedProjectionRows * sizeof(uint16_t),
            "D233 reset retained rocBLAS projection output"
        );
        d233_zero_device(
            z_f32_.get(),
            kD232LinearZElements * sizeof(float),
            "D233 reset linear Z F32"
        );
        d233_zero_device(
            a_f32_.get(),
            kD232LinearAbElements * sizeof(float),
            "D233 reset linear A F32"
        );
        d233_zero_device(
            b_f32_.get(),
            kD232LinearAbElements * sizeof(float),
            "D233 reset linear B F32"
        );
        d233_zero_device(
            gate_.get(),
            kD232LinearGateElements * sizeof(float),
            "D233 reset linear gate"
        );
        d233_zero_device(
            postconv_.get(),
            kD232LinearQkvElements * sizeof(float),
            "D233 reset linear post-convolution"
        );
        d233_zero_device(
            core_.get(),
            kD232LinearCoreElements * sizeof(float),
            "D233 reset linear core"
        );
        d233_zero_device(
            gated_.get(),
            kD232LinearCoreElements * sizeof(uint16_t),
            "D233 reset linear gated output"
        );
        d233_zero_device(
            residual_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset linear residual"
        );
        d233_zero_device(
            postnorm_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset linear postnorm"
        );
        d233_zero_device(
            final_state_.get(),
            static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements * sizeof(float),
            "D233 reset linear final state"
        );
        d233_zero_device(
            early_final_ring_.get(),
            static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements * sizeof(float),
            "D233 reset early linear final ring"
        );
        d233_zero_device(
            retained_final_ring_.get(),
            static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements * sizeof(uint16_t),
            "D233 reset retained linear final ring"
        );
    }

    template <typename T, bool GateValuesAreDecay>
    void launch(
        const qrt_qwen36_q1024_owner_layer_v1_t &layer,
        unsigned int linear_ordinal,
        const float *hidden_input,
        float *hidden_output,
        uint64_t *layer0_stage_digests,
        unsigned int stage_digest_token,
        hipEvent_t projection_end_event,
        hipEvent_t core_end_event,
        hipEvent_t moe_begin_event
    ) {
        const bool use_exact_q1_gdn =
            exact_q1_gdn_ &&
            (exact_q1_gdn_layer_mask_ &
             (UINT64_C(1) << layer.layer_index)) != UINT64_C(0);
        T *qkv = nullptr;
        T *z_native = nullptr;
        T *a_native = nullptr;
        T *b_native = nullptr;
        T *out_native = nullptr;
        T *final_ring = nullptr;
        if constexpr (std::is_same<T, float>::value) {
            qkv = early_qkv_.get();
            z_native = early_z_.get();
            a_native = early_a_.get();
            b_native = early_b_.get();
            out_native = early_out_.get();
            final_ring =
                early_final_ring_.get() +
                static_cast<size_t>(linear_ordinal) *
                    kD232LinearRingElements;
        } else {
            qkv = retained_qkv_.get();
            z_native = retained_z_.get();
            a_native = retained_a_.get();
            b_native = retained_b_.get();
            out_native = retained_out_.get();
            const unsigned int retained_ordinal =
                linear_ordinal - kD232EarlyLinearLayers;
            final_ring =
                retained_final_ring_.get() +
                static_cast<size_t>(retained_ordinal) *
                    kD232LinearRingElements;
        }

        const dim3 block(kThreads);
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                layer.linear_prefix_state,
                kD232LinearStateElements,
                layer0_stage_digests + 0u
            );
            d233_launch_fnv1a64_f32_digest(
                static_cast<const float *>(layer.linear_prefix_ring),
                kD232LinearRingElements,
                layer0_stage_digests + 1u
            );
        }
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            layer.input_norm,
            input_norm_f32_.get()
        );
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                input_norm_f32_.get() +
                    static_cast<size_t>(stage_digest_token) * kLayerHidden,
                kLayerHidden,
                layer0_stage_digests + 2u
            );
        }
        hipLaunchKernelGGL(
            f32_to_bf16_sublayer_kernel,
            d232_linear_grid(kD232HiddenElements),
            block,
            0u,
            nullptr,
            input_norm_f32_.get(),
            input_norm_bf16_.get(),
            kD232HiddenElements
        );
        if constexpr (std::is_same<T, float>::value) {
            d233_launch_early_f32_projection_exact(
                layer.linear_qkv,
                input_norm_f32_.get(),
                qkv,
                kQkvRows
            );
            d233_launch_early_f32_projection_exact(
                layer.linear_z,
                input_norm_f32_.get(),
                z_native,
                kZRows
            );
            d233_launch_early_f32_projection_exact(
                layer.linear_a,
                input_norm_f32_.get(),
                a_native,
                kAbRows
            );
            d233_launch_early_f32_projection_exact(
                layer.linear_b,
                input_norm_f32_.get(),
                b_native,
                kAbRows
            );
        } else {
            const uintptr_t qkv_address =
                reinterpret_cast<uintptr_t>(layer.linear_qkv);
            const uintptr_t expected_z =
                qkv_address +
                static_cast<uintptr_t>(kQkvRows) *
                    kLayerHidden * sizeof(uint16_t);
            const uintptr_t expected_a =
                expected_z +
                static_cast<uintptr_t>(kZRows) *
                    kLayerHidden * sizeof(uint16_t);
            const uintptr_t expected_b =
                expected_a +
                static_cast<uintptr_t>(kAbRows) *
                    kLayerHidden * sizeof(uint16_t);
            if (reinterpret_cast<uintptr_t>(layer.linear_z) != expected_z ||
                reinterpret_cast<uintptr_t>(layer.linear_a) != expected_a ||
                reinterpret_cast<uintptr_t>(layer.linear_b) != expected_b) {
                throw std::runtime_error(
                    "D233 retained QKVZ+A/B weights are not one contiguous q1-compatible matrix"
                );
            }
            if (rocblas_projections_ &&
                layer.layer_index >= exact_projection_layer_count_) {
                blas_.matmul(
                    layer.linear_qkv,
                    input_norm_bf16_.get(),
                    retained_projection_bf16_.get(),
                    kD233RetainedProjectionRows,
                    kLayerHidden
                );
                hipLaunchKernelGGL(
                    d233_publish_rocblas_retained_projection_kernel,
                    d232_linear_grid(
                        static_cast<size_t>(kSuffixTokens) *
                        kD233RetainedProjectionRows
                    ),
                    block,
                    0u,
                    nullptr,
                    retained_projection_bf16_.get(),
                    qkv,
                    early_qkv_.get(),
                    z_f32_.get(),
                    a_f32_.get(),
                    b_f32_.get()
                );
            } else {
                projection_aot_.launch(
                    layer.linear_qkv,
                    input_norm_bf16_.get(),
                    early_qkv_.get(),
                    retained_aux_f32_.get()
                );
                hipLaunchKernelGGL(
                    d233_publish_retained_projection_kernel,
                    d232_linear_grid(kD232LinearQkvElements),
                    block,
                    0u,
                    nullptr,
                    early_qkv_.get(),
                    retained_aux_f32_.get(),
                    qkv,
                    z_f32_.get(),
                    a_f32_.get(),
                    b_f32_.get()
                );
            }
        }
        if (layer0_stage_digests != nullptr) {
            if constexpr (std::is_same<T, float>::value) {
                d233_launch_fnv1a64_f32_digest(
                    qkv +
                        static_cast<size_t>(stage_digest_token) * kQkvRows,
                    kQkvRows,
                    layer0_stage_digests + 3u
                );
            } else {
                d233_launch_fnv1a64_f32_digest(
                    early_qkv_.get() +
                        static_cast<size_t>(stage_digest_token) * kQkvRows,
                    kQkvRows,
                    layer0_stage_digests + 3u
                );
            }
        }
        if constexpr (std::is_same<T, float>::value) {
            hipLaunchKernelGGL(
                native_to_f32_sublayer_kernel<T>,
                d232_linear_grid(kD232LinearZElements),
                block,
                0u,
                nullptr,
                z_native,
                z_f32_.get(),
                kD232LinearZElements
            );
            hipLaunchKernelGGL(
                native_to_f32_sublayer_kernel<T>,
                d232_linear_grid(kD232LinearAbElements),
                block,
                0u,
                nullptr,
                a_native,
                a_f32_.get(),
                kD232LinearAbElements
            );
            hipLaunchKernelGGL(
                native_to_f32_sublayer_kernel<T>,
                d232_linear_grid(kD232LinearAbElements),
                block,
                0u,
                nullptr,
                b_native,
                b_f32_.get(),
                kD232LinearAbElements
            );
        }
        if (use_exact_q1_gdn) {
            // The q1 reference publishes log(decay) to an F32 carrier before
            // the recurrent kernel reloads it and evaluates expf().  Keeping
            // that store/reload boundary matters after several state updates:
            // computing expf(log_decay) in this producer can retain a
            // different intermediate and only happens to round to the same
            // token-0 BF16 output.
            hipLaunchKernelGGL(
                gate_from_ab_kernel<false>,
                dim3(1u, kSuffixTokens),
                block,
                0u,
                nullptr,
                a_f32_.get(),
                b_f32_.get(),
                layer.linear_a_log,
                layer.linear_dt_bias,
                gate_.get()
            );
        } else {
            hipLaunchKernelGGL(
                gate_from_ab_kernel<GateValuesAreDecay>,
                dim3(1u, kSuffixTokens),
                block,
                0u,
                nullptr,
                a_f32_.get(),
                b_f32_.get(),
                layer.linear_a_log,
                layer.linear_dt_bias,
                gate_.get()
            );
        }
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                gate_.get() +
                    static_cast<size_t>(stage_digest_token) *
                        kGateOutputRows,
                kGateRows,
                layer0_stage_digests + 4u
            );
        }
        if (projection_end_event != nullptr) {
            check_hip(
                hipEventRecord(projection_end_event, nullptr),
                "D233 record linear projection profile boundary"
            );
        }
        if (use_exact_q1_gdn) {
            check_hip(
                hipMemcpyAsync(
                    final_ring,
                    layer.linear_prefix_ring,
                    kD232LinearRingElements * sizeof(T),
                    hipMemcpyDeviceToDevice,
                    nullptr
                ),
                "D233 initialize exact q1 convolution ring"
            );
            if (kD233PersistentExactQ1Conv &&
                layer0_stage_digests == nullptr) {
                hipLaunchKernelGGL(
                    d233_exact_q1_conv_suffix_kernel<T>,
                    dim3(
                        (kQkvRows + kThreads - 1u) / kThreads
                    ),
                    block,
                    0u,
                    nullptr,
                    qkv,
                    final_ring,
                    layer.linear_conv,
                    postconv_.get(),
                    static_cast<size_t>(kPrefixTokens)
                );
            } else {
                for (unsigned int token = 0u;
                     token < kSuffixTokens;
                     ++token) {
                    if constexpr (std::is_same<T, float>::value) {
                        if (layer0_stage_digests != nullptr &&
                            token == stage_digest_token) {
                            d233_launch_fnv1a64_f32_digest(
                                final_ring,
                                kD232LinearRingElements,
                                layer0_stage_digests + 11u
                            );
                            d233_launch_fnv1a64_bytes_digest(
                                layer.linear_conv,
                                static_cast<size_t>(kQkvRows) *
                                    kConvTaps * sizeof(uint16_t),
                                layer0_stage_digests + 12u
                            );
                        }
                    }
                    hipLaunchKernelGGL(
                        d233_exact_q1_conv_token_step_kernel<T>,
                        dim3(
                            (kQkvRows + kThreads - 1u) / kThreads
                        ),
                        block,
                        0u,
                        nullptr,
                        qkv +
                            static_cast<size_t>(token) * kQkvRows,
                        final_ring,
                        layer.linear_conv,
                        postconv_.get() +
                            static_cast<size_t>(token) * kQkvRows,
                        static_cast<size_t>(kPrefixTokens) + token
                    );
                }
            }
        } else {
            hipLaunchKernelGGL(
                suffix_halo_conv_kernel<T>,
                dim3(
                    (kQkvRows + kThreads - 1u) / kThreads,
                    kSuffixTokens
                ),
                block,
                0u,
                nullptr,
                static_cast<const T *>(layer.linear_prefix_ring),
                qkv,
                layer.linear_conv,
                postconv_.get()
            );
        }
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                postconv_.get() +
                    static_cast<size_t>(stage_digest_token) * kQkvRows,
                kQkvRows,
                layer0_stage_digests + 10u
            );
        }
        if (use_exact_q1_gdn) {
            hipLaunchKernelGGL(
                d233_postconv_qk_q1_serial_exact_kernel,
                dim3(kSuffixTokens, kKeyHeads),
                dim3(kKeyDim),
                0u,
                nullptr,
                postconv_.get()
            );
        } else {
            hipLaunchKernelGGL(
                postconv_qk_inplace_kernel,
                dim3(kSuffixTokens, kKeyHeads),
                dim3(kKeyDim),
                0u,
                nullptr,
                postconv_.get(),
                kSuffixTokens
            );
        }
        hipLaunchKernelGGL(
            postconv_value_inplace_kernel,
            dim3(
                (kValueFeatures + kThreads - 1u) / kThreads,
                kSuffixTokens
            ),
            block,
            0u,
            nullptr,
            postconv_.get(),
            kSuffixTokens
        );
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                postconv_.get() +
                    static_cast<size_t>(stage_digest_token) * kQkvRows,
                kQkvRows,
                layer0_stage_digests + 9u
            );
        }
        float *final_state =
            final_state_.get() +
            static_cast<size_t>(linear_ordinal) *
                kD232LinearStateElements;
        if (use_exact_q1_gdn) {
            check_hip(
                hipMemcpyAsync(
                    final_state,
                    layer.linear_prefix_state,
                    kD232LinearStateElements * sizeof(float),
                    hipMemcpyDeviceToDevice,
                    nullptr
                ),
                "D233 initialize exact q1 GDN state"
            );
            if constexpr (kD233PersistentExactQ1Gdn) {
                hipLaunchKernelGGL(
                    (
                        d233_exact_q1_gdn_suffix_kernel<
                            false,
                            kD233PersistentExactQ1GdnSharedQk
                        >
                    ),
                    dim3(kValueHeads),
                    dim3(kValueDim),
                    0u,
                    nullptr,
                    postconv_.get(),
                    gate_.get(),
                    final_state,
                    core_.get()
                );
            } else {
                for (unsigned int token = 0u;
                     token < kSuffixTokens;
                     ++token) {
                    hipLaunchKernelGGL(
                        d233_exact_q1_gdn_token_step_kernel<false>,
                        dim3(kValueHeads),
                        dim3(kValueDim),
                        0u,
                        nullptr,
                        postconv_.get() +
                            static_cast<size_t>(token) * kQkvRows,
                        gate_.get() +
                            static_cast<size_t>(token) * kGateOutputRows,
                        final_state,
                        core_.get() +
                            static_cast<size_t>(token) * kValueFeatures
                    );
                }
            }
        } else if (gdn_.q1024_seeded_launch_async(
                       postconv_.get(),
                       gate_.get(),
                       layer.linear_prefix_state,
                       core_.get(),
                       final_state,
                       GateValuesAreDecay ? 1 : 0,
                       nullptr
                   ) == 0) {
            throw std::runtime_error(
                std::string("D233 seeded GDN launch failed: ") +
                gdn_.last_error()
            );
        }
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                core_.get() +
                    static_cast<size_t>(stage_digest_token) * kValueFeatures,
                kValueFeatures,
                layer0_stage_digests + 5u
            );
        }
        hipLaunchKernelGGL(
            gated_rmsnorm_sublayer_kernel,
            dim3(kValueHeads, kSuffixTokens),
            dim3(kValueDim),
            0u,
            nullptr,
            core_.get(),
            z_f32_.get(),
            layer.linear_norm,
            gated_.get()
        );
        if (core_end_event != nullptr) {
            check_hip(
                hipEventRecord(core_end_event, nullptr),
                "D233 record linear core profile boundary"
            );
        }
        float *attention_update = early_out_.get();
        d233_launch_early_output_projection_exact(
            layer.linear_out,
            gated_.get(),
            attention_update
        );
        hipLaunchKernelGGL(
            residual_postnorm_sublayer_kernel<float>,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            attention_update,
            layer.post_attention_norm,
            residual_.get(),
            postnorm_.get()
        );
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                residual_.get() +
                    static_cast<size_t>(stage_digest_token) * kLayerHidden,
                kLayerHidden,
                layer0_stage_digests + 6u
            );
            d233_launch_fnv1a64_f32_digest(
                postnorm_.get() +
                    static_cast<size_t>(stage_digest_token) * kLayerHidden,
                kLayerHidden,
                layer0_stage_digests + 7u
            );
        }
        hipLaunchKernelGGL(
            publish_final_ring_kernel<T>,
            d232_linear_grid(kD232LinearRingElements),
            block,
            0u,
            nullptr,
            qkv,
            final_ring
        );
        if (moe_begin_event != nullptr) {
            check_hip(
                hipEventRecord(moe_begin_event, nullptr),
                "D233 record linear MoE profile boundary"
            );
        }
        const MoeApi &layer_moe =
            (fast_moe_layer_mask_ &
                    (UINT64_C(1) << layer.layer_index)) != 0u &&
                    linear_ordinal != 1u
                ? fast_moe_
                : moe_;
        MoeLaunchFunction moe_launch = linear_ordinal == 1u
            ? layer_moe.launch_q1024_early_f32_v1
            : layer_moe.launch_v2;
        if (moe_launch == nullptr) {
            throw std::runtime_error(
                "D233 q1024 early-F32 MoE export is unavailable"
            );
        }
        if (moe_launch(
                postnorm_.get(),
                residual_.get(),
                layer.moe_router,
                layer.moe_expert_gate_up,
                layer.moe_expert_down,
                layer.moe_shared_gate,
                layer.moe_shared_gate_projection,
                layer.moe_shared_up_projection,
                layer.moe_shared_down,
                hidden_output,
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("D233 ordered linear MoE launch failed: ") +
                layer_moe.last_error()
            );
        }
        if (layer0_stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                hidden_output +
                    static_cast<size_t>(stage_digest_token) * kLayerHidden,
                kLayerHidden,
                layer0_stage_digests + 8u
            );
            std::array<uint32_t, kD230Routes> host_topk_ids{};
            std::array<float, kD230Routes> host_topk_weights{};
            if (layer_moe.copy_topk_debug(
                    host_topk_ids.data(),
                    host_topk_weights.data()
                ) == 0) {
                throw std::runtime_error(
                    "D233 ordered linear MoE top-k debug copy failed"
                );
            }
            D233Token0RouteMetadata selected_routes{};
            const size_t selected_route_offset =
                static_cast<size_t>(stage_digest_token) * kD230TopK;
            std::copy_n(
                host_topk_ids.begin() + selected_route_offset,
                kD230TopK,
                selected_routes.expert_ids
            );
            std::copy_n(
                host_topk_weights.begin() + selected_route_offset,
                kD230TopK,
                selected_routes.weights
            );
            std::cerr
                << "BATCH_MARK qwen36_q1024_owner_layer0_router"
                << " token=" << stage_digest_token
                << " digest=" << std::hex
                << fnv1a64(
                    &selected_routes,
                    sizeof(selected_routes)
                )
                << std::dec
                << " expert_ids=";
            for (unsigned int route = 0u;
                 route < kD230TopK;
                 ++route) {
                std::cerr
                    << (route == 0u ? "" : ",")
                    << selected_routes.expert_ids[route];
            }
            std::cerr << " weights=";
            for (unsigned int route = 0u;
                 route < kD230TopK;
                 ++route) {
                std::cerr
                    << (route == 0u ? "" : ",")
                    << std::hexfloat
                    << selected_routes.weights[route];
            }
            std::cerr
                << std::defaultfloat
                << " diagnostic_only=1"
                << std::endl;
            if (layer_moe.copy_token0_stage_debug != nullptr) {
                std::array<float, kLayerHidden> routed_combined{};
                std::array<uint16_t, kLayerHidden> shared_down{};
                float shared_gate_scale = 0.0f;
                if (layer_moe.copy_token0_stage_debug(
                        routed_combined.data(),
                        shared_down.data(),
                        &shared_gate_scale
                    ) == 0) {
                    throw std::runtime_error(
                        "D233 ordered linear MoE stage debug copy failed"
                    );
                }
                std::cerr
                    << "BATCH_MARK qwen36_q1024_owner_layer0_moe_stages"
                    << " routed_digest=" << std::hex
                    << fnv1a64(
                        routed_combined.data(),
                        routed_combined.size() *
                            sizeof(routed_combined[0])
                    )
                    << " shared_down_bf16_digest="
                    << fnv1a64(
                        shared_down.data(),
                        shared_down.size() * sizeof(shared_down[0])
                    )
                    << std::dec
                    << " shared_gate_scale=" << std::hexfloat
                    << shared_gate_scale
                    << std::defaultfloat
                    << " diagnostic_only=1"
                    << std::endl;
            }
            if (linear_ordinal == 1u &&
                layer_moe
                        .copy_q1024_early_f32_token0_activation_debug !=
                    nullptr) {
                std::array<
                    float,
                    kD230TopK * kD230Intermediate
                > routed_activated{};
                if (layer_moe
                        .copy_q1024_early_f32_token0_activation_debug(
                            routed_activated.data()
                        ) == 0) {
                    throw std::runtime_error(
                        "D233 q1024 early-F32 activation debug copy failed"
                    );
                }
                std::cerr
                    << "BATCH_MARK "
                    << "qwen36_q1024_owner_early_f32_activation"
                    << " digest=" << std::hex
                    << fnv1a64(
                        routed_activated.data(),
                        routed_activated.size() *
                            sizeof(routed_activated[0])
                    )
                    << std::dec
                    << " elements=" << routed_activated.size()
                    << " diagnostic_only=1"
                    << std::endl;
            }
        }
        check_hip(
            hipMemcpyAsync(
                layer.linear_final_state,
                final_state,
                kD232LinearStateElements * sizeof(float),
                hipMemcpyDeviceToDevice,
                nullptr
            ),
            "D233 publish linear final state"
        );
        check_hip(
            hipMemcpyAsync(
                layer.linear_final_ring,
                final_ring,
                kD232LinearRingElements * sizeof(T),
                hipMemcpyDeviceToDevice,
                nullptr
            ),
            "D233 publish linear final ring"
        );
    }

private:
    const ProviderApi &gdn_;
    const MoeApi &moe_;
    const MoeApi &fast_moe_;
    unsigned int fast_moe_min_layer_ = kD232Layers;
    unsigned int fast_moe_max_layer_ = kD232Layers;
    uint64_t fast_moe_layer_mask_ = UINT64_C(0);
    const RocblasContext &blas_;
    const D233LinearProjectionAot &projection_aot_;
    bool exact_q1_gdn_ = false;
    uint64_t exact_q1_gdn_layer_mask_ = UINT64_C(0);
    bool rocblas_projections_ = false;
    unsigned int exact_projection_layer_count_ = 0u;
    DeviceBuffer<float> input_norm_f32_;
    DeviceBuffer<uint16_t> input_norm_bf16_;
    DeviceBuffer<float> early_qkv_;
    DeviceBuffer<float> early_z_;
    DeviceBuffer<float> early_a_;
    DeviceBuffer<float> early_b_;
    DeviceBuffer<float> early_out_;
    DeviceBuffer<uint16_t> retained_qkv_;
    DeviceBuffer<uint16_t> retained_z_;
    DeviceBuffer<uint16_t> retained_a_;
    DeviceBuffer<uint16_t> retained_b_;
    DeviceBuffer<uint16_t> retained_out_;
    DeviceBuffer<float> retained_aux_f32_;
    DeviceBuffer<uint16_t> retained_projection_bf16_;
    DeviceBuffer<float> z_f32_;
    DeviceBuffer<float> a_f32_;
    DeviceBuffer<float> b_f32_;
    DeviceBuffer<float> gate_;
    DeviceBuffer<float> postconv_;
    DeviceBuffer<float> core_;
    DeviceBuffer<uint16_t> gated_;
    DeviceBuffer<float> residual_;
    DeviceBuffer<float> postnorm_;
    DeviceBuffer<float> final_state_;
    DeviceBuffer<float> early_final_ring_;
    DeviceBuffer<uint16_t> retained_final_ring_;
};

class D233FullOwner {
public:
    D233FullOwner(
        const CkApi &ck,
        const MoeApi &moe,
        const MoeApi &fast_moe,
        unsigned int fast_moe_min_layer,
        unsigned int fast_moe_max_layer,
        const RocblasContext &blas,
        const D233FullProjectionAot &projection_aot,
        const D233Q1024AttentionAot &q1024_attention_aot
    )
        : ck_(ck),
          moe_(moe),
          fast_moe_(fast_moe),
          fast_moe_min_layer_(fast_moe_min_layer),
          fast_moe_max_layer_(fast_moe_max_layer),
          fast_moe_layer_mask_(d233_fast_moe_layer_mask(
              fast_moe_min_layer,
              fast_moe_max_layer
          )),
          blas_(blas),
          projection_aot_(projection_aot),
          q1024_attention_aot_(q1024_attention_aot),
          rocblas_projections_(d233_rocblas_projections_enabled()),
          exact_projection_layer_count_(
              d233_exact_projection_layer_count()
          ),
          input_norm_f32_(kD232HiddenElements),
          input_norm_bf16_(kD232HiddenElements),
          qkv_projection_(kD233FullQkvProjectionElements),
          q_projection_(kD233FullQProjectionElements),
          k_projection_(kD233FullKProjectionElements),
          q_(kD232FullQElements),
          gate_(kD232FullQElements),
          full_k_(
              static_cast<size_t>(kD232FullLayers) *
              kD232FullKvElements
          ),
          full_v_(
              static_cast<size_t>(kD232FullLayers) *
              kD232FullKvElements
          ),
          context_(kD232FullQElements),
          gated_(kD232FullQElements),
          residual_(kD232HiddenElements),
          postnorm_(kD232HiddenElements),
          triton_q1024_attention_(q1024_attention_aot.enabled()),
          grouped_q1_attention_queries_(
              d233_grouped_q1_attention_queries_enabled()
          ),
          grouped_q1_attention_exact_pv_(
              d233_grouped_q1_attention_exact_pv_enabled()
          ),
          grouped_q1_attention_exact_pv_prefix_solution_(
              d233_grouped_q1_attention_exact_pv_prefix_solution()
          ),
          batched_q1_attention_(
              d233_batched_q1_attention_enabled() ||
              grouped_q1_attention_queries_ ||
              triton_q1024_attention_
          ),
          batched_q1_attention_full_layer_mask_(
              d233_batched_q1_attention_full_layer_mask()
          ),
          exact_q1_attention_qk_tail_(
              d233_exact_q1_attention_tail_component_enabled(
                  "QRT_QWEN36_Q1024_OWNER_EXACT_Q1_ATTENTION_QK_TAIL"
              )
          ),
          exact_q1_attention_pv_tail_(
              d233_exact_q1_attention_tail_component_enabled(
                  "QRT_QWEN36_Q1024_OWNER_EXACT_Q1_ATTENTION_PV_TAIL"
              )
          ),
          q1_attention_scores_(kD233Q1AttentionScoreElements),
          grouped_q1_queries_(
              static_cast<size_t>(kD233Q1AttentionQueryChunk) *
              kD231QFeatures
          ),
          grouped_q1_context_(
              static_cast<size_t>(kD233Q1AttentionQueryChunk) *
              kD231QFeatures
          ) {
        if (grouped_q1_attention_queries_ &&
            triton_q1024_attention_) {
            throw std::runtime_error(
                "D233 grouped-query and Triton q1024 attention conflict"
            );
        }
        if (grouped_q1_attention_exact_pv_ &&
            !grouped_q1_attention_queries_) {
            throw std::runtime_error(
                "D233 grouped-query exact PV requires grouped queries"
            );
        }
        if (batched_q1_attention_ && !triton_q1024_attention_) {
            check_rocblas(
                rocblas_create_handle(&q1_attention_handle_),
                "D233 create batched q1 attention rocBLAS handle"
            );
            check_rocblas(
                rocblas_set_stream(q1_attention_handle_, nullptr),
                "D233 set batched q1 attention rocBLAS stream"
            );
            check_rocblas(
                rocblas_set_pointer_mode(
                    q1_attention_handle_,
                    rocblas_pointer_mode_host
                ),
                "D233 set batched q1 attention pointer mode"
            );
#if QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS > 1
            if ((exact_q1_attention_qk_tail_ ||
                 exact_q1_attention_pv_tail_) &&
                grouped_q1_attention_queries_) {
                check_hip(
                    hipEventCreateWithFlags(
                        &exact_tail_ready_event_,
                        hipEventDisableTiming
                    ),
                    "D233 create exact-tail ready event"
                );
                for (unsigned int index = 0u;
                     index < kD233ExactTailStreams;
                     ++index) {
                    check_hip(
                        hipStreamCreateWithFlags(
                            &exact_tail_streams_[index],
                            hipStreamNonBlocking
                        ),
                        "D233 create exact-tail stream"
                    );
                    check_hip(
                        hipEventCreateWithFlags(
                            &exact_tail_done_events_[index],
                            hipEventDisableTiming
                        ),
                        "D233 create exact-tail done event"
                    );
                    check_rocblas(
                        rocblas_create_handle(
                            &exact_tail_handles_[index]
                        ),
                        "D233 create exact-tail rocBLAS handle"
                    );
                    check_rocblas(
                        rocblas_set_stream(
                            exact_tail_handles_[index],
                            exact_tail_streams_[index]
                        ),
                        "D233 set exact-tail rocBLAS stream"
                    );
                    check_rocblas(
                        rocblas_set_pointer_mode(
                            exact_tail_handles_[index],
                            rocblas_pointer_mode_host
                        ),
                        "D233 set exact-tail pointer mode"
                    );
                }
            }
#endif
            if (d233_profile_layers_enabled() &&
                grouped_q1_attention_queries_) {
                for (auto &chunk_events :
                     q1_attention_profile_events_) {
                    for (hipEvent_t &event : chunk_events) {
                        check_hip(
                            hipEventCreateWithFlags(
                                &event,
                                hipEventDefault
                            ),
                            "D233 create attention subphase profile event"
                        );
                    }
                }
            }
        }
    }

    ~D233FullOwner() {
        for (auto &chunk_events : q1_attention_profile_events_) {
            for (hipEvent_t &event : chunk_events) {
                if (event != nullptr) {
                    (void)hipEventDestroy(event);
                    event = nullptr;
                }
            }
        }
#if QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS > 1
        for (unsigned int index = 0u;
             index < kD233ExactTailStreams;
             ++index) {
            if (exact_tail_handles_[index] != nullptr) {
                (void)rocblas_destroy_handle(
                    exact_tail_handles_[index]
                );
            }
            if (exact_tail_done_events_[index] != nullptr) {
                (void)hipEventDestroy(
                    exact_tail_done_events_[index]
                );
            }
            if (exact_tail_streams_[index] != nullptr) {
                (void)hipStreamDestroy(exact_tail_streams_[index]);
            }
        }
        if (exact_tail_ready_event_ != nullptr) {
            (void)hipEventDestroy(exact_tail_ready_event_);
        }
#endif
        if (q1_attention_handle_ != nullptr) {
            (void)rocblas_destroy_handle(q1_attention_handle_);
        }
    }

    bool uses_batched_q1_attention() const {
        return batched_q1_attention_;
    }

    bool uses_triton_q1024_attention() const {
        return triton_q1024_attention_;
    }

    bool uses_grouped_q1_attention_queries() const {
        return grouped_q1_attention_queries_;
    }

    bool uses_grouped_q1_attention_exact_pv() const {
        return grouped_q1_attention_exact_pv_;
    }

    rocblas_int grouped_q1_attention_exact_pv_prefix_solution() const {
        return grouped_q1_attention_exact_pv_prefix_solution_;
    }

    unsigned int batched_q1_attention_full_layer_mask() const {
        return batched_q1_attention_full_layer_mask_;
    }

    bool uses_exact_q1_attention_tail() const {
        return batched_q1_attention_ && !triton_q1024_attention_ &&
            exact_q1_attention_qk_tail_ &&
            exact_q1_attention_pv_tail_;
    }

    bool uses_exact_q1_attention_qk_tail() const {
        return batched_q1_attention_ && !triton_q1024_attention_ &&
            exact_q1_attention_qk_tail_;
    }

    bool uses_exact_q1_attention_pv_tail() const {
        return batched_q1_attention_ && !triton_q1024_attention_ &&
            exact_q1_attention_pv_tail_;
    }

    unsigned int exact_q1_attention_tail_stream_count() const {
        return (uses_exact_q1_attention_qk_tail() ||
                uses_exact_q1_attention_pv_tail()) &&
                grouped_q1_attention_queries_
            ? kD233ExactTailStreams
            : 1u;
    }

    bool uses_rocblas_projections() const {
        return rocblas_projections_;
    }

    unsigned int exact_projection_layer_count() const {
        return exact_projection_layer_count_;
    }

    void reset() {
        d233_zero_device(
            input_norm_f32_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset full input norm F32"
        );
        d233_zero_device(
            input_norm_bf16_.get(),
            kD232HiddenElements * sizeof(uint16_t),
            "D233 reset full input norm BF16"
        );
        d233_zero_device(
            qkv_projection_.get(),
            kD233FullQkvProjectionElements * sizeof(uint16_t),
            "D233 reset full combined QKV projection"
        );
        d233_zero_device(
            q_projection_.get(),
            kD233FullQProjectionElements * sizeof(uint16_t),
            "D233 reset full Q projection"
        );
        d233_zero_device(
            k_projection_.get(),
            kD233FullKProjectionElements * sizeof(uint16_t),
            "D233 reset full K projection"
        );
        d233_zero_device(
            q_.get(),
            kD232FullQElements * sizeof(uint16_t),
            "D233 reset full Q"
        );
        d233_zero_device(
            gate_.get(),
            kD232FullQElements * sizeof(uint16_t),
            "D233 reset full gate"
        );
        d233_zero_device(
            full_k_.get(),
            static_cast<size_t>(kD232FullLayers) *
                kD232FullKvElements * sizeof(uint16_t),
            "D233 reset combined K"
        );
        d233_zero_device(
            full_v_.get(),
            static_cast<size_t>(kD232FullLayers) *
                kD232FullKvElements * sizeof(uint16_t),
            "D233 reset combined V"
        );
        d233_zero_device(
            context_.get(),
            kD232FullQElements * sizeof(float),
            "D233 reset full context"
        );
        d233_zero_device(
            gated_.get(),
            kD232FullQElements * sizeof(uint16_t),
            "D233 reset full gated output"
        );
        d233_zero_device(
            residual_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset full residual"
        );
        d233_zero_device(
            postnorm_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset full postnorm"
        );
    }

    uint64_t copy_prefix(
        const qrt_qwen36_q1024_owner_layer_v1_t *layers
    ) {
        uint64_t copied = UINT64_C(0);
        unsigned int full_ordinal = 0u;
        for (unsigned int layer_index = 0u;
             layer_index < kD232Layers;
             ++layer_index) {
            if ((layer_index % 4u) != 3u) {
                continue;
            }
            const qrt_qwen36_q1024_owner_layer_v1_t &layer =
                layers[layer_index];
            uint16_t *destination_k =
                full_k_.get() +
                static_cast<size_t>(full_ordinal) *
                    kD232FullKvElements;
            uint16_t *destination_v =
                full_v_.get() +
                static_cast<size_t>(full_ordinal) *
                    kD232FullKvElements;
            const size_t bytes =
                kD232FullPrefixKvElements * sizeof(uint16_t);
            check_hip(
                hipMemcpyAsync(
                    destination_k,
                    layer.full_prefix_k,
                    bytes,
                    hipMemcpyDeviceToDevice,
                    nullptr
                ),
                "D233 copy full prefix K"
            );
            check_hip(
                hipMemcpyAsync(
                    destination_v,
                    layer.full_prefix_v,
                    bytes,
                    hipMemcpyDeviceToDevice,
                    nullptr
                ),
                "D233 copy full prefix V"
            );
            copied += 2u * static_cast<uint64_t>(bytes);
            ++full_ordinal;
        }
        if (full_ordinal != kD232FullLayers) {
            throw std::runtime_error(
                "D233 full prefix layer accounting failed"
            );
        }
        return copied;
    }

    void launch(
        const qrt_qwen36_q1024_owner_layer_v1_t &layer,
        unsigned int full_ordinal,
        unsigned int absolute_position_base,
        const float *hidden_input,
        float *hidden_output,
        uint64_t *stage_digests,
        unsigned int stage_token,
        hipEvent_t projection_end_event,
        hipEvent_t core_end_event,
        hipEvent_t moe_begin_event
    ) {
        if (full_ordinal >= kD232FullLayers) {
            throw std::runtime_error("D233 full ordinal is out of range");
        }
        const dim3 block(kThreads);
        const size_t hidden_stage_offset =
            static_cast<size_t>(stage_token) * kLayerHidden;
        const size_t qkv_stage_offset =
            static_cast<size_t>(stage_token) * kD233FullQkvRows;
        const size_t q_stage_offset =
            static_cast<size_t>(stage_token) * kD231QFeatures;
        const size_t kv_stage_offset =
            static_cast<size_t>(stage_token) * kD231KvFeatures;
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                hidden_input + hidden_stage_offset,
                kLayerHidden,
                stage_digests + 0u
            );
        }
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            layer.input_norm,
            input_norm_f32_.get()
        );
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                input_norm_f32_.get() + hidden_stage_offset,
                kLayerHidden,
                stage_digests + 1u
            );
        }
        hipLaunchKernelGGL(
            f32_to_bf16_sublayer_kernel,
            d232_linear_grid(kD232HiddenElements),
            block,
            0u,
            nullptr,
            input_norm_f32_.get(),
            input_norm_bf16_.get(),
            kD232HiddenElements
        );
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_bytes_digest(
                input_norm_bf16_.get() + hidden_stage_offset,
                kLayerHidden * sizeof(uint16_t),
                stage_digests + 2u
            );
        }
        uint16_t *layer_k =
            full_k_.get() +
            static_cast<size_t>(full_ordinal) * kD232FullKvElements;
        uint16_t *layer_v =
            full_v_.get() +
            static_cast<size_t>(full_ordinal) * kD232FullKvElements;
        uint16_t *suffix_k =
            layer_k +
            static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
        uint16_t *suffix_v =
            layer_v +
            static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
        if (rocblas_projections_ &&
            layer.layer_index >= exact_projection_layer_count_) {
            blas_.matmul(
                layer.full_q,
                input_norm_bf16_.get(),
                q_projection_.get(),
                kD231QProjectionRows,
                kLayerHidden
            );
            blas_.matmul(
                layer.full_k,
                input_norm_bf16_.get(),
                k_projection_.get(),
                kD231KProjectionRows,
                kLayerHidden
            );
            blas_.matmul(
                layer.full_v,
                input_norm_bf16_.get(),
                suffix_v,
                kD231VProjectionRows,
                kLayerHidden
            );
            hipLaunchKernelGGL(
                d233_pack_rocblas_full_projection_kernel,
                d232_linear_grid(kD233FullQkvProjectionElements),
                block,
                0u,
                nullptr,
                q_projection_.get(),
                k_projection_.get(),
                suffix_v,
                qkv_projection_.get()
            );
        } else {
            projection_aot_.launch(
                layer.full_q,
                layer.full_k,
                layer.full_v,
                input_norm_bf16_.get(),
                qkv_projection_.get()
            );
            hipLaunchKernelGGL(
                d233_publish_full_projection_kernel,
                d232_linear_grid(kD233FullQkvProjectionElements),
                block,
                0u,
                nullptr,
                qkv_projection_.get(),
                q_projection_.get(),
                k_projection_.get(),
                suffix_v
            );
        }
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_bytes_digest(
                qkv_projection_.get() + qkv_stage_offset,
                kD233FullQkvRows * sizeof(uint16_t),
                stage_digests + 3u
            );
        }
        hipLaunchKernelGGL(
            d233_prepare_q_gate_kernel,
            dim3(kD231QueryHeads, kSuffixTokens),
            dim3(kD231HeadDim),
            0u,
            nullptr,
            q_projection_.get(),
            layer.full_q_norm,
            q_.get(),
            gate_.get()
        );
        hipLaunchKernelGGL(
            d233_prepare_k_kernel,
            dim3(kD231KvHeads, kSuffixTokens),
            dim3(kD231HeadDim),
            0u,
            nullptr,
            k_projection_.get(),
            layer.full_k_norm,
            suffix_k
        );
        hipLaunchKernelGGL(
            d233_rope_bf16_kernel,
            dim3(kD231QueryHeads, kSuffixTokens),
            dim3(kD231RotaryDim / 2u),
            0u,
            nullptr,
            q_.get(),
            kD231QueryHeads,
            kD231QFeatures,
            absolute_position_base
        );
        hipLaunchKernelGGL(
            d233_rope_bf16_kernel,
            dim3(kD231KvHeads, kSuffixTokens),
            dim3(kD231RotaryDim / 2u),
            0u,
            nullptr,
            suffix_k,
            kD231KvHeads,
            kD231KvFeatures,
            absolute_position_base
        );
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_bytes_digest(
                q_.get() + q_stage_offset,
                kD231QFeatures * sizeof(uint16_t),
                stage_digests + 4u
            );
            d233_launch_fnv1a64_bytes_digest(
                gate_.get() + q_stage_offset,
                kD231QFeatures * sizeof(uint16_t),
                stage_digests + 5u
            );
            d233_launch_fnv1a64_bytes_digest(
                suffix_k + kv_stage_offset,
                kD231KvFeatures * sizeof(uint16_t),
                stage_digests + 6u
            );
            d233_launch_fnv1a64_bytes_digest(
                suffix_v + kv_stage_offset,
                kD231KvFeatures * sizeof(uint16_t),
                stage_digests + 7u
            );
        }
        if (projection_end_event != nullptr) {
            check_hip(
                hipEventRecord(projection_end_event, nullptr),
                "D233 record full projection profile boundary"
            );
        }
        const bool use_batched_q1_attention =
            batched_q1_attention_ &&
            (batched_q1_attention_full_layer_mask_ &
             (1u << full_ordinal)) != 0u;
        if (use_batched_q1_attention) {
            launch_batched_q1_attention(
                layer_k,
                layer_v,
                suffix_k,
                suffix_v
            );
        } else {
            if (ck_.suffix_launch(
                    q_.get(),
                    layer_k,
                    layer_v,
                    context_.get(),
                    nullptr
                ) != 0) {
                throw std::runtime_error(
                    "D233 CK suffix attention launch failed"
                );
            }
        }
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                context_.get() + q_stage_offset,
                kD231QFeatures,
                stage_digests + 8u
            );
        }
        hipLaunchKernelGGL(
            d231_attention_gate_kernel,
            d232_linear_grid(kD232FullQElements),
            block,
            0u,
            nullptr,
            context_.get(),
            gate_.get(),
            gated_.get(),
            kD232FullQElements
        );
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_bytes_digest(
                gated_.get() + q_stage_offset,
                kD231QFeatures * sizeof(uint16_t),
                stage_digests + 9u
            );
        }
        if (core_end_event != nullptr) {
            check_hip(
                hipEventRecord(core_end_event, nullptr),
                "D233 record full attention profile boundary"
            );
        }
        d233_launch_early_output_projection_exact(
            layer.full_out,
            gated_.get(),
            context_.get()
        );
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                context_.get() + hidden_stage_offset,
                kLayerHidden,
                stage_digests + 10u
            );
        }
        hipLaunchKernelGGL(
            residual_postnorm_sublayer_kernel<float>,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            context_.get(),
            layer.post_attention_norm,
            residual_.get(),
            postnorm_.get()
        );
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                residual_.get() + hidden_stage_offset,
                kLayerHidden,
                stage_digests + 11u
            );
            d233_launch_fnv1a64_f32_digest(
                postnorm_.get() + hidden_stage_offset,
                kLayerHidden,
                stage_digests + 12u
            );
        }
        if (moe_begin_event != nullptr) {
            check_hip(
                hipEventRecord(moe_begin_event, nullptr),
                "D233 record full MoE profile boundary"
            );
        }
        const MoeApi &layer_moe =
            (fast_moe_layer_mask_ &
                    (UINT64_C(1) << layer.layer_index)) != 0u
                ? fast_moe_
                : moe_;
        if (layer_moe.launch_v2(
                postnorm_.get(),
                residual_.get(),
                layer.moe_router,
                layer.moe_expert_gate_up,
                layer.moe_expert_down,
                layer.moe_shared_gate,
                layer.moe_shared_gate_projection,
                layer.moe_shared_up_projection,
                layer.moe_shared_down,
                hidden_output,
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("D233 ordered full MoE launch failed: ") +
                layer_moe.last_error()
            );
        }
        if (stage_digests != nullptr) {
            d233_launch_fnv1a64_f32_digest(
                hidden_output + hidden_stage_offset,
                kLayerHidden,
                stage_digests + 13u
            );
        }
        const size_t suffix_bytes =
            kD232FullSuffixKvElements * sizeof(uint16_t);
        check_hip(
            hipMemcpyAsync(
                layer.full_suffix_k,
                suffix_k,
                suffix_bytes,
                hipMemcpyDeviceToDevice,
                nullptr
            ),
            "D233 publish full suffix K"
        );
        check_hip(
            hipMemcpyAsync(
                layer.full_suffix_v,
                suffix_v,
                suffix_bytes,
                hipMemcpyDeviceToDevice,
                nullptr
            ),
            "D233 publish full suffix V"
        );
    }

private:
    rocblas_handle exact_tail_handle(unsigned int local_query) const {
#if QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS > 1
        return exact_tail_handles_[
            local_query % kD233ExactTailStreams
        ];
#else
        (void)local_query;
        return q1_attention_handle_;
#endif
    }

    void begin_exact_tail_batch() {
#if QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS > 1
        check_hip(
            hipEventRecord(exact_tail_ready_event_, nullptr),
            "D233 record exact-tail ready event"
        );
        for (unsigned int index = 0u;
             index < kD233ExactTailStreams;
             ++index) {
            check_hip(
                hipStreamWaitEvent(
                    exact_tail_streams_[index],
                    exact_tail_ready_event_,
                    0u
                ),
                "D233 wait exact-tail ready event"
            );
        }
#endif
    }

    void end_exact_tail_batch() {
#if QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS > 1
        for (unsigned int index = 0u;
             index < kD233ExactTailStreams;
             ++index) {
            check_hip(
                hipEventRecord(
                    exact_tail_done_events_[index],
                    exact_tail_streams_[index]
                ),
                "D233 record exact-tail done event"
            );
            check_hip(
                hipStreamWaitEvent(
                    nullptr,
                    exact_tail_done_events_[index],
                    0u
                ),
                "D233 wait exact-tail done event"
            );
        }
#endif
    }

    void launch_triton_q1024_attention(
        const uint16_t *layer_k,
        const uint16_t *layer_v,
        const uint16_t *suffix_k,
        const uint16_t *suffix_v
    ) {
        for (unsigned int query_base = 0u;
             query_base < kSuffixTokens;
             query_base += kD233Q1AttentionQueryChunk) {
            const unsigned int query_count = std::min(
                kD233Q1AttentionQueryChunk,
                kSuffixTokens - query_base
            );
            float *scores = q1_attention_scores_.get();
            const uint16_t *query =
                q_.get() +
                static_cast<size_t>(query_base) * kD231QFeatures;
            q1024_attention_aot_.launch_qk(
                query,
                layer_k,
                suffix_k,
                scores,
                query_base,
                query_count
            );
            hipLaunchKernelGGL(
                d233_q1_attention_softmax_kernel<false>,
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
            hipLaunchKernelGGL(
                d233_q1_attention_probability_bf16_kernel<false>,
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
            check_hip(
                hipGetLastError(),
                "D233 q1024 Triton attention softmax/probability"
            );
            const uint16_t *probabilities =
                reinterpret_cast<const uint16_t *>(scores);
            float *context =
                context_.get() +
                static_cast<size_t>(query_base) * kD231QFeatures;
            q1024_attention_aot_.launch_pv(
                probabilities,
                layer_v,
                suffix_v,
                context,
                query_base,
                query_count
            );
        }
    }

    void launch_grouped_q1_attention_queries(
        const uint16_t *layer_k,
        const uint16_t *layer_v,
        const uint16_t *suffix_k,
        const uint16_t *suffix_v
    ) {
        if (q1_attention_handle_ == nullptr) {
            throw std::runtime_error(
                "D233 grouped-query attention handle is unavailable"
            );
        }
        constexpr float kQkAlpha = 0.0625f;
        constexpr float kZero = 0.0f;
        constexpr float kOne = 1.0f;
        constexpr rocblas_int kQueryLeadingDimension =
            static_cast<rocblas_int>(kD231HeadDim);
        constexpr rocblas_int kScoreLeadingDimension =
            static_cast<rocblas_int>(kD233Q1AttentionScoreStride);
        constexpr rocblas_int kProbabilityLeadingDimension =
            static_cast<rocblas_int>(
                2u * kD233Q1AttentionScoreStride
            );
        constexpr rocblas_int kContextLeadingDimension =
            static_cast<rocblas_int>(kD231HeadDim);
        const dim3 block(kThreads);
        const bool profile_attention_subphases =
            q1_attention_profile_events_[0u][0u] != nullptr;
        const auto record_profile_boundary =
            [&](unsigned int query_base,
                unsigned int boundary_index) {
                if (!profile_attention_subphases) {
                    return;
                }
                const unsigned int chunk_index =
                    query_base / kD233Q1AttentionQueryChunk;
                check_hip(
                    hipEventRecord(
                        q1_attention_profile_events_
                            [chunk_index][boundary_index],
                        nullptr
                    ),
                    "D233 record attention subphase profile boundary"
                );
            };

        for (unsigned int query_base = 0u;
             query_base < kSuffixTokens;
             query_base += kD233Q1AttentionQueryChunk) {
            const unsigned int query_count = std::min(
                kD233Q1AttentionQueryChunk,
                kSuffixTokens - query_base
            );
            const size_t query_elements =
                static_cast<size_t>(query_count) * kD231QFeatures;
            record_profile_boundary(query_base, 0u);
            const uint16_t *query =
                q_.get() +
                static_cast<size_t>(query_base) * kD231QFeatures;
            hipLaunchKernelGGL(
                d233_pack_grouped_q1_queries_kernel,
                d232_linear_grid(query_elements),
                block,
                0u,
                nullptr,
                query,
                grouped_q1_queries_.get(),
                query_count
            );
            record_profile_boundary(query_base, 1u);

            float *scores = q1_attention_scores_.get();
            const rocblas_int grouped_columns =
                static_cast<rocblas_int>(
                    query_count * kD233HeadsPerKv
                );
            const size_t grouped_query_elements =
                static_cast<size_t>(query_count) * kD233HeadsPerKv *
                kD231HeadDim;
            const size_t grouped_score_elements =
                static_cast<size_t>(query_count) * kD233HeadsPerKv *
                kD233Q1AttentionScoreStride;
            for (unsigned int kv_head = 0u;
                 kv_head < kD231KvHeads;
                 ++kv_head) {
                const size_t kv_head_offset =
                    static_cast<size_t>(kv_head) * kD231HeadDim;
                const uint16_t *grouped_query =
                    grouped_q1_queries_.get() +
                    static_cast<size_t>(kv_head) *
                        grouped_query_elements;
                float *grouped_scores =
                    scores +
                    static_cast<size_t>(kv_head) *
                        grouped_score_elements;
                check_rocblas(
                    rocblas_gemm_ex(
                        q1_attention_handle_,
                        rocblas_operation_transpose,
                        rocblas_operation_none,
                        static_cast<rocblas_int>(kPrefixTokens),
                        grouped_columns,
                        static_cast<rocblas_int>(kD231HeadDim),
                        &kQkAlpha,
                        layer_k + kv_head_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(kD231KvFeatures),
                        grouped_query,
                        rocblas_datatype_bf16_r,
                        kQueryLeadingDimension,
                        &kZero,
                        grouped_scores,
                        rocblas_datatype_f32_r,
                        kScoreLeadingDimension,
                        grouped_scores,
                        rocblas_datatype_f32_r,
                        kScoreLeadingDimension,
                        rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard,
                        0,
                        rocblas_gemm_flags_none
                    ),
                    "D233 grouped-query prefix QK"
                );
                if (!exact_q1_attention_qk_tail_) {
                    check_rocblas(
                        rocblas_gemm_ex(
                            q1_attention_handle_,
                            rocblas_operation_transpose,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(kSuffixTokens),
                            grouped_columns,
                            static_cast<rocblas_int>(kD231HeadDim),
                            &kQkAlpha,
                            suffix_k + kv_head_offset,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            grouped_query,
                            rocblas_datatype_bf16_r,
                            kQueryLeadingDimension,
                            &kZero,
                            grouped_scores + kPrefixTokens,
                            rocblas_datatype_f32_r,
                            kScoreLeadingDimension,
                            grouped_scores + kPrefixTokens,
                            rocblas_datatype_f32_r,
                            kScoreLeadingDimension,
                            rocblas_datatype_f32_r,
                            rocblas_gemm_algo_standard,
                            0,
                            rocblas_gemm_flags_none
                        ),
                        "D233 grouped-query suffix QK"
                    );
                }
            }
            record_profile_boundary(query_base, 2u);
            if (exact_q1_attention_qk_tail_) {
                begin_exact_tail_batch();
                for (unsigned int local_query = 0u;
                     local_query < query_count;
                     ++local_query) {
                    const unsigned int tail_tokens =
                        query_base + local_query + 1u;
                    const uint16_t *single_query =
                        grouped_q1_queries_.get() +
                        static_cast<size_t>(local_query) *
                            kD233HeadsPerKv * kD231HeadDim;
                    float *single_scores =
                        scores +
                        static_cast<size_t>(local_query) *
                            kD233HeadsPerKv *
                            kD233Q1AttentionScoreStride +
                        kPrefixTokens;
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                            exact_tail_handle(local_query),
                            rocblas_operation_transpose,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(tail_tokens),
                            static_cast<rocblas_int>(
                                kD233HeadsPerKv
                            ),
                            static_cast<rocblas_int>(kD231HeadDim),
                            &kQkAlpha,
                            suffix_k,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            static_cast<rocblas_stride>(kD231HeadDim),
                            single_query,
                            rocblas_datatype_bf16_r,
                            kQueryLeadingDimension,
                            static_cast<rocblas_stride>(
                                grouped_query_elements
                            ),
                            &kZero,
                            single_scores,
                            rocblas_datatype_f32_r,
                            kScoreLeadingDimension,
                            static_cast<rocblas_stride>(
                                grouped_score_elements
                            ),
                            single_scores,
                            rocblas_datatype_f32_r,
                            kScoreLeadingDimension,
                            static_cast<rocblas_stride>(
                                grouped_score_elements
                            ),
                            static_cast<rocblas_int>(kD231KvHeads),
                            rocblas_datatype_f32_r,
                            rocblas_gemm_algo_standard,
                            0,
                            rocblas_gemm_flags_none
                        ),
                        "D233 grouped-query exact suffix QK"
                    );
                }
                end_exact_tail_batch();
            }
            record_profile_boundary(query_base, 3u);

#if QRT_QWEN36_Q1024_OWNER_FUSED_SOFTMAX_PROBABILITY
            hipLaunchKernelGGL(
                (d233_q1_attention_softmax_kernel<true, true>),
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
#else
            hipLaunchKernelGGL(
                (d233_q1_attention_softmax_kernel<true, false>),
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
            hipLaunchKernelGGL(
                d233_q1_attention_probability_bf16_kernel<true>,
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
#endif
            check_hip(
                hipGetLastError(),
                "D233 grouped-query attention softmax/probability"
            );
            record_profile_boundary(query_base, 4u);

            const uint16_t *probabilities =
                reinterpret_cast<const uint16_t *>(scores);
            const size_t grouped_probability_elements =
                static_cast<size_t>(query_count) * kD233HeadsPerKv *
                2u * kD233Q1AttentionScoreStride;
            const size_t grouped_context_elements =
                static_cast<size_t>(query_count) * kD233HeadsPerKv *
                kD231HeadDim;
            constexpr rocblas_stride kExactProbabilityQueryStride =
                static_cast<rocblas_stride>(
                    kD233HeadsPerKv *
                    2u * kD233Q1AttentionScoreStride
                );
            constexpr rocblas_stride kExactContextQueryStride =
                static_cast<rocblas_stride>(
                    kD233HeadsPerKv * kD231HeadDim
                );
            for (unsigned int kv_head = 0u;
                 kv_head < kD231KvHeads;
                 ++kv_head) {
                const size_t kv_head_offset =
                    static_cast<size_t>(kv_head) * kD231HeadDim;
                const uint16_t *grouped_probabilities =
                    probabilities +
                    static_cast<size_t>(kv_head) *
                        grouped_probability_elements;
                float *grouped_context =
                    grouped_q1_context_.get() +
                    static_cast<size_t>(kv_head) *
                        grouped_context_elements;
                if (grouped_q1_attention_exact_pv_) {
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                            q1_attention_handle_,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_int>(
                                kD233HeadsPerKv
                            ),
                            static_cast<rocblas_int>(kPrefixTokens),
                            &kOne,
                            layer_v + kv_head_offset,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            0,
                            grouped_probabilities,
                            rocblas_datatype_bf16_r,
                            kProbabilityLeadingDimension,
                            kExactProbabilityQueryStride,
                            &kZero,
                            grouped_context,
                            rocblas_datatype_f32_r,
                            kContextLeadingDimension,
                            kExactContextQueryStride,
                            grouped_context,
                            rocblas_datatype_f32_r,
                            kContextLeadingDimension,
                            kExactContextQueryStride,
                            static_cast<rocblas_int>(query_count),
                            rocblas_datatype_f32_r,
                            grouped_q1_attention_exact_pv_prefix_solution_ != 0
                                ? rocblas_gemm_algo_solution_index
                                : rocblas_gemm_algo_standard,
                            grouped_q1_attention_exact_pv_prefix_solution_,
                            rocblas_gemm_flags_none
                        ),
                        "D233 grouped-query exact-shape prefix PV"
                    );
                } else {
                    check_rocblas(
                        rocblas_gemm_ex(
                            q1_attention_handle_,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(kD231HeadDim),
                            grouped_columns,
                            static_cast<rocblas_int>(kPrefixTokens),
                            &kOne,
                            layer_v + kv_head_offset,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            grouped_probabilities,
                            rocblas_datatype_bf16_r,
                            kProbabilityLeadingDimension,
                            &kZero,
                            grouped_context,
                            rocblas_datatype_f32_r,
                            kContextLeadingDimension,
                            grouped_context,
                            rocblas_datatype_f32_r,
                            kContextLeadingDimension,
                            rocblas_datatype_f32_r,
                            kD233WidePrefixPvSolution != 0
                                ? rocblas_gemm_algo_solution_index
                                : rocblas_gemm_algo_standard,
                            kD233WidePrefixPvSolution,
                            rocblas_gemm_flags_none
                        ),
                        "D233 grouped-query prefix PV"
                    );
                }
                if (!exact_q1_attention_pv_tail_) {
                    if (grouped_q1_attention_exact_pv_) {
                        check_rocblas(
                            rocblas_gemm_strided_batched_ex(
                                q1_attention_handle_,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                static_cast<rocblas_int>(kD231HeadDim),
                                static_cast<rocblas_int>(
                                    kD233HeadsPerKv
                                ),
                                static_cast<rocblas_int>(kSuffixTokens),
                                &kOne,
                                suffix_v + kv_head_offset,
                                rocblas_datatype_bf16_r,
                                static_cast<rocblas_int>(kD231KvFeatures),
                                0,
                                grouped_probabilities + kPrefixTokens,
                                rocblas_datatype_bf16_r,
                                kProbabilityLeadingDimension,
                                kExactProbabilityQueryStride,
                                &kOne,
                                grouped_context,
                                rocblas_datatype_f32_r,
                                kContextLeadingDimension,
                                kExactContextQueryStride,
                                grouped_context,
                                rocblas_datatype_f32_r,
                                kContextLeadingDimension,
                                kExactContextQueryStride,
                                static_cast<rocblas_int>(query_count),
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none
                            ),
                            "D233 grouped-query exact-shape suffix PV"
                        );
                    } else {
                        check_rocblas(
                            rocblas_gemm_ex(
                                q1_attention_handle_,
                                rocblas_operation_none,
                                rocblas_operation_none,
                                static_cast<rocblas_int>(kD231HeadDim),
                                grouped_columns,
                                static_cast<rocblas_int>(kSuffixTokens),
                                &kOne,
                                suffix_v + kv_head_offset,
                                rocblas_datatype_bf16_r,
                                static_cast<rocblas_int>(kD231KvFeatures),
                                grouped_probabilities + kPrefixTokens,
                                rocblas_datatype_bf16_r,
                                kProbabilityLeadingDimension,
                                &kOne,
                                grouped_context,
                                rocblas_datatype_f32_r,
                                kContextLeadingDimension,
                                grouped_context,
                                rocblas_datatype_f32_r,
                                kContextLeadingDimension,
                                rocblas_datatype_f32_r,
                                rocblas_gemm_algo_standard,
                                0,
                                rocblas_gemm_flags_none
                            ),
                            "D233 grouped-query suffix PV"
                        );
                    }
                }
            }
            record_profile_boundary(query_base, 5u);
            if (exact_q1_attention_pv_tail_) {
                begin_exact_tail_batch();
                for (unsigned int local_query = 0u;
                     local_query < query_count;
                     ++local_query) {
                    const unsigned int tail_tokens =
                        query_base + local_query + 1u;
                    const uint16_t *single_probabilities =
                        probabilities +
                        static_cast<size_t>(local_query) *
                            kD233HeadsPerKv *
                            2u * kD233Q1AttentionScoreStride +
                        kPrefixTokens;
                    float *single_context =
                        grouped_q1_context_.get() +
                        static_cast<size_t>(local_query) *
                            kD233HeadsPerKv * kD231HeadDim;
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                            exact_tail_handle(local_query),
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_int>(
                                kD233HeadsPerKv
                            ),
                            static_cast<rocblas_int>(tail_tokens),
                            &kOne,
                            suffix_v,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            static_cast<rocblas_stride>(kD231HeadDim),
                            single_probabilities,
                            rocblas_datatype_bf16_r,
                            kProbabilityLeadingDimension,
                            static_cast<rocblas_stride>(
                                grouped_probability_elements
                            ),
                            &kOne,
                            single_context,
                            rocblas_datatype_f32_r,
                            kContextLeadingDimension,
                            static_cast<rocblas_stride>(
                                grouped_context_elements
                            ),
                            single_context,
                            rocblas_datatype_f32_r,
                            kContextLeadingDimension,
                            static_cast<rocblas_stride>(
                                grouped_context_elements
                            ),
                            static_cast<rocblas_int>(kD231KvHeads),
                            rocblas_datatype_f32_r,
                            rocblas_gemm_algo_standard,
                            0,
                            rocblas_gemm_flags_none
                        ),
                        "D233 grouped-query exact suffix PV"
                    );
                }
                end_exact_tail_batch();
            }
            record_profile_boundary(query_base, 6u);

            float *context =
                context_.get() +
                static_cast<size_t>(query_base) * kD231QFeatures;
            hipLaunchKernelGGL(
                d233_scatter_grouped_q1_context_kernel,
                d232_linear_grid(query_elements),
                block,
                0u,
                nullptr,
                grouped_q1_context_.get(),
                context,
                query_count
            );
            check_hip(
                hipGetLastError(),
                "D233 grouped-query context scatter"
            );
            record_profile_boundary(query_base, 7u);
        }
        if (profile_attention_subphases) {
            check_hip(
                hipEventSynchronize(
                    q1_attention_profile_events_
                        [kD233Q1AttentionChunkCount - 1u]
                        [kD233Q1AttentionProfileBoundaryCount - 1u]
                ),
                "D233 synchronize attention subphase profile"
            );
            std::array<
                double,
                kD233Q1AttentionProfileBoundaryCount - 1u
            > phase_ms{};
            for (const auto &chunk_events :
                 q1_attention_profile_events_) {
                for (unsigned int phase = 0u;
                     phase + 1u <
                         kD233Q1AttentionProfileBoundaryCount;
                     ++phase) {
                    float elapsed_ms = 0.0f;
                    check_hip(
                        hipEventElapsedTime(
                            &elapsed_ms,
                            chunk_events[phase],
                            chunk_events[phase + 1u]
                        ),
                        "D233 read attention subphase profile interval"
                    );
                    phase_ms[phase] +=
                        static_cast<double>(elapsed_ms);
                }
            }
            double attention_ms = 0.0;
            for (double elapsed_ms : phase_ms) {
                attention_ms += elapsed_ms;
            }
            const uint64_t call_ordinal =
                ++q1_attention_profile_call_ordinal_;
            const uint64_t run_ordinal =
                (call_ordinal - 1u) / kD232FullLayers + 1u;
            const uint64_t full_ordinal =
                (call_ordinal - 1u) % kD232FullLayers;
            std::cerr
                << "BATCH_MARK qwen36_q1024_owner_attention_profile"
                << " call_ordinal=" << call_ordinal
                << " run_ordinal=" << run_ordinal
                << " full_ordinal=" << full_ordinal
                << " layer=" << (full_ordinal * 4u + 3u)
                << " chunks=" << kD233Q1AttentionChunkCount
                << " pack_gpu_ms=" << phase_ms[0u]
                << " prefix_qk_gpu_ms=" << phase_ms[1u]
                << " tail_qk_gpu_ms=" << phase_ms[2u]
                << " softmax_probability_gpu_ms=" << phase_ms[3u]
                << " prefix_pv_gpu_ms=" << phase_ms[4u]
                << " tail_pv_gpu_ms=" << phase_ms[5u]
                << " scatter_gpu_ms=" << phase_ms[6u]
                << " attention_gpu_ms=" << attention_ms
                << " diagnostic_only=1"
                << std::endl;
        }
    }

    void launch_batched_q1_attention(
        const uint16_t *layer_k,
        const uint16_t *layer_v,
        const uint16_t *suffix_k,
        const uint16_t *suffix_v
    ) {
        if (triton_q1024_attention_) {
            launch_triton_q1024_attention(
                layer_k,
                layer_v,
                suffix_k,
                suffix_v
            );
            return;
        }
        if (grouped_q1_attention_queries_) {
            launch_grouped_q1_attention_queries(
                layer_k,
                layer_v,
                suffix_k,
                suffix_v
            );
            return;
        }
        if (q1_attention_handle_ == nullptr) {
            throw std::runtime_error(
                "D233 batched q1 attention handle is unavailable"
            );
        }
        constexpr unsigned int kHeadsPerKv =
            kD231QueryHeads / kD231KvHeads;
        const float qk_alpha =
            1.0f / sqrtf(static_cast<float>(kD231HeadDim));
        const float one = 1.0f;
        const float zero = 0.0f;
        const rocblas_stride query_stride =
            static_cast<rocblas_stride>(kD231QFeatures);
        const rocblas_stride score_query_stride =
            static_cast<rocblas_stride>(
                kD231QueryHeads * kD233Q1AttentionScoreStride
            );
        const rocblas_stride probability_head_stride =
            static_cast<rocblas_stride>(
                2u * kD233Q1AttentionScoreStride
            );
        const rocblas_stride probability_query_stride =
            static_cast<rocblas_stride>(
                2u * kD231QueryHeads *
                kD233Q1AttentionScoreStride
            );
        const rocblas_stride context_query_stride =
            static_cast<rocblas_stride>(kD231QFeatures);

        for (unsigned int query_base = 0u;
             query_base < kSuffixTokens;
             query_base += kD233Q1AttentionQueryChunk) {
            const unsigned int query_count = std::min(
                kD233Q1AttentionQueryChunk,
                kSuffixTokens - query_base
            );
            float *scores = q1_attention_scores_.get();
            const uint16_t *query =
                q_.get() +
                static_cast<size_t>(query_base) * kD231QFeatures;

            for (unsigned int kv_head = 0u;
                 kv_head < kD231KvHeads;
                 ++kv_head) {
                const size_t query_group_offset =
                    static_cast<size_t>(kv_head) *
                    kHeadsPerKv * kD231HeadDim;
                const size_t score_group_offset =
                    static_cast<size_t>(kv_head) *
                    kHeadsPerKv * kD233Q1AttentionScoreStride;
                const size_t kv_head_offset =
                    static_cast<size_t>(kv_head) * kD231HeadDim;
                check_rocblas(
                    rocblas_gemm_strided_batched_ex(
                        q1_attention_handle_,
                        rocblas_operation_transpose,
                        rocblas_operation_none,
                        static_cast<rocblas_int>(kPrefixTokens),
                        static_cast<rocblas_int>(kHeadsPerKv),
                        static_cast<rocblas_int>(kD231HeadDim),
                        &qk_alpha,
                        layer_k + kv_head_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(kD231KvFeatures),
                        0,
                        query + query_group_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(kD231HeadDim),
                        query_stride,
                        &zero,
                        scores + score_group_offset,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(
                            kD233Q1AttentionScoreStride
                        ),
                        score_query_stride,
                        scores + score_group_offset,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(
                            kD233Q1AttentionScoreStride
                        ),
                        score_query_stride,
                        static_cast<rocblas_int>(query_count),
                        rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard,
                        0,
                        rocblas_gemm_flags_none
                    ),
                    "D233 batched q1 attention prefix QK"
                );
                if (!exact_q1_attention_qk_tail_) {
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                        q1_attention_handle_,
                        rocblas_operation_transpose,
                        rocblas_operation_none,
                        static_cast<rocblas_int>(kSuffixTokens),
                        static_cast<rocblas_int>(kHeadsPerKv),
                        static_cast<rocblas_int>(kD231HeadDim),
                        &qk_alpha,
                        suffix_k + kv_head_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(kD231KvFeatures),
                        0,
                        query + query_group_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(kD231HeadDim),
                        query_stride,
                        &zero,
                        scores + score_group_offset + kPrefixTokens,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(
                            kD233Q1AttentionScoreStride
                        ),
                        score_query_stride,
                        scores + score_group_offset + kPrefixTokens,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(
                            kD233Q1AttentionScoreStride
                        ),
                        score_query_stride,
                        static_cast<rocblas_int>(query_count),
                        rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard,
                        0,
                        rocblas_gemm_flags_none
                        ),
                        "D233 batched q1 attention suffix QK"
                    );
                }
            }
            if (exact_q1_attention_qk_tail_) {
                for (unsigned int local_query = 0u;
                     local_query < query_count;
                     ++local_query) {
                    const unsigned int tail_tokens =
                        query_base + local_query + 1u;
                    const uint16_t *single_query =
                        query +
                        static_cast<size_t>(local_query) *
                            kD231QFeatures;
                    float *single_scores =
                        scores +
                        static_cast<size_t>(local_query) *
                            kD231QueryHeads *
                            kD233Q1AttentionScoreStride +
                        kPrefixTokens;
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                            q1_attention_handle_,
                            rocblas_operation_transpose,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(tail_tokens),
                            static_cast<rocblas_int>(kHeadsPerKv),
                            static_cast<rocblas_int>(kD231HeadDim),
                            &qk_alpha,
                            suffix_k,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            static_cast<rocblas_stride>(kD231HeadDim),
                            single_query,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_stride>(
                                kHeadsPerKv * kD231HeadDim
                            ),
                            &zero,
                            single_scores,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(
                                kD233Q1AttentionScoreStride
                            ),
                            static_cast<rocblas_stride>(
                                kHeadsPerKv *
                                kD233Q1AttentionScoreStride
                            ),
                            single_scores,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(
                                kD233Q1AttentionScoreStride
                            ),
                            static_cast<rocblas_stride>(
                                kHeadsPerKv *
                                kD233Q1AttentionScoreStride
                            ),
                            static_cast<rocblas_int>(kD231KvHeads),
                            rocblas_datatype_f32_r,
                            rocblas_gemm_algo_standard,
                            0,
                            rocblas_gemm_flags_none
                        ),
                        "D233 exact q1 attention suffix QK"
                    );
                }
            }

            hipLaunchKernelGGL(
                d233_q1_attention_softmax_kernel<false>,
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
            hipLaunchKernelGGL(
                d233_q1_attention_probability_bf16_kernel<false>,
                dim3(kD231QueryHeads, query_count),
                dim3(256u),
                0u,
                nullptr,
                scores,
                query_base,
                query_count
            );
            check_hip(
                hipGetLastError(),
                "D233 batched q1 attention softmax/probability"
            );

            const uint16_t *probabilities =
                reinterpret_cast<const uint16_t *>(scores);
            float *context =
                context_.get() +
                static_cast<size_t>(query_base) * kD231QFeatures;
            for (unsigned int kv_head = 0u;
                 kv_head < kD231KvHeads;
                 ++kv_head) {
                const size_t probability_group_offset =
                    static_cast<size_t>(kv_head) *
                    kHeadsPerKv * probability_head_stride;
                const size_t context_group_offset =
                    static_cast<size_t>(kv_head) *
                    kHeadsPerKv * kD231HeadDim;
                const size_t kv_head_offset =
                    static_cast<size_t>(kv_head) * kD231HeadDim;
                check_rocblas(
                    rocblas_gemm_strided_batched_ex(
                        q1_attention_handle_,
                        rocblas_operation_none,
                        rocblas_operation_none,
                        static_cast<rocblas_int>(kD231HeadDim),
                        static_cast<rocblas_int>(kHeadsPerKv),
                        static_cast<rocblas_int>(kPrefixTokens),
                        &one,
                        layer_v + kv_head_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(kD231KvFeatures),
                        0,
                        probabilities + probability_group_offset,
                        rocblas_datatype_bf16_r,
                        static_cast<rocblas_int>(
                            probability_head_stride
                        ),
                        probability_query_stride,
                        &zero,
                        context + context_group_offset,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(kD231HeadDim),
                        context_query_stride,
                        context + context_group_offset,
                        rocblas_datatype_f32_r,
                        static_cast<rocblas_int>(kD231HeadDim),
                        context_query_stride,
                        static_cast<rocblas_int>(query_count),
                        rocblas_datatype_f32_r,
                        rocblas_gemm_algo_standard,
                        0,
                        rocblas_gemm_flags_none
                    ),
                    "D233 batched q1 attention prefix PV"
                );
                if (!exact_q1_attention_pv_tail_) {
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                            q1_attention_handle_,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_int>(kHeadsPerKv),
                            static_cast<rocblas_int>(kSuffixTokens),
                            &one,
                            suffix_v + kv_head_offset,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            0,
                            probabilities +
                                probability_group_offset +
                                kPrefixTokens,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(
                                probability_head_stride
                            ),
                            probability_query_stride,
                            &one,
                            context + context_group_offset,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(kD231HeadDim),
                            context_query_stride,
                            context + context_group_offset,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(kD231HeadDim),
                            context_query_stride,
                            static_cast<rocblas_int>(query_count),
                            rocblas_datatype_f32_r,
                            rocblas_gemm_algo_standard,
                            0,
                            rocblas_gemm_flags_none
                        ),
                        "D233 batched q1 attention suffix PV"
                    );
                }
            }
            if (exact_q1_attention_pv_tail_) {
                for (unsigned int local_query = 0u;
                     local_query < query_count;
                     ++local_query) {
                    const unsigned int tail_tokens =
                        query_base + local_query + 1u;
                    const uint16_t *single_probabilities =
                        probabilities +
                        static_cast<size_t>(local_query) *
                            2u * kD231QueryHeads *
                            kD233Q1AttentionScoreStride +
                        kPrefixTokens;
                    float *single_context =
                        context +
                        static_cast<size_t>(local_query) *
                            kD231QFeatures;
                    check_rocblas(
                        rocblas_gemm_strided_batched_ex(
                            q1_attention_handle_,
                            rocblas_operation_none,
                            rocblas_operation_none,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_int>(kHeadsPerKv),
                            static_cast<rocblas_int>(tail_tokens),
                            &one,
                            suffix_v,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(kD231KvFeatures),
                            static_cast<rocblas_stride>(kD231HeadDim),
                            single_probabilities,
                            rocblas_datatype_bf16_r,
                            static_cast<rocblas_int>(
                                probability_head_stride
                            ),
                            static_cast<rocblas_stride>(
                                kHeadsPerKv *
                                probability_head_stride
                            ),
                            &one,
                            single_context,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_stride>(
                                kHeadsPerKv * kD231HeadDim
                            ),
                            single_context,
                            rocblas_datatype_f32_r,
                            static_cast<rocblas_int>(kD231HeadDim),
                            static_cast<rocblas_stride>(
                                kHeadsPerKv * kD231HeadDim
                            ),
                            static_cast<rocblas_int>(kD231KvHeads),
                            rocblas_datatype_f32_r,
                            rocblas_gemm_algo_standard,
                            0,
                            rocblas_gemm_flags_none
                        ),
                        "D233 exact q1 attention suffix PV"
                    );
                }
            }
        }
    }

    const CkApi &ck_;
    const MoeApi &moe_;
    const MoeApi &fast_moe_;
    unsigned int fast_moe_min_layer_ = kD232Layers;
    unsigned int fast_moe_max_layer_ = kD232Layers;
    uint64_t fast_moe_layer_mask_ = UINT64_C(0);
    const RocblasContext &blas_;
    const D233FullProjectionAot &projection_aot_;
    const D233Q1024AttentionAot &q1024_attention_aot_;
    bool rocblas_projections_ = false;
    unsigned int exact_projection_layer_count_ = 0u;
    DeviceBuffer<float> input_norm_f32_;
    DeviceBuffer<uint16_t> input_norm_bf16_;
    DeviceBuffer<uint16_t> qkv_projection_;
    DeviceBuffer<uint16_t> q_projection_;
    DeviceBuffer<uint16_t> k_projection_;
    DeviceBuffer<uint16_t> q_;
    DeviceBuffer<uint16_t> gate_;
    DeviceBuffer<uint16_t> full_k_;
    DeviceBuffer<uint16_t> full_v_;
    DeviceBuffer<float> context_;
    DeviceBuffer<uint16_t> gated_;
    DeviceBuffer<float> residual_;
    DeviceBuffer<float> postnorm_;
    bool triton_q1024_attention_ = false;
    bool grouped_q1_attention_queries_ = false;
    bool grouped_q1_attention_exact_pv_ = false;
    rocblas_int grouped_q1_attention_exact_pv_prefix_solution_ = 0;
    bool batched_q1_attention_ = false;
    unsigned int batched_q1_attention_full_layer_mask_ =
        UINT32_C(0x3ff);
    bool exact_q1_attention_qk_tail_ = false;
    bool exact_q1_attention_pv_tail_ = false;
    rocblas_handle q1_attention_handle_ = nullptr;
#if QRT_QWEN36_Q1024_OWNER_EXACT_TAIL_STREAMS > 1
    std::array<hipStream_t, kD233ExactTailStreams>
        exact_tail_streams_{};
    std::array<hipEvent_t, kD233ExactTailStreams>
        exact_tail_done_events_{};
    std::array<rocblas_handle, kD233ExactTailStreams>
        exact_tail_handles_{};
    hipEvent_t exact_tail_ready_event_ = nullptr;
#endif
    DeviceBuffer<float> q1_attention_scores_;
    DeviceBuffer<uint16_t> grouped_q1_queries_;
    DeviceBuffer<float> grouped_q1_context_;
    std::array<
        std::array<
            hipEvent_t,
            kD233Q1AttentionProfileBoundaryCount
        >,
        kD233Q1AttentionChunkCount
    > q1_attention_profile_events_{};
    uint64_t q1_attention_profile_call_ordinal_ = 0u;
};

class D233Owner {
public:
    D233Owner(
        const ProviderApi &gdn,
        const CkApi &ck,
        const MoeApi &moe,
        const MoeApi &fast_moe,
        unsigned int fast_moe_min_layer,
        unsigned int fast_moe_max_layer,
        const char *module_directory
    )
        : fast_moe_min_layer_(fast_moe_min_layer),
          fast_moe_max_layer_(fast_moe_max_layer),
          fast_moe_layer_mask_(d233_fast_moe_layer_mask(
              fast_moe_min_layer,
              fast_moe_max_layer
          )),
          token_ids_(kSuffixTokens),
          initial_hidden_(kD232HiddenElements),
          work_a_(kD232HiddenElements),
          work_b_(kD232HiddenElements),
          final_norm_f32_(kD232HiddenElements),
          final_norm_bf16_(kD232HiddenElements),
          logits_(kD233LogitElements),
          top1_ids_(kSuffixTokens),
          top1_logits_(kSuffixTokens),
          layer_digests_(kD233LayerDigestCount),
          token0_layer_digests_(kD233LayerDigestCount),
          layer0_stage_digests_(kD233Layer0StageDigestCount),
          blas_(),
          linear_projection_aot_(module_directory),
          linear_(
              gdn,
              moe,
              fast_moe,
              fast_moe_min_layer,
              fast_moe_max_layer,
              blas_,
              linear_projection_aot_
          ),
          full_projection_aot_(module_directory),
          q1024_attention_aot_(
              module_directory,
              d233_triton_q1024_attention_enabled()
          ),
          full_(
              ck,
              moe,
              fast_moe,
              fast_moe_min_layer,
              fast_moe_max_layer,
              blas_,
              full_projection_aot_,
              q1024_attention_aot_
          ) {}

    void run(
        const qrt_qwen36_q1024_owner_request_v1_t &request,
        qrt_qwen36_q1024_owner_result_v1_t *result
    ) {
        const auto total_start = std::chrono::steady_clock::now();
        d233_zero_device(
            token_ids_.get(),
            static_cast<size_t>(kSuffixTokens) * sizeof(uint32_t),
            "D233 reset token IDs"
        );
        d233_zero_device(
            initial_hidden_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset initial hidden"
        );
        d233_zero_device(
            work_a_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset hidden A"
        );
        d233_zero_device(
            work_b_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset hidden B"
        );
        d233_zero_device(
            final_norm_f32_.get(),
            kD232HiddenElements * sizeof(float),
            "D233 reset final norm F32"
        );
        d233_zero_device(
            final_norm_bf16_.get(),
            kD232HiddenElements * sizeof(uint16_t),
            "D233 reset final norm BF16"
        );
        d233_zero_device(
            logits_.get(),
            kD233LogitElements * sizeof(uint16_t),
            "D233 reset logits"
        );
        d233_zero_device(
            top1_ids_.get(),
            static_cast<size_t>(kSuffixTokens) * sizeof(uint32_t),
            "D233 reset top1 IDs"
        );
        d233_zero_device(
            top1_logits_.get(),
            static_cast<size_t>(kSuffixTokens) * sizeof(float),
            "D233 reset top1 logits"
        );
        linear_.reset();
        full_.reset();
        const bool emit_layer_digests = d233_layer_digests_enabled();
        const bool profile_layers = d233_profile_layers_enabled();
        std::array<hipEvent_t, kD232Layers + 1u> layer_events{};
        std::array<hipEvent_t, kD232Layers> projection_end_events{};
        std::array<hipEvent_t, kD232Layers> core_end_events{};
        std::array<hipEvent_t, kD232Layers> moe_begin_events{};
        if (profile_layers) {
            for (hipEvent_t &event : layer_events) {
                check_hip(
                    hipEventCreateWithFlags(&event, hipEventDefault),
                    "D233 create layer profile event"
                );
            }
            for (hipEvent_t &event : moe_begin_events) {
                check_hip(
                    hipEventCreateWithFlags(&event, hipEventDefault),
                    "D233 create MoE profile event"
                );
            }
            for (hipEvent_t &event : projection_end_events) {
                check_hip(
                    hipEventCreateWithFlags(&event, hipEventDefault),
                    "D233 create projection profile event"
                );
            }
            for (hipEvent_t &event : core_end_events) {
                check_hip(
                    hipEventCreateWithFlags(&event, hipEventDefault),
                    "D233 create core profile event"
                );
            }
        }
        const unsigned int stage_digest_layer =
            d233_stage_digest_layer();
        const unsigned int stage_digest_token =
            d233_stage_digest_token();
        if (emit_layer_digests) {
            d233_zero_device(
                layer_digests_.get(),
                static_cast<size_t>(kD233LayerDigestCount) *
                    sizeof(uint64_t),
                "D233 reset layer digests"
            );
            d233_zero_device(
                token0_layer_digests_.get(),
                static_cast<size_t>(kD233LayerDigestCount) *
                    sizeof(uint64_t),
                "D233 reset token zero layer digests"
            );
            d233_zero_device(
                layer0_stage_digests_.get(),
                static_cast<size_t>(kD233Layer0StageDigestCount) *
                    sizeof(uint64_t),
                "D233 reset layer zero stage digests"
            );
        }
        check_hip(
            hipMemcpyAsync(
                token_ids_.get(),
                request.suffix_token_ids,
                static_cast<size_t>(kSuffixTokens) * sizeof(uint32_t),
                hipMemcpyHostToDevice,
                nullptr
            ),
            "D233 copy suffix token IDs"
        );
        hipLaunchKernelGGL(
            d233_gather_embeddings_kernel,
            d232_linear_grid(kD232HiddenElements),
            dim3(kThreads),
            0u,
            nullptr,
            token_ids_.get(),
            request.token_embedding_weights,
            initial_hidden_.get()
        );
        result->prefix_kv_copy_bytes = full_.copy_prefix(request.layers);
        check_hip(
            hipMemcpyAsync(
                work_a_.get(),
                initial_hidden_.get(),
                kD232HiddenElements * sizeof(float),
                hipMemcpyDeviceToDevice,
                nullptr
            ),
            "D233 initialize hidden frontier"
        );
        if (emit_layer_digests) {
            d233_launch_layer_digest(
                initial_hidden_.get(),
                kD232HiddenElements,
                layer_digests_.get()
            );
            d233_launch_fnv1a64_f32_digest(
                initial_hidden_.get() +
                    static_cast<size_t>(stage_digest_token) * kLayerHidden,
                kLayerHidden,
                token0_layer_digests_.get()
            );
        }

        float *current = work_a_.get();
        float *next = work_b_.get();
        unsigned int linear_ordinal = 0u;
        unsigned int full_ordinal = 0u;
        std::cerr
            << "BATCH_MARK qwen36_q1024_owner_linear_route"
            << " owner="
            << (
                linear_.uses_exact_q1_gdn()
                    ? "exact_q1_sequential_gdn"
                    : "aiter_seeded_gdn"
            )
            << " layers=" << kD232LinearLayers
            << " exact_layer_mask=0x" << std::hex
            << linear_.exact_q1_gdn_layer_mask() << std::dec
            << " state_layout=value_head_key_value_f32"
            << " weight_bits=16"
            << " quantized=0"
            << " dflash_active=0"
            << " mtp_active=0"
            << " speculative_decode=0"
            << std::endl;
        std::cerr
            << "BATCH_MARK qwen36_q1024_owner_full_attention_route"
            << " owner="
            << (
                full_.uses_triton_q1024_attention()
                    ? "triton_q1024_grouped_bf16"
                    : (
                          full_.uses_grouped_q1_attention_queries()
                              ? "rocblas_grouped_q1_queries_bf16"
                              : (
                                    full_.uses_batched_q1_attention()
                                        ? "batched_q1_grouped_bf16"
                                        : "ck_fmha"
                                )
                      )
            )
            << " query_chunk="
            << (
                full_.uses_batched_q1_attention()
                    ? kD233Q1AttentionQueryChunk
                    : 0u
            )
            << " score_stride="
            << (
                full_.uses_batched_q1_attention()
                    ? kD233Q1AttentionScoreStride
                    : 0u
            )
            << " exact_q1_tail="
            << (full_.uses_exact_q1_attention_tail() ? 1 : 0)
            << " exact_q1_qk_tail="
            << (full_.uses_exact_q1_attention_qk_tail() ? 1 : 0)
            << " exact_q1_pv_tail="
            << (full_.uses_exact_q1_attention_pv_tail() ? 1 : 0)
            << " exact_q1_tail_streams="
            << full_.exact_q1_attention_tail_stream_count()
            << " grouped_queries="
            << (full_.uses_grouped_q1_attention_queries() ? 1 : 0)
            << " grouped_exact_pv="
            << (full_.uses_grouped_q1_attention_exact_pv() ? 1 : 0)
            << " grouped_exact_pv_prefix_solution="
            << full_.grouped_q1_attention_exact_pv_prefix_solution()
            << " grouped_wide_pv_solution="
            << kD233WidePrefixPvSolution
            << " fused_softmax_probability="
            << (kD233FusedSoftmaxProbability ? 1 : 0)
            << " softmax_register_denominator="
            << (kD233SoftmaxRegisterDenominator ? 1 : 0)
            << " batched_full_layer_mask="
            << full_.batched_q1_attention_full_layer_mask()
            << " full_layers=" << kD232FullLayers
            << " weight_bits=16"
            << " quantized=0"
            << " dflash_active=0"
            << " mtp_active=0"
            << " speculative_decode=0"
            << std::endl;
        if (linear_.uses_rocblas_projections() !=
                full_.uses_rocblas_projections() ||
            linear_.exact_projection_layer_count() !=
                full_.exact_projection_layer_count()) {
            throw std::runtime_error(
                "D233 linear/full projection routes disagree"
            );
        }
        std::cerr
            << "BATCH_MARK qwen36_q1024_owner_projection_route"
            << " owner="
            << (
                linear_.uses_rocblas_projections()
                    ? (
                          linear_.exact_projection_layer_count() == 0u
                              ? "rocblas_bf16_gemm"
                              : "exact_prefix_then_rocblas_bf16_gemm"
                      )
                    : "triton_scalar_row_exact"
            )
            << " exact_projection_layer_count="
            << linear_.exact_projection_layer_count()
            << " linear_projection_token_pack="
            << linear_projection_aot_.token_pack()
            << " full_projection_token_pack="
            << full_projection_aot_.token_pack()
            << " early_linear_owner=exact_f32"
            << " persistent_exact_q1_conv="
            << (kD233PersistentExactQ1Conv ? 1 : 0)
            << " persistent_exact_q1_gdn="
            << (kD233PersistentExactQ1Gdn ? 1 : 0)
            << " persistent_exact_q1_gdn_shared_qk="
            << (kD233PersistentExactQ1GdnSharedQk ? 1 : 0)
            << " retained_linear_layers="
            << kD232RetainedLinearLayers
            << " full_layers=" << kD232FullLayers
            << " weight_bits=16"
            << " quantized=0"
            << " dflash_active=0"
            << " mtp_active=0"
            << " speculative_decode=0"
            << std::endl;
        std::cerr
            << "BATCH_MARK qwen36_q1024_owner_moe_route"
            << " primary_layers=0-"
            << (
                fast_moe_min_layer_ == 0u
                    ? 0u
                    : fast_moe_min_layer_ - 1u
            )
            << " fast_min_layer=" << fast_moe_min_layer_
            << " fast_max_layer=" << fast_moe_max_layer_
            << " fast_layer_mask=0x" << std::hex
            << fast_moe_layer_mask_ << std::dec
            << " fast_active="
            << (
                fast_moe_layer_mask_ != UINT64_C(0)
                    ? 1
                    : 0
            )
            << " early_f32_layer=1"
            << " early_f32_primary=1"
            << " layer_count=" << kD232Layers
            << " diagnostic_only=0"
            << std::endl;
        if (profile_layers) {
            check_hip(
                hipEventRecord(layer_events[0], nullptr),
                "D233 record layer profile begin"
            );
        }
        for (unsigned int layer_index = 0u;
             layer_index < kD232Layers;
             ++layer_index) {
            const qrt_qwen36_q1024_owner_layer_v1_t &layer =
                request.layers[layer_index];
            if ((layer_index % 4u) == 3u) {
                full_.launch(
                    layer,
                    full_ordinal,
                    request.absolute_position_base,
                    current,
                    next,
                        emit_layer_digests &&
                                layer_index == stage_digest_layer
                            ? layer0_stage_digests_.get()
                            : nullptr,
                        stage_digest_token,
                        profile_layers
                            ? projection_end_events[layer_index]
                            : nullptr,
                        profile_layers
                            ? core_end_events[layer_index]
                            : nullptr,
                        profile_layers
                            ? moe_begin_events[layer_index]
                            : nullptr
                );
                ++full_ordinal;
            } else {
                if (layer_index < kD232EarlyLinearLayers) {
                    linear_.launch<float, true>(
                        layer,
                        linear_ordinal,
                        current,
                        next,
                        emit_layer_digests &&
                                layer_index == stage_digest_layer
                            ? layer0_stage_digests_.get()
                            : nullptr,
                        stage_digest_token,
                        profile_layers
                            ? projection_end_events[layer_index]
                            : nullptr,
                        profile_layers
                            ? core_end_events[layer_index]
                            : nullptr,
                        profile_layers
                            ? moe_begin_events[layer_index]
                            : nullptr
                    );
                } else {
                    linear_.launch<uint16_t, false>(
                        layer,
                        linear_ordinal,
                        current,
                        next,
                        emit_layer_digests &&
                                layer_index == stage_digest_layer
                            ? layer0_stage_digests_.get()
                            : nullptr,
                        stage_digest_token,
                        profile_layers
                            ? projection_end_events[layer_index]
                            : nullptr,
                        profile_layers
                            ? core_end_events[layer_index]
                            : nullptr,
                        profile_layers
                            ? moe_begin_events[layer_index]
                            : nullptr
                    );
                }
                ++linear_ordinal;
            }
            if (emit_layer_digests) {
                d233_launch_layer_digest(
                    next,
                    kD232HiddenElements,
                    layer_digests_.get() + layer_index + 1u
                );
                d233_launch_fnv1a64_f32_digest(
                    next +
                        static_cast<size_t>(stage_digest_token) * kLayerHidden,
                    kLayerHidden,
                    token0_layer_digests_.get() + layer_index + 1u
                );
            }
            if (profile_layers) {
                check_hip(
                    hipEventRecord(
                        layer_events[layer_index + 1u],
                        nullptr
                    ),
                    "D233 record layer profile boundary"
                );
            }
            std::swap(current, next);
        }
        if (linear_ordinal != kD232LinearLayers ||
            full_ordinal != kD232FullLayers ||
            current != work_a_.get()) {
            throw std::runtime_error(
                "D233 ordered layer accounting failed"
            );
        }
        check_hip(hipGetLastError(), "D233 ordered layer stack launch");
        check_hip(
            hipDeviceSynchronize(),
            "D233 ordered layer stack synchronize"
        );
        result->layer_stack_elapsed_ns = d233_elapsed_ns(total_start);
        if (profile_layers) {
            const uint64_t profile_ordinal = ++profile_run_ordinal_;
            double linear_ms = 0.0;
            double full_ms = 0.0;
            double pre_moe_ms = 0.0;
            double projection_ms = 0.0;
            double core_ms = 0.0;
            double output_ms = 0.0;
            double moe_ms = 0.0;
            for (unsigned int layer_index = 0u;
                 layer_index < kD232Layers;
                 ++layer_index) {
                float elapsed_ms = 0.0f;
                check_hip(
                    hipEventElapsedTime(
                        &elapsed_ms,
                        layer_events[layer_index],
                        layer_events[layer_index + 1u]
                    ),
                    "D233 read layer profile interval"
                );
                float layer_pre_moe_ms = 0.0f;
                float layer_projection_ms = 0.0f;
                float layer_core_ms = 0.0f;
                float layer_output_ms = 0.0f;
                float layer_moe_ms = 0.0f;
                check_hip(
                    hipEventElapsedTime(
                        &layer_pre_moe_ms,
                        layer_events[layer_index],
                        moe_begin_events[layer_index]
                    ),
                    "D233 read pre-MoE profile interval"
                );
                check_hip(
                    hipEventElapsedTime(
                        &layer_projection_ms,
                        layer_events[layer_index],
                        projection_end_events[layer_index]
                    ),
                    "D233 read projection profile interval"
                );
                check_hip(
                    hipEventElapsedTime(
                        &layer_core_ms,
                        projection_end_events[layer_index],
                        core_end_events[layer_index]
                    ),
                    "D233 read core profile interval"
                );
                check_hip(
                    hipEventElapsedTime(
                        &layer_output_ms,
                        core_end_events[layer_index],
                        moe_begin_events[layer_index]
                    ),
                    "D233 read output profile interval"
                );
                check_hip(
                    hipEventElapsedTime(
                        &layer_moe_ms,
                        moe_begin_events[layer_index],
                        layer_events[layer_index + 1u]
                    ),
                    "D233 read MoE profile interval"
                );
                pre_moe_ms += static_cast<double>(layer_pre_moe_ms);
                projection_ms +=
                    static_cast<double>(layer_projection_ms);
                core_ms += static_cast<double>(layer_core_ms);
                output_ms += static_cast<double>(layer_output_ms);
                moe_ms += static_cast<double>(layer_moe_ms);
                const bool full_attention =
                    (layer_index % 4u) == 3u;
                if (full_attention) {
                    full_ms += static_cast<double>(elapsed_ms);
                } else {
                    linear_ms += static_cast<double>(elapsed_ms);
                }
                std::cerr
                    << "BATCH_MARK qwen36_q1024_owner_layer_profile"
                    << " run_ordinal=" << profile_ordinal
                    << " layer=" << layer_index
                    << " kind="
                    << (full_attention ? "full" : "linear")
                    << " gpu_ms=" << elapsed_ms
                    << " pre_moe_gpu_ms=" << layer_pre_moe_ms
                    << " projection_gpu_ms=" << layer_projection_ms
                    << " core_gpu_ms=" << layer_core_ms
                    << " output_gpu_ms=" << layer_output_ms
                    << " moe_gpu_ms=" << layer_moe_ms
                    << " diagnostic_only=1"
                    << std::endl;
            }
            std::cerr
                << "BATCH_MARK qwen36_q1024_owner_layer_profile_summary"
                << " run_ordinal=" << profile_ordinal
                << " linear_layers=" << kD232LinearLayers
                << " linear_gpu_ms=" << linear_ms
                << " full_layers=" << kD232FullLayers
                << " full_gpu_ms=" << full_ms
                << " pre_moe_gpu_ms=" << pre_moe_ms
                << " projection_gpu_ms=" << projection_ms
                << " core_gpu_ms=" << core_ms
                << " output_gpu_ms=" << output_ms
                << " moe_gpu_ms=" << moe_ms
                << " total_gpu_ms=" << (linear_ms + full_ms)
                << " diagnostic_only=1"
                << std::endl;
            for (hipEvent_t &event : layer_events) {
                check_hip(
                    hipEventDestroy(event),
                    "D233 destroy layer profile event"
                );
                event = nullptr;
            }
            for (hipEvent_t &event : moe_begin_events) {
                check_hip(
                    hipEventDestroy(event),
                    "D233 destroy MoE profile event"
                );
                event = nullptr;
            }
            for (hipEvent_t &event : projection_end_events) {
                check_hip(
                    hipEventDestroy(event),
                    "D233 destroy projection profile event"
                );
                event = nullptr;
            }
            for (hipEvent_t &event : core_end_events) {
                check_hip(
                    hipEventDestroy(event),
                    "D233 destroy core profile event"
                );
                event = nullptr;
            }
        }
        if (emit_layer_digests) {
            std::array<uint64_t, kD233LayerDigestCount> host_digests{};
            check_hip(
                hipMemcpy(
                    host_digests.data(),
                    layer_digests_.get(),
                    host_digests.size() * sizeof(host_digests[0]),
                    hipMemcpyDeviceToHost
                ),
                "D233 copy layer digests"
            );
            const uint64_t run_ordinal = ++run_ordinal_;
            for (unsigned int checkpoint = 0u;
                 checkpoint < kD233LayerDigestCount;
                 ++checkpoint) {
                std::cerr
                    << "BATCH_MARK qwen36_q1024_owner_layer_digest"
                    << " run_ordinal=" << run_ordinal
                    << " checkpoint=" << checkpoint
                    << " completed_layers=" << checkpoint
                    << " digest=" << std::hex
                    << host_digests[checkpoint] << std::dec
                    << " diagnostic_only=1"
                    << std::endl;
            }
            std::array<
                uint64_t,
                kD233LayerDigestCount
            > host_token0_digests{};
            check_hip(
                hipMemcpy(
                    host_token0_digests.data(),
                    token0_layer_digests_.get(),
                    host_token0_digests.size() *
                        sizeof(host_token0_digests[0]),
                    hipMemcpyDeviceToHost
                ),
                "D233 copy token zero layer digests"
            );
            for (unsigned int checkpoint = 0u;
                 checkpoint < kD233LayerDigestCount;
                 ++checkpoint) {
                std::cerr
                    << "BATCH_MARK qwen36_q1024_owner_selected_token_layer_digest"
                    << " run_ordinal=" << run_ordinal
                    << " token=" << stage_digest_token
                    << " checkpoint=" << checkpoint
                    << " completed_layers=" << checkpoint
                    << " digest=" << std::hex
                    << host_token0_digests[checkpoint] << std::dec
                    << " algorithm=fnv1a64_f32"
                    << " diagnostic_only=1"
                    << std::endl;
            }
            std::array<
                uint64_t,
                kD233Layer0StageDigestCount
            > host_stage_digests{};
            check_hip(
                hipMemcpy(
                    host_stage_digests.data(),
                    layer0_stage_digests_.get(),
                    host_stage_digests.size() *
                        sizeof(host_stage_digests[0]),
                    hipMemcpyDeviceToHost
                ),
                "D233 copy layer zero stage digests"
            );
            constexpr const char *kLinearStageNames[
                kD233Layer0StageDigestCount
            ] = {
                "prefix_state",
                "prefix_ring",
                "input_norm",
                "qkv_projection",
                "gate",
                "seeded_gdn_core",
                "attention_residual",
                "moe_input_postnorm",
                "moe_output",
                "postconv_f32",
                "conv_f32",
                "conv_ring_before",
                "conv_weights_bf16",
                "unused_13"
            };
            constexpr const char *kFullStageNames[
                kD233Layer0StageDigestCount
            ] = {
                "residual_input_f32",
                "input_norm_f32",
                "input_norm_bf16",
                "qkv_projection_bf16",
                "q_rope_bf16",
                "gate_bf16",
                "k_rope_bf16",
                "v_bf16",
                "attention_context_f32",
                "gated_context_bf16",
                "attention_update_f32",
                "attention_residual_f32",
                "moe_input_postnorm_f32",
                "moe_output_f32"
            };
            const bool full_stage =
                (stage_digest_layer % 4u) == 3u;
            for (unsigned int stage = 0u;
                 stage < kD233Layer0StageDigestCount;
                 ++stage) {
                std::cerr
                    << "BATCH_MARK qwen36_q1024_owner_layer0_stage_digest"
                    << " run_ordinal=" << run_ordinal
                    << " layer=" << stage_digest_layer
                    << " token=" << stage_digest_token
                    << " stage=" << stage
                    << " stage_name="
                    << (
                        full_stage
                            ? kFullStageNames[stage]
                            : kLinearStageNames[stage]
                    )
                    << " digest=" << std::hex
                    << host_stage_digests[stage] << std::dec
                    << " algorithm=fnv1a64_bytes"
                    << " selected_token_only="
                    << (full_stage || stage >= 2u ? 1 : 0)
                    << " diagnostic_only=1"
                    << std::endl;
            }
        }

        const auto output_start = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            dim3(kThreads),
            0u,
            nullptr,
            current,
            request.final_norm,
            final_norm_f32_.get()
        );
        hipLaunchKernelGGL(
            f32_to_bf16_sublayer_kernel,
            d232_linear_grid(kD232HiddenElements),
            dim3(kThreads),
            0u,
            nullptr,
            final_norm_f32_.get(),
            final_norm_bf16_.get(),
            kD232HiddenElements
        );
        blas_.matmul(
            request.lm_head,
            final_norm_bf16_.get(),
            logits_.get(),
            kD233Vocab,
            kLayerHidden
        );
        hipLaunchKernelGGL(
            d233_top1_kernel,
            dim3(kSuffixTokens),
            dim3(kThreads),
            0u,
            nullptr,
            logits_.get(),
            top1_ids_.get(),
            top1_logits_.get()
        );
        check_hip(hipGetLastError(), "D233 output head launch");
        std::array<uint32_t, kSuffixTokens> host_ids{};
        std::array<float, kSuffixTokens> host_logits{};
        check_hip(
            hipMemcpy(
                host_ids.data(),
                top1_ids_.get(),
                host_ids.size() * sizeof(host_ids[0]),
                hipMemcpyDeviceToHost
            ),
            "D233 copy teacher prediction IDs"
        );
        check_hip(
            hipMemcpy(
                host_logits.data(),
                top1_logits_.get(),
                host_logits.size() * sizeof(host_logits[0]),
                hipMemcpyDeviceToHost
            ),
            "D233 copy teacher prediction logits"
        );
        result->output_head_elapsed_ns = d233_elapsed_ns(output_start);
        result->nonfinite_top1_count = 0u;
        for (size_t token = 0u; token < host_ids.size(); ++token) {
            if (host_ids[token] >= kD233Vocab ||
                !std::isfinite(host_logits[token])) {
                ++result->nonfinite_top1_count;
            }
            result->teacher_prediction_tokens[token] = host_ids[token];
        }
        result->teacher_prediction_count = kSuffixTokens;
        result->teacher_prediction_ids_fnv1a64 = fnv1a64(
            result->teacher_prediction_tokens,
            static_cast<size_t>(kSuffixTokens) *
                sizeof(result->teacher_prediction_tokens[0])
        );
        result->first_continuation_token =
            host_ids[kSuffixTokens - 1u];
        result->first_continuation_logit =
            host_logits[kSuffixTokens - 1u];
        result->linear_state_publish_bytes =
            static_cast<uint64_t>(kD232LinearLayers) *
            kD232LinearStateElements * sizeof(float);
        result->linear_ring_publish_bytes =
            static_cast<uint64_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements * sizeof(float) +
            static_cast<uint64_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements * sizeof(uint16_t);
        result->full_kv_publish_bytes =
            static_cast<uint64_t>(kD232FullLayers) * 2u *
            kD232FullSuffixKvElements * sizeof(uint16_t);
        result->total_elapsed_ns = d233_elapsed_ns(total_start);
    }

private:
    unsigned int fast_moe_min_layer_ = kD232Layers;
    unsigned int fast_moe_max_layer_ = kD232Layers;
    uint64_t fast_moe_layer_mask_ = UINT64_C(0);
    DeviceBuffer<uint32_t> token_ids_;
    DeviceBuffer<float> initial_hidden_;
    DeviceBuffer<float> work_a_;
    DeviceBuffer<float> work_b_;
    DeviceBuffer<float> final_norm_f32_;
    DeviceBuffer<uint16_t> final_norm_bf16_;
    DeviceBuffer<uint16_t> logits_;
    DeviceBuffer<uint32_t> top1_ids_;
    DeviceBuffer<float> top1_logits_;
    DeviceBuffer<uint64_t> layer_digests_;
    DeviceBuffer<uint64_t> token0_layer_digests_;
    DeviceBuffer<uint64_t> layer0_stage_digests_;
    uint64_t run_ordinal_ = 0u;
    uint64_t profile_run_ordinal_ = 0u;
    RocblasContext blas_;
    D233LinearProjectionAot linear_projection_aot_;
    D233LinearOwner linear_;
    D233FullProjectionAot full_projection_aot_;
    D233Q1024AttentionAot q1024_attention_aot_;
    D233FullOwner full_;
};

std::mutex g_d233_mutex;
std::string g_d233_last_error;
ProviderApi g_d233_gdn;
CkApi g_d233_ck;
MoeApi g_d233_moe;
MoeApi g_d233_fast_moe;
std::unique_ptr<D233Owner> g_d233_owner;
bool g_d233_prepared = false;

void d233_release_locked() {
    g_d233_owner.reset();
    if (g_d233_moe.release != nullptr) {
        g_d233_moe.release();
    }
    if (g_d233_moe.module != nullptr) {
        (void)FreeLibrary(g_d233_moe.module);
    }
    if (g_d233_fast_moe.release != nullptr) {
        g_d233_fast_moe.release();
    }
    if (g_d233_fast_moe.module != nullptr) {
        (void)FreeLibrary(g_d233_fast_moe.module);
    }
    if (g_d233_ck.release != nullptr) {
        (void)g_d233_ck.release();
    }
    if (g_d233_ck.module != nullptr) {
        (void)FreeLibrary(g_d233_ck.module);
    }
    if (g_d233_gdn.release != nullptr) {
        g_d233_gdn.release();
    }
    if (g_d233_gdn.module != nullptr) {
        (void)FreeLibrary(g_d233_gdn.module);
    }
    g_d233_gdn = ProviderApi{};
    g_d233_ck = CkApi{};
    g_d233_moe = MoeApi{};
    g_d233_fast_moe = MoeApi{};
    g_d233_prepared = false;
}

bool d233_reserved_zero(const uint64_t *values, size_t count) {
    if (values == nullptr) {
        return false;
    }
    for (size_t index = 0u; index < count; ++index) {
        if (values[index] != UINT64_C(0)) {
            return false;
        }
    }
    return true;
}

void d233_copy_text(char *destination, size_t capacity, const char *text) {
    if (destination == nullptr || capacity == 0u) {
        return;
    }
    (void)snprintf(
        destination,
        capacity,
        "%s",
        text != nullptr ? text : ""
    );
    destination[capacity - 1u] = '\0';
}

int d233_fail(
    qrt_qwen36_q1024_owner_result_v1_t *result,
    const char *stage,
    const std::string &message,
    const std::chrono::steady_clock::time_point &start
) {
    g_d233_last_error = message;
    if (result != nullptr) {
        result->status = -1;
        result->completed = 0u;
        result->total_elapsed_ns = d233_elapsed_ns(start);
        d233_copy_text(
            result->failure_stage,
            sizeof(result->failure_stage),
            stage
        );
        d233_copy_text(
            result->failure,
            sizeof(result->failure),
            message.c_str()
        );
    }
    return 0;
}

bool d233_validate_request(
    const qrt_qwen36_q1024_owner_request_v1_t &request,
    qrt_qwen36_q1024_owner_result_v1_t *result,
    std::string *failure
) {
    if (request.struct_size != sizeof(request) ||
        request.abi_version != QRT_QWEN36_Q1024_OWNER_ABI_VERSION ||
        request.flags != 0u ||
        request.batch_size != 1u ||
        request.prefix_tokens != kPrefixTokens ||
        request.suffix_tokens != kSuffixTokens ||
        request.absolute_position_base != kPrefixTokens ||
        request.layer_count != kD232Layers ||
        request.session_generation == UINT64_C(0) ||
        request.prompt_token_ids_fnv1a64 == UINT64_C(0) ||
        request.suffix_token_ids_fnv1a64 == UINT64_C(0) ||
        request.suffix_token_ids == nullptr ||
        request.token_embedding_weights == nullptr ||
        request.final_norm == nullptr ||
        request.lm_head == nullptr ||
        request.layers == nullptr ||
        !d233_reserved_zero(
            request.reserved,
            sizeof(request.reserved) / sizeof(request.reserved[0])
        )) {
        *failure =
            "D233 request ABI, fixed shape, identity, pointer, or reserved contract is invalid";
        return false;
    }
    if (fnv1a64(
            request.suffix_token_ids,
            static_cast<size_t>(kSuffixTokens) * sizeof(uint32_t)
        ) != request.suffix_token_ids_fnv1a64) {
        *failure = "D233 suffix token digest does not match the request";
        return false;
    }
    for (unsigned int token = 0u; token < kSuffixTokens; ++token) {
        if (request.suffix_token_ids[token] >= kD233Vocab) {
            *failure = "D233 request contains an out-of-vocabulary token";
            return false;
        }
    }

    uint32_t linear_count = 0u;
    uint32_t full_count = 0u;
    uint32_t fixed_count = 0u;
    uint32_t routed_count = 0u;
    for (unsigned int layer_index = 0u;
         layer_index < kD232Layers;
         ++layer_index) {
        const qrt_qwen36_q1024_owner_layer_v1_t &layer =
            request.layers[layer_index];
        const bool full_attention = (layer_index % 4u) == 3u;
        const bool common_valid =
            layer.struct_size == sizeof(layer) &&
            layer.abi_version == QRT_QWEN36_Q1024_OWNER_ABI_VERSION &&
            layer.layer_index == layer_index &&
            layer.input_norm != nullptr &&
            layer.post_attention_norm != nullptr &&
            layer.moe_router != nullptr &&
            layer.moe_expert_gate_up != nullptr &&
            layer.moe_expert_down != nullptr &&
            layer.moe_shared_gate != nullptr &&
            layer.moe_shared_gate_projection != nullptr &&
            layer.moe_shared_up_projection != nullptr &&
            layer.moe_shared_down != nullptr &&
            layer.reserved0 == 0u &&
            layer.reserved1 == 0u &&
            d233_reserved_zero(
                layer.reserved,
                sizeof(layer.reserved) / sizeof(layer.reserved[0])
            );
        if (!common_valid) {
            *failure =
                "D233 layer common weights, routed weights, identity, or reserved contract is invalid at layer " +
                std::to_string(layer_index);
            return false;
        }
        fixed_count += 7u;
        routed_count += 2u;
        if (full_attention) {
            const bool valid =
                layer.attention_kind ==
                    QRT_QWEN36_Q1024_OWNER_ATTENTION_FULL &&
                layer.qkv_ring_element_kind ==
                    QRT_QWEN36_Q1024_OWNER_ELEMENT_NONE &&
                layer.linear_gate_values_are_decay == 0u &&
                layer.full_q != nullptr &&
                layer.full_k != nullptr &&
                layer.full_v != nullptr &&
                layer.full_q_norm != nullptr &&
                layer.full_k_norm != nullptr &&
                layer.full_out != nullptr &&
                layer.full_prefix_k != nullptr &&
                layer.full_prefix_v != nullptr &&
                layer.full_suffix_k != nullptr &&
                layer.full_suffix_v != nullptr;
            if (!valid) {
                *failure =
                    "D233 full-attention descriptor is invalid at layer " +
                    std::to_string(layer_index);
                return false;
            }
            fixed_count += 6u;
            ++full_count;
        } else {
            const uint32_t expected_element_kind =
                layer_index < kD232EarlyLinearLayers
                    ? QRT_QWEN36_Q1024_OWNER_ELEMENT_F32
                    : QRT_QWEN36_Q1024_OWNER_ELEMENT_BF16;
            const uint32_t expected_decay =
                layer_index < kD232EarlyLinearLayers ? 1u : 0u;
            const bool valid =
                layer.attention_kind ==
                    QRT_QWEN36_Q1024_OWNER_ATTENTION_LINEAR &&
                layer.qkv_ring_element_kind == expected_element_kind &&
                layer.linear_gate_values_are_decay == expected_decay &&
                layer.linear_qkv != nullptr &&
                layer.linear_z != nullptr &&
                layer.linear_a != nullptr &&
                layer.linear_b != nullptr &&
                layer.linear_conv != nullptr &&
                layer.linear_a_log != nullptr &&
                layer.linear_dt_bias != nullptr &&
                layer.linear_norm != nullptr &&
                layer.linear_out != nullptr &&
                layer.linear_prefix_state != nullptr &&
                layer.linear_prefix_ring != nullptr &&
                layer.linear_final_state != nullptr &&
                layer.linear_final_ring != nullptr;
            if (!valid) {
                *failure =
                    "D233 linear-attention descriptor is invalid at layer " +
                    std::to_string(layer_index);
                return false;
            }
            fixed_count += 9u;
            ++linear_count;
        }
    }
    if (linear_count != kD232LinearLayers ||
        full_count != kD232FullLayers ||
        fixed_count != 610u ||
        routed_count != 80u) {
        *failure =
            "D233 descriptor counts do not cover the exact forty-layer real-weight graph";
        return false;
    }
    result->layer_count = kD232Layers;
    result->linear_layer_count = linear_count;
    result->full_layer_count = full_count;
    result->fixed_weight_pointer_count = fixed_count;
    result->routed_weight_pointer_count = routed_count;
    result->session_generation = request.session_generation;
    result->prompt_token_ids_fnv1a64 =
        request.prompt_token_ids_fnv1a64;
    result->suffix_token_ids_fnv1a64 =
        request.suffix_token_ids_fnv1a64;
    return true;
}

}  // namespace

QRT_Q1024_OWNER_EXPORT int QRT_Q1024_OWNER_CALL
qrt_qwen36_q1024_owner_prepare_v1(
    const qrt_qwen36_q1024_owner_prepare_v1_t *request
) {
    std::lock_guard<std::mutex> lock(g_d233_mutex);
    g_d233_last_error.clear();
    if (request == nullptr ||
        request->struct_size != sizeof(*request) ||
        request->abi_version != QRT_QWEN36_Q1024_OWNER_ABI_VERSION ||
        request->flags != 0u ||
        request->reserved0 != 0u ||
        request->gdn_kernel_dir == nullptr ||
        request->gdn_provider_dll == nullptr ||
        request->ck_provider_dll == nullptr ||
        request->moe_kernel_dir == nullptr ||
        request->moe_provider_dll == nullptr ||
        request->gdn_kernel_dir[0] == '\0' ||
        request->gdn_provider_dll[0] == '\0' ||
        request->ck_provider_dll[0] == '\0' ||
        request->moe_kernel_dir[0] == '\0' ||
        request->moe_provider_dll[0] == '\0' ||
        !d233_reserved_zero(
            request->reserved,
            sizeof(request->reserved) / sizeof(request->reserved[0])
        )) {
        g_d233_last_error =
            "D233 prepare request is null or violates its ABI/path contract";
        return 0;
    }
    if (g_d233_prepared && g_d233_owner != nullptr) {
        return 1;
    }
    d233_release_locked();
    try {
        if (!load_provider(request->gdn_provider_dll, &g_d233_gdn)) {
            throw std::runtime_error(
                "D233 could not load the GDN provider exports"
            );
        }
        if (g_d233_gdn.prepare(request->gdn_kernel_dir) == 0) {
            throw std::runtime_error(
                std::string("D233 GDN prepare failed: ") +
                g_d233_gdn.last_error()
            );
        }
        if (!load_ck_provider(request->ck_provider_dll, &g_d233_ck)) {
            throw std::runtime_error(
                "D233 could not load the CK suffix-attention exports"
            );
        }
        if (g_d233_ck.prepare() != 0) {
            throw std::runtime_error("D233 CK prepare failed");
        }
        if (!load_moe_provider(
                request->moe_provider_dll,
                &g_d233_moe
            )) {
            throw std::runtime_error(
                "D233 could not load the Q1024 MoE provider exports"
            );
        }
        if (g_d233_moe.prepare(request->moe_kernel_dir) == 0) {
            throw std::runtime_error(
                std::string("D233 MoE prepare failed: ") +
                g_d233_moe.last_error()
            );
        }
        const unsigned int fast_moe_min_layer =
            d233_fast_moe_min_layer();
        const unsigned int fast_moe_max_layer =
            d233_fast_moe_max_layer();
        if (fast_moe_min_layer < fast_moe_max_layer) {
            const std::string fast_moe_dll =
                d233_fast_moe_dll_path();
            if (fast_moe_dll.empty()) {
                throw std::runtime_error(
                    "D233 fast MoE layer route requires "
                    "QRT_QWEN36_Q1024_OWNER_FAST_MOE_DLL"
                );
            }
            if (!load_moe_provider(
                    fast_moe_dll.c_str(),
                    &g_d233_fast_moe
                )) {
                throw std::runtime_error(
                    "D233 could not load the fast Q1024 MoE provider exports"
                );
            }
            if (g_d233_fast_moe.prepare(request->moe_kernel_dir) == 0) {
                throw std::runtime_error(
                    std::string("D233 fast MoE prepare failed: ") +
                    g_d233_fast_moe.last_error()
                );
            }
        }
        g_d233_owner = std::make_unique<D233Owner>(
            g_d233_gdn,
            g_d233_ck,
            g_d233_moe,
            g_d233_fast_moe,
            fast_moe_min_layer,
            fast_moe_max_layer,
            request->moe_kernel_dir
        );
        g_d233_prepared = true;
        return 1;
    } catch (const std::exception &error) {
        g_d233_last_error = error.what();
        d233_release_locked();
        return 0;
    }
}

QRT_Q1024_OWNER_EXPORT int QRT_Q1024_OWNER_CALL
qrt_qwen36_q1024_owner_run_v1(
    const qrt_qwen36_q1024_owner_request_v1_t *request,
    qrt_qwen36_q1024_owner_result_v1_t *result
) {
    const auto start = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(g_d233_mutex);
    if (result == nullptr) {
        g_d233_last_error = "D233 result pointer is null";
        return 0;
    }
    *result = qrt_qwen36_q1024_owner_result_v1_t{};
    result->struct_size = static_cast<uint32_t>(sizeof(*result));
    result->abi_version = QRT_QWEN36_Q1024_OWNER_ABI_VERSION;
    result->status = -1;
    if (!g_d233_prepared || g_d233_owner == nullptr) {
        return d233_fail(
            result,
            "d233_owner_not_prepared",
            "D233 owner run requires a successful prepare",
            start
        );
    }
    if (request == nullptr) {
        return d233_fail(
            result,
            "d233_owner_request",
            "D233 owner request pointer is null",
            start
        );
    }
    std::string validation_failure;
    if (!d233_validate_request(*request, result, &validation_failure)) {
        return d233_fail(
            result,
            "d233_owner_descriptor_contract",
            validation_failure,
            start
        );
    }
    try {
        g_d233_owner->run(*request, result);
        if (result->teacher_prediction_count != kSuffixTokens ||
            result->teacher_prediction_ids_fnv1a64 == UINT64_C(0) ||
            result->first_continuation_token >= kD233Vocab ||
            !std::isfinite(result->first_continuation_logit) ||
            result->nonfinite_top1_count != 0u) {
            return d233_fail(
                result,
                "d233_owner_output_contract",
                "D233 output head produced an invalid teacher prediction surface",
                start
            );
        }
        result->status = 0;
        result->completed = 1u;
        result->total_elapsed_ns = d233_elapsed_ns(start);
        g_d233_last_error.clear();
        return 1;
    } catch (const std::exception &error) {
        return d233_fail(
            result,
            "d233_owner_execution",
            error.what(),
            start
        );
    }
}

QRT_Q1024_OWNER_EXPORT const char *QRT_Q1024_OWNER_CALL
qrt_qwen36_q1024_owner_last_error_v1(void) {
    std::lock_guard<std::mutex> lock(g_d233_mutex);
    return g_d233_last_error.c_str();
}

QRT_Q1024_OWNER_EXPORT void QRT_Q1024_OWNER_CALL
qrt_qwen36_q1024_owner_release_v1(void) {
    std::lock_guard<std::mutex> lock(g_d233_mutex);
    d233_release_locked();
    g_d233_last_error.clear();
}
