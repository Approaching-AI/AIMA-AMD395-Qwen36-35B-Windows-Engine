#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#if QRT_TRITON_MOE_ROCBLAS_ROUTER
#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>
#endif

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#if defined(_WIN32)
#define QRT_TRITON_MOE_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_TRITON_MOE_EXPORT extern "C"
#endif

#ifndef QRT_TRITON_MOE_BLOCK_M
#define QRT_TRITON_MOE_BLOCK_M 64
#endif
#ifndef QRT_TRITON_MOE_BLOCK_N
#define QRT_TRITON_MOE_BLOCK_N 64
#endif
#ifndef QRT_TRITON_MOE_THREADS
#define QRT_TRITON_MOE_THREADS 128
#endif
#ifndef QRT_TRITON_MOE_GATE_BLOCK_N
#define QRT_TRITON_MOE_GATE_BLOCK_N QRT_TRITON_MOE_BLOCK_N
#endif
#ifndef QRT_TRITON_MOE_DOWN_BLOCK_N
#define QRT_TRITON_MOE_DOWN_BLOCK_N QRT_TRITON_MOE_BLOCK_N
#endif
#ifndef QRT_TRITON_MOE_ROUTE_THREADS
#define QRT_TRITON_MOE_ROUTE_THREADS QRT_TRITON_MOE_THREADS
#endif
#ifndef QRT_TRITON_MOE_GATE_THREADS
#define QRT_TRITON_MOE_GATE_THREADS QRT_TRITON_MOE_THREADS
#endif
#ifndef QRT_TRITON_MOE_DOWN_THREADS
#define QRT_TRITON_MOE_DOWN_THREADS QRT_TRITON_MOE_THREADS
#endif
#ifndef QRT_TRITON_MOE_GATE_SHARED_BYTES
#define QRT_TRITON_MOE_GATE_SHARED_BYTES 8192
#endif
#ifndef QRT_TRITON_MOE_DOWN_SHARED_BYTES
#define QRT_TRITON_MOE_DOWN_SHARED_BYTES 8192
#endif
#ifndef QRT_TRITON_MOE_GROUP_M
#define QRT_TRITON_MOE_GROUP_M 8
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_ROUTED
#define QRT_TRITON_MOE_NATIVE_WMMA_ROUTED 0
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_GATE
#define QRT_TRITON_MOE_NATIVE_WMMA_GATE QRT_TRITON_MOE_NATIVE_WMMA_ROUTED
#endif
#ifndef QRT_TRITON_MOE_NATIVE_WMMA_DOWN
#define QRT_TRITON_MOE_NATIVE_WMMA_DOWN QRT_TRITON_MOE_NATIVE_WMMA_ROUTED
#endif
#ifndef QRT_TRITON_MOE_TRANSPOSED_ROUTER
#define QRT_TRITON_MOE_TRANSPOSED_ROUTER 0
#endif
#ifndef QRT_TRITON_MOE_ROUTER_THREADS
#define QRT_TRITON_MOE_ROUTER_THREADS 256
#endif
#ifndef QRT_TRITON_MOE_ROUTER_TOKEN_TILE
#define QRT_TRITON_MOE_ROUTER_TOKEN_TILE 1
#endif
#ifndef QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE
#define QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE 0
#endif
#ifndef QRT_TRITON_MOE_FUSED_COMBINE_WIDTH
#define QRT_TRITON_MOE_FUSED_COMBINE_WIDTH 1
#endif
#ifndef QRT_TRITON_MOE_TOKENS
#define QRT_TRITON_MOE_TOKENS 8192
#endif
#ifndef QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS
#define QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS 16
#endif
#ifndef QRT_TRITON_MOE_ZERO_REQUEST_SCRATCH
#define QRT_TRITON_MOE_ZERO_REQUEST_SCRATCH 0
#endif
#ifndef QRT_TRITON_MOE_ROCBLAS_ROUTER
#define QRT_TRITON_MOE_ROCBLAS_ROUTER 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_EXACT_ROUTED
#define QRT_TRITON_MOE_Q1024_EXACT_ROUTED 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED
#define QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
#define QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS
#define QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN
#define QRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
#define QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_EXACT_SHARED
#define QRT_TRITON_MOE_Q1024_EXACT_SHARED 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_EARLY_F32
#define QRT_TRITON_MOE_Q1024_EARLY_F32 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
#define QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE_ROWS
#define QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE_ROWS 64
#endif
#ifndef QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
#define QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE 0
#endif
#ifndef QRT_TRITON_MOE_Q1024_STATIC_F32_COMBINE
#define QRT_TRITON_MOE_Q1024_STATIC_F32_COMBINE 0
#endif
#ifndef QRT_TRITON_MOE_KERNEL_TOKEN_LABEL
#define QRT_TRITON_MOE_KERNEL_TOKEN_LABEL q8192
#endif
#define QRT_TRITON_MOE_STRINGIFY_INNER(value) #value
#define QRT_TRITON_MOE_STRINGIFY(value) \
    QRT_TRITON_MOE_STRINGIFY_INNER(value)
#define QRT_TRITON_MOE_KERNEL_PREFIX \
    QRT_TRITON_MOE_STRINGIFY(QRT_TRITON_MOE_KERNEL_TOKEN_LABEL) \
    "_selected_moe"
namespace {

constexpr uint32_t kTokens = QRT_TRITON_MOE_TOKENS;
constexpr uint32_t kTopK = 8;
constexpr uint32_t kRoutes = kTokens * kTopK;
constexpr uint32_t kRoutesPerSortProgram = 256;
constexpr uint32_t kRoutePrograms =
    (kRoutes + kRoutesPerSortProgram - 1u) / kRoutesPerSortProgram;
constexpr uint32_t kExperts = 256;
constexpr uint32_t kRouteScatterPrograms =
    kRoutePrograms > kExperts ? kRoutePrograms : kExperts;
constexpr uint32_t kHidden = 2048;
constexpr uint32_t kIntermediate = 512;
constexpr uint32_t kSharedGateRows = 1;
constexpr uint32_t kBlockM = QRT_TRITON_MOE_BLOCK_M;
constexpr uint32_t kGateBlockN = QRT_TRITON_MOE_GATE_BLOCK_N;
constexpr uint32_t kDownBlockN = QRT_TRITON_MOE_DOWN_BLOCK_N;
constexpr uint32_t kMaxSortedRoutes = kRoutes + kExperts * kBlockM - kTopK;
constexpr uint32_t kMaxRouteBlocks =
    (kMaxSortedRoutes + kBlockM - 1u) / kBlockM;
constexpr uint32_t kSortedExactRoutePack = 16u;
constexpr uint32_t kMaxSortedExactRouteGroups =
    (
        kMaxSortedRoutes + kSortedExactRoutePack - 1u
    ) / kSortedExactRoutePack;
constexpr uint32_t kRouteThreads = QRT_TRITON_MOE_ROUTE_THREADS;
constexpr uint32_t kGateThreads = QRT_TRITON_MOE_GATE_THREADS;
constexpr uint32_t kDownThreads = QRT_TRITON_MOE_DOWN_THREADS;
constexpr uint32_t kRouterThreads = QRT_TRITON_MOE_ROUTER_THREADS;
constexpr uint32_t kRouterTokenTile = QRT_TRITON_MOE_ROUTER_TOKEN_TILE;
constexpr uint32_t kFusedCombineWidth = QRT_TRITON_MOE_FUSED_COMBINE_WIDTH;
constexpr uint32_t kGroupM = QRT_TRITON_MOE_GROUP_M;
constexpr uint32_t kGateUpGridN = (2u * kIntermediate) / kGateBlockN;
constexpr uint32_t kDownGridN = kHidden / kDownBlockN;
constexpr uint32_t kGateSharedBytes = QRT_TRITON_MOE_GATE_SHARED_BYTES;
constexpr uint32_t kDownSharedBytes = QRT_TRITON_MOE_DOWN_SHARED_BYTES;
static_assert((2u * kIntermediate) % kGateBlockN == 0u);
static_assert(kHidden % kDownBlockN == 0u);
static_assert(kHidden % 32u == 0u);
static_assert(kExperts % 32u == 0u);
static_assert(kRouterThreads > 0u && kExperts % kRouterThreads == 0u);
static_assert(kRouterTokenTile > 0u && kTokens % kRouterTokenTile == 0u);
static_assert(kRouterTokenTile <= kRouterThreads);
static_assert(kRoutes % kRoutesPerSortProgram == 0u);
static_assert(kFusedCombineWidth > 0u && kHidden % kFusedCombineWidth == 0u);
#if !QRT_TRITON_MOE_TRANSPOSED_ROUTER
static_assert(
    kRouterThreads == kExperts && kRouterTokenTile == 1u,
    "row-major exact router requires one thread per expert and token tile 1"
);
#endif
#if QRT_TRITON_MOE_NATIVE_WMMA_GATE || QRT_TRITON_MOE_NATIVE_WMMA_DOWN
static_assert(
    kBlockM == 64u,
    "native WMMA selected-MoE kernels require QRT_TRITON_MOE_BLOCK_M=64"
);
#endif
#if QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
static_assert(
    kBlockM == 64u &&
        kBlockM % kSortedExactRoutePack == 0u &&
        kTokens == 1024u,
    "sorted packed exact MoE requires q1024, block-M 64, and route pack 16"
);
#endif
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
constexpr uint32_t kEarlyF32SortedRoutePack = 4u;
constexpr uint32_t kEarlyF32SortedPacksPerBlock =
    kBlockM / kEarlyF32SortedRoutePack;
static_assert(
    kTokens == 1024u &&
        kBlockM == 64u &&
        QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE &&
        QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE_ROWS == 64u &&
        kBlockM % kEarlyF32SortedRoutePack == 0u,
    "sorted early-F32 MoE requires q1024, block-M 64, and 64-row tiles"
);
#endif
constexpr uint32_t kNativeThreads = 256;
constexpr size_t kCountElements =
    static_cast<size_t>(kRoutePrograms + 1u) * kExperts;
constexpr size_t kInputElements = static_cast<size_t>(kTokens) * kHidden;
constexpr size_t kRouterWeightElements =
    static_cast<size_t>(kExperts) * kHidden;
constexpr size_t kActivatedElements = static_cast<size_t>(kRoutes) * kIntermediate;
constexpr size_t kRouteOutputElements = static_cast<size_t>(kRoutes) * kHidden;
constexpr size_t kOutputElements = static_cast<size_t>(kTokens) * kHidden;
constexpr size_t kSharedProjectionElements =
    static_cast<size_t>(kTokens) * kIntermediate;
constexpr size_t kMatrixWorkspaceLimit =
    static_cast<size_t>(256u) * 1024u * 1024u;
constexpr size_t kFullV3EventSlots =
    QRT_TRITON_MOE_FULL_V3_EVENT_SLOTS;
static_assert(kFullV3EventSlots > 0u);
#if QRT_TRITON_MOE_NATIVE_WMMA_GATE || QRT_TRITON_MOE_NATIVE_WMMA_DOWN
constexpr uint32_t kNativeWmmaWaveThreads = 32u;
constexpr uint32_t kNativeWmmaTile = 16u;
constexpr uint32_t kNativeWmmaThreads = 256u;
constexpr uint32_t kNativeWmmaKStage = 64u;
constexpr uint32_t kNativeWmmaSharedStride = 72u;
constexpr uint32_t kNativeWmmaGateMacroN = 64u;
constexpr uint32_t kNativeWmmaDownMacroN = 128u;
constexpr uint32_t kNativeWmmaGateGridN =
    kIntermediate / kNativeWmmaGateMacroN;
constexpr uint32_t kNativeWmmaDownGridN =
    kHidden / kNativeWmmaDownMacroN;
static_assert(kHidden % kNativeWmmaKStage == 0u);
static_assert(kIntermediate % kNativeWmmaKStage == 0u);
static_assert(kNativeWmmaKStage % kNativeWmmaTile == 0u);
static_assert(kIntermediate % kNativeWmmaGateMacroN == 0u);
static_assert(kHidden % kNativeWmmaDownMacroN == 0u);

using NativeWmmaBf16x16 =
    uint16_t __attribute__((ext_vector_type(16)));
using NativeWmmaF32x8 =
    float __attribute__((ext_vector_type(8)));
using NativeWmmaU32x4 =
    uint32_t __attribute__((ext_vector_type(4)));

struct alignas(16) NativeWmmaSharedStorage {
    int32_t routes[kBlockM];
    int32_t expert;
    int32_t padding[3];
    alignas(16) uint16_t a[kBlockM * kNativeWmmaSharedStride];
};

static_assert(sizeof(NativeWmmaSharedStorage) == 9488u);
static_assert(kGroupM > 0u);
#endif

struct ModuleKernel {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    uint32_t grid_x = 0;
    uint32_t threads = 0;
    uint32_t shared_bytes = 0;
};

struct MatrixPlan {
    hipblasLtMatmulDesc_t operation = nullptr;
    hipblasLtMatrixLayout_t weight_layout = nullptr;
    hipblasLtMatrixLayout_t input_layout = nullptr;
    hipblasLtMatrixLayout_t output_layout = nullptr;
    hipblasLtMatmulPreference_t preference = nullptr;
    hipblasLtMatmulAlgo_t algorithm{};
    size_t workspace_bytes = 0;
    uint32_t output_features = 0;
    uint32_t input_features = 0;
};

struct FullV3EventSlot {
    hipEvent_t input_ready = nullptr;
    hipEvent_t shared_done = nullptr;
    hipEvent_t caller_done = nullptr;
    bool in_flight = false;
};

struct ProviderState {
    ModuleKernel count;
    ModuleKernel prefix;
    ModuleKernel padded_prefix;
    ModuleKernel scatter;
    ModuleKernel gate_up;
    ModuleKernel down;
#if QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS
    ModuleKernel grouped_bf16_down;
#endif
#if QRT_TRITON_MOE_Q1024_EXACT_ROUTED || \
    QRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN || \
    QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
    ModuleKernel exact_gate_up;
    ModuleKernel exact_down;
#endif
#if QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED
    ModuleKernel packed_exact_gate_up;
    ModuleKernel packed_exact_down;
#endif
#if QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
    ModuleKernel sorted_packed_exact_gate_up;
    ModuleKernel sorted_packed_exact_down;
    ModuleKernel sorted_packed_exact_combine;
#endif
#if QRT_TRITON_MOE_Q1024_EXACT_SHARED
    ModuleKernel exact_shared_gate_up;
    ModuleKernel exact_shared_gate_logit;
    ModuleKernel exact_shared_down;
#endif
    uint16_t *input_bf16 = nullptr;
    uint16_t *transposed_router_weights = nullptr;
    uint16_t *router_logits_bf16 = nullptr;
#if QRT_TRITON_MOE_ROCBLAS_ROUTER
    rocblas_handle router_handle = nullptr;
#endif
    int32_t *topk_ids = nullptr;
    float *topk_weights = nullptr;
    int32_t *counts = nullptr;
    int32_t *cumsum = nullptr;
    int32_t *total_post_pad = nullptr;
    int32_t *sorted_routes = nullptr;
    int32_t *block_experts = nullptr;
    uint16_t *activated = nullptr;
    float *route_outputs = nullptr;
    float *routed_combined = nullptr;
    uint16_t *shared_gate_logits = nullptr;
    uint16_t *shared_gate_projection = nullptr;
    uint16_t *shared_up_projection = nullptr;
    uint16_t *shared_activated = nullptr;
    uint16_t *shared_down_projection = nullptr;
    float *shared_gate_scales = nullptr;
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    float *early_f32_activated = nullptr;
#endif
    hipblasLtHandle_t matrix_handle = nullptr;
    void *matrix_workspace = nullptr;
    size_t matrix_workspace_bytes = 0;
    MatrixPlan router_plan;
    MatrixPlan shared_gate_plan;
    MatrixPlan shared_projection_plan;
    MatrixPlan shared_down_plan;
    hipStream_t full_v3_shared_stream = nullptr;
    std::array<FullV3EventSlot, kFullV3EventSlots> full_v3_slots{};
    size_t full_v3_next_slot = 0u;
    hipStream_t full_v3_owner_stream = nullptr;
    bool full_v3_owner_valid = false;
    bool full_v3_poisoned = false;
    bool prepared = false;
    char kernel_dir[1024]{};
    char error[512]{};
};

ProviderState g_state;
std::atomic_flag g_full_v3_in_flight = ATOMIC_FLAG_INIT;

__device__ uint16_t float_to_bf16(float value) {
    uint32_t bits = __float_as_uint(value);
    bits += UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>(bits >> 16u);
}

__device__ float bf16_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16u);
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32
__device__ float early_f32_mul_add_separate(
    float accumulator,
    float left,
    float right
) {
    float product;
    float sum;
    asm("v_mul_f32 %0, %1, %2"
        : "=v"(product)
        : "v"(left), "v"(right));
    asm("v_add_f32 %0, %1, %2"
        : "=v"(sum)
        : "v"(accumulator), "v"(product));
    return sum;
}

__device__ float early_f32_add_separate(float left, float right) {
    float sum;
    asm("v_add_f32 %0, %1, %2"
        : "=v"(sum)
        : "v"(left), "v"(right));
    return sum;
}

__device__ float early_f32_silu_times_up(float gate, float up) {
    const float silu = static_cast<float>(
        static_cast<double>(gate) /
        (1.0 + exp(-static_cast<double>(gate)))
    );
    return silu * up;
}

__global__ void q1024_early_f32_router_topk_kernel(
    const float *post_attention,
    const uint16_t *router_weights,
    float *logits,
    int32_t *topk_ids,
    float *topk_weights
) {
    __shared__ float shared_logits[kExperts];
    const uint32_t token = blockIdx.x;
    const uint32_t expert = threadIdx.x;
    if (token >= kTokens || expert >= kExperts) {
        return;
    }

    const float *token_input =
        post_attention + static_cast<size_t>(token) * kHidden;
    const uint16_t *row_weights =
        router_weights + static_cast<size_t>(expert) * kHidden;
    float accumulator = 0.0f;
    for (uint32_t column = 0u; column < kHidden; ++column) {
        accumulator +=
            bf16_to_float(row_weights[column]) * token_input[column];
    }
    shared_logits[expert] = accumulator;
    logits[static_cast<size_t>(token) * kExperts + expert] = accumulator;
    __syncthreads();

    if (expert == 0u) {
        uint32_t best_ids[kTopK];
        double exponential_values[kTopK];
        double maximum = -1.0e300;
        double denominator = 0.0;
        const size_t route_base = static_cast<size_t>(token) * kTopK;
        for (uint32_t route = 0u; route < kTopK; ++route) {
            uint32_t best_expert = kExperts;
            for (uint32_t candidate = 0u; candidate < kExperts;
                 ++candidate) {
                bool selected = false;
                for (uint32_t prior = 0u; prior < route; ++prior) {
                    selected = selected || best_ids[prior] == candidate;
                }
                if (selected) {
                    continue;
                }
                if (best_expert == kExperts ||
                    shared_logits[candidate] >
                        shared_logits[best_expert] ||
                    (shared_logits[candidate] ==
                         shared_logits[best_expert] &&
                     candidate < best_expert)) {
                    best_expert = candidate;
                }
            }
            best_ids[route] = best_expert;
            topk_ids[route_base + route] =
                static_cast<int32_t>(best_expert);
            if (static_cast<double>(shared_logits[best_expert]) >
                maximum) {
                maximum =
                    static_cast<double>(shared_logits[best_expert]);
            }
        }
        for (uint32_t route = 0u; route < kTopK; ++route) {
            const double value = exp(
                static_cast<double>(shared_logits[best_ids[route]]) -
                maximum
            );
            exponential_values[route] = value;
            denominator += value;
        }
        for (uint32_t route = 0u; route < kTopK; ++route) {
            topk_weights[route_base + route] = static_cast<float>(
                exponential_values[route] / denominator
            );
        }
    }
}

__global__ void q1024_early_f32_routed_gate_up_kernel(
    const float *post_attention,
    const uint16_t *gate_up_weights,
    const int32_t *topk_ids,
    float *activated
) {
    const uint32_t route = blockIdx.x;
    const uint32_t intermediate =
        blockIdx.y * blockDim.x + threadIdx.x;
    if (route >= kRoutes || intermediate >= kIntermediate) {
        return;
    }
    const uint32_t token = route / kTopK;
    const uint32_t expert = static_cast<uint32_t>(topk_ids[route]);
    const float *input =
        post_attention + static_cast<size_t>(token) * kHidden;
    const uint16_t *expert_weights =
        gate_up_weights +
        static_cast<size_t>(expert) * 2u * kIntermediate * kHidden;
    const uint16_t *gate_row =
        expert_weights + static_cast<size_t>(intermediate) * kHidden;
    const uint16_t *up_row =
        expert_weights +
        static_cast<size_t>(kIntermediate + intermediate) * kHidden;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t column = 0u; column < kHidden; ++column) {
        const float input_value = input[column];
        gate = early_f32_mul_add_separate(
            gate,
            input_value,
            bf16_to_float(gate_row[column])
        );
        up = early_f32_mul_add_separate(
            up,
            input_value,
            bf16_to_float(up_row[column])
        );
    }
    activated[
        static_cast<size_t>(route) * kIntermediate + intermediate
    ] = early_f32_silu_times_up(gate, up);
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
// The exact early-F32 path must retain its left-to-right, non-fused reduction.
// Transpose each 64-row by 32-column weight tile through LDS so global reads
// are contiguous within a wave while every output thread still consumes
// columns 0..kHidden-1 in the original order.
__global__ void q1024_early_f32_routed_gate_up_transposed_tile_kernel(
    const float *post_attention,
    const uint16_t *gate_up_weights,
    const int32_t *topk_ids,
    float *activated
) {
    constexpr uint32_t kRows =
        QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE_ROWS;
    constexpr uint32_t kColumns = 32u;
    constexpr uint32_t kLdsStride = kColumns + 1u;
    constexpr uint32_t kWaves = kRows / kColumns;
    static_assert(kRows >= kColumns && kRows <= 256u);
    static_assert((kRows % kColumns) == 0u);
    __shared__ float input_tile[kColumns];
    __shared__ uint16_t gate_tile[kRows][kLdsStride];
    __shared__ uint16_t up_tile[kRows][kLdsStride];

    const uint32_t route = blockIdx.x;
    const uint32_t row_base = blockIdx.y * kRows;
    const uint32_t thread = threadIdx.x;
    const uint32_t row = row_base + thread;
    const uint32_t lane = thread & (kColumns - 1u);
    const uint32_t wave = thread / kColumns;
    if (route >= kRoutes) {
        return;
    }
    const uint32_t token = route / kTopK;
    const uint32_t expert = static_cast<uint32_t>(topk_ids[route]);
    const float *input =
        post_attention + static_cast<size_t>(token) * kHidden;
    const uint16_t *expert_weights =
        gate_up_weights +
        static_cast<size_t>(expert) * 2u * kIntermediate * kHidden;
    const uint16_t *gate_weights = expert_weights;
    const uint16_t *up_weights =
        expert_weights + static_cast<size_t>(kIntermediate) * kHidden;
    float gate = 0.0f;
    float up = 0.0f;

    for (uint32_t column_base = 0u; column_base < kHidden;
         column_base += kColumns) {
        if (thread < kColumns) {
            input_tile[thread] = input[column_base + thread];
        }
        for (uint32_t row_group = 0u; row_group < kRows / kWaves;
             ++row_group) {
            const uint32_t load_row = row_group * kWaves + wave;
            const size_t index =
                static_cast<size_t>(row_base + load_row) * kHidden +
                column_base + lane;
            gate_tile[load_row][lane] = gate_weights[index];
            up_tile[load_row][lane] = up_weights[index];
        }
        __syncthreads();
        for (uint32_t column = 0u; column < kColumns; ++column) {
            const float input_value = input_tile[column];
            gate = early_f32_mul_add_separate(
                gate,
                input_value,
                bf16_to_float(gate_tile[thread][column])
            );
            up = early_f32_mul_add_separate(
                up,
                input_value,
                bf16_to_float(up_tile[thread][column])
            );
        }
        __syncthreads();
    }
    if (row < kIntermediate) {
        activated[
            static_cast<size_t>(route) * kIntermediate + row
        ] = early_f32_silu_times_up(gate, up);
    }
}
#endif

__global__ void q1024_early_f32_routed_down_kernel(
    const float *activated,
    const uint16_t *down_weights,
    const uint32_t *topk_ids,
    const float *topk_weights,
    float *outputs,
    unsigned int route_count,
    int scale_after_down
) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= kHidden) {
        return;
    }
    float total = 0.0f;
    for (uint32_t route = 0u; route < route_count; ++route) {
        const uint32_t expert = topk_ids[route];
        const float *route_activated =
            activated + static_cast<size_t>(route) * kIntermediate;
        const uint16_t *down_row =
            down_weights +
            static_cast<size_t>(expert) * kHidden * kIntermediate +
            static_cast<size_t>(row) * kIntermediate;
        float route_total = 0.0f;
        for (uint32_t intermediate = 0u;
             intermediate < kIntermediate;
             ++intermediate) {
            route_total = early_f32_mul_add_separate(
                route_total,
                route_activated[intermediate],
                bf16_to_float(down_row[intermediate])
            );
        }
        if (scale_after_down) {
            total += static_cast<float>(
                static_cast<double>(topk_weights[route]) *
                static_cast<double>(route_total)
            );
        } else {
            total += route_total;
        }
    }
    outputs[row] = total;
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
__global__ void q1024_early_f32_routed_down_transposed_tile_kernel(
    const float *activated,
    const uint16_t *down_weights,
    const uint32_t *topk_ids,
    const float *topk_weights,
    float *outputs,
    unsigned int route_count,
    int scale_after_down
) {
    constexpr uint32_t kRows =
        QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE_ROWS;
    constexpr uint32_t kColumns = 32u;
    constexpr uint32_t kLdsStride = kColumns + 1u;
    constexpr uint32_t kWaves = kRows / kColumns;
    static_assert(kRows >= kColumns && kRows <= 256u);
    static_assert((kRows % kColumns) == 0u);
    __shared__ float activated_tile[kColumns];
    __shared__ uint16_t weight_tile[kRows][kLdsStride];

    const uint32_t token = blockIdx.y;
    const uint32_t row_base = blockIdx.x * kRows;
    const uint32_t thread = threadIdx.x;
    const uint32_t row = row_base + thread;
    const uint32_t lane = thread & (kColumns - 1u);
    const uint32_t wave = thread / kColumns;
    const float *token_activated =
        activated +
        static_cast<size_t>(token) * kTopK * kIntermediate;
    const uint32_t *token_topk_ids =
        topk_ids + static_cast<size_t>(token) * kTopK;
    const float *token_topk_weights =
        topk_weights + static_cast<size_t>(token) * kTopK;
    float *token_outputs =
        outputs + static_cast<size_t>(token) * kHidden;
    float total = 0.0f;
    for (uint32_t route = 0u; route < route_count; ++route) {
        const uint32_t expert = token_topk_ids[route];
        const float *route_activated =
            token_activated +
            static_cast<size_t>(route) * kIntermediate;
        const uint16_t *expert_weights =
            down_weights +
            static_cast<size_t>(expert) * kHidden * kIntermediate;
        float route_total = 0.0f;
        for (uint32_t column_base = 0u; column_base < kIntermediate;
             column_base += kColumns) {
            if (thread < kColumns) {
                activated_tile[thread] =
                    route_activated[column_base + thread];
            }
            for (uint32_t row_group = 0u; row_group < kRows / kWaves;
                 ++row_group) {
                const uint32_t load_row = row_group * kWaves + wave;
                const size_t index =
                    static_cast<size_t>(row_base + load_row) *
                        kIntermediate +
                    column_base + lane;
                weight_tile[load_row][lane] = expert_weights[index];
            }
            __syncthreads();
            for (uint32_t column = 0u; column < kColumns; ++column) {
                route_total = early_f32_mul_add_separate(
                    route_total,
                    activated_tile[column],
                    bf16_to_float(weight_tile[thread][column])
                );
            }
            __syncthreads();
        }
        if (scale_after_down) {
            total += static_cast<float>(
                static_cast<double>(token_topk_weights[route]) *
                static_cast<double>(route_total)
            );
        } else {
            total += route_total;
        }
    }
    if (row < kHidden) {
        token_outputs[row] = total;
    }
}
#endif

#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
// Four routes in a block share one expert and therefore one weight tile.
// Each route/row thread still consumes columns in ascending order with the
// same separate FP32 multiply/add sequence as the scalar exact path.
__global__ void q1024_early_f32_sorted_routed_gate_up_kernel(
    const float *post_attention,
    const uint16_t *gate_up_weights,
    const int32_t *sorted_route_ids,
    const int32_t *block_expert_ids,
    const int32_t *total_post_pad,
    float *activated
) {
    constexpr uint32_t kRows = 64u;
    constexpr uint32_t kColumns = 32u;
    constexpr uint32_t kLdsStride = kColumns + 1u;
    constexpr uint32_t kRoutePack = kEarlyF32SortedRoutePack;
    constexpr uint32_t kThreads = kRows * kRoutePack;
    static_assert(kThreads == 256u);
    __shared__ float input_tile[kRoutePack][kColumns];
    __shared__ uint16_t gate_tile[kRows][kLdsStride];
    __shared__ uint16_t up_tile[kRows][kLdsStride];

    const uint32_t route_program = blockIdx.x;
    const uint32_t route_block =
        route_program / kEarlyF32SortedPacksPerBlock;
    const uint32_t route_pack =
        route_program % kEarlyF32SortedPacksPerBlock;
    const uint32_t sorted_base =
        route_block * kBlockM + route_pack * kRoutePack;
    const uint32_t padded_routes =
        static_cast<uint32_t>(*total_post_pad);
    if (route_block * kBlockM >= padded_routes) {
        return;
    }

    const uint32_t thread = threadIdx.x;
    const uint32_t route_slot = thread / kRows;
    const uint32_t row_slot = thread % kRows;
    const uint32_t row = blockIdx.y * kRows + row_slot;
    const uint32_t lane = thread & (kColumns - 1u);
    const uint32_t wave = thread / kColumns;
    const int32_t route_index_value =
        sorted_route_ids[sorted_base + route_slot];
    const bool route_valid =
        route_index_value >= 0 &&
        static_cast<uint32_t>(route_index_value) < kRoutes;
    const uint32_t route_index = route_valid
        ? static_cast<uint32_t>(route_index_value)
        : 0u;
    const uint32_t expert =
        static_cast<uint32_t>(block_expert_ids[route_block]);
    const uint16_t *expert_weights =
        gate_up_weights +
        static_cast<size_t>(expert) * 2u * kIntermediate * kHidden;
    const uint16_t *gate_weights = expert_weights;
    const uint16_t *up_weights =
        expert_weights + static_cast<size_t>(kIntermediate) * kHidden;
    float gate = 0.0f;
    float up = 0.0f;

    for (uint32_t column_base = 0u; column_base < kHidden;
         column_base += kColumns) {
        if (thread < kRoutePack * kColumns) {
            const uint32_t load_route_slot = thread / kColumns;
            const uint32_t load_column = thread % kColumns;
            const int32_t load_route_value =
                sorted_route_ids[sorted_base + load_route_slot];
            const bool load_valid =
                load_route_value >= 0 &&
                static_cast<uint32_t>(load_route_value) < kRoutes;
            const uint32_t token = load_valid
                ? static_cast<uint32_t>(load_route_value) / kTopK
                : 0u;
            input_tile[load_route_slot][load_column] = load_valid
                ? post_attention[
                      static_cast<size_t>(token) * kHidden +
                      column_base + load_column
                  ]
                : 0.0f;
        }
        for (uint32_t row_group = 0u; row_group < kRows / 8u;
             ++row_group) {
            const uint32_t load_row = row_group * 8u + wave;
            const size_t index =
                static_cast<size_t>(
                    blockIdx.y * kRows + load_row
                ) * kHidden + column_base + lane;
            gate_tile[load_row][lane] = gate_weights[index];
            up_tile[load_row][lane] = up_weights[index];
        }
        __syncthreads();
        for (uint32_t column = 0u; column < kColumns; ++column) {
            const float input_value = input_tile[route_slot][column];
            gate = early_f32_mul_add_separate(
                gate,
                input_value,
                bf16_to_float(gate_tile[row_slot][column])
            );
            up = early_f32_mul_add_separate(
                up,
                input_value,
                bf16_to_float(up_tile[row_slot][column])
            );
        }
        __syncthreads();
    }
    if (route_valid) {
        activated[
            static_cast<size_t>(route_index) * kIntermediate + row
        ] = early_f32_silu_times_up(gate, up);
    }
}

__global__ void q1024_early_f32_sorted_routed_down_kernel(
    const float *activated,
    const uint16_t *down_weights,
    const int32_t *sorted_route_ids,
    const int32_t *block_expert_ids,
    const int32_t *total_post_pad,
    float *route_outputs
) {
    constexpr uint32_t kRows = 64u;
    constexpr uint32_t kColumns = 32u;
    constexpr uint32_t kLdsStride = kColumns + 1u;
    constexpr uint32_t kRoutePack = kEarlyF32SortedRoutePack;
    constexpr uint32_t kThreads = kRows * kRoutePack;
    static_assert(kThreads == 256u);
    __shared__ float activated_tile[kRoutePack][kColumns];
    __shared__ uint16_t weight_tile[kRows][kLdsStride];

    const uint32_t route_program = blockIdx.x;
    const uint32_t route_block =
        route_program / kEarlyF32SortedPacksPerBlock;
    const uint32_t route_pack =
        route_program % kEarlyF32SortedPacksPerBlock;
    const uint32_t sorted_base =
        route_block * kBlockM + route_pack * kRoutePack;
    const uint32_t padded_routes =
        static_cast<uint32_t>(*total_post_pad);
    if (route_block * kBlockM >= padded_routes) {
        return;
    }

    const uint32_t thread = threadIdx.x;
    const uint32_t route_slot = thread / kRows;
    const uint32_t row_slot = thread % kRows;
    const uint32_t row = blockIdx.y * kRows + row_slot;
    const uint32_t lane = thread & (kColumns - 1u);
    const uint32_t wave = thread / kColumns;
    const int32_t route_index_value =
        sorted_route_ids[sorted_base + route_slot];
    const bool route_valid =
        route_index_value >= 0 &&
        static_cast<uint32_t>(route_index_value) < kRoutes;
    const uint32_t route_index = route_valid
        ? static_cast<uint32_t>(route_index_value)
        : 0u;
    const uint32_t expert =
        static_cast<uint32_t>(block_expert_ids[route_block]);
    const uint16_t *expert_weights =
        down_weights +
        static_cast<size_t>(expert) * kHidden * kIntermediate;
    float route_total = 0.0f;

    for (uint32_t column_base = 0u; column_base < kIntermediate;
         column_base += kColumns) {
        if (thread < kRoutePack * kColumns) {
            const uint32_t load_route_slot = thread / kColumns;
            const uint32_t load_column = thread % kColumns;
            const int32_t load_route_value =
                sorted_route_ids[sorted_base + load_route_slot];
            const bool load_valid =
                load_route_value >= 0 &&
                static_cast<uint32_t>(load_route_value) < kRoutes;
            activated_tile[load_route_slot][load_column] = load_valid
                ? activated[
                      static_cast<size_t>(load_route_value) *
                          kIntermediate +
                      column_base + load_column
                  ]
                : 0.0f;
        }
        for (uint32_t row_group = 0u; row_group < kRows / 8u;
             ++row_group) {
            const uint32_t load_row = row_group * 8u + wave;
            const size_t index =
                static_cast<size_t>(
                    blockIdx.y * kRows + load_row
                ) * kIntermediate + column_base + lane;
            weight_tile[load_row][lane] = expert_weights[index];
        }
        __syncthreads();
        for (uint32_t column = 0u; column < kColumns; ++column) {
            route_total = early_f32_mul_add_separate(
                route_total,
                activated_tile[route_slot][column],
                bf16_to_float(weight_tile[row_slot][column])
            );
        }
        __syncthreads();
    }
    if (route_valid) {
        route_outputs[
            static_cast<size_t>(route_index) * kHidden + row
        ] = route_total;
    }
}

__global__ void q1024_early_f32_sorted_route_combine_kernel(
    const float *route_outputs,
    const float *topk_weights,
    float *outputs
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputElements) {
        return;
    }
    const uint32_t token =
        static_cast<uint32_t>(index / kHidden);
    const uint32_t row =
        static_cast<uint32_t>(index % kHidden);
    float total = 0.0f;
    for (uint32_t route = 0u; route < kTopK; ++route) {
        const uint32_t route_index = token * kTopK + route;
        const float route_total = route_outputs[
            static_cast<size_t>(route_index) * kHidden + row
        ];
        const float scaled = static_cast<float>(
            static_cast<double>(topk_weights[route_index]) *
            static_cast<double>(route_total)
        );
        total = early_f32_add_separate(total, scaled);
    }
    outputs[index] = total;
}
#endif

__global__ void q1024_early_f32_shared_gate_kernel(
    const float *selected_inputs,
    const uint16_t *gate_weight,
    float *gate_logits,
    float *gate_scales
) {
    __shared__ float partial[kNativeThreads];
    const uint32_t token = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (token >= kTokens) {
        return;
    }
    const float *input =
        selected_inputs + static_cast<size_t>(token) * kHidden;
    float sum = 0.0f;
    for (uint32_t column = lane; column < kHidden;
         column += blockDim.x) {
        sum += input[column] * bf16_to_float(gate_weight[column]);
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u;
         stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        const float gate_logit = partial[0];
        gate_logits[token] = gate_logit;
        gate_scales[token] = 1.0f / (1.0f + expf(-gate_logit));
    }
}

__global__ void q1024_early_f32_shared_gate_up_kernel(
    const float *selected_inputs,
    const uint16_t *gate_weights,
    const uint16_t *up_weights,
    float *activated
) {
    __shared__ float gate_partial[kNativeThreads];
    __shared__ float up_partial[kNativeThreads];
    const uint32_t token = blockIdx.x;
    const uint32_t intermediate = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (token >= kTokens || intermediate >= kIntermediate) {
        return;
    }
    const float *input =
        selected_inputs + static_cast<size_t>(token) * kHidden;
    const uint16_t *gate_row =
        gate_weights + static_cast<size_t>(intermediate) * kHidden;
    const uint16_t *up_row =
        up_weights + static_cast<size_t>(intermediate) * kHidden;
    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t column = lane; column < kHidden;
         column += blockDim.x) {
        const float input_value = input[column];
        gate += input_value * bf16_to_float(gate_row[column]);
        up += input_value * bf16_to_float(up_row[column]);
    }
    gate_partial[lane] = gate;
    up_partial[lane] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u;
         stride >>= 1u) {
        if (lane < stride) {
            gate_partial[lane] += gate_partial[lane + stride];
            up_partial[lane] += up_partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        const float gate_value = gate_partial[0];
        activated[
            static_cast<size_t>(token) * kIntermediate + intermediate
        ] = gate_value / (1.0f + expf(-gate_value)) * up_partial[0];
    }
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
// Pack eight independent output rows into one block. Each row keeps the same
// lane-local accumulation and 256->1 reduction tree as the scalar-grid
// kernel; only block scheduling is shared.
__global__ void q1024_early_f32_shared_gate_up_rows8_kernel(
    const float *selected_inputs,
    const uint16_t *gate_weights,
    const uint16_t *up_weights,
    float *activated
) {
    constexpr uint32_t kRows = 8u;
    __shared__ float gate_partial[kRows][kNativeThreads];
    __shared__ float up_partial[kRows][kNativeThreads];
    const uint32_t token = blockIdx.x;
    const uint32_t row_base = blockIdx.y * kRows;
    const uint32_t lane = threadIdx.x;
    if (token >= kTokens) {
        return;
    }
    const float *input =
        selected_inputs + static_cast<size_t>(token) * kHidden;
#pragma unroll
    for (uint32_t row_slot = 0u; row_slot < kRows; ++row_slot) {
        const uint32_t intermediate = row_base + row_slot;
        float gate = 0.0f;
        float up = 0.0f;
        if (intermediate < kIntermediate) {
            const uint16_t *gate_row =
                gate_weights +
                static_cast<size_t>(intermediate) * kHidden;
            const uint16_t *up_row =
                up_weights +
                static_cast<size_t>(intermediate) * kHidden;
            for (uint32_t column = lane; column < kHidden;
                 column += blockDim.x) {
                const float input_value = input[column];
                gate += input_value * bf16_to_float(gate_row[column]);
                up += input_value * bf16_to_float(up_row[column]);
            }
        }
        gate_partial[row_slot][lane] = gate;
        up_partial[row_slot][lane] = up;
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u;
         stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t row_slot = 0u; row_slot < kRows; ++row_slot) {
                gate_partial[row_slot][lane] +=
                    gate_partial[row_slot][lane + stride];
                up_partial[row_slot][lane] +=
                    up_partial[row_slot][lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t row_slot = 0u; row_slot < kRows; ++row_slot) {
            const uint32_t intermediate = row_base + row_slot;
            if (intermediate < kIntermediate) {
                const float gate_value = gate_partial[row_slot][0];
                activated[
                    static_cast<size_t>(token) * kIntermediate +
                    intermediate
                ] = gate_value / (1.0f + expf(-gate_value)) *
                    up_partial[row_slot][0];
            }
        }
    }
}
#endif

__global__ void q1024_early_f32_shared_down_kernel(
    const float *activated,
    const float *gate_scales,
    const uint16_t *down_weights,
    const float *routed_inputs,
    float *shared_outputs,
    float *combined_outputs
) {
    __shared__ float partial[kNativeThreads];
    const uint32_t token = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (token >= kTokens || row >= kHidden) {
        return;
    }
    const float *token_activated =
        activated + static_cast<size_t>(token) * kIntermediate;
    const uint16_t *down_row =
        down_weights + static_cast<size_t>(row) * kIntermediate;
    float sum = 0.0f;
    for (uint32_t intermediate = lane; intermediate < kIntermediate;
         intermediate += blockDim.x) {
        sum += token_activated[intermediate] *
            bf16_to_float(down_row[intermediate]);
    }
    partial[lane] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u;
         stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        const size_t index =
            static_cast<size_t>(token) * kHidden + row;
        const float shared = gate_scales[token] * partial[0];
        shared_outputs[index] = shared;
        combined_outputs[index] = routed_inputs[index] + shared;
    }
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
__global__ void q1024_early_f32_shared_down_rows8_kernel(
    const float *activated,
    const float *gate_scales,
    const uint16_t *down_weights,
    const float *routed_inputs,
    float *shared_outputs,
    float *combined_outputs
) {
    constexpr uint32_t kRows = 8u;
    __shared__ float partial[kRows][kNativeThreads];
    const uint32_t token = blockIdx.x;
    const uint32_t row_base = blockIdx.y * kRows;
    const uint32_t lane = threadIdx.x;
    if (token >= kTokens) {
        return;
    }
    const float *token_activated =
        activated + static_cast<size_t>(token) * kIntermediate;
#pragma unroll
    for (uint32_t row_slot = 0u; row_slot < kRows; ++row_slot) {
        const uint32_t row = row_base + row_slot;
        float sum = 0.0f;
        if (row < kHidden) {
            const uint16_t *down_row =
                down_weights + static_cast<size_t>(row) * kIntermediate;
            for (uint32_t intermediate = lane;
                 intermediate < kIntermediate;
                 intermediate += blockDim.x) {
                sum += token_activated[intermediate] *
                    bf16_to_float(down_row[intermediate]);
            }
        }
        partial[row_slot][lane] = sum;
    }
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2u; stride > 0u;
         stride >>= 1u) {
        if (lane < stride) {
#pragma unroll
            for (uint32_t row_slot = 0u; row_slot < kRows; ++row_slot) {
                partial[row_slot][lane] +=
                    partial[row_slot][lane + stride];
            }
        }
        __syncthreads();
    }
    if (lane == 0u) {
#pragma unroll
        for (uint32_t row_slot = 0u; row_slot < kRows; ++row_slot) {
            const uint32_t row = row_base + row_slot;
            if (row < kHidden) {
                const size_t index =
                    static_cast<size_t>(token) * kHidden + row;
                const float shared =
                    gate_scales[token] * partial[row_slot][0];
                shared_outputs[index] = shared;
                combined_outputs[index] = routed_inputs[index] + shared;
            }
        }
    }
}
#endif

__global__ void q1024_early_f32_output_residual_kernel(
    const float *residual_inputs,
    const float *moe_updates,
    float *outputs
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < kOutputElements) {
        outputs[index] = residual_inputs[index] + moe_updates[index];
    }
}
#endif

#if QRT_TRITON_MOE_NATIVE_WMMA_GATE || QRT_TRITON_MOE_NATIVE_WMMA_DOWN
__device__ __forceinline__ NativeWmmaBf16x16
load_native_wmma_fragment(const uint16_t *source) {
    NativeWmmaBf16x16 fragment;
#pragma unroll
    for (uint32_t element = 0u; element < kNativeWmmaTile; ++element) {
        fragment[element] = source[element];
    }
    return fragment;
}

__device__ __forceinline__ bool native_wmma_grouped_program(
    uint32_t program,
    uint32_t padded_routes,
    uint32_t num_programs_n,
    uint32_t *program_m,
    uint32_t *program_n
) {
    const uint32_t num_programs_m =
        (padded_routes + kBlockM - 1u) / kBlockM;
    if (program >= num_programs_m * num_programs_n) {
        return false;
    }
    const uint32_t programs_per_group = kGroupM * num_programs_n;
    const uint32_t group = program / programs_per_group;
    const uint32_t first_m = group * kGroupM;
    const uint32_t remaining_m = num_programs_m - first_m;
    const uint32_t group_m = remaining_m < kGroupM ? remaining_m : kGroupM;
    const uint32_t program_in_group = program % programs_per_group;
    *program_m = first_m + program_in_group % group_m;
    *program_n = program_in_group / group_m;
    return true;
}
#endif

__global__ void convert_input_kernel(
    const float *input,
    uint16_t *output,
    size_t elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = float_to_bf16(input[index]);
    }
}

__global__ void router_bf16_logits_topk_kernel(
    const uint16_t *logits_bf16,
    int32_t *topk_ids,
    float *topk_weights
) {
    __shared__ float shared_logits[kExperts];
    const uint32_t token = blockIdx.x;
    const uint32_t expert = threadIdx.x;
    if (token >= kTokens || expert >= kExperts) {
        return;
    }
    const size_t logit_base = static_cast<size_t>(token) * kExperts;
    const size_t route_base = static_cast<size_t>(token) * kTopK;
    shared_logits[expert] = bf16_to_float(
        logits_bf16[logit_base + expert]
    );
    __syncthreads();

    if (expert == 0u) {
        uint32_t best_ids[kTopK];
        double exponential_values[kTopK];
        double maximum = -1.0e300;
        double denominator = 0.0;
        for (uint32_t route = 0u; route < kTopK; ++route) {
            uint32_t best_expert = kExperts;
            for (uint32_t candidate = 0u; candidate < kExperts;
                 ++candidate) {
                bool selected = false;
                for (uint32_t prior = 0u; prior < route; ++prior) {
                    selected = selected || best_ids[prior] == candidate;
                }
                if (selected) {
                    continue;
                }
                if (best_expert == kExperts ||
                    shared_logits[candidate] >
                        shared_logits[best_expert] ||
                    (shared_logits[candidate] ==
                         shared_logits[best_expert] &&
                     candidate < best_expert)) {
                    best_expert = candidate;
                }
            }
            best_ids[route] = best_expert;
            topk_ids[route_base + route] =
                static_cast<int32_t>(best_expert);
            maximum = fmax(
                maximum,
                static_cast<double>(shared_logits[best_expert])
            );
        }
        for (uint32_t route = 0u; route < kTopK; ++route) {
            const double value = exp(
                static_cast<double>(
                    shared_logits[best_ids[route]]
                ) - maximum
            );
            exponential_values[route] = value;
            denominator += value;
        }
        for (uint32_t route = 0u; route < kTopK; ++route) {
            topk_weights[route_base + route] = static_cast<float>(
                exponential_values[route] / denominator
            );
        }
    }
}

#if QRT_TRITON_MOE_NATIVE_WMMA_GATE || QRT_TRITON_MOE_NATIVE_WMMA_DOWN
__global__ void native_wmma_gate_up_silu_kernel(
    const uint16_t *post_attention_bf16,
    const uint16_t *gate_up_bf16,
    const int32_t *sorted_route_ids,
    const int32_t *block_expert_ids,
    const int32_t *total_post_pad,
    uint16_t *activated_bf16
) {
    __shared__ NativeWmmaSharedStorage shared;

    const uint32_t padded_routes =
        static_cast<uint32_t>(*total_post_pad);
    uint32_t route_block = 0u;
    uint32_t inter_macro = 0u;
    if (!native_wmma_grouped_program(
            blockIdx.x,
            padded_routes,
            kNativeWmmaGateGridN,
            &route_block,
            &inter_macro
        )) {
        return;
    }

    const uint32_t thread = threadIdx.x;
    if (thread < kBlockM) {
        shared.routes[thread] =
            sorted_route_ids[route_block * kBlockM + thread];
    }
    if (thread == 0u) {
        shared.expert = block_expert_ids[route_block];
    }
    __syncthreads();

    const uint32_t wave = threadIdx.x / kNativeWmmaWaveThreads;
    const uint32_t lane = threadIdx.x % kNativeWmmaWaveThreads;
    const uint32_t source_index = lane % kNativeWmmaTile;
    const uint32_t output_row_segment = lane / kNativeWmmaTile;
    const uint32_t m_group = wave & 1u;
    const uint32_t inter_group = wave >> 1u;
    const uint32_t m_base = m_group * (2u * kNativeWmmaTile);
    const uint32_t inter =
        inter_macro * kNativeWmmaGateMacroN +
        inter_group * kNativeWmmaTile + source_index;

    NativeWmmaF32x8 gate_accumulator_m0{};
    NativeWmmaF32x8 gate_accumulator_m1{};
    NativeWmmaF32x8 up_accumulator_m0{};
    NativeWmmaF32x8 up_accumulator_m1{};
#pragma unroll 1
    for (uint32_t k_base = 0u; k_base < kHidden;
         k_base += kNativeWmmaKStage) {
        constexpr uint32_t kChunksPerRow =
            kNativeWmmaKStage * sizeof(uint16_t) /
            sizeof(NativeWmmaU32x4);
        constexpr uint32_t kAChunks = kBlockM * kChunksPerRow;

        for (uint32_t chunk = thread; chunk < kAChunks;
             chunk += kNativeWmmaThreads) {
            const uint32_t row = chunk / kChunksPerRow;
            const uint32_t row_chunk = chunk % kChunksPerRow;
            const int32_t route = shared.routes[row];
            NativeWmmaU32x4 value{};
            if (route >= 0 && route < static_cast<int32_t>(kRoutes)) {
                const size_t source =
                    static_cast<size_t>(route / static_cast<int32_t>(kTopK)) *
                        kHidden +
                    k_base + row_chunk * 8u;
                value = *reinterpret_cast<const NativeWmmaU32x4 *>(
                    post_attention_bf16 + source
                );
            }
            *reinterpret_cast<NativeWmmaU32x4 *>(
                shared.a + row * kNativeWmmaSharedStride + row_chunk * 8u
            ) = value;
        }

        __syncthreads();

#pragma unroll
        for (uint32_t k_sub = 0u; k_sub < kNativeWmmaKStage;
             k_sub += kNativeWmmaTile) {
            const NativeWmmaBf16x16 input_fragment_m0 =
                load_native_wmma_fragment(
                    shared.a +
                    (m_base + source_index) * kNativeWmmaSharedStride +
                    k_sub
                );
            const NativeWmmaBf16x16 input_fragment_m1 =
                load_native_wmma_fragment(
                    shared.a +
                    (m_base + kNativeWmmaTile + source_index) *
                        kNativeWmmaSharedStride +
                    k_sub
                );
            NativeWmmaBf16x16 weight_fragment =
                load_native_wmma_fragment(
                    gate_up_bf16 +
                    (static_cast<size_t>(shared.expert) *
                         (2u * kIntermediate) +
                     inter) *
                        kHidden +
                    k_base + k_sub
                );
            gate_accumulator_m0 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m0,
                    weight_fragment,
                    gate_accumulator_m0
                );
            gate_accumulator_m1 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m1,
                    weight_fragment,
                    gate_accumulator_m1
                );
            weight_fragment = load_native_wmma_fragment(
                gate_up_bf16 +
                (static_cast<size_t>(shared.expert) *
                     (2u * kIntermediate) +
                 kIntermediate + inter) *
                    kHidden +
                k_base + k_sub
            );
            up_accumulator_m0 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m0,
                    weight_fragment,
                    up_accumulator_m0
                );
            up_accumulator_m1 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m1,
                    weight_fragment,
                    up_accumulator_m1
                );
        }
        if (k_base + kNativeWmmaKStage < kHidden) {
            __syncthreads();
        }
    }

#pragma unroll
    for (uint32_t output_element = 0u; output_element < 8u;
         ++output_element) {
        const uint32_t row_in_tile =
            2u * output_element + output_row_segment;
        const uint32_t output_rows[2] = {
            m_base + row_in_tile,
            m_base + kNativeWmmaTile + row_in_tile,
        };
        const NativeWmmaF32x8 gate_accumulators[2] = {
            gate_accumulator_m0,
            gate_accumulator_m1,
        };
        const NativeWmmaF32x8 up_accumulators[2] = {
            up_accumulator_m0,
            up_accumulator_m1,
        };
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) {
            const int32_t output_route = shared.routes[output_rows[m]];
            if (output_route >= 0 &&
                output_route < static_cast<int32_t>(kRoutes)) {
                const float gate = bf16_to_float(
                    float_to_bf16(gate_accumulators[m][output_element])
                );
                const float up = bf16_to_float(
                    float_to_bf16(up_accumulators[m][output_element])
                );
                const float exponent = -(gate * 1.44269504089f);
                const float silu = gate /
                    (1.0f + __builtin_amdgcn_exp2f(exponent));
                activated_bf16[
                    static_cast<size_t>(output_route) * kIntermediate + inter
                ] = float_to_bf16(silu * up);
            }
        }
    }
}

__global__ void native_wmma_down_kernel(
    const uint16_t *activated_bf16,
    const uint16_t *down_bf16,
    const int32_t *sorted_route_ids,
    const int32_t *block_expert_ids,
    const int32_t *total_post_pad,
    float *route_outputs_f32
) {
    __shared__ NativeWmmaSharedStorage shared;

    const uint32_t padded_routes =
        static_cast<uint32_t>(*total_post_pad);
    uint32_t route_block = 0u;
    uint32_t output_macro = 0u;
    if (!native_wmma_grouped_program(
            blockIdx.x,
            padded_routes,
            kNativeWmmaDownGridN,
            &route_block,
            &output_macro
        )) {
        return;
    }

    const uint32_t thread = threadIdx.x;
    if (thread < kBlockM) {
        shared.routes[thread] =
            sorted_route_ids[route_block * kBlockM + thread];
    }
    if (thread == 0u) {
        shared.expert = block_expert_ids[route_block];
    }
    __syncthreads();

    const uint32_t wave = threadIdx.x / kNativeWmmaWaveThreads;
    const uint32_t lane = threadIdx.x % kNativeWmmaWaveThreads;
    const uint32_t source_index = lane % kNativeWmmaTile;
    const uint32_t output_row_segment = lane / kNativeWmmaTile;
    const uint32_t m_group = wave & 1u;
    const uint32_t n_group = wave >> 1u;
    const uint32_t m_base = m_group * (2u * kNativeWmmaTile);
    const uint32_t output_base =
        output_macro * kNativeWmmaDownMacroN +
        n_group * (2u * kNativeWmmaTile);

    NativeWmmaF32x8 accumulator_m0_n0{};
    NativeWmmaF32x8 accumulator_m0_n1{};
    NativeWmmaF32x8 accumulator_m1_n0{};
    NativeWmmaF32x8 accumulator_m1_n1{};
#pragma unroll 1
    for (uint32_t k_base = 0u; k_base < kIntermediate;
         k_base += kNativeWmmaKStage) {
        constexpr uint32_t kChunksPerRow =
            kNativeWmmaKStage * sizeof(uint16_t) /
            sizeof(NativeWmmaU32x4);
        constexpr uint32_t kAChunks = kBlockM * kChunksPerRow;

        for (uint32_t chunk = thread; chunk < kAChunks;
             chunk += kNativeWmmaThreads) {
            const uint32_t row = chunk / kChunksPerRow;
            const uint32_t row_chunk = chunk % kChunksPerRow;
            const int32_t route = shared.routes[row];
            NativeWmmaU32x4 value{};
            if (route >= 0 && route < static_cast<int32_t>(kRoutes)) {
                const size_t source =
                    static_cast<size_t>(route) * kIntermediate + k_base +
                    row_chunk * 8u;
                value = *reinterpret_cast<const NativeWmmaU32x4 *>(
                    activated_bf16 + source
                );
            }
            *reinterpret_cast<NativeWmmaU32x4 *>(
                shared.a + row * kNativeWmmaSharedStride + row_chunk * 8u
            ) = value;
        }

        __syncthreads();

#pragma unroll
        for (uint32_t k_sub = 0u; k_sub < kNativeWmmaKStage;
             k_sub += kNativeWmmaTile) {
            const NativeWmmaBf16x16 input_fragment_m0 =
                load_native_wmma_fragment(
                    shared.a +
                    (m_base + source_index) * kNativeWmmaSharedStride +
                    k_sub
                );
            const NativeWmmaBf16x16 input_fragment_m1 =
                load_native_wmma_fragment(
                    shared.a +
                    (m_base + kNativeWmmaTile + source_index) *
                        kNativeWmmaSharedStride +
                    k_sub
                );
            NativeWmmaBf16x16 weight_fragment =
                load_native_wmma_fragment(
                    down_bf16 +
                    (static_cast<size_t>(shared.expert) * kHidden +
                     output_base + source_index) *
                        kIntermediate +
                    k_base + k_sub
                );
            accumulator_m0_n0 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m0,
                    weight_fragment,
                    accumulator_m0_n0
                );
            accumulator_m1_n0 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m1,
                    weight_fragment,
                    accumulator_m1_n0
                );
            weight_fragment = load_native_wmma_fragment(
                down_bf16 +
                (static_cast<size_t>(shared.expert) * kHidden +
                 output_base + kNativeWmmaTile + source_index) *
                    kIntermediate +
                k_base + k_sub
            );
            accumulator_m0_n1 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m0,
                    weight_fragment,
                    accumulator_m0_n1
                );
            accumulator_m1_n1 =
                __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    input_fragment_m1,
                    weight_fragment,
                    accumulator_m1_n1
                );
        }
        if (k_base + kNativeWmmaKStage < kIntermediate) {
            __syncthreads();
        }
    }

#pragma unroll
    for (uint32_t output_element = 0u; output_element < 8u;
         ++output_element) {
        const uint32_t row_in_tile =
            2u * output_element + output_row_segment;
        const uint32_t output_rows[2] = {
            m_base + row_in_tile,
            m_base + kNativeWmmaTile + row_in_tile,
        };
        const NativeWmmaF32x8 accumulators_n0[2] = {
            accumulator_m0_n0,
            accumulator_m1_n0,
        };
        const NativeWmmaF32x8 accumulators_n1[2] = {
            accumulator_m0_n1,
            accumulator_m1_n1,
        };
#pragma unroll
        for (uint32_t m = 0u; m < 2u; ++m) {
            const int32_t output_route = shared.routes[output_rows[m]];
            if (output_route >= 0 &&
                output_route < static_cast<int32_t>(kRoutes)) {
                route_outputs_f32[
                    static_cast<size_t>(output_route) * kHidden +
                    output_base + source_index
                ] = accumulators_n0[m][output_element];
                route_outputs_f32[
                    static_cast<size_t>(output_route) * kHidden +
                    output_base + kNativeWmmaTile + source_index
                ] = accumulators_n1[m][output_element];
            }
        }
    }
}
#endif

__global__ void combine_route_order_kernel(
    const float *route_outputs,
    const float *topk_weights,
    float *outputs,
    size_t output_elements
) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= output_elements) {
        return;
    }
    const size_t token = index / kHidden;
    const size_t column = index - token * kHidden;
    const size_t route_base = token * kTopK;
    float value = 0.0f;
    for (uint32_t route_order = 0; route_order < kTopK; ++route_order) {
        const size_t route = route_base + route_order;
        const float contribution =
            topk_weights[route] *
            route_outputs[route * kHidden + column];
        value = __fadd_rn(value, contribution);
    }
#if QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS
    outputs[index] = bf16_to_float(float_to_bf16(value));
#else
    outputs[index] = value;
#endif
}

#if QRT_TRITON_MOE_TRANSPOSED_ROUTER
__global__ void transpose_router_weights_kernel(
    const uint16_t *router_weights,
    uint16_t *transposed_router_weights
) {
    __shared__ uint16_t tile[32][33];

    const uint32_t lane_x = threadIdx.x;
    const uint32_t lane_y = threadIdx.y;
    const uint32_t source_column = blockIdx.x * 32u + lane_x;
    const uint32_t source_expert_base = blockIdx.y * 32u + lane_y;
#pragma unroll
    for (uint32_t row_offset = 0u; row_offset < 32u;
         row_offset += 8u) {
        tile[lane_y + row_offset][lane_x] = router_weights[
            static_cast<size_t>(source_expert_base + row_offset) * kHidden +
            source_column
        ];
    }
    __syncthreads();

    const uint32_t output_expert = blockIdx.y * 32u + lane_x;
    const uint32_t output_column_base = blockIdx.x * 32u + lane_y;
#pragma unroll
    for (uint32_t column_offset = 0u; column_offset < 32u;
         column_offset += 8u) {
        transposed_router_weights[
            static_cast<size_t>(output_column_base + column_offset) *
                kExperts +
            output_expert
        ] = tile[lane_x][lane_y + column_offset];
    }
}
#endif

__global__ void router_topk_kernel(
    const float *post_attention,
    const uint16_t *router_weights,
    int32_t *topk_ids,
    float *topk_weights,
    bool bf16_logit_endpoint
) {
    __shared__ float shared_logits[kRouterTokenTile][kExperts];

    const uint32_t token_base = blockIdx.x * kRouterTokenTile;
    const uint32_t worker = threadIdx.x;
    if (token_base >= kTokens || worker >= kRouterThreads) {
        return;
    }
#if QRT_TRITON_MOE_TRANSPOSED_ROUTER
    constexpr uint32_t kExpertsPerThread = kExperts / kRouterThreads;
    float accumulators[kRouterTokenTile][kExpertsPerThread]{};
    for (uint32_t column = 0; column < kHidden; ++column) {
        float inputs[kRouterTokenTile];
#pragma unroll
        for (uint32_t token_slot = 0u; token_slot < kRouterTokenTile;
             ++token_slot) {
            // vLLM's router projection consumes the BF16 RMSNorm output.
            // The provider retains that surface as F32 for the surrounding
            // residual path, so restore the model-visible BF16 endpoint here
            // before accumulating router logits.  Rounding only the completed
            // logit is too late: near-tied experts can otherwise cross the
            // top-k boundary.
            inputs[token_slot] = bf16_to_float(float_to_bf16(
                post_attention[
                    static_cast<size_t>(token_base + token_slot) * kHidden +
                    column
                ]
            ));
        }
#pragma unroll
        for (uint32_t slot = 0u; slot < kExpertsPerThread; ++slot) {
            const uint32_t expert = worker + slot * kRouterThreads;
            const uint16_t weight = router_weights[
                static_cast<size_t>(column) * kExperts + expert
            ];
            const float weight_f32 = bf16_to_float(weight);
#pragma unroll
            for (uint32_t token_slot = 0u;
                 token_slot < kRouterTokenTile;
                 ++token_slot) {
                accumulators[token_slot][slot] +=
                    weight_f32 * inputs[token_slot];
            }
        }
    }
#pragma unroll
    for (uint32_t slot = 0u; slot < kExpertsPerThread; ++slot) {
        const uint32_t expert = worker + slot * kRouterThreads;
#pragma unroll
        for (uint32_t token_slot = 0u; token_slot < kRouterTokenTile;
             ++token_slot) {
            const float logit = accumulators[token_slot][slot];
            shared_logits[token_slot][expert] = bf16_logit_endpoint
                ? bf16_to_float(float_to_bf16(logit))
                : logit;
        }
    }
#else
    const uint32_t token = token_base;
    const float *token_input =
        post_attention + static_cast<size_t>(token) * kHidden;
    const uint32_t expert = worker;
    const uint16_t *row_weights =
        router_weights + static_cast<size_t>(expert) * kHidden;
    float accumulator = 0.0f;
    for (uint32_t column = 0; column < kHidden; ++column) {
        const uint16_t weight = row_weights[column];
        const float input_value = bf16_to_float(float_to_bf16(
            token_input[column]
        ));
        accumulator +=
            bf16_to_float(weight) * input_value;
    }
    shared_logits[0][expert] = bf16_logit_endpoint
        ? bf16_to_float(float_to_bf16(accumulator))
        : accumulator;
#endif
    __syncthreads();

#if QRT_TRITON_MOE_TRANSPOSED_ROUTER
    if (worker < kRouterTokenTile) {
        const uint32_t token_slot = worker;
#else
    if (worker == 0u) {
        constexpr uint32_t token_slot = 0u;
#endif
        const uint32_t token = token_base + token_slot;
        uint32_t best_ids[kTopK];
        double exponential_values[kTopK];
        double maximum = -1.0e300;
        double denominator = 0.0;
        for (uint32_t route = 0; route < kTopK; ++route) {
            uint32_t best_expert = kExperts;
            for (uint32_t candidate = 0; candidate < kExperts;
                 ++candidate) {
                bool selected = false;
                for (uint32_t prior = 0; prior < route; ++prior) {
                    selected = selected || best_ids[prior] == candidate;
                }
                if (selected) {
                    continue;
                }
                if (best_expert == kExperts ||
                    shared_logits[token_slot][candidate] >
                        shared_logits[token_slot][best_expert] ||
                    (shared_logits[token_slot][candidate] ==
                         shared_logits[token_slot][best_expert] &&
                     candidate < best_expert)) {
                    best_expert = candidate;
                }
            }
            best_ids[route] = best_expert;
            topk_ids[static_cast<size_t>(token) * kTopK + route] =
                static_cast<int32_t>(best_expert);
            maximum = fmax(
                maximum,
                static_cast<double>(shared_logits[token_slot][best_expert])
            );
        }
        for (uint32_t route = 0; route < kTopK; ++route) {
            const double value = exp(
                static_cast<double>(
                    shared_logits[token_slot][best_ids[route]]
                ) - maximum
            );
            exponential_values[route] = value;
            denominator += value;
        }
        for (uint32_t route = 0; route < kTopK; ++route) {
            topk_weights[static_cast<size_t>(token) * kTopK + route] =
                static_cast<float>(exponential_values[route] / denominator);
        }
    }
}

__global__ void shared_gate_scale_kernel(
    const uint16_t *gate_logits,
    float *gate_scales
) {
    const size_t token =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= kTokens) {
        return;
    }
    const float gate = bf16_to_float(gate_logits[token]);
    gate_scales[token] = 1.0f / (1.0f + expf(-gate));
}

__global__ void shared_activation_kernel(
    const uint16_t *gate_projection,
    const uint16_t *up_projection,
    uint16_t *activated
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kSharedProjectionElements) {
        return;
    }
    const float gate = bf16_to_float(gate_projection[index]);
    const float up = bf16_to_float(up_projection[index]);
    const float silu = gate / (1.0f + expf(-gate));
    activated[index] = float_to_bf16(silu * up);
}

__global__ void shared_combine_residual_kernel(
    const uint16_t *down_projection,
    const float *gate_scales,
    const float *residual_hidden,
    const float *routed_combined,
    float *output_hidden,
    bool vllm_bf16_residual
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= kOutputElements) {
        return;
    }
    const size_t token = index / kHidden;
#if QRT_TRITON_MOE_Q1024_EXACT_SHARED
    float combined;
    float output;
    asm("v_fma_f32 %0, %1, %2, %3"
        : "=v"(combined)
        : "v"(gate_scales[token]),
          "v"(bf16_to_float(down_projection[index])),
          "v"(routed_combined[index]));
    asm("v_add_f32 %0, %1, %2"
        : "=v"(output)
        : "v"(residual_hidden[index]), "v"(combined));
    output_hidden[index] = output;
#else
    const float shared = __fmul_rn(
        gate_scales[token],
        bf16_to_float(down_projection[index])
    );
#if QRT_TRITON_MOE_Q1024_STATIC_F32_COMBINE
    (void)vllm_bf16_residual;
    const float combined = __fadd_rn(routed_combined[index], shared);
    output_hidden[index] = __fadd_rn(residual_hidden[index], combined);
#else
    if (vllm_bf16_residual) {
        const float routed_bf16 =
            bf16_to_float(float_to_bf16(routed_combined[index]));
        const float shared_bf16 = bf16_to_float(float_to_bf16(shared));
        const float combined_bf16 = bf16_to_float(float_to_bf16(
            __fadd_rn(routed_bf16, shared_bf16)
        ));
        const float residual_bf16 =
            bf16_to_float(float_to_bf16(residual_hidden[index]));
        output_hidden[index] = bf16_to_float(float_to_bf16(
            __fadd_rn(residual_bf16, combined_bf16)
        ));
    } else {
        const float combined = __fadd_rn(routed_combined[index], shared);
        output_hidden[index] = __fadd_rn(residual_hidden[index], combined);
    }
#endif
#endif
}

__global__ void full_v3_fused_combine_residual_kernel(
    const float *route_outputs,
    const float *topk_weights,
    const int32_t *topk_ids,
    const uint16_t *down_projection,
    const float *gate_scales,
    const float *residual_hidden,
    float *output_hidden,
    bool vllm_bf16_residual,
    bool vllm_routed_bf16_endpoint,
    bool vllm_route_sum_vt4,
    bool vllm_route_sum_bf16_endpoint,
    bool vllm_sorted_bf16_route_sum
) {
    const size_t work_item =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t scalar_base = work_item * kFusedCombineWidth;
    if (scalar_base >= kOutputElements) {
        return;
    }
    const size_t token = scalar_base / kHidden;
    const size_t column_base = scalar_base - token * kHidden;
    const size_t route_base = token * kTopK;
    float routed[kFusedCombineWidth]{};
    if (vllm_sorted_bf16_route_sum) {
        uint32_t route_order[kTopK]{};
        uint32_t previous_expert = 0u;
        for (uint32_t step = 0u; step < kTopK; ++step) {
            uint32_t selected_route = kTopK;
            uint32_t selected_expert = kExperts;
            for (uint32_t route = 0u; route < kTopK; ++route) {
                const uint32_t expert = static_cast<uint32_t>(
                    topk_ids[route_base + route]
                );
                if ((step == 0u || expert > previous_expert) &&
                    expert < selected_expert) {
                    selected_expert = expert;
                    selected_route = route;
                }
            }
            route_order[step] = selected_route;
            previous_expert = selected_expert;
        }
        for (uint32_t step = 0u; step < kTopK; ++step) {
            const size_t route = route_base + route_order[step];
#pragma unroll
            for (uint32_t lane = 0u; lane < kFusedCombineWidth; ++lane) {
                const float down_bf16 = bf16_to_float(float_to_bf16(
                    route_outputs[route * kHidden + column_base + lane]
                ));
                const float contribution_bf16 = bf16_to_float(float_to_bf16(
                    topk_weights[route] * down_bf16
                ));
                routed[lane] = bf16_to_float(float_to_bf16(
                    __fadd_rn(routed[lane], contribution_bf16)
                ));
            }
        }
    } else if (vllm_route_sum_vt4) {
        // PyTorch a36e1d39's 16-bit CUDA reduction uses vt0=4.  For the
        // eight non-contiguous top-k rows, each output thread pairs routes
        // 0/4, 1/5, 2/6, and 3/7, then adds those four accumulators in order.
#pragma unroll
        for (uint32_t lane = 0u; lane < kFusedCombineWidth; ++lane) {
            float pair_sums[4]{};
#pragma unroll
            for (uint32_t pair = 0u; pair < 4u; ++pair) {
                const size_t route0 = route_base + pair;
                const size_t route1 = route0 + 4u;
                const float contribution0_f32 =
                    topk_weights[route0] * route_outputs[
                        route0 * kHidden + column_base + lane
                    ];
                const float contribution1_f32 =
                    topk_weights[route1] * route_outputs[
                        route1 * kHidden + column_base + lane
                    ];
                const float contribution0 = vllm_routed_bf16_endpoint
                    ? bf16_to_float(float_to_bf16(contribution0_f32))
                    : contribution0_f32;
                const float contribution1 = vllm_routed_bf16_endpoint
                    ? bf16_to_float(float_to_bf16(contribution1_f32))
                    : contribution1_f32;
                pair_sums[pair] = __fadd_rn(
                    contribution0,
                    contribution1
                );
            }
            float sum = __fadd_rn(pair_sums[0], pair_sums[1]);
            sum = __fadd_rn(sum, pair_sums[2]);
            sum = __fadd_rn(sum, pair_sums[3]);
            routed[lane] = sum;
        }
    } else {
        for (uint32_t route_order = 0u; route_order < kTopK;
             ++route_order) {
            const size_t route = route_base + route_order;
            const float weight = topk_weights[route];
#pragma unroll
            for (uint32_t lane = 0u; lane < kFusedCombineWidth; ++lane) {
                const float contribution_f32 = weight * route_outputs[
                    route * kHidden + column_base + lane
                ];
                const float contribution = vllm_routed_bf16_endpoint
                    ? bf16_to_float(float_to_bf16(contribution_f32))
                    : contribution_f32;
                routed[lane] = __fadd_rn(routed[lane], contribution);
            }
        }
    }
    if (vllm_route_sum_bf16_endpoint) {
#pragma unroll
        for (uint32_t lane = 0u; lane < kFusedCombineWidth; ++lane) {
            routed[lane] = bf16_to_float(float_to_bf16(routed[lane]));
        }
    }
#pragma unroll
    for (uint32_t lane = 0u; lane < kFusedCombineWidth; ++lane) {
        const size_t index = scalar_base + lane;
        const float shared = __fmul_rn(
            gate_scales[token],
            bf16_to_float(down_projection[index])
        );
        if (vllm_bf16_residual) {
            const float routed_bf16 =
                bf16_to_float(float_to_bf16(routed[lane]));
            const float shared_bf16 = bf16_to_float(float_to_bf16(shared));
            const float combined_bf16 = bf16_to_float(float_to_bf16(
                __fadd_rn(routed_bf16, shared_bf16)
            ));
            const float residual_bf16 =
                bf16_to_float(float_to_bf16(residual_hidden[index]));
            output_hidden[index] = bf16_to_float(float_to_bf16(
                __fadd_rn(residual_bf16, combined_bf16)
            ));
        } else {
            const float combined = __fadd_rn(routed[lane], shared);
            output_hidden[index] = __fadd_rn(
                residual_hidden[index],
                combined
            );
        }
    }
}

void set_error(const char *stage, hipError_t status) {
    std::snprintf(
        g_state.error,
        sizeof(g_state.error),
        "%s: hip status %d (%s)",
        stage,
        static_cast<int>(status),
        hipGetErrorString(status)
    );
}

void set_error_text(const char *message) {
    std::snprintf(g_state.error, sizeof(g_state.error), "%s", message);
}

void set_matrix_error(const char *stage, hipblasStatus_t status) {
    std::snprintf(
        g_state.error,
        sizeof(g_state.error),
        "%s: hipBLASLt status %d",
        stage,
        static_cast<int>(status)
    );
}

void release_matrix_plan(MatrixPlan *plan) {
    if (plan->preference != nullptr) {
        (void)hipblasLtMatmulPreferenceDestroy(plan->preference);
    }
    if (plan->output_layout != nullptr) {
        (void)hipblasLtMatrixLayoutDestroy(plan->output_layout);
    }
    if (plan->input_layout != nullptr) {
        (void)hipblasLtMatrixLayoutDestroy(plan->input_layout);
    }
    if (plan->weight_layout != nullptr) {
        (void)hipblasLtMatrixLayoutDestroy(plan->weight_layout);
    }
    if (plan->operation != nullptr) {
        (void)hipblasLtMatmulDescDestroy(plan->operation);
    }
    *plan = MatrixPlan{};
}

bool ensure_matrix_workspace(size_t bytes) {
    if (bytes <= g_state.matrix_workspace_bytes) {
        return true;
    }
    if (bytes > kMatrixWorkspaceLimit) {
        set_error_text("hipBLASLt selected workspace above provider limit");
        return false;
    }
    if (g_state.matrix_workspace != nullptr) {
        (void)hipFree(g_state.matrix_workspace);
        g_state.matrix_workspace = nullptr;
        g_state.matrix_workspace_bytes = 0;
    }
    if (bytes == 0) {
        return true;
    }
    const hipError_t status = hipMalloc(&g_state.matrix_workspace, bytes);
    if (status != hipSuccess) {
        set_error("hipMalloc(matrix_workspace)", status);
        return false;
    }
    g_state.matrix_workspace_bytes = bytes;
    return true;
}

bool ensure_matrix_plan(
    MatrixPlan *plan,
    uint32_t output_features,
    uint32_t input_features
) {
    if (plan->operation != nullptr) {
        return plan->output_features == output_features &&
            plan->input_features == input_features;
    }
    if (g_state.matrix_handle == nullptr) {
        const hipblasStatus_t status = hipblasLtCreate(&g_state.matrix_handle);
        if (status != HIPBLAS_STATUS_SUCCESS) {
            set_matrix_error("hipblasLtCreate", status);
            return false;
        }
    }
    plan->output_features = output_features;
    plan->input_features = input_features;
    hipblasStatus_t status = hipblasLtMatmulDescCreate(
        &plan->operation,
        HIPBLAS_COMPUTE_32F,
        HIP_R_32F
    );
    if (status != HIPBLAS_STATUS_SUCCESS) {
        set_matrix_error("hipblasLtMatmulDescCreate", status);
        return false;
    }
    hipblasOperation_t transpose_weight = HIPBLAS_OP_T;
    hipblasOperation_t transpose_input = HIPBLAS_OP_N;
    status = hipblasLtMatmulDescSetAttribute(
        plan->operation,
        HIPBLASLT_MATMUL_DESC_TRANSA,
        &transpose_weight,
        sizeof(transpose_weight)
    );
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulDescSetAttribute(
            plan->operation,
            HIPBLASLT_MATMUL_DESC_TRANSB,
            &transpose_input,
            sizeof(transpose_input)
        );
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatrixLayoutCreate(
            &plan->weight_layout,
            HIP_R_16BF,
            input_features,
            output_features,
            input_features
        );
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatrixLayoutCreate(
            &plan->input_layout,
            HIP_R_16BF,
            input_features,
            kTokens,
            input_features
        );
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatrixLayoutCreate(
            &plan->output_layout,
            HIP_R_16BF,
            output_features,
            kTokens,
            output_features
        );
    }
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulPreferenceCreate(&plan->preference);
    }
    const uint64_t workspace_limit =
        static_cast<uint64_t>(kMatrixWorkspaceLimit);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        status = hipblasLtMatmulPreferenceSetAttribute(
            plan->preference,
            HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_limit,
            sizeof(workspace_limit)
        );
    }
    if (status != HIPBLAS_STATUS_SUCCESS) {
        set_matrix_error("hipBLASLt matrix plan", status);
        return false;
    }
    constexpr int kRequestedAlgorithms = 16;
    std::array<hipblasLtMatmulHeuristicResult_t, kRequestedAlgorithms>
        heuristics{};
    int returned_algorithms = 0;
    status = hipblasLtMatmulAlgoGetHeuristic(
        g_state.matrix_handle,
        plan->operation,
        plan->weight_layout,
        plan->input_layout,
        plan->output_layout,
        plan->output_layout,
        plan->preference,
        kRequestedAlgorithms,
        heuristics.data(),
        &returned_algorithms
    );
    if (status != HIPBLAS_STATUS_SUCCESS) {
        set_matrix_error("hipblasLtMatmulAlgoGetHeuristic", status);
        return false;
    }
    if (returned_algorithms <= 0) {
        set_error_text("hipBLASLt returned no fixed q8192 BF16 algorithm");
        return false;
    }
    plan->algorithm = heuristics[0].algo;
    plan->workspace_bytes = heuristics[0].workspaceSize;
    return ensure_matrix_workspace(plan->workspace_bytes);
}

bool launch_matrix(
    MatrixPlan *plan,
    const uint16_t *weights,
    const uint16_t *inputs,
    uint16_t *outputs,
    uint32_t output_features,
    uint32_t input_features,
    hipStream_t stream,
    const char *stage
) {
    if (!ensure_matrix_plan(plan, output_features, input_features)) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const hipblasStatus_t status = hipblasLtMatmul(
        g_state.matrix_handle,
        plan->operation,
        &alpha,
        weights,
        plan->weight_layout,
        inputs,
        plan->input_layout,
        &beta,
        outputs,
        plan->output_layout,
        outputs,
        plan->output_layout,
        &plan->algorithm,
        g_state.matrix_workspace,
        plan->workspace_bytes,
        stream
    );
    if (status != HIPBLAS_STATUS_SUCCESS) {
        set_matrix_error(stage, status);
        return false;
    }
    return true;
}

void release_kernel(ModuleKernel *kernel) {
    if (kernel->module != nullptr) {
        (void)hipModuleUnload(kernel->module);
    }
    *kernel = ModuleKernel{};
}

bool release_full_v3_execution_state() {
    hipError_t synchronization_status = hipSuccess;
    const char *synchronization_stage = nullptr;
    if (g_state.full_v3_poisoned) {
        synchronization_status = hipDeviceSynchronize();
        synchronization_stage = "full-v3 poisoned-state release";
    } else {
        for (FullV3EventSlot &slot : g_state.full_v3_slots) {
            if (!slot.in_flight || slot.caller_done == nullptr) {
                continue;
            }
            synchronization_status = hipEventSynchronize(slot.caller_done);
            if (synchronization_status != hipSuccess) {
                synchronization_stage = "full-v3 caller completion release";
                break;
            }
        }
        if (synchronization_status == hipSuccess &&
            g_state.full_v3_shared_stream != nullptr) {
            synchronization_status = hipStreamSynchronize(
                g_state.full_v3_shared_stream
            );
            synchronization_stage = "full-v3 shared stream release";
        }
    }
    if (synchronization_status != hipSuccess) {
        // A device-wide retry is the only safe way to cover a partially
        // enqueued slot whose caller_done record itself failed.
        const hipError_t device_status = hipDeviceSynchronize();
        if (device_status != hipSuccess) {
            set_error(
                synchronization_stage != nullptr ?
                    synchronization_stage : "full-v3 release",
                device_status
            );
            return false;
        }
    }
    if (g_state.full_v3_shared_stream != nullptr) {
        (void)hipStreamDestroy(g_state.full_v3_shared_stream);
        g_state.full_v3_shared_stream = nullptr;
    }
    for (FullV3EventSlot &slot : g_state.full_v3_slots) {
        if (slot.caller_done != nullptr) {
            (void)hipEventDestroy(slot.caller_done);
        }
        if (slot.shared_done != nullptr) {
            (void)hipEventDestroy(slot.shared_done);
        }
        if (slot.input_ready != nullptr) {
            (void)hipEventDestroy(slot.input_ready);
        }
        slot = FullV3EventSlot{};
    }
    g_state.full_v3_next_slot = 0u;
    g_state.full_v3_owner_stream = nullptr;
    g_state.full_v3_owner_valid = false;
    g_state.full_v3_poisoned = false;
    return true;
}

bool release_state() {
    if (!release_full_v3_execution_state()) {
        return false;
    }
    release_matrix_plan(&g_state.shared_down_plan);
    release_matrix_plan(&g_state.shared_projection_plan);
    release_matrix_plan(&g_state.shared_gate_plan);
    release_matrix_plan(&g_state.router_plan);
    if (g_state.matrix_workspace != nullptr) {
        (void)hipFree(g_state.matrix_workspace);
    }
    if (g_state.matrix_handle != nullptr) {
        (void)hipblasLtDestroy(g_state.matrix_handle);
    }
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    if (g_state.early_f32_activated != nullptr) {
        (void)hipFree(g_state.early_f32_activated);
    }
#endif
    if (g_state.shared_gate_scales != nullptr) {
        (void)hipFree(g_state.shared_gate_scales);
    }
    if (g_state.shared_down_projection != nullptr) {
        (void)hipFree(g_state.shared_down_projection);
    }
    if (g_state.shared_activated != nullptr) {
        (void)hipFree(g_state.shared_activated);
    }
    if (g_state.shared_up_projection != nullptr) {
        (void)hipFree(g_state.shared_up_projection);
    }
    if (g_state.shared_gate_projection != nullptr) {
        (void)hipFree(g_state.shared_gate_projection);
    }
    if (g_state.shared_gate_logits != nullptr) {
        (void)hipFree(g_state.shared_gate_logits);
    }
    if (g_state.routed_combined != nullptr) {
        (void)hipFree(g_state.routed_combined);
    }
    if (g_state.route_outputs != nullptr) (void)hipFree(g_state.route_outputs);
    if (g_state.activated != nullptr) (void)hipFree(g_state.activated);
    if (g_state.block_experts != nullptr) (void)hipFree(g_state.block_experts);
    if (g_state.sorted_routes != nullptr) (void)hipFree(g_state.sorted_routes);
    if (g_state.total_post_pad != nullptr) (void)hipFree(g_state.total_post_pad);
    if (g_state.cumsum != nullptr) (void)hipFree(g_state.cumsum);
    if (g_state.counts != nullptr) (void)hipFree(g_state.counts);
    if (g_state.topk_weights != nullptr) (void)hipFree(g_state.topk_weights);
    if (g_state.topk_ids != nullptr) (void)hipFree(g_state.topk_ids);
#if QRT_TRITON_MOE_ROCBLAS_ROUTER
    if (g_state.router_handle != nullptr) {
        (void)rocblas_destroy_handle(g_state.router_handle);
    }
#endif
    if (g_state.router_logits_bf16 != nullptr) {
        (void)hipFree(g_state.router_logits_bf16);
    }
    if (g_state.transposed_router_weights != nullptr) {
        (void)hipFree(g_state.transposed_router_weights);
    }
    if (g_state.input_bf16 != nullptr) (void)hipFree(g_state.input_bf16);
    release_kernel(&g_state.down);
    release_kernel(&g_state.gate_up);
#if QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS
    release_kernel(&g_state.grouped_bf16_down);
#endif
#if QRT_TRITON_MOE_Q1024_EXACT_ROUTED || \
    QRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN || \
    QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
    release_kernel(&g_state.exact_down);
    release_kernel(&g_state.exact_gate_up);
#endif
#if QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED
    release_kernel(&g_state.packed_exact_down);
    release_kernel(&g_state.packed_exact_gate_up);
#endif
#if QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
    release_kernel(&g_state.sorted_packed_exact_combine);
    release_kernel(&g_state.sorted_packed_exact_down);
    release_kernel(&g_state.sorted_packed_exact_gate_up);
#endif
#if QRT_TRITON_MOE_Q1024_EXACT_SHARED
    release_kernel(&g_state.exact_shared_down);
    release_kernel(&g_state.exact_shared_gate_logit);
    release_kernel(&g_state.exact_shared_gate_up);
#endif
    release_kernel(&g_state.scatter);
    release_kernel(&g_state.padded_prefix);
    release_kernel(&g_state.prefix);
    release_kernel(&g_state.count);
    g_state = ProviderState{};
    return true;
}

bool load_kernel(
    const char *directory,
    const char *stem,
    const char *symbol,
    uint32_t grid_x,
    uint32_t threads,
    uint32_t shared_bytes,
    ModuleKernel *kernel
) {
    char path[1400];
    const int length = std::snprintf(
        path,
        sizeof(path),
        "%s\\" QRT_TRITON_MOE_KERNEL_PREFIX "_%s.hsaco",
        directory,
        stem
    );
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) {
        set_error_text("selected-MoE kernel path is too long");
        return false;
    }
    hipError_t status = hipModuleLoad(&kernel->module, path);
    if (status != hipSuccess) {
        set_error((std::string("hipModuleLoad(") + stem + ")").c_str(), status);
        return false;
    }
    status = hipModuleGetFunction(&kernel->function, kernel->module, symbol);
    if (status != hipSuccess) {
        set_error((std::string("hipModuleGetFunction(") + stem + ")").c_str(), status);
        return false;
    }
    kernel->grid_x = grid_x;
    kernel->threads = threads;
    kernel->shared_bytes = shared_bytes;
    return true;
}

bool load_named_kernel(
    const char *directory,
    const char *file,
    const char *symbol,
    uint32_t grid_x,
    uint32_t threads,
    uint32_t shared_bytes,
    ModuleKernel *kernel
) {
    char path[1400];
    const int length = std::snprintf(
        path,
        sizeof(path),
        "%s\\%s",
        directory,
        file
    );
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) {
        set_error_text("selected-MoE named kernel path is too long");
        return false;
    }
    hipError_t status = hipModuleLoad(&kernel->module, path);
    if (status != hipSuccess) {
        set_error(
            (std::string("hipModuleLoad(") + file + ")").c_str(),
            status
        );
        return false;
    }
    status = hipModuleGetFunction(&kernel->function, kernel->module, symbol);
    if (status != hipSuccess) {
        set_error(
            (std::string("hipModuleGetFunction(") + symbol + ")").c_str(),
            status
        );
        return false;
    }
    kernel->grid_x = grid_x;
    kernel->threads = threads;
    kernel->shared_bytes = shared_bytes;
    return true;
}

hipError_t launch_module(
    ModuleKernel &kernel,
    void **arguments,
    size_t argument_count,
    hipStream_t stream
) {
    void *global_scratch = nullptr;
    void *profile_scratch = nullptr;
    arguments[argument_count] = &global_scratch;
    arguments[argument_count + 1u] = &profile_scratch;
    return hipModuleLaunchKernel(
        kernel.function,
        kernel.grid_x,
        1u,
        1u,
        kernel.threads,
        1u,
        1u,
        kernel.shared_bytes,
        stream,
        arguments,
        nullptr
    );
}

template <typename T>
bool allocate(T **pointer, size_t bytes, const char *stage) {
    hipError_t status = hipMalloc(reinterpret_cast<void **>(pointer), bytes);
    if (status != hipSuccess) {
        set_error(stage, status);
        return false;
    }
    return true;
}

bool allocate_optional_transposed_router() {
#if QRT_TRITON_MOE_TRANSPOSED_ROUTER
    return allocate(
        &g_state.transposed_router_weights,
        kRouterWeightElements * sizeof(uint16_t),
        "hipMalloc(transposed_router_weights)"
    );
#else
    return true;
#endif
}

bool q8192_hipblaslt_bf16_router_requested() {
    const char *value = std::getenv(
        "QRT_QWEN36_Q8192_ROUTER_HIPBLASLT_BF16"
    );
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0;
}

bool allocate_optional_hipblaslt_router_logits() {
    if (!q8192_hipblaslt_bf16_router_requested()) {
        return true;
    }
    return allocate(
        &g_state.router_logits_bf16,
        static_cast<size_t>(kTokens) * kExperts * sizeof(uint16_t),
        "hipMalloc(router_logits_bf16)"
    );
}

bool ensure_optional_hipblaslt_router_plan() {
    return !q8192_hipblaslt_bf16_router_requested() ||
        ensure_matrix_plan(&g_state.router_plan, kExperts, kHidden);
}

bool ensure_full_v3_execution_state() {
    bool complete = g_state.full_v3_shared_stream != nullptr;
    for (const FullV3EventSlot &slot : g_state.full_v3_slots) {
        complete = complete && slot.input_ready != nullptr &&
            slot.shared_done != nullptr && slot.caller_done != nullptr;
    }
    if (complete) {
        return true;
    }
    (void)release_full_v3_execution_state();
    hipError_t status = hipStreamCreateWithFlags(
        &g_state.full_v3_shared_stream,
        hipStreamNonBlocking
    );
    for (FullV3EventSlot &slot : g_state.full_v3_slots) {
        if (status == hipSuccess) {
            status = hipEventCreateWithFlags(
                &slot.input_ready,
                hipEventDisableTiming
            );
        }
        if (status == hipSuccess) {
            status = hipEventCreateWithFlags(
                &slot.shared_done,
                hipEventDisableTiming
            );
        }
        if (status == hipSuccess) {
            status = hipEventCreateWithFlags(
                &slot.caller_done,
                hipEventDisableTiming
            );
        }
    }
    if (status != hipSuccess) {
        set_error("full-v3 stream/event creation", status);
        (void)release_full_v3_execution_state();
        return false;
    }
    return true;
}

bool launch_input_conversion(
    const float *post_attention_f32,
    hipStream_t stream
) {
    hipLaunchKernelGGL(
        convert_input_kernel,
        dim3(static_cast<uint32_t>(
            (kInputElements + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        post_attention_f32,
        g_state.input_bf16,
        kInputElements
    );
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("convert_input_kernel", status);
        return false;
    }
    return true;
}

bool zero_request_scratch(hipStream_t stream) {
#if QRT_TRITON_MOE_ZERO_REQUEST_SCRATCH
    struct ScratchRange {
        void *pointer;
        size_t bytes;
    };
    const ScratchRange ranges[] = {
        {
            g_state.topk_ids,
            static_cast<size_t>(kRoutes) * sizeof(int32_t)
        },
        {
            g_state.topk_weights,
            static_cast<size_t>(kRoutes) * sizeof(float)
        },
        {g_state.counts, kCountElements * sizeof(int32_t)},
        {
            g_state.cumsum,
            static_cast<size_t>(kExperts + 1u) * sizeof(int32_t)
        },
        {g_state.total_post_pad, sizeof(int32_t)},
        {
            g_state.sorted_routes,
            static_cast<size_t>(kMaxSortedRoutes) * sizeof(int32_t)
        },
        {
            g_state.block_experts,
            static_cast<size_t>(kMaxRouteBlocks) * sizeof(int32_t)
        },
        {
            g_state.activated,
            kActivatedElements * sizeof(uint16_t)
        },
        {
            g_state.route_outputs,
            kRouteOutputElements * sizeof(float)
        },
        {
            g_state.routed_combined,
            kOutputElements * sizeof(float)
        },
        {
            g_state.shared_gate_logits,
            static_cast<size_t>(kTokens) * sizeof(uint16_t)
        },
        {
            g_state.shared_gate_projection,
            kSharedProjectionElements * sizeof(uint16_t)
        },
        {
            g_state.shared_up_projection,
            kSharedProjectionElements * sizeof(uint16_t)
        },
        {
            g_state.shared_activated,
            kSharedProjectionElements * sizeof(uint16_t)
        },
        {
            g_state.shared_down_projection,
            kOutputElements * sizeof(uint16_t)
        },
        {
            g_state.shared_gate_scales,
            static_cast<size_t>(kTokens) * sizeof(float)
        }
    };
    for (const ScratchRange &range : ranges) {
        const hipError_t status = hipMemsetAsync(
            range.pointer,
            0,
            range.bytes,
            stream
        );
        if (status != hipSuccess) {
            set_error("zero request scratch", status);
            return false;
        }
    }
#else
    (void)stream;
#endif
    return true;
}

bool launch_route_sort(hipStream_t stream) {
    hipError_t status = hipMemsetAsync(
        g_state.counts,
        0,
        kCountElements * sizeof(int32_t),
        stream
    );
    if (status == hipSuccess) {
        status = hipMemsetAsync(
            g_state.cumsum,
            0,
            static_cast<size_t>(kExperts + 1u) * sizeof(int32_t),
            stream
        );
    }
    if (status == hipSuccess) {
        status = hipMemsetD32Async(
            reinterpret_cast<hipDeviceptr_t>(g_state.sorted_routes),
            static_cast<int>(kRoutes),
            kMaxSortedRoutes,
            stream
        );
    }
    if (status != hipSuccess) {
        set_error("selected-MoE setup", status);
        return false;
    }

    void *count_args[4] = {&g_state.topk_ids, &g_state.counts};
    status = launch_module(g_state.count, count_args, 2u, stream);
    void *prefix_args[3] = {&g_state.counts};
    if (status == hipSuccess) {
        status = launch_module(g_state.prefix, prefix_args, 1u, stream);
    }
    void *padded_args[5] = {
        &g_state.total_post_pad,
        &g_state.counts,
        &g_state.cumsum,
    };
    if (status == hipSuccess) {
        status = launch_module(
            g_state.padded_prefix,
            padded_args,
            3u,
            stream
        );
    }
    void *scatter_args[7] = {
        &g_state.topk_ids,
        &g_state.sorted_routes,
        &g_state.block_experts,
        &g_state.counts,
        &g_state.cumsum,
    };
    if (status == hipSuccess) {
        status = launch_module(g_state.scatter, scatter_args, 5u, stream);
    }
    if (status != hipSuccess) {
        set_error("selected-MoE route sort", status);
        return false;
    }
    return true;
}

bool launch_routed_matrices_after_input_conversion(
    const uint16_t *gate_up_bf16,
    const uint16_t *down_bf16,
    hipStream_t stream
) {
#if QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED
    const uint16_t *gate_up_pointer = gate_up_bf16;
    const uint16_t *input_pointer = g_state.input_bf16;
    const int32_t *topk_ids_pointer = g_state.topk_ids;
    uint16_t *activated_pointer = g_state.activated;
    void *gate_arguments[6] = {
        &gate_up_pointer,
        &input_pointer,
        &topk_ids_pointer,
        &activated_pointer,
    };
    hipError_t status = launch_module(
        g_state.packed_exact_gate_up,
        gate_arguments,
        4u,
        stream
    );
    const uint16_t *down_pointer = down_bf16;
    const float *topk_weights_pointer = g_state.topk_weights;
    float *output_pointer = g_state.routed_combined;
    void *down_arguments[7] = {
        &down_pointer,
        &activated_pointer,
        &topk_ids_pointer,
        &topk_weights_pointer,
        &output_pointer,
    };
    if (status == hipSuccess) {
        status = launch_module(
            g_state.packed_exact_down,
            down_arguments,
            5u,
            stream
        );
    }
    if (status != hipSuccess) {
        set_error("launch(q1024 packed-exact routed kernels)", status);
        return false;
    }
    return true;
#elif QRT_TRITON_MOE_Q1024_EXACT_ROUTED
    const uint16_t *gate_up_pointer = gate_up_bf16;
    const uint16_t *input_pointer = g_state.input_bf16;
    const int32_t *topk_ids_pointer = g_state.topk_ids;
    uint16_t *activated_pointer = g_state.activated;
    void *gate_arguments[6] = {
        &gate_up_pointer,
        &input_pointer,
        &topk_ids_pointer,
        &activated_pointer,
    };
    hipError_t status = launch_module(
        g_state.exact_gate_up,
        gate_arguments,
        4u,
        stream
    );
    const uint16_t *down_pointer = down_bf16;
    const float *topk_weights_pointer = g_state.topk_weights;
    float *output_pointer = g_state.routed_combined;
    void *down_arguments[7] = {
        &down_pointer,
        &activated_pointer,
        &topk_ids_pointer,
        &topk_weights_pointer,
        &output_pointer,
    };
    if (status == hipSuccess) {
        status = launch_module(
            g_state.exact_down,
            down_arguments,
            5u,
            stream
        );
    }
    if (status != hipSuccess) {
        set_error("launch(q1024 exact routed kernels)", status);
        return false;
    }
    return true;
#else
    if (!launch_route_sort(stream)) {
        return false;
    }
    hipError_t status = hipSuccess;
#if QRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN
    if (status == hipSuccess) {
        const uint16_t *gate_up_pointer = gate_up_bf16;
        const uint16_t *input_pointer = g_state.input_bf16;
        const int32_t *topk_ids_pointer = g_state.topk_ids;
        uint16_t *activated_pointer = g_state.activated;
        void *gate_arguments[6] = {
            &gate_up_pointer,
            &input_pointer,
            &topk_ids_pointer,
            &activated_pointer,
        };
        status = launch_module(
            g_state.exact_gate_up,
            gate_arguments,
            4u,
            stream
        );
    }
#elif QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
    if (status == hipSuccess) {
        const uint16_t *gate_up_pointer = gate_up_bf16;
        const uint16_t *input_pointer = g_state.input_bf16;
        const int32_t *sorted_pointer = g_state.sorted_routes;
        const int32_t *expert_pointer = g_state.block_experts;
        const int32_t *padded_pointer = g_state.total_post_pad;
        uint16_t *activated_pointer = g_state.activated;
        void *gate_arguments[8] = {
            &gate_up_pointer,
            &input_pointer,
            &sorted_pointer,
            &expert_pointer,
            &padded_pointer,
            &activated_pointer,
        };
        status = launch_module(
            g_state.sorted_packed_exact_gate_up,
            gate_arguments,
            6u,
            stream
        );
    }
#elif QRT_TRITON_MOE_NATIVE_WMMA_GATE
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            native_wmma_gate_up_silu_kernel,
            dim3(kMaxRouteBlocks * kNativeWmmaGateGridN),
            dim3(kNativeWmmaThreads),
            0,
            stream,
            g_state.input_bf16,
            gate_up_bf16,
            g_state.sorted_routes,
            g_state.block_experts,
            g_state.total_post_pad,
            g_state.activated
        );
        status = hipGetLastError();
    }
#else
    const uint16_t *gate_up_pointer = gate_up_bf16;
    void *gate_args[8] = {
        &g_state.input_bf16,
        &gate_up_pointer,
        &g_state.sorted_routes,
        &g_state.block_experts,
        &g_state.total_post_pad,
        &g_state.activated,
    };
    if (status == hipSuccess) {
        status = launch_module(g_state.gate_up, gate_args, 6u, stream);
    }
#endif
#if QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
    if (status == hipSuccess) {
        const uint16_t *down_pointer = down_bf16;
        const uint16_t *activated_pointer = g_state.activated;
        const int32_t *topk_ids_pointer = g_state.topk_ids;
        const float *topk_weights_pointer = g_state.topk_weights;
        float *output_pointer = g_state.routed_combined;
        void *down_arguments[7] = {
            &down_pointer,
            &activated_pointer,
            &topk_ids_pointer,
            &topk_weights_pointer,
            &output_pointer,
        };
        status = launch_module(
            g_state.exact_down,
            down_arguments,
            5u,
            stream
        );
    }
#elif QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
    if (status == hipSuccess) {
        const uint16_t *down_pointer = down_bf16;
        const uint16_t *activated_pointer = g_state.activated;
        const int32_t *sorted_pointer = g_state.sorted_routes;
        const int32_t *expert_pointer = g_state.block_experts;
        const int32_t *padded_pointer = g_state.total_post_pad;
        float *route_output_pointer = g_state.route_outputs;
        void *down_arguments[8] = {
            &down_pointer,
            &activated_pointer,
            &sorted_pointer,
            &expert_pointer,
            &padded_pointer,
            &route_output_pointer,
        };
        status = launch_module(
            g_state.sorted_packed_exact_down,
            down_arguments,
            6u,
            stream
        );
    }
    if (status == hipSuccess) {
        const float *route_output_pointer = g_state.route_outputs;
        const float *topk_weights_pointer = g_state.topk_weights;
        float *output_pointer = g_state.routed_combined;
        void *combine_arguments[5] = {
            &route_output_pointer,
            &topk_weights_pointer,
            &output_pointer,
        };
        status = launch_module(
            g_state.sorted_packed_exact_combine,
            combine_arguments,
            3u,
            stream
        );
    }
#elif QRT_TRITON_MOE_NATIVE_WMMA_DOWN
    if (status == hipSuccess) {
        hipLaunchKernelGGL(
            native_wmma_down_kernel,
            dim3(kMaxRouteBlocks * kNativeWmmaDownGridN),
            dim3(kNativeWmmaThreads),
            0,
            stream,
            g_state.activated,
            down_bf16,
            g_state.sorted_routes,
            g_state.block_experts,
            g_state.total_post_pad,
            g_state.route_outputs
        );
        status = hipGetLastError();
    }
#else
    const uint16_t *down_pointer = down_bf16;
    void *down_args[8] = {
        &g_state.activated,
        &down_pointer,
        &g_state.sorted_routes,
        &g_state.block_experts,
        &g_state.total_post_pad,
        &g_state.route_outputs,
    };
    if (status == hipSuccess) {
#if QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS
        status = launch_module(
            g_state.grouped_bf16_down,
            down_args,
            6u,
            stream
        );
#else
        status = launch_module(g_state.down, down_args, 6u, stream);
#endif
    }
#endif
    if (status != hipSuccess) {
        set_error("launch(selected-MoE routed matrix kernels)", status);
        return false;
    }
    return true;
#endif
}

bool launch_route_combine(float *output_f32, hipStream_t stream) {
    hipLaunchKernelGGL(
        combine_route_order_kernel,
        dim3(static_cast<uint32_t>(
            (kOutputElements + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        g_state.route_outputs,
        g_state.topk_weights,
        output_f32,
        kOutputElements
    );
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("combine_route_order_kernel", status);
        return false;
    }
    return true;
}

bool launch_routed_pipeline_after_input_conversion(
    const uint16_t *gate_up_bf16,
    const uint16_t *down_bf16,
    float *output_f32,
    hipStream_t stream
) {
    const bool matrices_launched =
        launch_routed_matrices_after_input_conversion(
               gate_up_bf16,
               down_bf16,
               stream
           );
#if QRT_TRITON_MOE_Q1024_EXACT_ROUTED || \
    QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED || \
    QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED || \
    QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
    (void)output_f32;
    return matrices_launched;
#else
    return matrices_launched &&
        launch_route_combine(output_f32, stream);
#endif
}

bool launch_routed_pipeline(
    const float *post_attention_f32,
    const uint16_t *gate_up_bf16,
    const uint16_t *down_bf16,
    float *output_f32,
    hipStream_t stream
) {
    return launch_input_conversion(post_attention_f32, stream) &&
        launch_routed_pipeline_after_input_conversion(
            gate_up_bf16,
            down_bf16,
            output_f32,
            stream
        );
}

bool launch_router(
    const float *post_attention_f32,
    const uint16_t *router_bf16,
    hipStream_t stream
) {
    if (q8192_hipblaslt_bf16_router_requested()) {
        if (g_state.router_logits_bf16 == nullptr ||
            !launch_input_conversion(post_attention_f32, stream) ||
            !launch_matrix(
                &g_state.router_plan,
                router_bf16,
                g_state.input_bf16,
                g_state.router_logits_bf16,
                kExperts,
                kHidden,
                stream,
                "hipblasLtMatmul(router_logits_bf16)"
            )) {
            return false;
        }
        hipLaunchKernelGGL(
            router_bf16_logits_topk_kernel,
            dim3(kTokens),
            dim3(kExperts),
            0,
            stream,
            g_state.router_logits_bf16,
            g_state.topk_ids,
            g_state.topk_weights
        );
        const hipError_t status = hipGetLastError();
        if (status != hipSuccess) {
            set_error("router_bf16_logits_topk_kernel", status);
            return false;
        }
        return true;
    }
#if QRT_TRITON_MOE_ROCBLAS_ROUTER
    if (!launch_input_conversion(post_attention_f32, stream)) {
        return false;
    }
    rocblas_status rocblas_status_value = rocblas_set_stream(
        g_state.router_handle,
        stream
    );
    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (rocblas_status_value == rocblas_status_success) {
        rocblas_status_value = rocblas_gemm_strided_batched_ex(
            g_state.router_handle,
            rocblas_operation_transpose,
            rocblas_operation_none,
            static_cast<rocblas_int>(kExperts),
            1,
            static_cast<rocblas_int>(kHidden),
            &alpha,
            router_bf16,
            rocblas_datatype_bf16_r,
            static_cast<rocblas_int>(kHidden),
            0,
            g_state.input_bf16,
            rocblas_datatype_bf16_r,
            static_cast<rocblas_int>(kHidden),
            static_cast<rocblas_stride>(kHidden),
            &beta,
            g_state.router_logits_bf16,
            rocblas_datatype_bf16_r,
            static_cast<rocblas_int>(kExperts),
            static_cast<rocblas_stride>(kExperts),
            g_state.router_logits_bf16,
            rocblas_datatype_bf16_r,
            static_cast<rocblas_int>(kExperts),
            static_cast<rocblas_stride>(kExperts),
            static_cast<rocblas_int>(kTokens),
            rocblas_datatype_f32_r,
            rocblas_gemm_algo_solution_index,
            INT32_C(-9),
            rocblas_gemm_flags_none
        );
    }
    if (rocblas_status_value != rocblas_status_success) {
        std::snprintf(
            g_state.error,
            sizeof(g_state.error),
            "rocBLAS router: status %d",
            static_cast<int>(rocblas_status_value)
        );
        return false;
    }
    hipLaunchKernelGGL(
        router_bf16_logits_topk_kernel,
        dim3(kTokens),
        dim3(kExperts),
        0,
        stream,
        g_state.router_logits_bf16,
        g_state.topk_ids,
        g_state.topk_weights
    );
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("router_bf16_logits_topk_kernel", status);
        return false;
    }
    return true;
#else
    const char *bf16_logit_endpoint_value = std::getenv(
        "QRT_QWEN36_Q8192_ROUTER_BF16_LOGIT_ENDPOINT"
    );
    const bool bf16_logit_endpoint =
        bf16_logit_endpoint_value != nullptr &&
        bf16_logit_endpoint_value[0] != '\0' &&
        std::strcmp(bf16_logit_endpoint_value, "0") != 0;
#if QRT_TRITON_MOE_TRANSPOSED_ROUTER
    hipLaunchKernelGGL(
        transpose_router_weights_kernel,
        dim3(kHidden / 32u, kExperts / 32u),
        dim3(32u, 8u),
        0,
        stream,
        router_bf16,
        g_state.transposed_router_weights
    );
    const hipError_t transpose_status = hipGetLastError();
    if (transpose_status != hipSuccess) {
        set_error("transpose_router_weights_kernel", transpose_status);
        return false;
    }
    const uint16_t *router_kernel_weights =
        g_state.transposed_router_weights;
#else
    const uint16_t *router_kernel_weights = router_bf16;
#endif
    hipLaunchKernelGGL(
        router_topk_kernel,
        dim3(kTokens / kRouterTokenTile),
        dim3(kRouterThreads),
        0,
        stream,
        post_attention_f32,
        router_kernel_weights,
        g_state.topk_ids,
        g_state.topk_weights,
        bf16_logit_endpoint
    );
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("router_topk_kernel", status);
        return false;
    }
    return true;
#endif
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32
bool check_q1024_early_f32_launch(const char *stage) {
    const hipError_t status = hipGetLastError();
    if (status == hipSuccess) {
        return true;
    }
    set_error(stage, status);
    return false;
}

bool launch_q1024_early_f32_full(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    hipStream_t stream
) {
    static_assert(
        kSharedProjectionElements + 2u * kOutputElements <=
            kRouteOutputElements,
        "q1024 early-F32 shared scratch exceeds route-output storage"
    );
    float *scratch_f32 = g_state.route_outputs;
    float *router_logits_f32 = scratch_f32;
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    float *routed_activated_f32 = g_state.early_f32_activated;
#else
    float *routed_activated_f32 = scratch_f32;
#endif

    hipLaunchKernelGGL(
        q1024_early_f32_router_topk_kernel,
        dim3(kTokens),
        dim3(kExperts),
        0u,
        stream,
        post_attention_f32,
        router_bf16,
        router_logits_f32,
        g_state.topk_ids,
        g_state.topk_weights
    );
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 router/top-k"
        )) {
        return false;
    }
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    if (!launch_route_sort(stream)) {
        return false;
    }
#endif

    constexpr uint32_t kRoutedRowThreads =
        QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
            ? QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE_ROWS
            : 64u;
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    hipLaunchKernelGGL(
        q1024_early_f32_sorted_routed_gate_up_kernel,
        dim3(
            kMaxRouteBlocks * kEarlyF32SortedPacksPerBlock,
            kIntermediate / 64u
        ),
        dim3(256u),
        0u,
        stream,
        post_attention_f32,
        routed_gate_up_bf16,
        g_state.sorted_routes,
        g_state.block_experts,
        g_state.total_post_pad,
        routed_activated_f32
    );
#elif QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
    hipLaunchKernelGGL(
        q1024_early_f32_routed_gate_up_transposed_tile_kernel,
        dim3(
            kRoutes,
            (kIntermediate + kRoutedRowThreads - 1u) /
                kRoutedRowThreads
        ),
        dim3(kRoutedRowThreads),
        0u,
        stream,
        post_attention_f32,
        routed_gate_up_bf16,
        g_state.topk_ids,
        routed_activated_f32
    );
#else
    hipLaunchKernelGGL(
        q1024_early_f32_routed_gate_up_kernel,
        dim3(
            kRoutes,
            (kIntermediate + kRoutedRowThreads - 1u) /
                kRoutedRowThreads
        ),
        dim3(kRoutedRowThreads),
        0u,
        stream,
        post_attention_f32,
        routed_gate_up_bf16,
        g_state.topk_ids,
        routed_activated_f32
    );
#endif
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 routed gate/up"
        )) {
        return false;
    }
    hipError_t status = hipMemcpyAsync(
        g_state.activated,
        routed_activated_f32,
        static_cast<size_t>(kTopK) * kIntermediate * sizeof(float),
        hipMemcpyDeviceToDevice,
        stream
    );
    if (status != hipSuccess) {
        set_error("q1024 early-F32 preserve token0 activation", status);
        return false;
    }
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    hipLaunchKernelGGL(
        q1024_early_f32_sorted_routed_down_kernel,
        dim3(
            kMaxRouteBlocks * kEarlyF32SortedPacksPerBlock,
            kHidden / 64u
        ),
        dim3(256u),
        0u,
        stream,
        routed_activated_f32,
        routed_down_bf16,
        g_state.sorted_routes,
        g_state.block_experts,
        g_state.total_post_pad,
        g_state.route_outputs
    );
    hipLaunchKernelGGL(
        q1024_early_f32_sorted_route_combine_kernel,
        dim3(static_cast<uint32_t>(
            (kOutputElements + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0u,
        stream,
        g_state.route_outputs,
        g_state.topk_weights,
        g_state.routed_combined
    );
#elif QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
    // Keep each token's exact row arithmetic unchanged while eliminating
    // 1,023 serialized host submissions. The token dimension is independent:
    // no state or reduction crosses blockIdx.y.
    hipLaunchKernelGGL(
        q1024_early_f32_routed_down_transposed_tile_kernel,
        dim3(
            (kHidden + kRoutedRowThreads - 1u) /
                kRoutedRowThreads,
            kTokens
        ),
        dim3(kRoutedRowThreads),
        0u,
        stream,
        routed_activated_f32,
        routed_down_bf16,
        reinterpret_cast<const uint32_t *>(g_state.topk_ids),
        g_state.topk_weights,
        g_state.routed_combined,
        static_cast<unsigned int>(kTopK),
        1
    );
#else
    for (uint32_t token = 0u; token < kTokens; ++token) {
        hipLaunchKernelGGL(
            q1024_early_f32_routed_down_kernel,
            dim3(
                (kHidden + kRoutedRowThreads - 1u) /
                    kRoutedRowThreads
            ),
            dim3(kRoutedRowThreads),
            0u,
            stream,
            routed_activated_f32 +
                static_cast<size_t>(token) * kTopK * kIntermediate,
            routed_down_bf16,
            reinterpret_cast<const uint32_t *>(g_state.topk_ids) +
                static_cast<size_t>(token) * kTopK,
            g_state.topk_weights + static_cast<size_t>(token) * kTopK,
            g_state.routed_combined +
                static_cast<size_t>(token) * kHidden,
            static_cast<unsigned int>(kTopK),
            1
        );
    }
#endif
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 routed down"
        )) {
        return false;
    }

    // The routed activation is dead once the preceding kernel completes.
    // Reuse its oversized route-output allocation for the three disjoint
    // exact-F32 shared surfaces.
    float *shared_activated_f32 = scratch_f32;
    float *shared_output_f32 =
        shared_activated_f32 + kSharedProjectionElements;
    float *combined_output_f32 =
        shared_output_f32 + kOutputElements;
    float *shared_gate_logits_f32 =
        reinterpret_cast<float *>(g_state.shared_gate_projection);

    hipLaunchKernelGGL(
        q1024_early_f32_shared_gate_kernel,
        dim3(kTokens),
        dim3(kNativeThreads),
        0u,
        stream,
        post_attention_f32,
        shared_gate_bf16,
        shared_gate_logits_f32,
        g_state.shared_gate_scales
    );
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 shared gate"
        )) {
        return false;
    }
#if QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
    hipLaunchKernelGGL(
        q1024_early_f32_shared_gate_up_rows8_kernel,
        dim3(kTokens, (kIntermediate + 7u) / 8u),
        dim3(kNativeThreads),
        0u,
        stream,
        post_attention_f32,
        shared_gate_projection_bf16,
        shared_up_projection_bf16,
        shared_activated_f32
    );
#else
    hipLaunchKernelGGL(
        q1024_early_f32_shared_gate_up_kernel,
        dim3(kTokens, kIntermediate),
        dim3(kNativeThreads),
        0u,
        stream,
        post_attention_f32,
        shared_gate_projection_bf16,
        shared_up_projection_bf16,
        shared_activated_f32
    );
#endif
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 shared gate/up"
        )) {
        return false;
    }
#if QRT_TRITON_MOE_Q1024_EARLY_F32_TRANSPOSED_TILE
    hipLaunchKernelGGL(
        q1024_early_f32_shared_down_rows8_kernel,
        dim3(kTokens, (kHidden + 7u) / 8u),
        dim3(kNativeThreads),
        0u,
        stream,
        shared_activated_f32,
        g_state.shared_gate_scales,
        shared_down_bf16,
        g_state.routed_combined,
        shared_output_f32,
        combined_output_f32
    );
#else
    hipLaunchKernelGGL(
        q1024_early_f32_shared_down_kernel,
        dim3(kTokens, kHidden),
        dim3(kNativeThreads),
        0u,
        stream,
        shared_activated_f32,
        g_state.shared_gate_scales,
        shared_down_bf16,
        g_state.routed_combined,
        shared_output_f32,
        combined_output_f32
    );
#endif
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 shared down/combine"
        )) {
        return false;
    }
    hipLaunchKernelGGL(
        convert_input_kernel,
        dim3(static_cast<uint32_t>(
            (kOutputElements + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0u,
        stream,
        shared_output_f32,
        g_state.shared_down_projection,
        kOutputElements
    );
    if (!check_q1024_early_f32_launch(
            "q1024 early-F32 shared debug endpoint"
        )) {
        return false;
    }
    hipLaunchKernelGGL(
        q1024_early_f32_output_residual_kernel,
        dim3(static_cast<uint32_t>(
            (kOutputElements + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0u,
        stream,
        residual_hidden_f32,
        combined_output_f32,
        output_f32
    );
    return check_q1024_early_f32_launch(
        "q1024 early-F32 output residual"
    );
}
#endif

bool launch_shared_pipeline(
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    hipStream_t stream
) {
#if QRT_TRITON_MOE_Q1024_EXACT_SHARED
    const uint16_t *gate_pointer = shared_gate_projection_bf16;
    const uint16_t *up_pointer = shared_up_projection_bf16;
    const uint16_t *input_pointer = g_state.input_bf16;
    uint16_t *activated_pointer = g_state.shared_activated;
    void *gate_up_arguments[6] = {
        &gate_pointer,
        &up_pointer,
        &input_pointer,
        &activated_pointer,
    };
    hipError_t status = launch_module(
        g_state.exact_shared_gate_up,
        gate_up_arguments,
        4u,
        stream
    );
    const uint16_t *gate_scalar_pointer = shared_gate_bf16;
    uint16_t *gate_logit_pointer = g_state.shared_gate_logits;
    void *gate_logit_arguments[5] = {
        &gate_scalar_pointer,
        &input_pointer,
        &gate_logit_pointer,
    };
    if (status == hipSuccess) {
        status = launch_module(
            g_state.exact_shared_gate_logit,
            gate_logit_arguments,
            3u,
            stream
        );
    }
    const uint16_t *down_pointer = shared_down_bf16;
    uint16_t *down_output_pointer = g_state.shared_down_projection;
    void *down_arguments[5] = {
        &down_pointer,
        &activated_pointer,
        &down_output_pointer,
    };
    if (status == hipSuccess) {
        status = launch_module(
            g_state.exact_shared_down,
            down_arguments,
            3u,
            stream
        );
    }
    if (status != hipSuccess) {
        set_error("launch(q1024 exact shared kernels)", status);
        return false;
    }
    hipLaunchKernelGGL(
        shared_gate_scale_kernel,
        dim3(static_cast<uint32_t>(
            (kTokens + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        g_state.shared_gate_logits,
        g_state.shared_gate_scales
    );
    status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("exact shared gate scale", status);
        return false;
    }
    return true;
#else
    if (!launch_matrix(
            &g_state.shared_gate_plan,
            shared_gate_bf16,
            g_state.input_bf16,
            g_state.shared_gate_logits,
            kSharedGateRows,
            kHidden,
            stream,
            "hipblasLtMatmul(shared_gate)"
        )) {
        return false;
    }
    hipLaunchKernelGGL(
        shared_gate_scale_kernel,
        dim3(static_cast<uint32_t>(
            (kTokens + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        g_state.shared_gate_logits,
        g_state.shared_gate_scales
    );
    hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("shared_gate_scale_kernel", status);
        return false;
    }
    if (!launch_matrix(
            &g_state.shared_projection_plan,
            shared_gate_projection_bf16,
            g_state.input_bf16,
            g_state.shared_gate_projection,
            kIntermediate,
            kHidden,
            stream,
            "hipblasLtMatmul(shared_gate_projection)"
        ) ||
        !launch_matrix(
            &g_state.shared_projection_plan,
            shared_up_projection_bf16,
            g_state.input_bf16,
            g_state.shared_up_projection,
            kIntermediate,
            kHidden,
            stream,
            "hipblasLtMatmul(shared_up_projection)"
        )) {
        return false;
    }
    hipLaunchKernelGGL(
        shared_activation_kernel,
        dim3(static_cast<uint32_t>(
            (kSharedProjectionElements + kNativeThreads - 1u) /
            kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        g_state.shared_gate_projection,
        g_state.shared_up_projection,
        g_state.shared_activated
    );
    status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("shared_activation_kernel", status);
        return false;
    }
    if (!launch_matrix(
            &g_state.shared_down_plan,
            shared_down_bf16,
            g_state.shared_activated,
            g_state.shared_down_projection,
            kHidden,
            kIntermediate,
            stream,
            "hipblasLtMatmul(shared_down_projection)"
        )) {
        return false;
    }
    return true;
#endif
}

bool q65536_vllm_bf16_residual_enabled() {
    const char *value = std::getenv(
        "QRT_QWEN36_Q65536_VLLM_BF16_RESIDUAL_NORM"
    );
    if (value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0) {
        return true;
    }
    value = std::getenv(
        "QRT_QWEN36_Q8192_VLLM_BF16_RESIDUAL_CARRIER"
    );
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0;
}

bool q65536_vllm_routed_bf16_endpoint_enabled() {
    const char *value = std::getenv(
        "QRT_QWEN36_Q65536_VLLM_ROUTED_BF16_ENDPOINT"
    );
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0;
}

bool q65536_vllm_route_sum_vt4_enabled() {
    const char *value = std::getenv(
        "QRT_QWEN36_Q65536_VLLM_ROUTE_SUM_VT4"
    );
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0;
}

bool q65536_vllm_route_sum_bf16_endpoint_enabled() {
    const char *value = std::getenv(
        "QRT_QWEN36_Q65536_VLLM_ROUTE_SUM_BF16_ENDPOINT"
    );
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0;
}

bool q8192_vllm_sorted_bf16_route_sum_enabled() {
    const char *value = std::getenv(
        "QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM"
    );
    return value != nullptr && value[0] != '\0' &&
        std::strcmp(value, "0") != 0;
}

bool launch_ordered_combine_residual(
    const float *residual_hidden_f32,
    float *output_f32,
    hipStream_t stream,
    bool vllm_bf16_residual
) {
    hipLaunchKernelGGL(
        shared_combine_residual_kernel,
        dim3(static_cast<uint32_t>(
            (kOutputElements + kNativeThreads - 1u) / kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        g_state.shared_down_projection,
        g_state.shared_gate_scales,
        residual_hidden_f32,
        g_state.routed_combined,
        output_f32,
        vllm_bf16_residual
    );
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("shared_combine_residual_kernel", status);
        return false;
    }
    return true;
}

bool launch_full_v3_fused_combine_residual(
    const float *residual_hidden_f32,
    float *output_f32,
    hipStream_t stream,
    bool vllm_bf16_residual
) {
    hipLaunchKernelGGL(
        full_v3_fused_combine_residual_kernel,
        dim3(static_cast<uint32_t>(
            (kOutputElements / kFusedCombineWidth + kNativeThreads - 1u) /
                kNativeThreads
        )),
        dim3(kNativeThreads),
        0,
        stream,
        g_state.route_outputs,
        g_state.topk_weights,
        g_state.topk_ids,
        g_state.shared_down_projection,
        g_state.shared_gate_scales,
        residual_hidden_f32,
        output_f32,
        vllm_bf16_residual,
        q65536_vllm_routed_bf16_endpoint_enabled(),
        q65536_vllm_route_sum_vt4_enabled(),
        q65536_vllm_route_sum_bf16_endpoint_enabled(),
        q8192_vllm_sorted_bf16_route_sum_enabled()
    );
    const hipError_t status = hipGetLastError();
    if (status != hipSuccess) {
        set_error("full_v3_fused_combine_residual_kernel", status);
        return false;
    }
    return true;
}

struct FullV3InFlightGuard {
    bool acquired = false;

    explicit FullV3InFlightGuard(bool wait_for_turn = false) {
        do {
            acquired = !g_full_v3_in_flight.test_and_set(
                std::memory_order_acquire
            );
            if (!acquired && wait_for_turn) {
                std::this_thread::yield();
            }
        } while (!acquired && wait_for_turn);
    }

    ~FullV3InFlightGuard() {
        if (acquired) {
            g_full_v3_in_flight.clear(std::memory_order_release);
        }
    }
};

void mark_full_v3_queue_idle() {
    for (FullV3EventSlot &slot : g_state.full_v3_slots) {
        slot.in_flight = false;
    }
    g_state.full_v3_owner_stream = nullptr;
    g_state.full_v3_owner_valid = false;
}

bool retire_completed_full_v3_slots() {
    bool any_in_flight = false;
    for (FullV3EventSlot &slot : g_state.full_v3_slots) {
        if (!slot.in_flight) {
            continue;
        }
        const hipError_t status = hipEventQuery(slot.caller_done);
        if (status == hipSuccess) {
            slot.in_flight = false;
        } else if (status == hipErrorNotReady) {
            any_in_flight = true;
        } else {
            set_error("full-v3 caller completion query", status);
            return false;
        }
    }
    if (!any_in_flight) {
        g_state.full_v3_owner_stream = nullptr;
        g_state.full_v3_owner_valid = false;
    }
    return true;
}

FullV3EventSlot *acquire_full_v3_event_slot(hipStream_t caller_stream) {
    if (g_state.full_v3_poisoned) {
        set_error_text("full selected-MoE v3 execution state is poisoned");
        return nullptr;
    }
    if (!retire_completed_full_v3_slots()) {
        return nullptr;
    }
    if (g_state.full_v3_owner_valid &&
        g_state.full_v3_owner_stream != caller_stream) {
        set_error_text(
            "full selected-MoE v3 async work is pending on another stream"
        );
        return nullptr;
    }
    for (size_t offset = 0u; offset < kFullV3EventSlots; ++offset) {
        const size_t index =
            (g_state.full_v3_next_slot + offset) % kFullV3EventSlots;
        FullV3EventSlot &slot = g_state.full_v3_slots[index];
        if (slot.in_flight) {
            continue;
        }
        // HIP explicitly makes re-recording an event while it is still being
        // recorded undefined.  A slot becomes reusable only after caller_done
        // proves that input_ready, shared_done, and the combine have completed.
        slot.in_flight = true;
        g_state.full_v3_next_slot = (index + 1u) % kFullV3EventSlots;
        g_state.full_v3_owner_stream = caller_stream;
        g_state.full_v3_owner_valid = true;
        return &slot;
    }
    // A long-context prefill can enqueue more independent tile/layer calls
    // than the fixed event ring before the GPU reaches the first completion.
    // Preserve the asynchronous pipeline by applying bounded backpressure to
    // the oldest slot instead of rejecting an otherwise valid launch.  The
    // caller_done event is downstream of input_ready, shared_done, and the
    // final combine, so successful synchronization makes every event in this
    // slot safe to re-record.
    FullV3EventSlot &oldest =
        g_state.full_v3_slots[g_state.full_v3_next_slot];
    if (!oldest.in_flight || oldest.caller_done == nullptr) {
        set_error_text(
            "full selected-MoE v3 async event queue bookkeeping is invalid"
        );
        return nullptr;
    }
    const hipError_t status = hipEventSynchronize(oldest.caller_done);
    if (status != hipSuccess) {
        set_error("full-v3 event queue backpressure", status);
        return nullptr;
    }
    const size_t acquired_index = g_state.full_v3_next_slot;
    g_state.full_v3_next_slot =
        (acquired_index + 1u) % kFullV3EventSlots;
    g_state.full_v3_owner_stream = caller_stream;
    g_state.full_v3_owner_valid = true;
    return &oldest;
}

bool drain_full_v3_streams(hipStream_t caller_stream) {
    hipError_t shared_status = hipSuccess;
    if (g_state.full_v3_shared_stream != nullptr) {
        shared_status = hipStreamSynchronize(g_state.full_v3_shared_stream);
    }
    const hipError_t caller_status = hipStreamSynchronize(caller_stream);
    if (shared_status != hipSuccess || caller_status != hipSuccess) {
        g_state.full_v3_poisoned = true;
        if (shared_status != hipSuccess) {
            set_error("full-v3 shared stream drain", shared_status);
        } else {
            set_error("full-v3 caller stream drain", caller_status);
        }
        return false;
    }
    mark_full_v3_queue_idle();
    return true;
}

int launch_full_v3_impl(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    void *stream_pointer,
    bool synchronize
) {
    FullV3InFlightGuard in_flight;
    if (!in_flight.acquired) {
        // The losing caller must not mutate the shared error buffer while the
        // accepted call is using the provider state.
        return 0;
    }
    if (!g_state.prepared || post_attention_f32 == nullptr ||
        residual_hidden_f32 == nullptr || router_bf16 == nullptr ||
        routed_gate_up_bf16 == nullptr || routed_down_bf16 == nullptr ||
        shared_gate_bf16 == nullptr ||
        shared_gate_projection_bf16 == nullptr ||
        shared_up_projection_bf16 == nullptr ||
        shared_down_bf16 == nullptr || output_f32 == nullptr) {
        set_error_text("full selected-MoE v3 launch received an invalid surface");
        return 0;
    }
    if (!ensure_full_v3_execution_state()) {
        return 0;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    FullV3EventSlot *slot = acquire_full_v3_event_slot(stream);
    if (slot == nullptr) {
        return 0;
    }
    if (!launch_input_conversion(post_attention_f32, stream)) {
        drain_full_v3_streams(stream);
        return 0;
    }
    hipError_t status = hipEventRecord(slot->input_ready, stream);
    if (status == hipSuccess) {
        status = hipStreamWaitEvent(
            g_state.full_v3_shared_stream,
            slot->input_ready,
            0u
        );
    }
    if (status != hipSuccess) {
        set_error("full-v3 input-ready dependency", status);
        drain_full_v3_streams(stream);
        return 0;
    }

    if (!launch_shared_pipeline(
            shared_gate_bf16,
            shared_gate_projection_bf16,
            shared_up_projection_bf16,
            shared_down_bf16,
            g_state.full_v3_shared_stream
        )) {
        drain_full_v3_streams(stream);
        return 0;
    }
    status = hipEventRecord(
        slot->shared_done,
        g_state.full_v3_shared_stream
    );
    if (status != hipSuccess) {
        set_error("full-v3 shared-done record", status);
        drain_full_v3_streams(stream);
        return 0;
    }

    if (!launch_router(post_attention_f32, router_bf16, stream) ||
        !launch_routed_matrices_after_input_conversion(
            routed_gate_up_bf16,
            routed_down_bf16,
            stream
        )) {
        drain_full_v3_streams(stream);
        return 0;
    }
#if !QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE && \
    !QRT_TRITON_MOE_Q1024_EXACT_ROUTED && \
    !QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED && \
    !QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED && \
    !QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
    if (!launch_route_combine(g_state.routed_combined, stream)) {
        drain_full_v3_streams(stream);
        return 0;
    }
#endif
    status = hipStreamWaitEvent(stream, slot->shared_done, 0u);
    if (status != hipSuccess) {
        set_error("full-v3 shared-done dependency", status);
        drain_full_v3_streams(stream);
        return 0;
    }
#if QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE
    if (!launch_full_v3_fused_combine_residual(
#else
    if (!launch_ordered_combine_residual(
#endif
            residual_hidden_f32,
            output_f32,
            stream,
            q65536_vllm_bf16_residual_enabled()
        )) {
        drain_full_v3_streams(stream);
        return 0;
    }
    status = hipEventRecord(slot->caller_done, stream);
    if (status != hipSuccess) {
        set_error("full-v3 caller completion record", status);
        drain_full_v3_streams(stream);
        return 0;
    }

    if (synchronize) {
        status = hipStreamSynchronize(stream);
        if (status != hipSuccess) {
            set_error("full-v3 completion", status);
            drain_full_v3_streams(stream);
            return 0;
        }
        // All v3 work is restricted to this caller stream while it is pending.
        // Its shared-done waits also prove that the auxiliary stream is done.
        mark_full_v3_queue_idle();
    }

    g_state.error[0] = '\0';
    return 1;
}

}  // namespace

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_prepare(const char *kernel_dir) {
    FullV3InFlightGuard lifecycle_guard(true);
    if (kernel_dir == nullptr || kernel_dir[0] == '\0') {
        set_error_text("selected-MoE kernel directory is empty");
        return 0;
    }
    if (g_state.prepared && !g_state.full_v3_poisoned &&
        std::strcmp(g_state.kernel_dir, kernel_dir) == 0) {
        return 1;
    }
    if (!release_state()) {
        return 0;
    }
    std::snprintf(g_state.kernel_dir, sizeof(g_state.kernel_dir), "%s", kernel_dir);
    if (!load_kernel(kernel_dir, "route_count", "_route_count_kernel", kRoutePrograms, kRouteThreads, 0u, &g_state.count) ||
        !load_kernel(kernel_dir, "route_prefix_by_program", "_route_prefix_by_program_kernel", 256u, kRouteThreads, 0u, &g_state.prefix) ||
        !load_kernel(kernel_dir, "route_padded_prefix", "_route_padded_prefix_kernel", 1u, kRouteThreads, 0u, &g_state.padded_prefix) ||
        !load_kernel(kernel_dir, "route_scatter", "_route_scatter_kernel", kRouteScatterPrograms, kRouteThreads, 0u, &g_state.scatter) ||
        !load_kernel(kernel_dir, "gate_up_silu", "_gate_up_silu_kernel", kMaxRouteBlocks * kGateUpGridN, kGateThreads, kGateSharedBytes, &g_state.gate_up) ||
        !load_kernel(kernel_dir, "down", "_down_kernel", kMaxRouteBlocks * kDownGridN, kDownThreads, kDownSharedBytes, &g_state.down) ||
#if QRT_TRITON_MOE_Q1024_GROUPED_BF16_ENDPOINTS
        !load_named_kernel(
            kernel_dir,
            "q1024_selected_moe_down_bf16_endpoint.hsaco",
            "_down_kernel",
            kMaxRouteBlocks * kDownGridN,
            kDownThreads,
            kDownSharedBytes,
            &g_state.grouped_bf16_down
        ) ||
#endif
#if QRT_TRITON_MOE_Q1024_EXACT_ROUTED || \
    QRT_TRITON_MOE_Q1024_EXACT_GATE_GROUPED_DOWN || \
    QRT_TRITON_MOE_Q1024_GROUPED_GATE_EXACT_DOWN
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_gate_up_silu_w8.hsaco",
            "_gate_up_silu_kernel",
            kTokens * kTopK * kIntermediate,
            256u,
            32u,
            &g_state.exact_gate_up
        ) ||
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_down_sum.hsaco",
            "_down_sum_kernel",
            kTokens * kHidden,
            32u,
            0u,
            &g_state.exact_down
        ) ||
#endif
#if QRT_TRITON_MOE_Q1024_PACKED_EXACT_ROUTED
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_packed_gate_up_silu_w8.hsaco",
            "_packed_gate_up_silu_kernel",
            kTokens * kTopK * (kIntermediate / 8u),
            256u,
            32u,
            &g_state.packed_exact_gate_up
        ) ||
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_packed_down_sum.hsaco",
            "_packed_down_sum_kernel",
            kTokens * (kHidden / 8u),
            32u,
            0u,
            &g_state.packed_exact_down
        ) ||
#endif
#if QRT_TRITON_MOE_Q1024_SORTED_PACKED_EXACT_ROUTED
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_sorted_packed_gate_up_silu_w8.hsaco",
            "_sorted_packed_gate_up_silu_kernel",
            kMaxSortedExactRouteGroups * kIntermediate,
            256u,
            32u,
            &g_state.sorted_packed_exact_gate_up
        ) ||
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_sorted_packed_down.hsaco",
            "_sorted_packed_down_kernel",
            kMaxSortedExactRouteGroups * kHidden,
            32u,
            0u,
            &g_state.sorted_packed_exact_down
        ) ||
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_sorted_packed_route_combine.hsaco",
            "_sorted_packed_route_combine_kernel",
            kTokens * kHidden,
            32u,
            0u,
            &g_state.sorted_packed_exact_combine
        ) ||
#endif
#if QRT_TRITON_MOE_Q1024_EXACT_SHARED
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_shared_gate_up_silu.hsaco",
            "_shared_gate_up_silu_kernel",
            kTokens * kIntermediate,
            128u,
            16u,
            &g_state.exact_shared_gate_up
        ) ||
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_shared_gate_logit.hsaco",
            "_shared_gate_logit_kernel",
            kTokens,
            128u,
            16u,
            &g_state.exact_shared_gate_logit
        ) ||
        !load_named_kernel(
            kernel_dir,
            "q1024_triton_0626_exact_shared_down.hsaco",
            "_shared_down_kernel",
            kTokens * kHidden,
            32u,
            0u,
            &g_state.exact_shared_down
        ) ||
#endif
        !allocate(&g_state.input_bf16, kInputElements * sizeof(uint16_t), "hipMalloc(input_bf16)") ||
        !allocate_optional_transposed_router() ||
        !allocate_optional_hipblaslt_router_logits() ||
#if QRT_TRITON_MOE_ROCBLAS_ROUTER
        (q8192_hipblaslt_bf16_router_requested() ? false : !allocate(
            &g_state.router_logits_bf16,
            static_cast<size_t>(kTokens) * kExperts * sizeof(uint16_t),
            "hipMalloc(router_logits_bf16)"
        )) ||
#endif
        !allocate(&g_state.topk_ids, static_cast<size_t>(kRoutes) * sizeof(int32_t), "hipMalloc(topk_ids)") ||
        !allocate(&g_state.topk_weights, static_cast<size_t>(kRoutes) * sizeof(float), "hipMalloc(topk_weights)") ||
        !allocate(&g_state.counts, kCountElements * sizeof(int32_t), "hipMalloc(counts)") ||
        !allocate(&g_state.cumsum, static_cast<size_t>(kExperts + 1u) * sizeof(int32_t), "hipMalloc(cumsum)") ||
        !allocate(&g_state.total_post_pad, sizeof(int32_t), "hipMalloc(total_post_pad)") ||
        !allocate(&g_state.sorted_routes, static_cast<size_t>(kMaxSortedRoutes) * sizeof(int32_t), "hipMalloc(sorted_routes)") ||
        !allocate(&g_state.block_experts, static_cast<size_t>(kMaxRouteBlocks) * sizeof(int32_t), "hipMalloc(block_experts)") ||
        !allocate(&g_state.activated, kActivatedElements * sizeof(uint16_t), "hipMalloc(activated)") ||
        !allocate(&g_state.route_outputs, kRouteOutputElements * sizeof(float), "hipMalloc(route_outputs_f32)") ||
        !allocate(&g_state.routed_combined, kOutputElements * sizeof(float), "hipMalloc(routed_combined_f32)") ||
        !allocate(&g_state.shared_gate_logits, static_cast<size_t>(kTokens) * sizeof(uint16_t), "hipMalloc(shared_gate_logits)") ||
        !allocate(&g_state.shared_gate_projection, kSharedProjectionElements * sizeof(uint16_t), "hipMalloc(shared_gate_projection)") ||
        !allocate(&g_state.shared_up_projection, kSharedProjectionElements * sizeof(uint16_t), "hipMalloc(shared_up_projection)") ||
        !allocate(&g_state.shared_activated, kSharedProjectionElements * sizeof(uint16_t), "hipMalloc(shared_activated)") ||
        !allocate(&g_state.shared_down_projection, kOutputElements * sizeof(uint16_t), "hipMalloc(shared_down_projection)") ||
        !allocate(&g_state.shared_gate_scales, static_cast<size_t>(kTokens) * sizeof(float), "hipMalloc(shared_gate_scales)") ||
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
        !allocate(
            &g_state.early_f32_activated,
            kActivatedElements * sizeof(float),
            "hipMalloc(early_f32_activated)"
        ) ||
#endif
        !ensure_optional_hipblaslt_router_plan() ||
        !ensure_matrix_plan(
            &g_state.shared_gate_plan,
            kSharedGateRows,
            kHidden
        ) ||
        !ensure_matrix_plan(
            &g_state.shared_projection_plan,
            kIntermediate,
            kHidden
        ) ||
        !ensure_matrix_plan(
            &g_state.shared_down_plan,
            kHidden,
            kIntermediate
        )) {
        const std::string prepare_error = g_state.error;
        (void)release_state();
        set_error_text(prepare_error.c_str());
        return 0;
    }
#if QRT_TRITON_MOE_ROCBLAS_ROUTER
    rocblas_status router_status =
        rocblas_create_handle(&g_state.router_handle);
    if (router_status == rocblas_status_success) {
        router_status = rocblas_set_pointer_mode(
            g_state.router_handle,
            rocblas_pointer_mode_host
        );
    }
    if (router_status != rocblas_status_success) {
        std::snprintf(
            g_state.error,
            sizeof(g_state.error),
            "rocBLAS router prepare: status %d",
            static_cast<int>(router_status)
        );
        const std::string prepare_error = g_state.error;
        (void)release_state();
        set_error_text(prepare_error.c_str());
        return 0;
    }
#endif
    g_state.prepared = true;
    g_state.error[0] = '\0';
    return 1;
}

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_launch(
    const float *post_attention_f32,
    const uint16_t *gate_up_bf16,
    const uint16_t *down_bf16,
    const uint32_t *topk_ids_host,
    const float *topk_weights_host,
    float *output_f32,
    void *stream_pointer
) {
    if (!g_state.prepared || post_attention_f32 == nullptr ||
        gate_up_bf16 == nullptr || down_bf16 == nullptr ||
        topk_ids_host == nullptr || topk_weights_host == nullptr ||
        output_f32 == nullptr) {
        set_error_text("selected-MoE launch received an invalid surface");
        return 0;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    hipError_t status = hipMemcpyAsync(
        g_state.topk_ids,
        topk_ids_host,
        static_cast<size_t>(kRoutes) * sizeof(uint32_t),
        hipMemcpyHostToDevice,
        stream
    );
    if (status == hipSuccess) {
        status = hipMemcpyAsync(
            g_state.topk_weights,
            topk_weights_host,
            static_cast<size_t>(kRoutes) * sizeof(float),
            hipMemcpyHostToDevice,
            stream
        );
    }
    if (status != hipSuccess) {
        set_error("hipMemcpyAsync(route metadata)", status);
        return 0;
    }

    if (!launch_routed_pipeline(
            post_attention_f32,
            gate_up_bf16,
            down_bf16,
            output_f32,
            stream
        )) {
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_launch_full_v2_async(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    void *stream_pointer
) {
    if (!g_state.prepared || post_attention_f32 == nullptr ||
        residual_hidden_f32 == nullptr || router_bf16 == nullptr ||
        routed_gate_up_bf16 == nullptr || routed_down_bf16 == nullptr ||
        shared_gate_bf16 == nullptr ||
        shared_gate_projection_bf16 == nullptr ||
        shared_up_projection_bf16 == nullptr ||
        shared_down_bf16 == nullptr || output_f32 == nullptr) {
        set_error_text("full selected-MoE v2 launch received an invalid surface");
        return 0;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    if (!zero_request_scratch(stream) ||
        !launch_router(post_attention_f32, router_bf16, stream) ||
        !launch_routed_pipeline(
            post_attention_f32,
            routed_gate_up_bf16,
            routed_down_bf16,
            g_state.routed_combined,
            stream
        )) {
        return 0;
    }
    if (!launch_shared_pipeline(
            shared_gate_bf16,
            shared_gate_projection_bf16,
            shared_up_projection_bf16,
            shared_down_bf16,
            stream
        )) {
        return 0;
    }
    if (!launch_ordered_combine_residual(
            residual_hidden_f32,
            output_f32,
            stream,
            q65536_vllm_bf16_residual_enabled()
        )) {
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_launch_full_v2(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    void *stream_pointer
) {
    if (qrt_triton_moe_q8192_launch_full_v2_async(
            post_attention_f32,
            residual_hidden_f32,
            router_bf16,
            routed_gate_up_bf16,
            routed_down_bf16,
            shared_gate_bf16,
            shared_gate_projection_bf16,
            shared_up_projection_bf16,
            shared_down_bf16,
            output_f32,
            stream_pointer
        ) == 0) {
        return 0;
    }

    const hipError_t status = hipStreamSynchronize(
        static_cast<hipStream_t>(stream_pointer)
    );
    if (status != hipSuccess) {
        set_error("shared_combine_residual_kernel", status);
        return 0;
    }

    g_state.error[0] = '\0';
    return 1;
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32
QRT_TRITON_MOE_EXPORT int
qrt_triton_moe_q8192_launch_full_q1024_early_f32_v1(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    void *stream_pointer
) {
    if (!g_state.prepared || post_attention_f32 == nullptr ||
        residual_hidden_f32 == nullptr || router_bf16 == nullptr ||
        routed_gate_up_bf16 == nullptr || routed_down_bf16 == nullptr ||
        shared_gate_bf16 == nullptr ||
        shared_gate_projection_bf16 == nullptr ||
        shared_up_projection_bf16 == nullptr ||
        shared_down_bf16 == nullptr || output_f32 == nullptr) {
        set_error_text(
            "q1024 early-F32 full launch received an invalid surface"
        );
        return 0;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    if (!launch_q1024_early_f32_full(
            post_attention_f32,
            residual_hidden_f32,
            router_bf16,
            routed_gate_up_bf16,
            routed_down_bf16,
            shared_gate_bf16,
            shared_gate_projection_bf16,
            shared_up_projection_bf16,
            shared_down_bf16,
            output_f32,
            stream
        )) {
        return 0;
    }
    const hipError_t status = hipStreamSynchronize(stream);
    if (status != hipSuccess) {
        set_error("q1024 early-F32 completion", status);
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}
#endif

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_launch_full_v3_async(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    void *stream_pointer
) {
    return launch_full_v3_impl(
        post_attention_f32,
        residual_hidden_f32,
        router_bf16,
        routed_gate_up_bf16,
        routed_down_bf16,
        shared_gate_bf16,
        shared_gate_projection_bf16,
        shared_up_projection_bf16,
        shared_down_bf16,
        output_f32,
        stream_pointer,
        false
    );
}

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_launch_full_v3(
    const float *post_attention_f32,
    const float *residual_hidden_f32,
    const uint16_t *router_bf16,
    const uint16_t *routed_gate_up_bf16,
    const uint16_t *routed_down_bf16,
    const uint16_t *shared_gate_bf16,
    const uint16_t *shared_gate_projection_bf16,
    const uint16_t *shared_up_projection_bf16,
    const uint16_t *shared_down_bf16,
    float *output_f32,
    void *stream_pointer
) {
    return launch_full_v3_impl(
        post_attention_f32,
        residual_hidden_f32,
        router_bf16,
        routed_gate_up_bf16,
        routed_down_bf16,
        shared_gate_bf16,
        shared_gate_projection_bf16,
        shared_up_projection_bf16,
        shared_down_bf16,
        output_f32,
        stream_pointer,
        true
    );
}

QRT_TRITON_MOE_EXPORT const char *qrt_triton_moe_q8192_last_error() {
    return g_state.error;
}

QRT_TRITON_MOE_EXPORT uint32_t qrt_triton_moe_q8192_backend_mask() {
    return
        (QRT_TRITON_MOE_NATIVE_WMMA_GATE ? UINT32_C(1) : UINT32_C(0)) |
        (QRT_TRITON_MOE_NATIVE_WMMA_DOWN ? UINT32_C(2) : UINT32_C(0)) |
        (QRT_TRITON_MOE_TRANSPOSED_ROUTER ? UINT32_C(4) : UINT32_C(0)) |
        (QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE ?
            UINT32_C(8) : UINT32_C(0)) |
        (QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE ?
            UINT32_C(16) : UINT32_C(0));
}

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_launch_router_debug(
    const float *post_attention_f32,
    const uint16_t *router_bf16,
    void *stream_pointer
) {
    if (!g_state.prepared || post_attention_f32 == nullptr ||
        router_bf16 == nullptr) {
        set_error_text("router debug launch received an invalid surface");
        return 0;
    }
    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    if (!launch_router(post_attention_f32, router_bf16, stream)) {
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}

QRT_TRITON_MOE_EXPORT int qrt_triton_moe_q8192_copy_topk_debug(
    uint32_t *topk_ids_host,
    float *topk_weights_host
) {
    if (!g_state.prepared || topk_ids_host == nullptr ||
        topk_weights_host == nullptr) {
        return 0;
    }
    hipError_t status = hipMemcpy(
        topk_ids_host,
        g_state.topk_ids,
        static_cast<size_t>(kRoutes) * sizeof(uint32_t),
        hipMemcpyDeviceToHost
    );
    if (status == hipSuccess) {
        status = hipMemcpy(
            topk_weights_host,
            g_state.topk_weights,
            static_cast<size_t>(kRoutes) * sizeof(float),
            hipMemcpyDeviceToHost
        );
    }
    return status == hipSuccess ? 1 : 0;
}

QRT_TRITON_MOE_EXPORT int
qrt_triton_moe_q8192_copy_token0_stage_debug(
    float *routed_combined_host,
    uint16_t *shared_down_projection_host,
    float *shared_gate_scale_host
) {
    if (!g_state.prepared || routed_combined_host == nullptr ||
        shared_down_projection_host == nullptr ||
        shared_gate_scale_host == nullptr) {
        return 0;
    }
    hipError_t status = hipMemcpy(
        routed_combined_host,
        g_state.routed_combined,
        static_cast<size_t>(kHidden) * sizeof(float),
        hipMemcpyDeviceToHost
    );
    if (status == hipSuccess) {
        status = hipMemcpy(
            shared_down_projection_host,
            g_state.shared_down_projection,
            static_cast<size_t>(kHidden) * sizeof(uint16_t),
            hipMemcpyDeviceToHost
        );
    }
    if (status == hipSuccess) {
        status = hipMemcpy(
            shared_gate_scale_host,
            g_state.shared_gate_scales,
            sizeof(float),
            hipMemcpyDeviceToHost
        );
    }
    return status == hipSuccess ? 1 : 0;
}

QRT_TRITON_MOE_EXPORT int
qrt_triton_moe_q8192_copy_token_stage_debug(
    uint32_t token_index,
    float *routed_combined_host,
    uint16_t *shared_down_projection_host,
    float *shared_gate_scale_host
) {
    if (!g_state.prepared || token_index >= kTokens ||
        routed_combined_host == nullptr ||
        shared_down_projection_host == nullptr ||
        shared_gate_scale_host == nullptr) {
        return 0;
    }
    const size_t hidden_offset =
        static_cast<size_t>(token_index) * kHidden;
    hipError_t status = hipSuccess;
#if QRT_TRITON_MOE_FULL_V3_FUSED_COMBINE
    std::array<float, kTopK * kHidden> route_outputs{};
    std::array<float, kTopK> route_weights{};
    const size_t route_base = static_cast<size_t>(token_index) * kTopK;
    status = hipMemcpy(
        route_outputs.data(),
        g_state.route_outputs + route_base * kHidden,
        route_outputs.size() * sizeof(float),
        hipMemcpyDeviceToHost
    );
    if (status == hipSuccess) {
        status = hipMemcpy(
            route_weights.data(),
            g_state.topk_weights + route_base,
            route_weights.size() * sizeof(float),
            hipMemcpyDeviceToHost
        );
    }
    if (status == hipSuccess) {
        for (size_t column = 0u; column < kHidden; ++column) {
            float sum = 0.0f;
            for (size_t route = 0u; route < kTopK; ++route) {
                sum += route_weights[route] *
                    route_outputs[route * kHidden + column];
            }
            routed_combined_host[column] = sum;
        }
    }
#else
    status = hipMemcpy(
        routed_combined_host,
        g_state.routed_combined + hidden_offset,
        static_cast<size_t>(kHidden) * sizeof(float),
        hipMemcpyDeviceToHost
    );
#endif
    if (status == hipSuccess) {
        status = hipMemcpy(
            shared_down_projection_host,
            g_state.shared_down_projection + hidden_offset,
            static_cast<size_t>(kHidden) * sizeof(uint16_t),
            hipMemcpyDeviceToHost
        );
    }
    if (status == hipSuccess) {
        status = hipMemcpy(
            shared_gate_scale_host,
            g_state.shared_gate_scales + token_index,
            sizeof(float),
            hipMemcpyDeviceToHost
        );
    }
    return status == hipSuccess ? 1 : 0;
}

#if QRT_TRITON_MOE_Q1024_EARLY_F32
QRT_TRITON_MOE_EXPORT int
qrt_triton_moe_q8192_copy_q1024_early_f32_token0_activation_debug(
    float *activated_host
) {
    if (!g_state.prepared || activated_host == nullptr) {
        return 0;
    }
    const hipError_t status = hipMemcpy(
        activated_host,
        g_state.activated,
        static_cast<size_t>(kTopK) * kIntermediate * sizeof(float),
        hipMemcpyDeviceToHost
    );
    return status == hipSuccess ? 1 : 0;
}
#endif

QRT_TRITON_MOE_EXPORT uint64_t qrt_triton_moe_q8192_scratch_bytes() {
    uint64_t bytes =
        kInputElements * sizeof(uint16_t) +
        static_cast<size_t>(kRoutes) * (sizeof(int32_t) + sizeof(float)) +
        kCountElements * sizeof(int32_t) +
        static_cast<size_t>(kExperts + 1u) * sizeof(int32_t) +
        sizeof(int32_t) +
        static_cast<size_t>(kMaxSortedRoutes) * sizeof(int32_t) +
        static_cast<size_t>(kMaxRouteBlocks) * sizeof(int32_t) +
        kActivatedElements * sizeof(uint16_t) +
        kRouteOutputElements * sizeof(float) +
        kOutputElements * sizeof(float) +
        static_cast<size_t>(kTokens) * sizeof(uint16_t) +
        3u * kSharedProjectionElements * sizeof(uint16_t) +
        kOutputElements * sizeof(uint16_t) +
        static_cast<size_t>(kTokens) * sizeof(float) +
        g_state.matrix_workspace_bytes;
#if QRT_TRITON_MOE_TRANSPOSED_ROUTER
    bytes += kRouterWeightElements * sizeof(uint16_t);
#endif
#if QRT_TRITON_MOE_Q1024_EARLY_F32_SORTED_TILE
    bytes += kActivatedElements * sizeof(float);
#endif
    return bytes;
}

QRT_TRITON_MOE_EXPORT void qrt_triton_moe_q8192_release() {
    FullV3InFlightGuard lifecycle_guard(true);
    (void)release_state();
}
