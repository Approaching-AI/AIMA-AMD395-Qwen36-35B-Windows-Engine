#include <hip/hip_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

#if defined(_WIN32)
#define QRT_FLA_GDN_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_FLA_GDN_EXPORT extern "C"
#endif

namespace {

constexpr uint32_t kQkHeads = 16u;
constexpr uint32_t kValueHeads = 32u;
constexpr uint32_t kKeyDim = 128u;
constexpr uint32_t kValueDim = 128u;
constexpr uint32_t kChunk = 64u;
constexpr uint32_t kQkPrepRows = 32u;
constexpr uint32_t kStateValueTiles = 8u;
constexpr uint32_t kOutputValueTiles = 4u;
constexpr uint32_t kGateRows = 64u;
constexpr uint32_t kQkvRows = 8192u;
constexpr uint32_t kValueFeatures = kValueHeads * kValueDim;
constexpr uint32_t kStateElements = kValueFeatures * kKeyDim;
constexpr int32_t kQ65536SegmentTokens = 1024;

constexpr int32_t kSmokeTokens = 64;
constexpr int32_t kQ8192Tokens = 8192;
constexpr int32_t kQ16384Tokens = 16384;
constexpr int32_t kQ17408Tokens = 17408;
constexpr int32_t kQ32768Tokens = 32768;
constexpr int32_t kQ65536Tokens = 65536;

constexpr uint64_t kCompactQkvBytesPerToken = 16384u;
constexpr uint64_t kGateAndBetaBytesPerToken = 192u;
constexpr uint64_t kAOrWBytesPerToken = 8192u;
constexpr uint64_t kAiOrVNewBytesPerToken = 8192u;
constexpr uint64_t kChunkStateBytesPerToken = 16384u;
constexpr uint64_t kScratchBytesPerToken =
    kCompactQkvBytesPerToken +
    kGateAndBetaBytesPerToken +
    kAOrWBytesPerToken +
    kAiOrVNewBytesPerToken +
    kChunkStateBytesPerToken;

enum class KernelIndex : size_t {
    kQkL2Norm = 0u,
    kVBetaCopy,
    kGateCumsum,
    kScaledDotKkt,
    kSolveTril64,
    kRecomputeWU,
    kChunkState,
    kChunkOutput,
    kCount,
};

struct KernelSpec {
    const char *file;
    const char *symbol;
    uint32_t threads;
    uint32_t dynamic_shared_bytes;
};

constexpr std::array<KernelSpec, static_cast<size_t>(KernelIndex::kCount)>
    kKernelSpecs{{
        {
            "q8192_fla_chunk_gdn_qk_l2norm.hsaco",
            "_fla_qk_l2norm_from_native_f32_kernel",
            128u,
            512u,
        },
        {
            "q8192_fla_chunk_gdn_v_beta_copy.hsaco",
            "_fla_v_beta_copy_from_native_f32_kernel",
            128u,
            0u,
        },
        {
            "q8192_fla_chunk_gdn_gate_cumsum.hsaco",
            "_fla_chunk_gate_cumsum_kernel",
            64u,
            8u,
        },
        {
            "q8192_fla_chunk_gdn_scaled_dot_kkt.hsaco",
            "_fla_chunk_scaled_dot_kkt_kernel",
            256u,
            8192u,
        },
        {
            "q8192_fla_chunk_gdn_solve_tril_64.hsaco",
            "_fla_solve_tril_64_kernel",
            256u,
            4096u,
        },
        {
            "q8192_fla_chunk_gdn_recompute_w_u.hsaco",
            "_fla_recompute_w_u_kernel",
            256u,
            8192u,
        },
        {
            "q8192_fla_chunk_gdn_chunk_state.hsaco",
            "_fla_chunk_state_kernel",
            128u,
            8192u,
        },
        {
            "q8192_fla_chunk_gdn_chunk_output.hsaco",
            "_fla_chunk_output_kernel",
            256u,
            8192u,
        },
    }};

struct ProviderState {
    std::array<hipModule_t, static_cast<size_t>(KernelIndex::kCount)> modules{};
    std::array<hipFunction_t, static_cast<size_t>(KernelIndex::kCount)>
        functions{};
    uint16_t *compact_qkv = nullptr;
    float *gate_and_beta = nullptr;
    void *a_or_w = nullptr;
    void *ai_or_v_new = nullptr;
    uint16_t *chunk_state = nullptr;
    int32_t scratch_tokens = 0;
    bool prepared = false;
    char kernel_dir[1024]{};
    char error[768]{};
};

ProviderState g_state;

size_t kernel_slot(KernelIndex index) {
    return static_cast<size_t>(index);
}

void set_error_text(const char *message) {
    std::snprintf(
        g_state.error,
        sizeof(g_state.error),
        "%s",
        message == nullptr ? "unknown error" : message
    );
}

void set_error(const char *stage, hipError_t status) {
    std::snprintf(
        g_state.error,
        sizeof(g_state.error),
        "%s failed: hip_status=%d hip_error=%s",
        stage,
        static_cast<int>(status),
        hipGetErrorString(status)
    );
}

void release_scratch() {
    if (g_state.chunk_state != nullptr) {
        (void)hipFree(g_state.chunk_state);
    }
    if (g_state.ai_or_v_new != nullptr) {
        (void)hipFree(g_state.ai_or_v_new);
    }
    if (g_state.a_or_w != nullptr) {
        (void)hipFree(g_state.a_or_w);
    }
    if (g_state.gate_and_beta != nullptr) {
        (void)hipFree(g_state.gate_and_beta);
    }
    if (g_state.compact_qkv != nullptr) {
        (void)hipFree(g_state.compact_qkv);
    }
    g_state.compact_qkv = nullptr;
    g_state.gate_and_beta = nullptr;
    g_state.a_or_w = nullptr;
    g_state.ai_or_v_new = nullptr;
    g_state.chunk_state = nullptr;
    g_state.scratch_tokens = 0;
}

void release_state() {
    release_scratch();
    for (size_t index = 0u; index < g_state.modules.size(); ++index) {
        if (g_state.modules[index] != nullptr) {
            (void)hipModuleUnload(g_state.modules[index]);
        }
    }
    g_state = ProviderState{};
}

bool checked_bytes(int32_t tokens, uint64_t bytes_per_token, size_t *bytes) {
    if (tokens <= 0 || bytes == nullptr ||
        static_cast<uint64_t>(tokens) >
            static_cast<uint64_t>(std::numeric_limits<size_t>::max()) /
                bytes_per_token) {
        return false;
    }
    *bytes = static_cast<size_t>(
        static_cast<uint64_t>(tokens) * bytes_per_token
    );
    return true;
}

bool ensure_scratch(int32_t tokens) {
    if (g_state.scratch_tokens == tokens &&
        g_state.compact_qkv != nullptr &&
        g_state.gate_and_beta != nullptr &&
        g_state.a_or_w != nullptr &&
        g_state.ai_or_v_new != nullptr &&
        g_state.chunk_state != nullptr) {
        return true;
    }
    size_t compact_qkv_bytes = 0u;
    size_t gate_and_beta_bytes = 0u;
    size_t a_or_w_bytes = 0u;
    size_t ai_or_v_new_bytes = 0u;
    size_t chunk_state_bytes = 0u;
    if (!checked_bytes(
            tokens,
            kCompactQkvBytesPerToken,
            &compact_qkv_bytes
        ) ||
        !checked_bytes(
            tokens,
            kGateAndBetaBytesPerToken,
            &gate_and_beta_bytes
        ) ||
        !checked_bytes(tokens, kAOrWBytesPerToken, &a_or_w_bytes) ||
        !checked_bytes(tokens, kAiOrVNewBytesPerToken, &ai_or_v_new_bytes) ||
        !checked_bytes(
            tokens,
            kChunkStateBytesPerToken,
            &chunk_state_bytes
        )) {
        set_error_text("FLA chunk-GDN scratch size overflow");
        return false;
    }

    release_scratch();
    hipError_t status = hipMalloc(
        reinterpret_cast<void **>(&g_state.compact_qkv),
        compact_qkv_bytes
    );
    if (status != hipSuccess) {
        set_error("hipMalloc(compact_qkv)", status);
        release_scratch();
        return false;
    }
    status = hipMalloc(
        reinterpret_cast<void **>(&g_state.gate_and_beta),
        gate_and_beta_bytes
    );
    if (status != hipSuccess) {
        set_error("hipMalloc(gate_and_beta)", status);
        release_scratch();
        return false;
    }
    status = hipMalloc(&g_state.a_or_w, a_or_w_bytes);
    if (status != hipSuccess) {
        set_error("hipMalloc(a_or_w)", status);
        release_scratch();
        return false;
    }
    status = hipMalloc(&g_state.ai_or_v_new, ai_or_v_new_bytes);
    if (status != hipSuccess) {
        set_error("hipMalloc(ai_or_v_new)", status);
        release_scratch();
        return false;
    }
    status = hipMalloc(
        reinterpret_cast<void **>(&g_state.chunk_state),
        chunk_state_bytes
    );
    if (status != hipSuccess) {
        set_error("hipMalloc(chunk_state)", status);
        release_scratch();
        return false;
    }
    g_state.scratch_tokens = tokens;
    return true;
}

bool load_kernels(const char *directory) {
    char path[1400];
    for (size_t index = 0u; index < kKernelSpecs.size(); ++index) {
        const KernelSpec &spec = kKernelSpecs[index];
        const int length = std::snprintf(
            path,
            sizeof(path),
            "%s\\%s",
            directory,
            spec.file
        );
        if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) {
            set_error_text("FLA chunk-GDN kernel path is too long");
            return false;
        }
        hipError_t status = hipModuleLoad(&g_state.modules[index], path);
        if (status != hipSuccess) {
            char stage[256];
            std::snprintf(
                stage,
                sizeof(stage),
                "hipModuleLoad(%s)",
                spec.file
            );
            set_error(stage, status);
            return false;
        }
        status = hipModuleGetFunction(
            &g_state.functions[index],
            g_state.modules[index],
            spec.symbol
        );
        if (status != hipSuccess) {
            char stage[256];
            std::snprintf(
                stage,
                sizeof(stage),
                "hipModuleGetFunction(%s)",
                spec.symbol
            );
            set_error(stage, status);
            return false;
        }
    }
    return true;
}

bool launch(
    KernelIndex index,
    uint32_t grid_x,
    uint32_t grid_y,
    uint32_t grid_z,
    hipStream_t stream,
    void **arguments
) {
    const size_t slot = kernel_slot(index);
    const KernelSpec &spec = kKernelSpecs[slot];
    const hipError_t status = hipModuleLaunchKernel(
        g_state.functions[slot],
        grid_x,
        grid_y,
        grid_z,
        spec.threads,
        1u,
        1u,
        spec.dynamic_shared_bytes,
        stream,
        arguments,
        nullptr
    );
    if (status != hipSuccess) {
        char stage[256];
        std::snprintf(
            stage,
            sizeof(stage),
            "hipModuleLaunchKernel(%s)",
            spec.symbol
        );
        set_error(stage, status);
        return false;
    }
    return true;
}

bool supported_tokens(int32_t tokens) {
    return tokens == kSmokeTokens ||
        tokens == kQ8192Tokens ||
        tokens == kQ16384Tokens ||
        tokens == kQ17408Tokens ||
        tokens == kQ32768Tokens ||
        tokens == kQ65536Tokens;
}

int launch_segment_async(
    const float *postconv_raw_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    void *stream_pointer,
    int32_t tokens,
    bool reset_state
) {
    if (!ensure_scratch(tokens)) {
        return 0;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    if (reset_state) {
        const hipError_t status = hipMemsetAsync(
            final_state_f32,
            0,
            static_cast<size_t>(kStateElements) * sizeof(float),
            stream
        );
        if (status != hipSuccess) {
            set_error("hipMemsetAsync(initial_state)", status);
            return 0;
        }
    }
    uint16_t *q_bf16 = g_state.compact_qkv;
    uint16_t *k_bf16 =
        q_bf16 + static_cast<size_t>(tokens) * kQkHeads * kKeyDim;
    uint16_t *v_bf16 =
        k_bf16 + static_cast<size_t>(tokens) * kQkHeads * kKeyDim;
    float *g_cumsum = g_state.gate_and_beta;
    uint16_t *beta_bf16 = reinterpret_cast<uint16_t *>(
        g_state.gate_and_beta +
        static_cast<size_t>(tokens) * kValueHeads
    );
    float *a_f32 = static_cast<float *>(g_state.a_or_w);
    uint16_t *a_inverse_bf16 =
        static_cast<uint16_t *>(g_state.ai_or_v_new);
    uint16_t *w_bf16 = static_cast<uint16_t *>(g_state.a_or_w);
    uint16_t *v_new_bf16 =
        static_cast<uint16_t *>(g_state.ai_or_v_new);
    uint16_t *u_bf16 = v_bf16;
    int32_t launch_tokens = tokens;
    void *global_scratch = nullptr;
    void *profile_scratch = nullptr;

    const float *raw_pointer = postconv_raw_f32;
    uint16_t *q_pointer = q_bf16;
    uint16_t *k_pointer = k_bf16;
    void *qk_arguments[] = {
        &raw_pointer,
        &q_pointer,
        &k_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kQkL2Norm,
            static_cast<uint32_t>(
                (static_cast<uint64_t>(tokens) * kQkHeads +
                 kQkPrepRows - 1u) /
                kQkPrepRows
            ),
            1u,
            1u,
            stream,
            qk_arguments
        )) {
        return 0;
    }

    const float *gate_pointer = gate_f32;
    uint16_t *v_pointer = v_bf16;
    uint16_t *beta_pointer = beta_bf16;
    void *v_beta_arguments[] = {
        &raw_pointer,
        &gate_pointer,
        &v_pointer,
        &beta_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kVBetaCopy,
            static_cast<uint32_t>(
                static_cast<uint64_t>(tokens) * kValueHeads
            ),
            1u,
            1u,
            stream,
            v_beta_arguments
        )) {
        return 0;
    }

    float *g_pointer = g_cumsum;
    void *cumsum_arguments[] = {
        &gate_pointer,
        &g_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    const uint32_t chunks =
        static_cast<uint32_t>(tokens / static_cast<int32_t>(kChunk));
    if (!launch(
            KernelIndex::kGateCumsum,
            chunks,
            kValueHeads,
            1u,
            stream,
            cumsum_arguments
        )) {
        return 0;
    }

    float *a_pointer = a_f32;
    void *kkt_arguments[] = {
        &k_pointer,
        &beta_pointer,
        &g_pointer,
        &a_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kScaledDotKkt,
            chunks,
            kValueHeads,
            1u,
            stream,
            kkt_arguments
        )) {
        return 0;
    }

    const size_t inverse_bytes =
        static_cast<size_t>(tokens) *
        kValueHeads *
        kChunk *
        sizeof(uint16_t);
    hipError_t status = hipMemsetAsync(
        a_inverse_bf16,
        0,
        inverse_bytes,
        stream
    );
    if (status != hipSuccess) {
        set_error("hipMemsetAsync(a_inverse_bf16)", status);
        return 0;
    }
    uint16_t *inverse_pointer = a_inverse_bf16;
    void *solve_arguments[] = {
        &a_pointer,
        &inverse_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kSolveTril64,
            chunks,
            kValueHeads,
            1u,
            stream,
            solve_arguments
        )) {
        return 0;
    }

    uint16_t *w_pointer = w_bf16;
    uint16_t *u_pointer = u_bf16;
    void *recompute_arguments[] = {
        &k_pointer,
        &v_pointer,
        &beta_pointer,
        &w_pointer,
        &u_pointer,
        &inverse_pointer,
        &g_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kRecomputeWU,
            chunks,
            kValueHeads,
            1u,
            stream,
            recompute_arguments
        )) {
        return 0;
    }

    uint16_t *v_new_pointer = v_new_bf16;
    uint16_t *chunk_state_pointer = g_state.chunk_state;
    const float *initial_state_pointer = final_state_f32;
    float *final_state_pointer = final_state_f32;
    void *state_arguments[] = {
        &k_pointer,
        &u_pointer,
        &w_pointer,
        &v_new_pointer,
        &g_pointer,
        &initial_state_pointer,
        &chunk_state_pointer,
        &final_state_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kChunkState,
            kStateValueTiles,
            kValueHeads,
            1u,
            stream,
            state_arguments
        )) {
        return 0;
    }

    float *output_pointer = output_f32;
    void *output_arguments[] = {
        &q_pointer,
        &k_pointer,
        &v_new_pointer,
        &chunk_state_pointer,
        &g_pointer,
        &output_pointer,
        &launch_tokens,
        &global_scratch,
        &profile_scratch,
    };
    if (!launch(
            KernelIndex::kChunkOutput,
            kOutputValueTiles,
            chunks,
            kValueHeads,
            stream,
            output_arguments
        )) {
        return 0;
    }

    g_state.error[0] = '\0';
    return 1;
}

int launch_pipeline_async(
    const float *postconv_raw_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens
) {
    if (!g_state.prepared || postconv_raw_f32 == nullptr ||
        gate_f32 == nullptr || output_f32 == nullptr ||
        final_state_f32 == nullptr || !supported_tokens(tokens) ||
        tokens % static_cast<int32_t>(kChunk) != 0 ||
        gate_values_are_decay != 0) {
        set_error_text(
            "FLA chunk-GDN launch requires a supported multiple-of-64 shape, "
            "raw log gates, and non-null surfaces"
        );
        return 0;
    }

    const int32_t segment_tokens =
        tokens == kQ65536Tokens ? kQ65536SegmentTokens : tokens;
    for (int32_t token_offset = 0; token_offset < tokens;
         token_offset += segment_tokens) {
        if (launch_segment_async(
                postconv_raw_f32 +
                    static_cast<size_t>(token_offset) * kQkvRows,
                gate_f32 +
                    static_cast<size_t>(token_offset) * kGateRows,
                output_f32 +
                    static_cast<size_t>(token_offset) * kValueFeatures,
                final_state_f32,
                stream_pointer,
                segment_tokens,
                token_offset == 0
            ) == 0) {
            return 0;
        }
    }
    g_state.error[0] = '\0';
    return 1;
}

int launch_pipeline_synchronous(
    const float *postconv_raw_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens
) {
    if (launch_pipeline_async(
            postconv_raw_f32,
            gate_f32,
            output_f32,
            final_state_f32,
            gate_values_are_decay,
            stream_pointer,
            tokens
        ) == 0) {
        return 0;
    }
    const hipError_t status = hipStreamSynchronize(
        static_cast<hipStream_t>(stream_pointer)
    );
    if (status != hipSuccess) {
        set_error("hipStreamSynchronize(fla_chunk_gdn)", status);
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}

}  // namespace

QRT_FLA_GDN_EXPORT int qrt_aiter_fused_gdn_q8192_prepare(
    const char *kernel_dir
) {
    if (kernel_dir == nullptr || kernel_dir[0] == '\0') {
        set_error_text("FLA chunk-GDN kernel directory is empty");
        return 0;
    }
    if (g_state.prepared &&
        std::strcmp(g_state.kernel_dir, kernel_dir) == 0) {
        return 1;
    }
    release_state();
    std::snprintf(
        g_state.kernel_dir,
        sizeof(g_state.kernel_dir),
        "%s",
        kernel_dir
    );
    if (!load_kernels(kernel_dir)) {
        const std::string prepare_error = g_state.error;
        release_state();
        set_error_text(prepare_error.c_str());
        return 0;
    }
    g_state.prepared = true;
    g_state.error[0] = '\0';
    return 1;
}

#define QRT_DEFINE_FLA_GDN_LAUNCH(SHAPE, TOKENS)                         \
    QRT_FLA_GDN_EXPORT int                                               \
        qrt_aiter_fused_gdn_##SHAPE##_launch_async(                     \
            const float *postconv_raw_f32,                              \
            const float *gate_f32,                                      \
            float *output_f32,                                          \
            float *final_state_f32,                                     \
            int gate_values_are_decay,                                  \
            void *stream_pointer                                        \
        ) {                                                              \
        return launch_pipeline_async(                                   \
            postconv_raw_f32,                                           \
            gate_f32,                                                   \
            output_f32,                                                 \
            final_state_f32,                                            \
            gate_values_are_decay,                                      \
            stream_pointer,                                             \
            TOKENS                                                      \
        );                                                              \
    }                                                                  \
    QRT_FLA_GDN_EXPORT int qrt_aiter_fused_gdn_##SHAPE##_launch(         \
        const float *postconv_raw_f32,                                  \
        const float *gate_f32,                                          \
        float *output_f32,                                              \
        float *final_state_f32,                                         \
        int gate_values_are_decay,                                      \
        void *stream_pointer                                            \
    ) {                                                                 \
        return launch_pipeline_synchronous(                             \
            postconv_raw_f32,                                           \
            gate_f32,                                                   \
            output_f32,                                                 \
            final_state_f32,                                            \
            gate_values_are_decay,                                      \
            stream_pointer,                                             \
            TOKENS                                                      \
        );                                                              \
    }

QRT_DEFINE_FLA_GDN_LAUNCH(q64, kSmokeTokens)
QRT_DEFINE_FLA_GDN_LAUNCH(q8192, kQ8192Tokens)
QRT_DEFINE_FLA_GDN_LAUNCH(q16384, kQ16384Tokens)
QRT_DEFINE_FLA_GDN_LAUNCH(q17408, kQ17408Tokens)
QRT_DEFINE_FLA_GDN_LAUNCH(q32768, kQ32768Tokens)
QRT_DEFINE_FLA_GDN_LAUNCH(q65536, kQ65536Tokens)

#undef QRT_DEFINE_FLA_GDN_LAUNCH

QRT_FLA_GDN_EXPORT uint64_t qrt_fla_chunk_gdn_scratch_bytes(
    int32_t tokens
) {
    const int32_t scratch_tokens =
        tokens == kQ65536Tokens ? kQ65536SegmentTokens : tokens;
    return scratch_tokens > 0
        ? static_cast<uint64_t>(scratch_tokens) * kScratchBytesPerToken
        : 0u;
}

QRT_FLA_GDN_EXPORT const char *qrt_aiter_fused_gdn_q8192_last_error() {
    return g_state.error;
}

QRT_FLA_GDN_EXPORT void qrt_aiter_fused_gdn_q8192_release() {
    release_state();
}
