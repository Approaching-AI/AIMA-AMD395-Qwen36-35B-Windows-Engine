#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_WIN32)
#define QRT_AITER_GDN_EXPORT extern "C" __declspec(dllexport)
#else
#define QRT_AITER_GDN_EXPORT extern "C"
#endif

namespace {

constexpr uint32_t kGridValueTiles = 16u;
constexpr uint32_t kGridValueHeads = 32u;
constexpr uint32_t kThreads = 32u;
constexpr uint32_t kDynamicSharedBytes = 4096u;
constexpr uint32_t kSeededDynamicSharedBytes = 512u;
constexpr int32_t kQ1024Tokens = 1024;
constexpr int32_t kQ8192Tokens = 8192;
constexpr int32_t kQ16384Tokens = 16384;
constexpr int32_t kQ17408Tokens = 17408;
constexpr int32_t kQ32768Tokens = 32768;
constexpr int32_t kQ32384Tokens = 32384;
constexpr int32_t kQ65536Tokens = 65536;
constexpr int32_t kQ129536Tokens = 129536;
constexpr int32_t kQ130560Tokens = 130560;
constexpr int32_t kQ131071Tokens = 131071;
constexpr int32_t kQ131072Tokens = 131072;
constexpr int32_t kQ131073Tokens = 131073;
constexpr int32_t kQ262143Tokens = 262143;
constexpr int32_t kQ262144Tokens = 262144;

struct ProviderState {
    hipModule_t module = nullptr;
    hipFunction_t function = nullptr;
    hipModule_t q262144_bf16_module = nullptr;
    hipFunction_t q262144_bf16_function = nullptr;
    hipModule_t seeded_module = nullptr;
    hipFunction_t seeded_function = nullptr;
    hipModule_t q32768_seeded_bf16_module = nullptr;
    hipFunction_t q32768_seeded_bf16_function = nullptr;
    bool prepared = false;
    char kernel_dir[1024]{};
    char error[512]{};
};

ProviderState g_state;

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

void release_state() {
    if (g_state.q32768_seeded_bf16_module != nullptr) {
        (void)hipModuleUnload(g_state.q32768_seeded_bf16_module);
    }
    if (g_state.seeded_module != nullptr) {
        (void)hipModuleUnload(g_state.seeded_module);
    }
    if (g_state.q262144_bf16_module != nullptr) {
        (void)hipModuleUnload(g_state.q262144_bf16_module);
    }
    if (g_state.module != nullptr) {
        (void)hipModuleUnload(g_state.module);
    }
    g_state = ProviderState{};
}

bool load_kernel(const char *directory) {
    char path[1400];
    const int length = std::snprintf(
        path,
        sizeof(path),
        "%s\\q8192_aiter_fused_gdn.hsaco",
        directory
    );
    if (length <= 0 || static_cast<size_t>(length) >= sizeof(path)) {
        set_error_text("AITER fused-GDN kernel path is too long");
        return false;
    }
    hipError_t status = hipModuleLoad(&g_state.module, path);
    if (status != hipSuccess) {
        set_error("hipModuleLoad(q8192_aiter_fused_gdn)", status);
        return false;
    }
    status = hipModuleGetFunction(
        &g_state.function,
        g_state.module,
        "_fixed_q8192_aiter_fused_gdn_kernel"
    );
    if (status != hipSuccess) {
        set_error("hipModuleGetFunction(q8192_aiter_fused_gdn)", status);
        return false;
    }

    const int q262144_bf16_length = std::snprintf(
        path,
        sizeof(path),
        "%s\\q262144_aiter_fused_gdn_bf16.hsaco",
        directory
    );
    if (q262144_bf16_length <= 0 ||
        static_cast<size_t>(q262144_bf16_length) >= sizeof(path)) {
        set_error_text("AITER q262144 BF16 fused-GDN kernel path is too long");
        return false;
    }
    status = hipModuleLoad(&g_state.q262144_bf16_module, path);
    if (status != hipSuccess) {
        set_error("hipModuleLoad(q262144_aiter_fused_gdn_bf16)", status);
        return false;
    }
    status = hipModuleGetFunction(
        &g_state.q262144_bf16_function,
        g_state.q262144_bf16_module,
        "_fixed_q8192_aiter_fused_gdn_kernel"
    );
    if (status != hipSuccess) {
        set_error(
            "hipModuleGetFunction(q262144_aiter_fused_gdn_bf16)",
            status
        );
        return false;
    }

    const int seeded_length = std::snprintf(
        path,
        sizeof(path),
        "%s\\q1024_seeded_aiter_fused_gdn.hsaco",
        directory
    );
    if (seeded_length <= 0 ||
        static_cast<size_t>(seeded_length) >= sizeof(path)) {
        set_error_text("AITER seeded fused-GDN kernel path is too long");
        return false;
    }
    status = hipModuleLoad(&g_state.seeded_module, path);
    if (status != hipSuccess) {
        set_error("hipModuleLoad(q1024_seeded_aiter_fused_gdn)", status);
        return false;
    }
    status = hipModuleGetFunction(
        &g_state.seeded_function,
        g_state.seeded_module,
        "_fixed_q1024_seeded_aiter_fused_gdn_kernel"
    );
    if (status != hipSuccess) {
        set_error(
            "hipModuleGetFunction(q1024_seeded_aiter_fused_gdn)",
            status
        );
        return false;
    }

    const int q32768_seeded_bf16_length = std::snprintf(
        path,
        sizeof(path),
        "%s\\q32768_seeded_aiter_fused_gdn_bf16.hsaco",
        directory
    );
    if (q32768_seeded_bf16_length <= 0 ||
        static_cast<size_t>(q32768_seeded_bf16_length) >= sizeof(path)) {
        set_error_text(
            "AITER q32768 seeded BF16 fused-GDN kernel path is too long"
        );
        return false;
    }
    status = hipModuleLoad(&g_state.q32768_seeded_bf16_module, path);
    if (status != hipSuccess) {
        set_error(
            "hipModuleLoad(q32768_seeded_aiter_fused_gdn_bf16)",
            status
        );
        return false;
    }
    status = hipModuleGetFunction(
        &g_state.q32768_seeded_bf16_function,
        g_state.q32768_seeded_bf16_module,
        "_fixed_q1024_seeded_aiter_fused_gdn_kernel"
    );
    if (status != hipSuccess) {
        set_error(
            "hipModuleGetFunction(q32768_seeded_aiter_fused_gdn_bf16)",
            status
        );
        return false;
    }
    return true;
}

}  // namespace

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q8192_prepare(
    const char *kernel_dir
) {
    if (kernel_dir == nullptr || kernel_dir[0] == '\0') {
        set_error_text("AITER fused-GDN kernel directory is empty");
        return 0;
    }
    if (g_state.prepared && std::strcmp(g_state.kernel_dir, kernel_dir) == 0) {
        return 1;
    }

    release_state();
    std::snprintf(
        g_state.kernel_dir,
        sizeof(g_state.kernel_dir),
        "%s",
        kernel_dir
    );
    if (!load_kernel(kernel_dir)) {
        const std::string prepare_error = g_state.error;
        release_state();
        set_error_text(prepare_error.c_str());
        return 0;
    }

    g_state.prepared = true;
    g_state.error[0] = '\0';
    return 1;
}

int launch_async_with_function(
    hipFunction_t function,
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens,
    const char *shape_name
) {
    if (!g_state.prepared || function == nullptr ||
        postconv_f32 == nullptr || gate_f32 == nullptr ||
        output_f32 == nullptr || final_state_f32 == nullptr ||
        (gate_values_are_decay != 0 && gate_values_are_decay != 1)) {
        set_error_text("AITER fused-GDN launch received an invalid surface");
        return 0;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    const float *postconv_pointer = postconv_f32;
    const float *gate_pointer = gate_f32;
    float *output_pointer = output_f32;
    float *state_pointer = final_state_f32;
    int32_t decay_flag = gate_values_are_decay;
    void *global_scratch = nullptr;
    void *profile_scratch = nullptr;
    void *arguments[] = {
        &postconv_pointer,
        &gate_pointer,
        &output_pointer,
        &state_pointer,
        &decay_flag,
        &tokens,
        &global_scratch,
        &profile_scratch,
    };
    hipError_t status = hipModuleLaunchKernel(
        function,
        kGridValueTiles,
        kGridValueHeads,
        1u,
        kThreads,
        1u,
        1u,
        kDynamicSharedBytes,
        stream,
        arguments,
        nullptr
    );
    if (status != hipSuccess) {
        char stage[128];
        std::snprintf(
            stage,
            sizeof(stage),
            "hipModuleLaunchKernel(%s_aiter_fused_gdn)",
            shape_name
        );
        set_error(stage, status);
        return 0;
    }

    g_state.error[0] = '\0';
    return 1;
}

int launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens,
    const char *shape_name
) {
    return launch_async_with_function(
        g_state.function,
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        tokens,
        shape_name
    );
}

int launch_synchronous(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens,
    const char *shape_name
) {
    if (launch_async(
            postconv_f32,
            gate_f32,
            output_f32,
            final_state_f32,
            gate_values_are_decay,
            stream_pointer,
            tokens,
            shape_name
        ) == 0) {
        return 0;
    }

    const hipError_t status = hipStreamSynchronize(
        static_cast<hipStream_t>(stream_pointer)
    );
    if (status != hipSuccess) {
        char stage[128];
        std::snprintf(
            stage,
            sizeof(stage),
            "hipStreamSynchronize(%s_aiter_fused_gdn)",
            shape_name
        );
        set_error(stage, status);
        return 0;
    }

    g_state.error[0] = '\0';
    return 1;
}

int launch_seeded_async_with_function(
    hipFunction_t function,
    const float *postconv_f32,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_f32,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens,
    const char *shape_name
) {
    if (!g_state.prepared || function == nullptr ||
        postconv_f32 == nullptr || gate_f32 == nullptr ||
        initial_state_key_major_f32 == nullptr || output_f32 == nullptr ||
        final_state_key_major_f32 == nullptr ||
        (gate_values_are_decay != 0 && gate_values_are_decay != 1)) {
        set_error_text(
            "AITER seeded fused-GDN launch received an invalid surface"
        );
        return 0;
    }

    hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
    const float *postconv_pointer = postconv_f32;
    const float *gate_pointer = gate_f32;
    const float *initial_state_pointer = initial_state_key_major_f32;
    float *output_pointer = output_f32;
    float *final_state_pointer = final_state_key_major_f32;
    int32_t decay_flag = gate_values_are_decay;
    void *global_scratch = nullptr;
    void *profile_scratch = nullptr;
    void *arguments[] = {
        &postconv_pointer,
        &gate_pointer,
        &initial_state_pointer,
        &output_pointer,
        &final_state_pointer,
        &decay_flag,
        &tokens,
        &global_scratch,
        &profile_scratch,
    };
    const hipError_t status = hipModuleLaunchKernel(
        function,
        kGridValueTiles,
        kGridValueHeads,
        1u,
        kThreads,
        1u,
        1u,
        kSeededDynamicSharedBytes,
        stream,
        arguments,
        nullptr
    );
    if (status != hipSuccess) {
        char stage[160];
        std::snprintf(
            stage,
            sizeof(stage),
            "hipModuleLaunchKernel(%s_seeded_aiter_fused_gdn)",
            shape_name
        );
        set_error(stage, status);
        return 0;
    }

    g_state.error[0] = '\0';
    return 1;
}

int launch_seeded_async(
    const float *postconv_f32,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_f32,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_seeded_async_with_function(
        g_state.seeded_function,
        postconv_f32,
        gate_f32,
        initial_state_key_major_f32,
        output_f32,
        final_state_key_major_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ1024Tokens,
        "q1024"
    );
}

int launch_seeded_synchronous(
    const float *postconv_f32,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_f32,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    if (launch_seeded_async(
            postconv_f32,
            gate_f32,
            initial_state_key_major_f32,
            output_f32,
            final_state_key_major_f32,
            gate_values_are_decay,
            stream_pointer
        ) == 0) {
        return 0;
    }
    const hipError_t status = hipStreamSynchronize(
        static_cast<hipStream_t>(stream_pointer)
    );
    if (status != hipSuccess) {
        set_error(
            "hipStreamSynchronize(q1024_seeded_aiter_fused_gdn)",
            status
        );
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q8192_launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_async(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ8192Tokens,
        "q8192"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q8192_launch(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_synchronous(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ8192Tokens,
        "q8192"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q16384_launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_async(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ16384Tokens,
        "q16384"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q16384_launch(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_synchronous(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ16384Tokens,
        "q16384"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q17408_launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_async(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ17408Tokens,
        "q17408"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q17408_launch(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_synchronous(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ17408Tokens,
        "q17408"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q1024_seeded_launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_f32,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_seeded_async(
        postconv_f32,
        gate_f32,
        initial_state_key_major_f32,
        output_f32,
        final_state_key_major_f32,
        gate_values_are_decay,
        stream_pointer
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q1024_seeded_launch(
    const float *postconv_f32,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_f32,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_seeded_synchronous(
        postconv_f32,
        gate_f32,
        initial_state_key_major_f32,
        output_f32,
        final_state_key_major_f32,
        gate_values_are_decay,
        stream_pointer
    );
}

QRT_AITER_GDN_EXPORT int
qrt_aiter_fused_gdn_q32768_seeded_bf16_launch_async(
    const float *postconv_bf16_abi,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_bf16_abi,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    // Preserve the provider's pointer ABI.  This code object interprets the
    // postconv/output pointers as BF16 and carries exact F32 key-major state
    // between bounded q32768 temporal chunks.
    return launch_seeded_async_with_function(
        g_state.q32768_seeded_bf16_function,
        postconv_bf16_abi,
        gate_f32,
        initial_state_key_major_f32,
        output_bf16_abi,
        final_state_key_major_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ32768Tokens,
        "q32768_bf16"
    );
}

QRT_AITER_GDN_EXPORT int
qrt_aiter_fused_gdn_q32384_seeded_bf16_launch_async(
    const float *postconv_bf16_abi,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_bf16_abi,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    // The seeded code object keeps the temporal length as a runtime scalar.
    // q129536 is exactly four q32384 chunks, so this uses the same BF16/F32
    // state contract without a ragged final launch.
    return launch_seeded_async_with_function(
        g_state.q32768_seeded_bf16_function,
        postconv_bf16_abi,
        gate_f32,
        initial_state_key_major_f32,
        output_bf16_abi,
        final_state_key_major_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ32384Tokens,
        "q32384_bf16"
    );
}

QRT_AITER_GDN_EXPORT int
qrt_aiter_fused_gdn_seeded_bf16_launch_async_dynamic(
    const float *postconv_bf16_abi,
    const float *gate_f32,
    const float *initial_state_key_major_f32,
    float *output_bf16_abi,
    float *final_state_key_major_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens
) {
    if (tokens <= 0 || tokens > kQ32768Tokens) {
        set_error_text(
            "AITER dynamic seeded BF16 fused-GDN requires 1..32768 tokens"
        );
        return 0;
    }
    return launch_seeded_async_with_function(
        g_state.q32768_seeded_bf16_function,
        postconv_bf16_abi,
        gate_f32,
        initial_state_key_major_f32,
        output_bf16_abi,
        final_state_key_major_f32,
        gate_values_are_decay,
        stream_pointer,
        tokens,
        "dynamic_bf16"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q32768_launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_async(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ32768Tokens,
        "q32768"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q32768_launch(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_synchronous(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ32768Tokens,
        "q32768"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q65536_launch_async(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_async(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ65536Tokens,
        "q65536"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q65536_launch(
    const float *postconv_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    return launch_synchronous(
        postconv_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ65536Tokens,
        "q65536"
    );
}

#define QRT_AITER_GDN_DEFINE_LONG_EXPORTS(token_count, shape_name)          \
    QRT_AITER_GDN_EXPORT int                                                \
        qrt_aiter_fused_gdn_##shape_name##_launch_async(                    \
            const float *postconv_f32,                                      \
            const float *gate_f32,                                          \
            float *output_f32,                                              \
            float *final_state_f32,                                         \
            int gate_values_are_decay,                                      \
            void *stream_pointer) {                                         \
        return launch_async(                                                \
            postconv_f32,                                                   \
            gate_f32,                                                       \
            output_f32,                                                     \
            final_state_f32,                                                \
            gate_values_are_decay,                                          \
            stream_pointer,                                                 \
            token_count,                                                    \
            #shape_name);                                                   \
    }                                                                       \
    QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_##shape_name##_launch(     \
        const float *postconv_f32,                                          \
        const float *gate_f32,                                              \
        float *output_f32,                                                  \
        float *final_state_f32,                                             \
        int gate_values_are_decay,                                          \
        void *stream_pointer) {                                             \
        return launch_synchronous(                                          \
            postconv_f32,                                                   \
            gate_f32,                                                       \
            output_f32,                                                     \
            final_state_f32,                                                \
            gate_values_are_decay,                                          \
            stream_pointer,                                                 \
            token_count,                                                    \
            #shape_name);                                                   \
    }

QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ129536Tokens, q129536)
QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ130560Tokens, q130560)
QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ131071Tokens, q131071)
QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ131072Tokens, q131072)
QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ131073Tokens, q131073)
QRT_AITER_GDN_DEFINE_LONG_EXPORTS(kQ262143Tokens, q262143)

#undef QRT_AITER_GDN_DEFINE_LONG_EXPORTS

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q262144_launch_async(
    const float *postconv_bf16_abi,
    const float *gate_f32,
    float *output_bf16_abi,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    // Preserve the established provider function-pointer ABI while the
    // q262144-only code object interprets the first and third pointers as
    // contiguous BF16 cells.  Shorter, authority-bearing shapes continue to
    // use the original F32-boundary module above.
    return launch_async_with_function(
        g_state.q262144_bf16_function,
        postconv_bf16_abi,
        gate_f32,
        output_bf16_abi,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        kQ262144Tokens,
        "q262144_bf16"
    );
}

QRT_AITER_GDN_EXPORT int qrt_aiter_fused_gdn_q262144_launch(
    const float *postconv_bf16_abi,
    const float *gate_f32,
    float *output_bf16_abi,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer
) {
    if (qrt_aiter_fused_gdn_q262144_launch_async(
            postconv_bf16_abi,
            gate_f32,
            output_bf16_abi,
            final_state_f32,
            gate_values_are_decay,
            stream_pointer
        ) == 0) {
        return 0;
    }
    const hipError_t status = hipStreamSynchronize(
        static_cast<hipStream_t>(stream_pointer)
    );
    if (status != hipSuccess) {
        set_error(
            "hipStreamSynchronize(q262144_bf16_aiter_fused_gdn)",
            status
        );
        return 0;
    }
    g_state.error[0] = '\0';
    return 1;
}

QRT_AITER_GDN_EXPORT uint64_t qrt_aiter_fused_gdn_q8192_scratch_bytes() {
    // Both direct-boundary kernels own recurrent state in registers /
    // compiler-managed private storage and write directly to caller outputs.
    // The seeded route reads/writes caller-owned key-major state. No provider
    // allocation is performed at prepare or request time.
    return 0u;
}

QRT_AITER_GDN_EXPORT const char *qrt_aiter_fused_gdn_q8192_last_error() {
    return g_state.error;
}

QRT_AITER_GDN_EXPORT void qrt_aiter_fused_gdn_q8192_release() {
    release_state();
}
