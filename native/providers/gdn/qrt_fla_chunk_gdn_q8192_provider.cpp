#include <hip/hip_runtime.h>
#include "blackwell_kkt.h"
#include "blackwell_state.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

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
constexpr size_t kBlackwellStateScratchBytes = kStateElements * sizeof(float) + kChunk * kValueFeatures * sizeof(uint16_t);
// Bound every recurrent dispatch to 16 chunks on WDDM. All segment boundaries
// are chunk boundaries and carry the unrounded F32 state on the same stream.
constexpr int32_t kSegmentTokens = 1024;

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
constexpr uint64_t kPaddedPostconvBytesPerToken =
    static_cast<uint64_t>(kQkvRows) * sizeof(float);
constexpr uint64_t kPaddedGateBytesPerToken =
    static_cast<uint64_t>(kGateRows) * sizeof(float);
constexpr uint64_t kPaddedOutputBytesPerToken =
    static_cast<uint64_t>(kValueFeatures) * sizeof(float);
constexpr uint64_t kMainScratchBytesPerToken =
    kCompactQkvBytesPerToken +
    kGateAndBetaBytesPerToken +
    kAOrWBytesPerToken +
    kAiOrVNewBytesPerToken +
    kChunkStateBytesPerToken;
constexpr uint64_t kTailPaddingBytes =
    static_cast<uint64_t>(kChunk) *
    (kPaddedPostconvBytesPerToken +
     kPaddedGateBytesPerToken +
     kPaddedOutputBytesPerToken);

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

// Supplied by the exact AOT build. Old FlashInfer-order objects are not ABI
// compatible with this Triton/FLA pipeline and must not share its directory.
#include "qrt_fla_gdn_kernel_specs.inc"

struct ProviderState {
    std::array<hipModule_t, static_cast<size_t>(KernelIndex::kCount)> modules{};
    std::array<hipFunction_t, static_cast<size_t>(KernelIndex::kCount)>
        functions{};
    uint16_t *compact_qkv = nullptr;
    float *gate_and_beta = nullptr;
    void *a_or_w = nullptr;
    void *ai_or_v_new = nullptr;
    uint16_t *chunk_state = nullptr;
    float *blackwell_temporary_state = nullptr;
    uint16_t *blackwell_residual = nullptr;
    float *padded_postconv = nullptr;
    float *padded_gate = nullptr;
    float *padded_output = nullptr;
    int32_t scratch_tokens = 0;
    bool prepared = false;
    bool q64_dumped = false;
    char kernel_dir[1024]{};
    char error[768]{};
};

ProviderState g_state;

bool blackwell_state_enabled() {
    const char* setting = std::getenv("QRT_FLA_GDN_STATE_BLACKWELL");
    return setting && std::strcmp(setting, "1") == 0;
}

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

bool dump_q64_stage(bool enabled, const char *name, const void *device, size_t bytes) {
    if (!enabled) return true;
    const char *directory = std::getenv("QRT_FLA_GDN_DUMP_Q64_DIR");
    if (directory == nullptr || device == nullptr || bytes == 0 || bytes > 2u * 1024u * 1024u) {
        set_error_text("q64 stage dump exceeds its bounded diagnostic surface");
        return false;
    }
    const std::string path = std::string(directory) + "\\stage-" + name + ".bin";
    if (std::ifstream(path, std::ios::binary).good()) {
        set_error_text("q64 stage dump refuses to overwrite an existing capture");
        return false;
    }
    std::vector<unsigned char> host(bytes);
    const hipError_t status = hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost);
    if (status != hipSuccess) {
        set_error("hipMemcpy(q64_stage_dump)", status);
        return false;
    }
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(host.data()), static_cast<std::streamsize>(bytes));
    if (!output.good()) {
        set_error_text("q64 stage dump write failed");
        return false;
    }
    return true;
}

bool launch_blackwell_kkt(const uint16_t* k, const uint16_t* beta,
                          const float* g, float* a, int32_t tokens,
                          hipStream_t stream, bool dump) {
    // This slow-exact KKT route never queues multiple chunks. Shape admission
    // bounds each launch to q64; completed device time also gates continuation.
    if (!k || !beta || !g || !a || tokens <= 0 || tokens > kSegmentTokens || tokens % kChunk) {
        set_error_text("Blackwell KKT requires checked chunk-aligned segment pointers");
        return false;
    }
    struct Event {
        hipEvent_t handle = nullptr;
        ~Event() { if (handle) (void)hipEventDestroy(handle); }
    } begin, end;
    hipError_t status = hipEventCreate(&begin.handle);
    if (status == hipSuccess) status = hipEventCreate(&end.handle);
    if (status != hipSuccess) { set_error("hipEventCreate(blackwell_kkt)", status); return false; }
    float maximum_ms = 0.0f;
    for (unsigned int chunk = 0; chunk < static_cast<unsigned int>(tokens) / kChunk; ++chunk) {
        status = hipEventRecord(begin.handle, stream);
        if (status != hipSuccess) { set_error("hipEventRecord(blackwell_kkt_begin)", status); return false; }
        hipLaunchKernelGGL(qrt_fla_blackwell::dot_kernel,
            dim3(kChunk * kChunk / (qrt_fla_blackwell::kThreads / qrt_fla_blackwell::kGroup), kValueHeads),
            dim3(qrt_fla_blackwell::kThreads), 0, stream, k, beta, a, chunk);
        status = hipGetLastError();
        if (status == hipSuccess) status = hipEventRecord(end.handle, stream);
        if (status == hipSuccess) status = hipEventSynchronize(end.handle);
        float milliseconds = 0.0f;
        if (status == hipSuccess) status = hipEventElapsedTime(&milliseconds, begin.handle, end.handle);
        if (status != hipSuccess) { set_error("blackwell_kkt_chunk", status); return false; }
        if (!(milliseconds <= 100.0f)) { set_error_text("Blackwell KKT chunk exceeded 100 ms; remaining chunks not submitted"); return false; }
        if (milliseconds > maximum_ms) maximum_ms = milliseconds;
    }
    if (!dump_q64_stage(dump, "a-dot-f32", a, 64u * 32u * 64u * 4u)) return false;
    const unsigned int elements = static_cast<unsigned int>(tokens) * kValueHeads * kChunk;
    hipLaunchKernelGGL(qrt_fla_blackwell::gate_kernel,
        dim3((elements + 255u) / 256u), dim3(256u), 0, stream,
        a, g, static_cast<unsigned int>(tokens));
    status = hipGetLastError();
    if (status == hipSuccess) status = hipStreamSynchronize(stream);
    if (status != hipSuccess) { set_error("blackwell_kkt_gate", status); return false; }
    std::fprintf(stderr, "FLA_KKT route=blackwell_group16_width26_k128 tokens=%d chunks=%u maximum_chunk_ms=%.6f guard_ms=100\n",
        tokens, static_cast<unsigned int>(tokens) / kChunk, static_cast<double>(maximum_ms));
    return true;
}

void release_scratch() {
    if (g_state.blackwell_temporary_state) (void)hipFree(g_state.blackwell_temporary_state);
    if (g_state.blackwell_residual) (void)hipFree(g_state.blackwell_residual);
    g_state.blackwell_temporary_state = nullptr;
    g_state.blackwell_residual = nullptr;
    if (g_state.padded_output != nullptr) {
        (void)hipFree(g_state.padded_output);
    }
    if (g_state.padded_gate != nullptr) {
        (void)hipFree(g_state.padded_gate);
    }
    if (g_state.padded_postconv != nullptr) {
        (void)hipFree(g_state.padded_postconv);
    }
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
    g_state.padded_postconv = nullptr;
    g_state.padded_gate = nullptr;
    g_state.padded_output = nullptr;
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
    if (g_state.scratch_tokens >= tokens &&
        g_state.compact_qkv != nullptr &&
        g_state.gate_and_beta != nullptr &&
        g_state.a_or_w != nullptr &&
        g_state.ai_or_v_new != nullptr &&
        g_state.chunk_state != nullptr &&
        g_state.padded_postconv != nullptr &&
        g_state.padded_gate != nullptr &&
        g_state.padded_output != nullptr) {
        return true;
    }
    size_t compact_qkv_bytes = 0u;
    size_t gate_and_beta_bytes = 0u;
    size_t a_or_w_bytes = 0u;
    size_t ai_or_v_new_bytes = 0u;
    size_t chunk_state_bytes = 0u;
    size_t padded_postconv_bytes = 0u;
    size_t padded_gate_bytes = 0u;
    size_t padded_output_bytes = 0u;
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
        ) ||
        !checked_bytes(
            static_cast<int32_t>(kChunk),
            kPaddedPostconvBytesPerToken,
            &padded_postconv_bytes
        ) ||
        !checked_bytes(
            static_cast<int32_t>(kChunk),
            kPaddedGateBytesPerToken,
            &padded_gate_bytes
        ) ||
        !checked_bytes(
            static_cast<int32_t>(kChunk),
            kPaddedOutputBytesPerToken,
            &padded_output_bytes
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
    status = hipMalloc(
        reinterpret_cast<void **>(&g_state.padded_postconv),
        padded_postconv_bytes
    );
    if (status != hipSuccess) {
        set_error("hipMalloc(padded_postconv)", status);
        release_scratch();
        return false;
    }
    status = hipMalloc(
        reinterpret_cast<void **>(&g_state.padded_gate),
        padded_gate_bytes
    );
    if (status != hipSuccess) {
        set_error("hipMalloc(padded_gate)", status);
        release_scratch();
        return false;
    }
    status = hipMalloc(
        reinterpret_cast<void **>(&g_state.padded_output),
        padded_output_bytes
    );
    if (status != hipSuccess) {
        set_error("hipMalloc(padded_output)", status);
        release_scratch();
        return false;
    }
    g_state.scratch_tokens = tokens;
    return true;
}

bool launch_blackwell_state(const uint16_t* k, const uint16_t* u, const uint16_t* w,
                            const float* g, uint16_t* h, uint16_t* v_new,
                            float* state, int32_t tokens, hipStream_t stream) {
    if (!k || !u || !w || !g || !h || !v_new || !state || tokens <= 0 || tokens > kSegmentTokens || tokens % kChunk) {
        set_error_text("Blackwell state requires checked chunk-aligned segment pointers"); return false;
    }
    if (!g_state.blackwell_temporary_state || !g_state.blackwell_residual) {
        size_t available = 0, total = 0;
        hipError_t status = hipMemGetInfo(&available, &total);
        if (status != hipSuccess) { set_error("hipMemGetInfo(blackwell_state)", status); return false; }
        if (available < kBlackwellStateScratchBytes + 512u * 1024u * 1024u) {
            set_error_text("Blackwell state device memory reserve failed"); return false;
        }
        if (!g_state.blackwell_temporary_state)
            status = hipMalloc(reinterpret_cast<void**>(&g_state.blackwell_temporary_state), kStateElements * sizeof(float));
        if (status == hipSuccess && !g_state.blackwell_residual)
            status = hipMalloc(reinterpret_cast<void**>(&g_state.blackwell_residual), kChunk * kValueFeatures * sizeof(uint16_t));
        if (status != hipSuccess) { set_error("hipMalloc(blackwell_state)", status); return false; }
    }
    struct Event { hipEvent_t handle = nullptr; ~Event() { if (handle) (void)hipEventDestroy(handle); } } begin, end;
    hipError_t status = hipEventCreate(&begin.handle);
    if (status == hipSuccess) status = hipEventCreate(&end.handle);
    if (status != hipSuccess) { set_error("hipEventCreate(blackwell_state)", status); return false; }
    float maximum_ms = 0;
    auto timed = [&](const char* name, auto operation) {
        hipError_t result = hipEventRecord(begin.handle, stream);
        if (result == hipSuccess) result = operation();
        if (result == hipSuccess) result = hipEventRecord(end.handle, stream);
        if (result == hipSuccess) result = hipEventSynchronize(end.handle);
        float milliseconds = 0;
        if (result == hipSuccess) result = hipEventElapsedTime(&milliseconds, begin.handle, end.handle);
        if (result != hipSuccess) { set_error(name, result); return false; }
        if (!(milliseconds <= 100.0f)) { set_error_text("Blackwell state dispatch exceeded 100 ms; remaining work not submitted"); return false; }
        if (milliseconds > maximum_ms) maximum_ms = milliseconds;
        return true;
    };
    float* initial = state; float* final = g_state.blackwell_temporary_state;
    for (int32_t offset = 0; offset < tokens; offset += kChunk) {
        const size_t value_offset = static_cast<size_t>(offset) * kValueFeatures;
        const float* gate = g + static_cast<size_t>(offset) * kValueHeads;
        if (!timed("blackwell_state_project", [&] {
                return qrt_fla_blackwell_state::project(w + value_offset, u + value_offset, gate, initial,
                    h + static_cast<size_t>(offset / kChunk) * kStateElements, v_new + value_offset,
                    g_state.blackwell_residual, kChunk, stream);
            })) return false;
        if (!timed("blackwell_state_update", [&] {
                return qrt_fla_blackwell_state::update(k + static_cast<size_t>(offset) * kQkHeads * kKeyDim,
                    g_state.blackwell_residual, gate, initial, final, kChunk, stream);
            })) return false;
        float* swap = initial; initial = final; final = swap;
    }
    // Odd chunk counts finish in the private buffer. Materialize the public
    // state before returning, including the single neutral-padded tail chunk.
    if (initial != state) {
        status = hipMemcpyAsync(state, initial, kStateElements * sizeof(float), hipMemcpyDeviceToDevice, stream);
        if (status == hipSuccess) status = hipStreamSynchronize(stream);
        if (status != hipSuccess) { set_error("hipMemcpyAsync(blackwell_final_state)", status); return false; }
    }
    std::fprintf(stderr, "FLA_STATE route=blackwell_group16_width26_k128_k64 tokens=%d chunks=%u maximum_dispatch_ms=%.6f guard_ms=100\n",
        tokens, static_cast<unsigned>(tokens / kChunk), static_cast<double>(maximum_ms));
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
    const char *sync_each_stage = std::getenv(
        "QRT_FLA_GDN_SYNC_EACH_STAGE"
    );
    const bool diagnose = sync_each_stage != nullptr &&
        sync_each_stage[0] != '\0' && sync_each_stage[0] != '0';
    if (diagnose) {
        std::fprintf(stderr, "FLA_STAGE begin=%s\n", spec.symbol);
        std::fflush(stderr);
    }
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
    if (diagnose) {
        const hipError_t sync_status = hipStreamSynchronize(stream);
        if (sync_status != hipSuccess) {
            char stage[256];
            std::snprintf(
                stage,
                sizeof(stage),
                "hipStreamSynchronize(%s)",
                spec.symbol
            );
            set_error(stage, sync_status);
            return false;
        }
        std::fprintf(stderr, "FLA_STAGE end=%s\n", spec.symbol);
        std::fflush(stderr);
    }
    return true;
}

bool supported_tokens(int32_t tokens) {
    return tokens > 0 && tokens <= kQ65536Tokens;
}

int32_t padded_tokens(int32_t tokens) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(tokens) + kChunk - 1u) /
            kChunk * kChunk
    );
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
    const char *dump_directory = std::getenv("QRT_FLA_GDN_DUMP_Q64_DIR");
    const bool dump = dump_directory != nullptr && dump_directory[0] != '\0' &&
        !g_state.q64_dumped && reset_state && tokens == kSmokeTokens;
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
    if (!dump_q64_stage(dump, "q-normalized-bf16", q_bf16, 64u * 2048u * 2u) ||
        !dump_q64_stage(dump, "k-normalized-bf16", k_bf16, 64u * 2048u * 2u) ||
        !dump_q64_stage(dump, "raw-f32", postconv_raw_f32, 64u * 8192u * 4u) ||
        !dump_q64_stage(dump, "gate-f32", gate_f32, 64u * 64u * 4u)) return 0;

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
    if (!dump_q64_stage(dump, "v-bf16", v_pointer, 64u * 4096u * 2u) ||
        !dump_q64_stage(dump, "beta-bf16", beta_pointer, 64u * 32u * 2u)) return 0;

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
    if (!dump_q64_stage(dump, "g-cumsum-f32", g_pointer, 64u * 32u * 4u)) return 0;

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
    const char* blackwell_kkt = std::getenv("QRT_FLA_GDN_KKT_BLACKWELL");
    if (blackwell_kkt != nullptr && std::strcmp(blackwell_kkt, "1") == 0) {
        if (!launch_blackwell_kkt(k_pointer, beta_pointer, g_pointer, a_pointer,
                                 tokens, stream, dump)) return 0;
    } else if (!launch(
            KernelIndex::kScaledDotKkt,
            chunks,
            kValueHeads,
            1u,
            stream,
            kkt_arguments
        )) {
        return 0;
    }
    if (!dump_q64_stage(dump, "a-f32", a_pointer, 64u * 32u * 64u * 4u)) return 0;

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
    if (!dump_q64_stage(dump, "a-inverse-bf16", inverse_pointer, 64u * 32u * 64u * 2u)) return 0;

    // A is dead after solve; inverse is dead after recompute. Each recompute
    // CTA owns one complete (chunk, value-head), so its V -> U alias has no
    // inter-program reader. State/output run only after recompute on this stream.
    uint16_t *w_pointer = static_cast<uint16_t *>(g_state.a_or_w);
    uint16_t *u_pointer = v_pointer;
    void *recompute_arguments[] = {
        &k_pointer, &v_pointer, &beta_pointer, &w_pointer, &u_pointer,
        &inverse_pointer, &g_pointer, &launch_tokens,
        &global_scratch, &profile_scratch,
    };
    if (!launch(
            KernelIndex::kRecomputeWU, chunks, kValueHeads, 1u,
            stream, recompute_arguments
        )) {
        return 0;
    }
    if (!dump_q64_stage(dump, "w-bf16", w_pointer, 64u * 4096u * 2u) ||
        !dump_q64_stage(dump, "u-bf16", u_pointer, 64u * 4096u * 2u)) return 0;

    uint16_t *v_new_pointer = static_cast<uint16_t *>(g_state.ai_or_v_new);
    uint16_t *chunk_state_pointer = g_state.chunk_state;
    const float *initial_state_pointer = final_state_f32;
    float *final_state_pointer = final_state_f32;
    void *state_arguments[] = {
        &k_pointer, &u_pointer, &w_pointer, &v_new_pointer, &g_pointer,
        &initial_state_pointer, &chunk_state_pointer, &final_state_pointer,
        &launch_tokens, &global_scratch, &profile_scratch,
    };
    if (blackwell_state_enabled()) {
        if (!launch_blackwell_state(k_pointer, u_pointer, w_pointer, g_pointer, chunk_state_pointer,
                                    v_new_pointer, final_state_f32, tokens, stream)) return 0;
    } else if (!launch(
            KernelIndex::kChunkState, kStateValueTiles, kValueHeads, 1u,
            stream, state_arguments
        )) {
        return 0;
    }
    if (!dump_q64_stage(dump, "v-new-bf16", v_new_pointer, 64u * 4096u * 2u) ||
        !dump_q64_stage(dump, "chunk-state-bf16", chunk_state_pointer, 32u * 128u * 128u * 2u)) return 0;

    float *output_pointer = output_f32;
    void *output_arguments[] = {
        &q_pointer, &k_pointer, &v_new_pointer, &chunk_state_pointer,
        &g_pointer, &output_pointer, &launch_tokens,
        &global_scratch, &profile_scratch,
    };
    if (!launch(
            KernelIndex::kChunkOutput, kOutputValueTiles, chunks, kValueHeads,
            stream, output_arguments
        )) {
        return 0;
    }
    if (dump) g_state.q64_dumped = true;

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
    const char *dump_directory = std::getenv("QRT_FLA_GDN_DUMP_Q64_DIR");
    if (dump_directory != nullptr && dump_directory[0] != '\0' && tokens != kSmokeTokens) {
        set_error_text("stage capture is restricted to a single q64 component probe");
        return 0;
    }
    if (!g_state.prepared || postconv_raw_f32 == nullptr ||
        gate_f32 == nullptr || output_f32 == nullptr ||
        final_state_f32 == nullptr || !supported_tokens(tokens) ||
        gate_values_are_decay != 0) {
        set_error_text(
            "FLA chunk-GDN launch requires 1..65536 tokens, raw log gates, "
            "and non-null surfaces"
        );
        return 0;
    }

    if (tokens % static_cast<int32_t>(kChunk) != 0) {
        const int32_t prefix_tokens =
            tokens / static_cast<int32_t>(kChunk) *
            static_cast<int32_t>(kChunk);
        const int32_t tail_tokens = tokens - prefix_tokens;
        // Scratch always covers a complete segment/chunk. Kernels only see
        // aligned extents, including the neutral final chunk; caller buffers
        // retain their exact logical extent and are never read past the end.
        const int32_t scratch_tokens =
            tokens > kSegmentTokens ? kSegmentTokens : padded_tokens(tokens);
        if (!ensure_scratch(scratch_tokens)) {
            return 0;
        }
        hipStream_t stream = static_cast<hipStream_t>(stream_pointer);
        for (int32_t offset = 0; offset < prefix_tokens;) {
            const int32_t remaining = prefix_tokens - offset;
            const int32_t count = remaining > kSegmentTokens
                ? kSegmentTokens : remaining;
            if (launch_segment_async(
                    postconv_raw_f32 + static_cast<size_t>(offset) * kQkvRows,
                    gate_f32 + static_cast<size_t>(offset) * kGateRows,
                    output_f32 + static_cast<size_t>(offset) * kValueFeatures,
                    final_state_f32, stream_pointer, count, offset == 0
                ) == 0) {
                return 0;
            }
            offset += count;
        }
        const size_t padded_postconv_bytes =
            static_cast<size_t>(kChunk) *
            static_cast<size_t>(kQkvRows) * sizeof(float);
        const size_t padded_gate_bytes =
            static_cast<size_t>(kChunk) *
            static_cast<size_t>(kGateRows) * sizeof(float);
        const size_t tail_postconv_bytes =
            static_cast<size_t>(tail_tokens) *
            static_cast<size_t>(kQkvRows) * sizeof(float);
        const size_t tail_gate_bytes =
            static_cast<size_t>(tail_tokens) *
            static_cast<size_t>(kGateRows) * sizeof(float);
        const size_t tail_output_bytes =
            static_cast<size_t>(tail_tokens) *
            static_cast<size_t>(kValueFeatures) * sizeof(float);
        hipError_t status = hipMemsetAsync(
            g_state.padded_postconv,
            0,
            padded_postconv_bytes,
            stream
        );
        if (status != hipSuccess) {
            set_error("hipMemsetAsync(padded_postconv)", status);
            return 0;
        }
        status = hipMemsetAsync(
            g_state.padded_gate,
            0,
            padded_gate_bytes,
            stream
        );
        if (status != hipSuccess) {
            set_error("hipMemsetAsync(padded_gate)", status);
            return 0;
        }
        status = hipMemcpyAsync(
            g_state.padded_postconv,
            postconv_raw_f32 +
                static_cast<size_t>(prefix_tokens) * kQkvRows,
            tail_postconv_bytes,
            hipMemcpyDeviceToDevice,
            stream
        );
        if (status != hipSuccess) {
            set_error("hipMemcpyAsync(padded_postconv)", status);
            return 0;
        }
        status = hipMemcpyAsync(
            g_state.padded_gate,
            gate_f32 +
                static_cast<size_t>(prefix_tokens) * kGateRows,
            tail_gate_bytes,
            hipMemcpyDeviceToDevice,
            stream
        );
        if (status != hipSuccess) {
            set_error("hipMemcpyAsync(padded_gate)", status);
            return 0;
        }
        if (launch_segment_async(
                g_state.padded_postconv,
                g_state.padded_gate,
                g_state.padded_output,
                final_state_f32,
                stream_pointer,
                static_cast<int32_t>(kChunk),
                prefix_tokens == 0
            ) == 0) {
            return 0;
        }
        status = hipMemcpyAsync(
            output_f32 +
                static_cast<size_t>(prefix_tokens) * kValueFeatures,
            g_state.padded_output,
            tail_output_bytes,
            hipMemcpyDeviceToDevice,
            stream
        );
        if (status != hipSuccess) {
            set_error("hipMemcpyAsync(unpadded_output)", status);
            return 0;
        }
        g_state.error[0] = '\0';
        return 1;
    }

    for (int32_t token_offset = 0; token_offset < tokens;) {
        const int32_t remaining = tokens - token_offset;
        const int32_t segment_tokens = remaining > kSegmentTokens
            ? kSegmentTokens : remaining;
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
        token_offset += segment_tokens;
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

QRT_FLA_GDN_EXPORT int qrt_aiter_fused_gdn_launch_async_dynamic(
    const float *postconv_raw_f32,
    const float *gate_f32,
    float *output_f32,
    float *final_state_f32,
    int gate_values_are_decay,
    void *stream_pointer,
    int32_t tokens
) {
    return launch_pipeline_async(
        postconv_raw_f32,
        gate_f32,
        output_f32,
        final_state_f32,
        gate_values_are_decay,
        stream_pointer,
        tokens
    );
}

QRT_FLA_GDN_EXPORT uint64_t qrt_fla_chunk_gdn_scratch_bytes(
    int32_t tokens
) {
    const int32_t scratch_tokens = !supported_tokens(tokens) ? 0
        : (tokens > kSegmentTokens ? kSegmentTokens : padded_tokens(tokens));
    return scratch_tokens > 0
        ? static_cast<uint64_t>(scratch_tokens) *
              kMainScratchBytesPerToken +
              kTailPaddingBytes +
              ((blackwell_state_enabled() || g_state.blackwell_temporary_state || g_state.blackwell_residual)
                  ? kBlackwellStateScratchBytes : 0u)
        : 0u;
}

QRT_FLA_GDN_EXPORT const char *qrt_aiter_fused_gdn_q8192_last_error() {
    return g_state.error;
}

QRT_FLA_GDN_EXPORT void qrt_aiter_fused_gdn_q8192_release() {
    release_state();
}
