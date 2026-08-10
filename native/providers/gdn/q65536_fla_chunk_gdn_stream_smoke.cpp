#include <hip/hip_runtime.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

constexpr size_t kQkvRows = 8192u;
constexpr size_t kGateRows = 64u;
constexpr size_t kValueFeatures = 4096u;
constexpr size_t kStateElements = 32u * 128u * 128u;

using PrepareFunction = int (*)(const char *);
using LaunchFunction = int (*)(
    const float *,
    const float *,
    float *,
    float *,
    int,
    void *
);
using ScratchBytesFunction = uint64_t (*)(int32_t);
using LastErrorFunction = const char *(*)();
using ReleaseFunction = void (*)();

bool check_hip(hipError_t status, const char *stage) {
    if (status == hipSuccess) {
        return true;
    }
    std::cerr << "q65536_fla_chunk_gdn_stream_smoke"
              << " stage=" << stage
              << " hip_status=" << static_cast<int>(status)
              << " hip_error=" << hipGetErrorString(status)
              << std::endl;
    return false;
}

bool parse_tokens(const char *text, int32_t *tokens) {
    if (text == nullptr || tokens == nullptr) {
        return false;
    }
    char *end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
        (value != 8192 && value != 65536)) {
        return false;
    }
    *tokens = static_cast<int32_t>(value);
    return true;
}

}  // namespace

int main(int argc, char **argv) {
#if !defined(_WIN32)
    (void)argc;
    (void)argv;
    return 2;
#else
    if (argc != 4) {
        std::cerr
            << "usage: q65536_fla_chunk_gdn_stream_smoke "
               "<kernel_dir> <provider_dll> <8192|65536>"
            << std::endl;
        return 2;
    }
    int32_t tokens = 0;
    if (!parse_tokens(argv[3], &tokens)) {
        std::cerr << "unsupported token count: " << argv[3] << std::endl;
        return 2;
    }

    HMODULE module = LoadLibraryA(argv[2]);
    if (module == nullptr) {
        std::cerr << "provider_load_error=" << GetLastError() << std::endl;
        return 1;
    }
    const auto prepare = reinterpret_cast<PrepareFunction>(
        GetProcAddress(module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    const std::string launch_name =
        "qrt_aiter_fused_gdn_q" + std::to_string(tokens) + "_launch_async";
    const auto launch = reinterpret_cast<LaunchFunction>(
        GetProcAddress(module, launch_name.c_str())
    );
    const auto scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
        GetProcAddress(module, "qrt_fla_chunk_gdn_scratch_bytes")
    );
    const auto last_error = reinterpret_cast<LastErrorFunction>(
        GetProcAddress(module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    const auto release = reinterpret_cast<ReleaseFunction>(
        GetProcAddress(module, "qrt_aiter_fused_gdn_q8192_release")
    );
    if (prepare == nullptr || launch == nullptr || scratch_bytes == nullptr ||
        last_error == nullptr || release == nullptr ||
        prepare(argv[1]) == 0) {
        std::cerr << "provider_prepare_error="
                  << (last_error != nullptr ? last_error() : "missing ABI")
                  << std::endl;
        (void)FreeLibrary(module);
        return 1;
    }

    const size_t raw_bytes =
        static_cast<size_t>(tokens) * kQkvRows * sizeof(float);
    const size_t gate_bytes =
        static_cast<size_t>(tokens) * kGateRows * sizeof(float);
    const size_t output_bytes =
        static_cast<size_t>(tokens) * kValueFeatures * sizeof(float);
    const size_t state_bytes = kStateElements * sizeof(float);
    float *raw = nullptr;
    float *gate = nullptr;
    float *output = nullptr;
    float *state = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    bool ok =
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&raw), raw_bytes),
            "hipMalloc(raw)"
        ) &&
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&gate), gate_bytes),
            "hipMalloc(gate)"
        ) &&
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&output), output_bytes),
            "hipMalloc(output)"
        ) &&
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&state), state_bytes),
            "hipMalloc(state)"
        ) &&
        check_hip(hipStreamCreate(&stream), "hipStreamCreate") &&
        check_hip(hipEventCreate(&start), "hipEventCreate(start)") &&
        check_hip(hipEventCreate(&stop), "hipEventCreate(stop)") &&
        check_hip(
            hipMemsetAsync(raw, 0, raw_bytes, stream),
            "hipMemsetAsync(raw)"
        ) &&
        check_hip(
            hipMemsetAsync(gate, 0, gate_bytes, stream),
            "hipMemsetAsync(gate)"
        ) &&
        check_hip(
            hipMemsetAsync(output, 0x7f, output_bytes, stream),
            "hipMemsetAsync(output)"
        ) &&
        check_hip(hipEventRecord(start, stream), "hipEventRecord(start)");
    if (ok) {
        ok = launch(raw, gate, output, state, 0, stream) != 0;
        if (!ok) {
            std::cerr << "provider_launch_error=" << last_error() << std::endl;
        }
    }
    if (ok) {
        ok =
            check_hip(hipEventRecord(stop, stream), "hipEventRecord(stop)") &&
            check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
    }

    float elapsed_ms = 0.0f;
    float first_output = NAN;
    float last_output = NAN;
    float first_state = NAN;
    float last_state = NAN;
    if (ok) {
        ok =
            check_hip(
                hipEventElapsedTime(&elapsed_ms, start, stop),
                "hipEventElapsedTime"
            ) &&
            check_hip(
                hipMemcpy(
                    &first_output,
                    output,
                    sizeof(first_output),
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(first_output)"
            ) &&
            check_hip(
                hipMemcpy(
                    &last_output,
                    output +
                        static_cast<size_t>(tokens) * kValueFeatures - 1u,
                    sizeof(last_output),
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(last_output)"
            ) &&
            check_hip(
                hipMemcpy(
                    &first_state,
                    state,
                    sizeof(first_state),
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(first_state)"
            ) &&
            check_hip(
                hipMemcpy(
                    &last_state,
                    state + kStateElements - 1u,
                    sizeof(last_state),
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(last_state)"
            );
    }
    ok = ok &&
        first_output == 0.0f &&
        last_output == 0.0f &&
        first_state == 0.0f &&
        last_state == 0.0f;

    std::cout << "q65536_fla_chunk_gdn_stream_smoke"
              << " pass=" << (ok ? 1 : 0)
              << " tokens=" << tokens
              << " elapsed_ms=" << std::fixed << std::setprecision(3)
              << elapsed_ms
              << " scratch_bytes=" << scratch_bytes(tokens)
              << " raw_bytes=" << raw_bytes
              << " output_bytes=" << output_bytes
              << " first_output=" << first_output
              << " last_output=" << last_output
              << " first_state=" << first_state
              << " last_state=" << last_state
              << std::endl;

    if (stop != nullptr) {
        (void)hipEventDestroy(stop);
    }
    if (start != nullptr) {
        (void)hipEventDestroy(start);
    }
    if (stream != nullptr) {
        (void)hipStreamDestroy(stream);
    }
    if (state != nullptr) {
        (void)hipFree(state);
    }
    if (output != nullptr) {
        (void)hipFree(output);
    }
    if (gate != nullptr) {
        (void)hipFree(gate);
    }
    if (raw != nullptr) {
        (void)hipFree(raw);
    }
    release();
    (void)FreeLibrary(module);
    return ok ? 0 : 1;
#endif
}
