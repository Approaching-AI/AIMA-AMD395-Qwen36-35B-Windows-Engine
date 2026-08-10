#include <hip/hip_runtime.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kTokens = 64;
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

struct ProviderApi {
#if defined(_WIN32)
    HMODULE module = nullptr;
#endif
    PrepareFunction prepare = nullptr;
    LaunchFunction launch = nullptr;
    LaunchFunction launch_async = nullptr;
    ScratchBytesFunction scratch_bytes = nullptr;
    LastErrorFunction last_error = nullptr;
    ReleaseFunction release = nullptr;
};

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16u) & 1u;
    return static_cast<uint16_t>(
        (bits + UINT32_C(0x7fff) + lsb) >> 16u
    );
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

float bf16_round(float value) {
    return bf16_to_float(float_to_bf16(value));
}

uint64_t fnv1a64(const void *data, size_t bytes) {
    const auto *cursor = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < bytes; ++index) {
        hash ^= cursor[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

bool load_provider(const char *path, ProviderApi *api) {
#if !defined(_WIN32)
    (void)path;
    (void)api;
    return false;
#else
    if (path == nullptr || api == nullptr) {
        return false;
    }
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) {
        std::cerr << "q64_fla_chunk_gdn_smoke load_library_error="
                  << GetLastError() << std::endl;
        return false;
    }
    api->prepare = reinterpret_cast<PrepareFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    api->launch = reinterpret_cast<LaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q64_launch")
    );
    api->launch_async = reinterpret_cast<LaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q64_launch_async")
    );
    api->scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
        GetProcAddress(api->module, "qrt_fla_chunk_gdn_scratch_bytes")
    );
    api->last_error = reinterpret_cast<LastErrorFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    api->release = reinterpret_cast<ReleaseFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_release")
    );
    return api->prepare != nullptr &&
        api->launch != nullptr &&
        api->launch_async != nullptr &&
        api->scratch_bytes != nullptr &&
        api->last_error != nullptr &&
        api->release != nullptr;
#endif
}

void unload_provider(ProviderApi *api) {
#if defined(_WIN32)
    if (api != nullptr && api->module != nullptr) {
        if (api->release != nullptr) {
            api->release();
        }
        (void)FreeLibrary(api->module);
        *api = ProviderApi{};
    }
#else
    (void)api;
#endif
}

bool check_hip(hipError_t status, const char *stage) {
    if (status == hipSuccess) {
        return true;
    }
    std::cerr << "q64_fla_chunk_gdn_smoke stage=" << stage
              << " hip_status=" << static_cast<int>(status)
              << " hip_error=" << hipGetErrorString(status)
              << std::endl;
    return false;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr
            << "usage: q64_fla_chunk_gdn_smoke <kernel_dir> <provider_dll>"
            << std::endl;
        return 2;
    }

    ProviderApi api;
    if (!load_provider(argv[2], &api)) {
        std::cerr << "q64_fla_chunk_gdn_smoke provider_symbols=missing"
                  << std::endl;
        unload_provider(&api);
        return 1;
    }
    if (api.prepare(argv[1]) == 0) {
        std::cerr << "q64_fla_chunk_gdn_smoke prepare_error="
                  << api.last_error() << std::endl;
        unload_provider(&api);
        return 1;
    }

    const size_t raw_elements = static_cast<size_t>(kTokens) * kQkvRows;
    const size_t gate_elements = static_cast<size_t>(kTokens) * kGateRows;
    const size_t output_elements =
        static_cast<size_t>(kTokens) * kValueFeatures;
    std::vector<float> raw(raw_elements);
    std::vector<float> gate(gate_elements);
    for (size_t index = 0u; index < raw.size(); ++index) {
        const size_t feature = index % kQkvRows;
        const float phase =
            static_cast<float>((index * 17u + feature * 13u) % 2048u) /
                2048.0f -
            0.5f;
        raw[index] = bf16_round(
            phase * (feature < 4096u ? 0.25f : 0.125f)
        );
    }
    for (int32_t token = 0; token < kTokens; ++token) {
        for (size_t head = 0u; head < 32u; ++head) {
            gate[static_cast<size_t>(token) * kGateRows + head] =
                -0.004f -
                0.0001f * static_cast<float>((token + head) % 17u);
            gate[
                static_cast<size_t>(token) * kGateRows + 32u + head
            ] = bf16_round(
                0.25f +
                0.5f *
                    static_cast<float>((token * 7u + head * 11u) % 31u) /
                    30.0f
            );
        }
    }

    float *device_raw = nullptr;
    float *device_gate = nullptr;
    float *device_output = nullptr;
    float *device_state = nullptr;
    hipStream_t stream = nullptr;
    const size_t raw_bytes = raw.size() * sizeof(float);
    const size_t gate_bytes = gate.size() * sizeof(float);
    const size_t output_bytes = output_elements * sizeof(float);
    const size_t state_bytes = kStateElements * sizeof(float);
    bool ok =
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&device_raw), raw_bytes),
            "hipMalloc(raw)"
        ) &&
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&device_gate), gate_bytes),
            "hipMalloc(gate)"
        ) &&
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&device_output), output_bytes),
            "hipMalloc(output)"
        ) &&
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&device_state), state_bytes),
            "hipMalloc(state)"
        ) &&
        check_hip(hipStreamCreate(&stream), "hipStreamCreate") &&
        check_hip(
            hipMemcpy(
                device_raw,
                raw.data(),
                raw_bytes,
                hipMemcpyHostToDevice
            ),
            "hipMemcpy(raw)"
        ) &&
        check_hip(
            hipMemcpy(
                device_gate,
                gate.data(),
                gate_bytes,
                hipMemcpyHostToDevice
            ),
            "hipMemcpy(gate)"
        );

    std::vector<float> sync_output(output_elements);
    std::vector<float> sync_state(kStateElements);
    std::vector<float> async_output(output_elements);
    std::vector<float> async_state(kStateElements);
    if (ok) {
        ok = api.launch(
                 device_raw,
                 device_gate,
                 device_output,
                 device_state,
                 0,
                 nullptr
             ) != 0;
        if (!ok) {
            std::cerr << "q64_fla_chunk_gdn_smoke sync_error="
                      << api.last_error() << std::endl;
        }
    }
    if (ok) {
        ok =
            check_hip(
                hipMemcpy(
                    sync_output.data(),
                    device_output,
                    output_bytes,
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(sync_output)"
            ) &&
            check_hip(
                hipMemcpy(
                    sync_state.data(),
                    device_state,
                    state_bytes,
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(sync_state)"
            ) &&
            check_hip(
                hipMemsetAsync(device_output, 0, output_bytes, stream),
                "hipMemsetAsync(output)"
            ) &&
            check_hip(
                hipMemsetAsync(device_state, 0, state_bytes, stream),
                "hipMemsetAsync(state)"
            );
    }
    if (ok) {
        ok = api.launch_async(
                 device_raw,
                 device_gate,
                 device_output,
                 device_state,
                 0,
                 stream
             ) != 0;
        if (!ok) {
            std::cerr << "q64_fla_chunk_gdn_smoke async_error="
                      << api.last_error() << std::endl;
        }
    }
    if (ok) {
        ok =
            check_hip(hipStreamSynchronize(stream), "hipStreamSynchronize") &&
            check_hip(
                hipMemcpy(
                    async_output.data(),
                    device_output,
                    output_bytes,
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(async_output)"
            ) &&
            check_hip(
                hipMemcpy(
                    async_state.data(),
                    device_state,
                    state_bytes,
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(async_state)"
            );
    }

    size_t output_nonfinite = 0u;
    size_t state_nonfinite = 0u;
    size_t output_nonzero = 0u;
    if (ok) {
        for (float value : sync_output) {
            output_nonfinite += !std::isfinite(value) ? 1u : 0u;
            output_nonzero += value != 0.0f ? 1u : 0u;
        }
        for (float value : sync_state) {
            state_nonfinite += !std::isfinite(value) ? 1u : 0u;
        }
        ok =
            output_nonfinite == 0u &&
            state_nonfinite == 0u &&
            output_nonzero > output_elements / 2u &&
            sync_output == async_output &&
            sync_state == async_state;
    }

    std::cout << "q64_fla_chunk_gdn_smoke"
              << " pass=" << (ok ? 1 : 0)
              << " sync_async_exact="
              << ((sync_output == async_output &&
                   sync_state == async_state)
                      ? 1
                      : 0)
              << " output_nonfinite=" << output_nonfinite
              << " state_nonfinite=" << state_nonfinite
              << " output_nonzero=" << output_nonzero
              << " output_hash=" << std::hex << std::setfill('0')
              << std::setw(16)
              << fnv1a64(sync_output.data(), output_bytes)
              << " state_hash=" << std::setw(16)
              << fnv1a64(sync_state.data(), state_bytes)
              << std::dec
              << " scratch_bytes=" << api.scratch_bytes(kTokens)
              << std::endl;

    if (stream != nullptr) {
        (void)hipStreamDestroy(stream);
    }
    if (device_state != nullptr) {
        (void)hipFree(device_state);
    }
    if (device_output != nullptr) {
        (void)hipFree(device_output);
    }
    if (device_gate != nullptr) {
        (void)hipFree(device_gate);
    }
    if (device_raw != nullptr) {
        (void)hipFree(device_raw);
    }
    unload_provider(&api);
    return ok ? 0 : 1;
}
