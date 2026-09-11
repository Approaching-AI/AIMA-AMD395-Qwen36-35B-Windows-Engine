#include <hip/hip_runtime.h>
#include "fla_checkpoint.h"

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

#if !defined(QRT_FLA_GDN_SMOKE_TOKENS)
#define QRT_FLA_GDN_SMOKE_TOKENS 64
#endif
constexpr int32_t kTokens = QRT_FLA_GDN_SMOKE_TOKENS;
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
using DynamicLaunchFunction = int (*)(
    const float *,
    const float *,
    float *,
    float *,
    int,
    void *,
    int32_t
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
    DynamicLaunchFunction launch_async_dynamic = nullptr;
    qrt_fla_checkpoint::Launch launch_checkpoints = nullptr;
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
    api->launch_async_dynamic = reinterpret_cast<DynamicLaunchFunction>(
        GetProcAddress(
            api->module,
            "qrt_aiter_fused_gdn_launch_async_dynamic"
        )
    );
    api->scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
        GetProcAddress(api->module, "qrt_fla_chunk_gdn_scratch_bytes")
    );
    api->launch_checkpoints = reinterpret_cast<qrt_fla_checkpoint::Launch>(
        GetProcAddress(api->module, "qrt_fla_gdn_launch_async_checkpoints_v1")
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
        (kTokens == 64 || api->launch_async_dynamic != nullptr) &&
        api->scratch_bytes != nullptr &&
        api->last_error != nullptr &&
        api->release != nullptr;
#endif
}

int launch_fixture(
    const ProviderApi &api,
    const float *raw,
    const float *gate,
    float *output,
    float *state,
    void *stream,
    bool asynchronous
) {
    if (kTokens == 64) {
        const LaunchFunction launch = asynchronous ? api.launch_async : api.launch;
        return launch(raw, gate, output, state, 0, stream);
    }
    return api.launch_async_dynamic(
        raw,
        gate,
        output,
        state,
        0,
        stream,
        kTokens
    );
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

bool write_binary_file(
    const std::string &path,
    const void *data,
    size_t bytes
) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        return false;
    }
    output.write(static_cast<const char *>(data), bytes);
    return static_cast<bool>(output);
}

std::string join_path(const std::string &directory, const char *name) {
#if defined(_WIN32)
    constexpr char kSeparator = '\\';
#else
    constexpr char kSeparator = '/';
#endif
    if (directory.empty() ||
        directory.back() == '/' || directory.back() == '\\') {
        return directory + name;
    }
    return directory + kSeparator + name;
}

bool read_binary_file_exact(
    const std::string &path,
    void *data,
    size_t bytes
) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input || input.tellg() != static_cast<std::streamoff>(bytes)) {
        return false;
    }
    input.seekg(0, std::ios::beg);
    input.read(static_cast<char *>(data), static_cast<std::streamsize>(bytes));
    return static_cast<bool>(input);
}

bool load_gb10_capture_fixture(
    const char *directory,
    std::vector<float> *raw,
    std::vector<float> *gate
) {
    if (directory == nullptr || directory[0] == '\0' ||
        raw == nullptr || gate == nullptr) {
        return false;
    }
    const size_t qk_elements =
        static_cast<size_t>(kTokens) * 16u * 128u;
    const size_t v_elements =
        static_cast<size_t>(kTokens) * 32u * 128u;
    const size_t gate_head_elements =
        static_cast<size_t>(kTokens) * 32u;
    std::vector<uint16_t> q(qk_elements);
    std::vector<uint16_t> k(qk_elements);
    std::vector<uint16_t> v(v_elements);
    std::vector<float> g(gate_head_elements);
    std::vector<uint16_t> beta(gate_head_elements);
    const std::string root(directory);
    const bool loaded =
        read_binary_file_exact(
            join_path(root, "full-q-bf16.bin"),
            q.data(),
            q.size() * sizeof(q[0])
        ) &&
        read_binary_file_exact(
            join_path(root, "full-k-bf16.bin"),
            k.data(),
            k.size() * sizeof(k[0])
        ) &&
        read_binary_file_exact(
            join_path(root, "full-v-bf16.bin"),
            v.data(),
            v.size() * sizeof(v[0])
        ) &&
        read_binary_file_exact(
            join_path(root, "full-g-f32.bin"),
            g.data(),
            g.size() * sizeof(g[0])
        ) &&
        read_binary_file_exact(
            join_path(root, "full-beta-bf16.bin"),
            beta.data(),
            beta.size() * sizeof(beta[0])
        );
    if (!loaded) {
        std::cerr << "q64_fla_chunk_gdn_smoke capture_fixture_error="
                  << root << std::endl;
        return false;
    }
    for (int32_t token = 0; token < kTokens; ++token) {
        const size_t qk_offset = static_cast<size_t>(token) * 16u * 128u;
        const size_t v_offset = static_cast<size_t>(token) * 32u * 128u;
        const size_t raw_offset = static_cast<size_t>(token) * kQkvRows;
        for (size_t index = 0u; index < 16u * 128u; ++index) {
            (*raw)[raw_offset + index] = bf16_to_float(q[qk_offset + index]);
            (*raw)[raw_offset + 2048u + index] =
                bf16_to_float(k[qk_offset + index]);
        }
        for (size_t index = 0u; index < 32u * 128u; ++index) {
            (*raw)[raw_offset + 4096u + index] =
                bf16_to_float(v[v_offset + index]);
        }
        const size_t gate_offset = static_cast<size_t>(token) * kGateRows;
        const size_t head_offset = static_cast<size_t>(token) * 32u;
        for (size_t head = 0u; head < 32u; ++head) {
            (*gate)[gate_offset + head] = g[head_offset + head];
            (*gate)[gate_offset + 32u + head] =
                bf16_to_float(beta[head_offset + head]);
        }
    }
    return true;
}

}  // namespace


bool check_fp32_checkpoints(const ProviderApi& api, const ProviderApi& reference,
    const std::vector<float>& raw, const std::vector<float>& gate,
    const float* device_raw, const float* device_gate, hipStream_t stream,
    const std::vector<float>& expected_output, const std::vector<float>& expected_state) {
    if (kTokens <= 64 || !api.launch_checkpoints || !reference.launch_async_dynamic) {
        std::cerr << "FLA_CHECKPOINT missing checkpoint/reference route or eligible prefix\n";
        return false;
    }
    using namespace qrt_fla_checkpoint;
    constexpr size_t guard = 64u;
    constexpr float sentinel = 12345.0f;
    Plan plan{}; plan.struct_size = sizeof(plan); plan.abi_version = kVersion;
    const unsigned last = (static_cast<unsigned>(kTokens) - 1u) / 64u * 64u;
    for (unsigned position : {64u, 1024u, last}) {
        if (position >= static_cast<unsigned>(kTokens) ||
            (plan.count && position <= plan.prefix_tokens[plan.count - 1u])) continue;
        plan.prefix_tokens[plan.count++] = position;
    }
    std::array<float*, kCapacity> allocations{};
    std::array<std::vector<float>, kCapacity> snapshots;
    float *out_allocation = nullptr, *state_allocation = nullptr;
    std::vector<float> output(expected_output.size() + 2u * guard, sentinel);
    std::vector<float> state(kStateElements + 2u * guard, sentinel);
    bool ok = check_hip(hipMalloc(reinterpret_cast<void**>(&out_allocation), output.size() * sizeof(float)), "checkpoint_output_allocate") &&
        check_hip(hipMalloc(reinterpret_cast<void**>(&state_allocation), state.size() * sizeof(float)), "checkpoint_state_allocate");
    auto upload = [&](float* dst, const std::vector<float>& src) {
        return check_hip(hipMemcpyAsync(dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice, stream), "checkpoint_upload");
    };
    auto download = [&](std::vector<float>& dst, const float* src) {
        return check_hip(hipMemcpyAsync(dst.data(), src, dst.size() * sizeof(float), hipMemcpyDeviceToHost, stream), "checkpoint_download") &&
            check_hip(hipStreamSynchronize(stream), "checkpoint_wait");
    };
    auto guards = [&](const std::vector<float>& values) {
        for (size_t i = 0u; i < guard; ++i)
            if (values[i] != sentinel || values[values.size() - 1u - i] != sentinel) return false;
        return true;
    };
    for (unsigned slot = 0u; ok && slot < plan.count; ++slot) {
        snapshots[slot] = state;
        ok = check_hip(hipMalloc(reinterpret_cast<void**>(&allocations[slot]), state.size() * sizeof(float)), "checkpoint_slot_allocate");
        if (ok) {
            plan.states[slot] = allocations[slot] + guard; plan.state_bytes[slot] = kStateBytes;
            ok = upload(allocations[slot], snapshots[slot]);
        }
    }
    if (ok) ok = upload(out_allocation, output) && upload(state_allocation, state);
    if (ok) {
        // Real API rejects invalid plans before touching output/state/checkpoints.
        for (unsigned fault = 0u; ok && fault < 4u; ++fault) {
            Plan invalid = plan;
            if (fault == 0u) invalid.count = kCapacity + 1u;
            if (fault == 1u) invalid.prefix_tokens[0] = 63u;
            if (fault == 2u) invalid.state_bytes[0] = kStateBytes - 1u;
            if (fault == 3u) invalid.states[0] = state_allocation + guard + 1u;
            ok = api.launch_checkpoints(device_raw, device_gate, out_allocation + guard,
                state_allocation + guard, 0, stream, kTokens, &invalid) == 0;
        }
        if (ok) ok = download(output, out_allocation) && download(state, state_allocation);
        if (ok) ok = std::all_of(output.begin(), output.end(), [&](float x){return x == sentinel;}) &&
            std::all_of(state.begin(), state.end(), [&](float x){return x == sentinel;});
        for (unsigned slot = 0u; ok && slot < plan.count; ++slot) {
            ok = download(snapshots[slot], allocations[slot]);
            if (ok) ok = std::all_of(snapshots[slot].begin(), snapshots[slot].end(), [&](float x){return x == sentinel;});
        }
    }
    const auto begin = std::chrono::steady_clock::now();
    if (ok) ok = api.launch_checkpoints(device_raw, device_gate, out_allocation + guard,
        state_allocation + guard, 0, stream, kTokens, &plan) != 0;
    if (ok) ok = check_hip(hipStreamSynchronize(stream), "checkpoint_capture_wait");
    const double capture_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    if (ok) ok = download(output, out_allocation) && download(state, state_allocation);
    if (ok) ok = guards(output) && guards(state) &&
        std::memcmp(output.data() + guard, expected_output.data(), expected_output.size() * sizeof(float)) == 0 &&
        std::memcmp(state.data() + guard, expected_state.data(), kStateBytes) == 0;
    for (unsigned slot = 0u; ok && slot < plan.count; ++slot) {
        ok = download(snapshots[slot], allocations[slot]) && guards(snapshots[slot]);
        const unsigned prefix = plan.prefix_tokens[slot];
        std::fill(output.begin(), output.end(), sentinel); std::fill(state.begin(), state.end(), sentinel);
        if (ok) ok = upload(out_allocation, output) && upload(state_allocation, state);
        // A separately executed qualified old provider computes only this
        // prefix. It never sees or consumes the captured checkpoint buffer.
        if (ok) ok = reference.launch_async_dynamic(device_raw, device_gate, out_allocation + guard,
            state_allocation + guard, 0, stream, static_cast<int32_t>(prefix)) != 0;
        if (ok) ok = download(state, state_allocation) && download(output, out_allocation);
        size_t state_errors = 0u, output_errors = 0u;
        if (ok) {
            for (size_t i = 0u; i < kStateElements; ++i)
                state_errors += std::memcmp(&state[i + guard], &snapshots[slot][i + guard], sizeof(float)) != 0;
            for (size_t i = 0u; i < size_t(prefix) * kValueFeatures; ++i)
                output_errors += std::memcmp(&output[i + guard], &expected_output[i], sizeof(float)) != 0;
            ok = guards(state) && guards(output) && state_errors == 0u && output_errors == 0u &&
                std::all_of(output.begin() + guard + size_t(prefix) * kValueFeatures,
                            output.end(), [&](float x){return x == sentinel;});
        }
        std::cout << "{\"kind\":\"fla_fp32_checkpoint\",\"tokens\":" << kTokens
                  << ",\"prefix_tokens\":" << prefix << ",\"state_elements\":" << kStateElements
                  << ",\"state_bit_mismatches\":" << state_errors << ",\"prefix_output_bit_mismatches\":" << output_errors
                  << ",\"guarded_prefix_pass\":" << (ok ? "true" : "false")
                  << ",\"model_checkpoint_qualified\":false}" << std::endl;
        const char* dump = std::getenv("QRT_FLA_GDN_CHECKPOINT_DUMP_PREFIX");
        if (ok && dump && *dump) ok = write_binary_file(std::string(dump) + "-prefix" + std::to_string(prefix) + "-state-f32.bin",
            snapshots[slot].data() + guard, kStateBytes);
    }
    // Repeated shorter calls cannot mutate any saved checkpoint or input.
    for (unsigned slot = 0u; ok && slot < plan.count; ++slot) {
        std::vector<float> after(snapshots[slot].size());
        ok = download(after, allocations[slot]) && std::memcmp(after.data(), snapshots[slot].data(), after.size() * sizeof(float)) == 0;
    }
    if (ok) {
        std::vector<float> after(raw.size()); ok = download(after, device_raw) &&
            std::memcmp(after.data(), raw.data(), raw.size() * sizeof(float)) == 0;
        after.resize(gate.size()); if (ok) ok = download(after, device_gate) &&
            std::memcmp(after.data(), gate.data(), gate.size() * sizeof(float)) == 0;
    }
    std::cout << "{\"kind\":\"fla_fp32_checkpoint_capture\",\"tokens\":" << kTokens << ",\"checkpoints\":" << plan.count
              << ",\"capture_wall_ms\":" << capture_ms << ",\"full_output_elements\":" << expected_output.size()
              << ",\"unchanged_output_state_and_inputs\":" << (ok ? "true" : "false")
              << ",\"inference_acceptance\":false}" << std::endl;
    if (!ok) std::cerr << "FLA_CHECKPOINT error=" << api.last_error() << std::endl;
    (void)hipStreamSynchronize(stream);
    for (auto allocation : allocations) if (allocation) (void)hipFree(allocation);
    if (state_allocation) (void)hipFree(state_allocation);
    if (out_allocation) (void)hipFree(out_allocation);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) {
        std::cerr
            << "usage: q64_fla_chunk_gdn_smoke <kernel_dir> <provider_dll> "
            << "[reference_kernel_dir reference_provider_dll]"
            << std::endl;
        return 2;
    }

    ProviderApi api;
    const char* exact_setting = std::getenv("QRT_FLA_GDN_SMOKE_REQUIRE_REFERENCE_EXACT");
    const bool require_reference_exact = exact_setting && std::strcmp(exact_setting, "1") == 0;
    if (require_reference_exact && argc != 5) {
        std::cerr << "exact component comparison requires a reference provider" << std::endl;
        return 2;
    }
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
    ProviderApi reference_api;
    const bool compare_reference = argc == 5;
    if (compare_reference) {
        if (!load_provider(argv[4], &reference_api)) {
            std::cerr << "q64_fla_chunk_gdn_smoke reference_symbols=missing"
                      << std::endl;
            unload_provider(&api);
            return 1;
        }
        if (reference_api.prepare(argv[3]) == 0) {
            std::cerr << "q64_fla_chunk_gdn_smoke reference_prepare_error="
                      << reference_api.last_error() << std::endl;
            unload_provider(&reference_api);
            unload_provider(&api);
            return 1;
        }
    }

    const size_t raw_elements = static_cast<size_t>(kTokens) * kQkvRows;
    const size_t gate_elements = static_cast<size_t>(kTokens) * kGateRows;
    const size_t output_elements =
        static_cast<size_t>(kTokens) * kValueFeatures;
    std::vector<float> raw(raw_elements);
    std::vector<float> gate(gate_elements);
    const char *capture_fixture_directory =
        std::getenv("QRT_FLA_GDN_SMOKE_INPUT_DIR");
    const bool use_capture_fixture =
        capture_fixture_directory != nullptr &&
        capture_fixture_directory[0] != '\0';
    if (use_capture_fixture) {
        if (!load_gb10_capture_fixture(
                capture_fixture_directory,
                &raw,
                &gate
            )) {
            unload_provider(&api);
            if (compare_reference) {
                unload_provider(&reference_api);
            }
            return 1;
        }
    } else {
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
    std::vector<float> reference_output(output_elements);
    std::vector<float> reference_state(kStateElements);
    if (ok) {
        ok = launch_fixture(
                 api,
                 device_raw,
                 device_gate,
                 device_output,
                 device_state,
                 nullptr,
                 false
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
    if (ok && compare_reference) {
        ok = check_hip(
            hipStreamSynchronize(stream),
            "hipStreamSynchronize(before_reference)"
        );
    }
    if (ok && compare_reference) {
        ok = launch_fixture(
                 reference_api,
                 device_raw,
                 device_gate,
                 device_output,
                 device_state,
                 nullptr,
                 false
             ) != 0;
        if (!ok) {
            std::cerr << "q64_fla_chunk_gdn_smoke reference_launch_error="
                      << reference_api.last_error() << std::endl;
        }
    }
    if (ok && compare_reference) {
        ok =
            check_hip(
                hipMemcpy(
                    reference_output.data(),
                    device_output,
                    output_bytes,
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(reference_output)"
            ) &&
            check_hip(
                hipMemcpy(
                    reference_state.data(),
                    device_state,
                    state_bytes,
                    hipMemcpyDeviceToHost
                ),
                "hipMemcpy(reference_state)"
            );
    }
    if (ok) {
        ok = launch_fixture(
                 api,
                 device_raw,
                 device_gate,
                 device_output,
                 device_state,
                 stream,
                 true
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
    double output_square_error = 0.0;
    double output_reference_square = 0.0;
    double output_absolute_error = 0.0;
    double state_square_error = 0.0;
    double state_reference_square = 0.0;
    double state_absolute_error = 0.0;
    float output_max_absolute_error = 0.0f;
    float state_max_absolute_error = 0.0f;
    size_t output_max_error_index = 0u;
    size_t state_max_error_index = 0u;
    double output_candidate_square = 0.0;
    double state_candidate_square = 0.0;
    size_t output_reference_bit_mismatches = 0u, state_reference_bit_mismatches = 0u;
    if (ok) {
        for (float value : sync_output) {
            output_nonfinite += !std::isfinite(value) ? 1u : 0u;
            output_nonzero += value != 0.0f ? 1u : 0u;
        }
        for (float value : sync_state) {
            state_nonfinite += !std::isfinite(value) ? 1u : 0u;
        }
        if (compare_reference) {
            for (size_t index = 0u; index < sync_output.size(); ++index) {
                output_reference_bit_mismatches += std::memcmp(&sync_output[index], &reference_output[index], sizeof(float)) != 0;
                const double delta = static_cast<double>(sync_output[index]) -
                    static_cast<double>(reference_output[index]);
                output_square_error += delta * delta;
                output_reference_square +=
                    static_cast<double>(reference_output[index]) *
                    static_cast<double>(reference_output[index]);
                output_candidate_square +=
                    static_cast<double>(sync_output[index]) *
                    static_cast<double>(sync_output[index]);
                output_absolute_error += std::abs(delta);
                if (std::abs(delta) > output_max_absolute_error) {
                    output_max_absolute_error =
                        static_cast<float>(std::abs(delta));
                    output_max_error_index = index;
                }
            }
            for (size_t index = 0u; index < sync_state.size(); ++index) {
                state_reference_bit_mismatches += std::memcmp(&sync_state[index], &reference_state[index], sizeof(float)) != 0;
                const double delta = static_cast<double>(sync_state[index]) -
                    static_cast<double>(reference_state[index]);
                state_square_error += delta * delta;
                state_reference_square +=
                    static_cast<double>(reference_state[index]) *
                    static_cast<double>(reference_state[index]);
                state_candidate_square +=
                    static_cast<double>(sync_state[index]) *
                    static_cast<double>(sync_state[index]);
                state_absolute_error += std::abs(delta);
                if (std::abs(delta) > state_max_absolute_error) {
                    state_max_absolute_error =
                        static_cast<float>(std::abs(delta));
                    state_max_error_index = index;
                }
            }
        }
        ok =
            output_nonfinite == 0u &&
            state_nonfinite == 0u &&
            output_nonzero > output_elements / 2u &&
            sync_output == async_output &&
            sync_state == async_state &&
            (!require_reference_exact ||
             (output_reference_bit_mismatches == 0u && state_reference_bit_mismatches == 0u));
    }

    const char* checkpoint_setting = std::getenv("QRT_FLA_GDN_SMOKE_CHECKPOINTS");
    if (ok && checkpoint_setting && std::strcmp(checkpoint_setting, "1") == 0)
        ok = check_fp32_checkpoints(api, reference_api, raw, gate, device_raw, device_gate,
            stream, sync_output, sync_state);

    const char *dump_prefix = std::getenv("QRT_FLA_GDN_SMOKE_DUMP_PREFIX");
    if (ok && dump_prefix != nullptr && dump_prefix[0] != '\0') {
        std::vector<uint16_t> output_bf16(sync_output.size());
        std::transform(
            sync_output.begin(),
            sync_output.end(),
            output_bf16.begin(),
            float_to_bf16
        );
        const std::string prefix(dump_prefix);
        const std::string output_path = prefix + "-output-bf16.bin";
        const std::string state_path = prefix + "-state-f32.bin";
        ok = write_binary_file(
                 output_path,
                 output_bf16.data(),
                 output_bf16.size() * sizeof(output_bf16[0])
             ) &&
            write_binary_file(
                 state_path,
                 sync_state.data(),
                 sync_state.size() * sizeof(sync_state[0])
             );
        if (!ok) {
            std::cerr << "q64_fla_chunk_gdn_smoke dump_error prefix="
                      << prefix << std::endl;
        }
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
              << " fixture=" << (use_capture_fixture ? "gb10_capture" : "synthetic")
              << " reference_compared=" << (compare_reference ? 1 : 0);
    std::cout << " reference_exact_required=" << (require_reference_exact ? 1 : 0)
              << " output_reference_bit_mismatches=" << output_reference_bit_mismatches
              << " state_reference_bit_mismatches=" << state_reference_bit_mismatches;
    if (compare_reference) {
        std::cout
              << " output_reference_relative_l2="
              << std::sqrt(output_square_error /
                   std::max(output_reference_square, 1.0e-30))
              << " output_reference_mean_abs="
              << output_absolute_error /
                   static_cast<double>(sync_output.size())
              << " output_reference_max_abs="
              << output_max_absolute_error
              << " output_candidate_l2="
              << std::sqrt(output_candidate_square)
              << " output_reference_l2="
              << std::sqrt(output_reference_square)
              << " output_max_error_index=" << output_max_error_index
              << " output_max_error_candidate="
              << sync_output[output_max_error_index]
              << " output_max_error_reference="
              << reference_output[output_max_error_index]
              << " state_reference_relative_l2="
              << std::sqrt(state_square_error /
                   std::max(state_reference_square, 1.0e-30))
              << " state_reference_mean_abs="
              << state_absolute_error /
                   static_cast<double>(sync_state.size())
              << " state_reference_max_abs="
              << state_max_absolute_error
              << " state_candidate_l2="
              << std::sqrt(state_candidate_square)
              << " state_reference_l2="
              << std::sqrt(state_reference_square)
              << " state_max_error_index=" << state_max_error_index
              << " state_max_error_candidate="
              << sync_state[state_max_error_index]
              << " state_max_error_reference="
              << reference_state[state_max_error_index];
    }
    std::cout
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
    unload_provider(&reference_api);
    unload_provider(&api);
    return ok ? 0 : 1;
}
