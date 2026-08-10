#include <hip/hip_runtime.h>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

constexpr unsigned int kFullTokens = 130560u;
constexpr unsigned int kChunkTokens = 1024u;
constexpr unsigned int kFullChunks = kFullTokens / kChunkTokens;
constexpr unsigned int kTailTokens = kFullTokens % kChunkTokens;
constexpr unsigned int kQkvRows = 8192u;
constexpr unsigned int kKeyHeads = 16u;
constexpr unsigned int kKeyDim = 128u;
constexpr unsigned int kKeyFeatures = kKeyHeads * kKeyDim;
constexpr unsigned int kValueHeads = 32u;
constexpr unsigned int kValueDim = 128u;
constexpr unsigned int kValueFeatures = kValueHeads * kValueDim;
constexpr unsigned int kGateRows = 32u;
constexpr unsigned int kGateOutputRows = 64u;
constexpr unsigned int kThreads = 256u;
constexpr float kQScale = 0.08838834764831845f;

static_assert(kFullChunks == 127u);
static_assert(kTailTokens == 512u);

using PrepareFunction = int (__cdecl *)(const char *);
using ZeroStateLaunchFunction = int (__cdecl *)(
    const float *,
    const float *,
    float *,
    float *,
    int,
    void *);
using SeededLaunchFunction = int (__cdecl *)(
    const float *,
    const float *,
    const float *,
    float *,
    float *,
    int,
    void *);
using LastErrorFunction = const char *(__cdecl *)();
using ReleaseFunction = void (__cdecl *)();

struct ProviderApi {
    HMODULE module = nullptr;
    PrepareFunction prepare = nullptr;
    ZeroStateLaunchFunction q130560_launch = nullptr;
    SeededLaunchFunction q1024_seeded_launch = nullptr;
    LastErrorFunction last_error = nullptr;
    ReleaseFunction release = nullptr;
};

struct Comparison {
    unsigned long long mismatches = 0u;
    unsigned long long nonfinite = 0u;
    unsigned long long first_mismatch =
        (std::numeric_limits<unsigned long long>::max)();
    unsigned int max_abs_bits = 0u;
};

struct ModeResult {
    int gate_values_are_decay = 0;
    Comparison output{};
    Comparison state{};
    float full_ms = 0.0f;
    float chunked_ms = 0.0f;
    bool pass = false;
};

__device__ uint16_t float_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        uint16_t upper = static_cast<uint16_t>(bits >> 16u);
        if ((bits & 0x007fffffu) != 0u) {
            upper |= 0x0040u;
        }
        return upper;
    }
    const uint32_t lsb = (bits >> 16u) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16u);
}

__device__ float bf16_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16u);
}

__device__ float bf16_round(float value) {
    return bf16_to_float(float_to_bf16(value));
}

__device__ uint32_t mix_index(uint32_t value) {
    value ^= value >> 16u;
    value *= UINT32_C(0x7feb352d);
    value ^= value >> 15u;
    value *= UINT32_C(0x846ca68b);
    return value ^ (value >> 16u);
}

__global__ void fill_postconv_kernel(float *postconv, size_t elements) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int feature =
        static_cast<unsigned int>(index % kQkvRows);
    const uint32_t mixed = mix_index(static_cast<uint32_t>(index));
    const int centered = static_cast<int>(mixed & UINT32_C(0xffff)) - 32768;
    const float unit = static_cast<float>(centered) / 32768.0f;
    if (feature < kKeyFeatures) {
        postconv[index] = bf16_round(unit * 0.125f) * kQScale;
    } else if (feature < 2u * kKeyFeatures) {
        postconv[index] = bf16_round(unit * 0.125f);
    } else {
        postconv[index] = bf16_round(unit * 0.0625f);
    }
}

__global__ void fill_gate_kernel(
    float *gate,
    size_t elements,
    int gate_values_are_decay
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const unsigned int token =
        static_cast<unsigned int>(index / kGateOutputRows);
    const unsigned int row =
        static_cast<unsigned int>(index % kGateOutputRows);
    const unsigned int head = row % kGateRows;
    if (row < kGateRows) {
        const unsigned int phase = (37u * token + 101u * head) & 1023u;
        const float log_g = -(
            0.000125f +
            0.031125f * static_cast<float>(phase) / 1023.0f
        );
        gate[index] = gate_values_are_decay != 0 ? expf(log_g) : log_g;
    } else {
        const float beta =
            0.05f +
            0.9f *
                static_cast<float>((13u * token + 17u * head) % 997u) /
                996.0f;
        gate[index] = bf16_round(beta);
    }
}

__global__ void fill_direct_identity_padding_kernel(float *gate) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements =
        static_cast<size_t>(kChunkTokens - kTailTokens) * kGateRows;
    if (index >= elements) {
        return;
    }
    const size_t token = index / kGateRows + kTailTokens;
    const size_t head = index % kGateRows;
    gate[token * kGateOutputRows + head] = 1.0f;
}

__device__ void record_comparison(
    float reference,
    float candidate,
    size_t index,
    Comparison *comparison
) {
    const unsigned int reference_bits = __float_as_uint(reference);
    const unsigned int candidate_bits = __float_as_uint(candidate);
    if (reference_bits != candidate_bits) {
        atomicAdd(&comparison->mismatches, 1ull);
        atomicMin(
            &comparison->first_mismatch,
            static_cast<unsigned long long>(index)
        );
    }
    if (!isfinite(reference) || !isfinite(candidate)) {
        atomicAdd(&comparison->nonfinite, 1ull);
        return;
    }
    atomicMax(
        &comparison->max_abs_bits,
        __float_as_uint(fabsf(reference - candidate))
    );
}

__global__ void compare_output_kernel(
    const float *full_output,
    const float *chunk_output,
    size_t elements,
    size_t global_offset,
    Comparison *comparison
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        record_comparison(
            full_output[index],
            chunk_output[index],
            global_offset + index,
            comparison
        );
    }
}

__global__ void compare_state_kernel(
    const float *full_row_major,
    const float *chunked_key_major,
    size_t elements,
    Comparison *comparison
) {
    const size_t row_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row_index >= elements) {
        return;
    }
    const unsigned int key =
        static_cast<unsigned int>(row_index % kKeyDim);
    const unsigned int value_index =
        static_cast<unsigned int>(row_index / kKeyDim);
    const unsigned int value_head = value_index / kValueDim;
    const unsigned int value = value_index % kValueDim;
    const size_t key_index =
        static_cast<size_t>(value_head) * kKeyDim * kValueDim +
        static_cast<size_t>(key) * kValueDim + value;
    record_comparison(
        full_row_major[row_index],
        chunked_key_major[key_index],
        row_index,
        comparison
    );
}

bool load_provider(const char *path, ProviderApi *api) {
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) {
        return false;
    }
    api->prepare = reinterpret_cast<PrepareFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    api->q130560_launch = reinterpret_cast<ZeroStateLaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q130560_launch")
    );
    api->q1024_seeded_launch = reinterpret_cast<SeededLaunchFunction>(
        GetProcAddress(
            api->module,
            "qrt_aiter_fused_gdn_q1024_seeded_launch"
        )
    );
    api->last_error = reinterpret_cast<LastErrorFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    api->release = reinterpret_cast<ReleaseFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_release")
    );
    return api->prepare != nullptr && api->q130560_launch != nullptr &&
        api->q1024_seeded_launch != nullptr && api->last_error != nullptr &&
        api->release != nullptr;
}

bool reset_comparison(Comparison *device) {
    const Comparison initial{};
    return hipMemcpy(
        device,
        &initial,
        sizeof(initial),
        hipMemcpyHostToDevice
    ) == hipSuccess;
}

float bits_to_float(unsigned int bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int fail(const char *stage, hipError_t status) {
    std::cerr
        << "q130560_chunked1024_aiter_fused_gdn_smoke"
        << " stage=" << stage
        << " hip_status=" << static_cast<int>(status)
        << " hip_error=" << hipGetErrorString(status)
        << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr
            << "usage: q130560_chunked1024_aiter_fused_gdn_smoke "
               "KERNEL_DIR PROVIDER_DLL"
            << std::endl;
        return 2;
    }

    ProviderApi api;
    if (!load_provider(argv[2], &api)) {
        std::cerr
            << "q130560_chunked1024_aiter_fused_gdn_smoke"
            << " stage=load_provider windows_error=" << GetLastError()
            << std::endl;
        return 1;
    }
    if (api.prepare(argv[1]) == 0) {
        std::cerr
            << "q130560_chunked1024_aiter_fused_gdn_smoke"
            << " stage=prepare error=" << api.last_error()
            << std::endl;
        FreeLibrary(api.module);
        return 1;
    }

    const size_t postconv_elements =
        static_cast<size_t>(kFullTokens) * kQkvRows;
    const size_t gate_elements =
        static_cast<size_t>(kFullTokens) * kGateOutputRows;
    const size_t full_output_elements =
        static_cast<size_t>(kFullTokens) * kValueFeatures;
    const size_t chunk_postconv_elements =
        static_cast<size_t>(kChunkTokens) * kQkvRows;
    const size_t chunk_gate_elements =
        static_cast<size_t>(kChunkTokens) * kGateOutputRows;
    const size_t chunk_output_elements =
        static_cast<size_t>(kChunkTokens) * kValueFeatures;
    const size_t tail_output_elements =
        static_cast<size_t>(kTailTokens) * kValueFeatures;
    const size_t state_elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;

    float *postconv = nullptr;
    float *gate = nullptr;
    float *full_output = nullptr;
    float *chunk_postconv = nullptr;
    float *chunk_gate = nullptr;
    float *chunk_output = nullptr;
    float *full_state_row = nullptr;
    float *state_key_a = nullptr;
    float *state_key_b = nullptr;
    Comparison *output_comparison = nullptr;
    Comparison *state_comparison = nullptr;

    auto allocate = [](auto **pointer, size_t bytes) {
        return hipMalloc(reinterpret_cast<void **>(pointer), bytes);
    };
    hipError_t status =
        allocate(&postconv, postconv_elements * sizeof(float));
    if (status == hipSuccess) {
        status = allocate(&gate, gate_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(&full_output, full_output_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(
            &chunk_postconv,
            chunk_postconv_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(&chunk_gate, chunk_gate_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(
            &chunk_output,
            chunk_output_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(&full_state_row, state_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(&state_key_a, state_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(&state_key_b, state_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(&output_comparison, sizeof(Comparison));
    }
    if (status == hipSuccess) {
        status = allocate(&state_comparison, sizeof(Comparison));
    }
    if (status != hipSuccess) {
        return fail("hipMalloc", status);
    }

    fill_postconv_kernel<<<
        dim3(static_cast<unsigned int>(
            (postconv_elements + kThreads - 1u) / kThreads
        )),
        dim3(kThreads)>>>(postconv, postconv_elements);
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        return fail("fill_postconv", status);
    }

    ModeResult results[2];
    bool all_ok = true;
    for (int mode = 0; mode <= 1; ++mode) {
        ModeResult &result = results[mode];
        result.gate_values_are_decay = mode;
        fill_gate_kernel<<<
            dim3(static_cast<unsigned int>(
                (gate_elements + kThreads - 1u) / kThreads
            )),
            dim3(kThreads)>>>(gate, gate_elements, mode);
        status = hipMemset(state_key_a, 0, state_elements * sizeof(float));
        if (status == hipSuccess) {
            status = hipMemset(state_key_b, 0, state_elements * sizeof(float));
        }
        if (status == hipSuccess && !reset_comparison(output_comparison)) {
            status = hipErrorUnknown;
        }
        if (status == hipSuccess && !reset_comparison(state_comparison)) {
            status = hipErrorUnknown;
        }
        if (status == hipSuccess) {
            status = hipDeviceSynchronize();
        }
        if (status != hipSuccess) {
            all_ok = false;
            break;
        }

        hipEvent_t full_start = nullptr;
        hipEvent_t full_stop = nullptr;
        status = hipEventCreate(&full_start);
        if (status == hipSuccess) {
            status = hipEventCreate(&full_stop);
        }
        if (status == hipSuccess) {
            status = hipEventRecord(full_start);
        }
        const bool full_launch_ok =
            status == hipSuccess &&
            api.q130560_launch(
                postconv,
                gate,
                full_output,
                full_state_row,
                mode,
                nullptr
            ) != 0;
        if (full_launch_ok) {
            status = hipEventRecord(full_stop);
        }
        if (full_launch_ok && status == hipSuccess) {
            status = hipEventSynchronize(full_stop);
        }
        if (full_launch_ok && status == hipSuccess) {
            status = hipEventElapsedTime(
                &result.full_ms,
                full_start,
                full_stop
            );
        }
        if (full_start != nullptr) {
            (void)hipEventDestroy(full_start);
        }
        if (full_stop != nullptr) {
            (void)hipEventDestroy(full_stop);
        }
        if (!full_launch_ok || status != hipSuccess) {
            std::cerr
                << "q130560_chunked1024_aiter_fused_gdn_smoke"
                << " stage=full mode=" << mode
                << " error=" << api.last_error() << std::endl;
            all_ok = false;
            break;
        }

        hipEvent_t chunk_start = nullptr;
        hipEvent_t chunk_stop = nullptr;
        status = hipEventCreate(&chunk_start);
        if (status == hipSuccess) {
            status = hipEventCreate(&chunk_stop);
        }
        if (status == hipSuccess) {
            status = hipEventRecord(chunk_start);
        }
        float *state_in = state_key_a;
        float *state_out = state_key_b;
        for (unsigned int chunk = 0u;
             status == hipSuccess && chunk < kFullChunks;
             ++chunk) {
            const size_t token_offset =
                static_cast<size_t>(chunk) * kChunkTokens;
            const float *chunk_postconv_source =
                postconv + token_offset * kQkvRows;
            const float *chunk_gate_source =
                gate + token_offset * kGateOutputRows;
            if (api.q1024_seeded_launch(
                    chunk_postconv_source,
                    chunk_gate_source,
                    state_in,
                    chunk_output,
                    state_out,
                    mode,
                    nullptr
                ) == 0) {
                status = hipErrorUnknown;
                break;
            }
            compare_output_kernel<<<
                dim3(static_cast<unsigned int>(
                    (chunk_output_elements + kThreads - 1u) / kThreads
                )),
                dim3(kThreads)>>>(
                    full_output + token_offset * kValueFeatures,
                    chunk_output,
                    chunk_output_elements,
                    token_offset * kValueFeatures,
                    output_comparison
                );
            status = hipGetLastError();
            std::swap(state_in, state_out);
        }

        const size_t tail_token_offset =
            static_cast<size_t>(kFullChunks) * kChunkTokens;
        if (status == hipSuccess) {
            status = hipMemset(
                chunk_postconv,
                0,
                chunk_postconv_elements * sizeof(float)
            );
        }
        if (status == hipSuccess) {
            status = hipMemset(
                chunk_gate,
                0,
                chunk_gate_elements * sizeof(float)
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                chunk_postconv,
                postconv + tail_token_offset * kQkvRows,
                static_cast<size_t>(kTailTokens) * kQkvRows * sizeof(float),
                hipMemcpyDeviceToDevice
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                chunk_gate,
                gate + tail_token_offset * kGateOutputRows,
                static_cast<size_t>(kTailTokens) *
                    kGateOutputRows * sizeof(float),
                hipMemcpyDeviceToDevice
            );
        }
        if (status == hipSuccess && mode != 0) {
            const size_t padding_decay_elements =
                static_cast<size_t>(kChunkTokens - kTailTokens) * kGateRows;
            fill_direct_identity_padding_kernel<<<
                dim3(static_cast<unsigned int>(
                    (padding_decay_elements + kThreads - 1u) / kThreads
                )),
                dim3(kThreads)>>>(chunk_gate);
            status = hipGetLastError();
        }
        if (status == hipSuccess &&
            api.q1024_seeded_launch(
                chunk_postconv,
                chunk_gate,
                state_in,
                chunk_output,
                state_out,
                mode,
                nullptr
            ) == 0) {
            status = hipErrorUnknown;
        }
        if (status == hipSuccess) {
            compare_output_kernel<<<
                dim3(static_cast<unsigned int>(
                    (tail_output_elements + kThreads - 1u) / kThreads
                )),
                dim3(kThreads)>>>(
                    full_output + tail_token_offset * kValueFeatures,
                    chunk_output,
                    tail_output_elements,
                    tail_token_offset * kValueFeatures,
                    output_comparison
                );
            compare_state_kernel<<<
                dim3(static_cast<unsigned int>(
                    (state_elements + kThreads - 1u) / kThreads
                )),
                dim3(kThreads)>>>(
                    full_state_row,
                    state_out,
                    state_elements,
                    state_comparison
                );
            status = hipGetLastError();
        }
        if (status == hipSuccess) {
            status = hipEventRecord(chunk_stop);
        }
        if (status == hipSuccess) {
            status = hipEventSynchronize(chunk_stop);
        }
        if (status == hipSuccess) {
            status = hipEventElapsedTime(
                &result.chunked_ms,
                chunk_start,
                chunk_stop
            );
        }
        if (chunk_start != nullptr) {
            (void)hipEventDestroy(chunk_start);
        }
        if (chunk_stop != nullptr) {
            (void)hipEventDestroy(chunk_stop);
        }
        if (status != hipSuccess) {
            std::cerr
                << "q130560_chunked1024_aiter_fused_gdn_smoke"
                << " stage=chunked mode=" << mode
                << " error=" << api.last_error() << std::endl;
            all_ok = false;
            break;
        }

        status = hipMemcpy(
            &result.output,
            output_comparison,
            sizeof(Comparison),
            hipMemcpyDeviceToHost
        );
        if (status == hipSuccess) {
            status = hipMemcpy(
                &result.state,
                state_comparison,
                sizeof(Comparison),
                hipMemcpyDeviceToHost
            );
        }
        if (status != hipSuccess) {
            all_ok = false;
            break;
        }
        result.pass =
            result.output.mismatches == 0u &&
            result.output.nonfinite == 0u &&
            result.output.max_abs_bits == 0u &&
            result.state.mismatches == 0u &&
            result.state.nonfinite == 0u &&
            result.state.max_abs_bits == 0u;
        all_ok = all_ok && result.pass;
    }

    for (const ModeResult &result : results) {
        std::cout
            << std::fixed << std::setprecision(6)
            << "q130560_chunked1024_aiter_fused_gdn_smoke"
            << " gate_values_are_decay=" << result.gate_values_are_decay
            << " full_tokens=" << kFullTokens
            << " chunk_tokens=" << kChunkTokens
            << " full_chunks=" << kFullChunks
            << " tail_tokens=" << kTailTokens
            << " output_elements=" << full_output_elements
            << " output_mismatches=" << result.output.mismatches
            << " output_nonfinite=" << result.output.nonfinite
            << " output_first_mismatch=" << result.output.first_mismatch
            << " output_max_abs="
            << bits_to_float(result.output.max_abs_bits)
            << " state_layout=key_major"
            << " state_elements=" << state_elements
            << " state_mismatches=" << result.state.mismatches
            << " state_nonfinite=" << result.state.nonfinite
            << " state_first_mismatch=" << result.state.first_mismatch
            << " state_max_abs="
            << bits_to_float(result.state.max_abs_bits)
            << " full_ms=" << result.full_ms
            << " chunked_ms=" << result.chunked_ms
            << " pass=" << (result.pass ? 1 : 0)
            << std::endl;
    }

    (void)hipFree(state_comparison);
    (void)hipFree(output_comparison);
    (void)hipFree(state_key_b);
    (void)hipFree(state_key_a);
    (void)hipFree(full_state_row);
    (void)hipFree(chunk_output);
    (void)hipFree(chunk_gate);
    (void)hipFree(chunk_postconv);
    (void)hipFree(full_output);
    (void)hipFree(gate);
    (void)hipFree(postconv);
    api.release();
    FreeLibrary(api.module);
    return all_ok ? 0 : 1;
}
