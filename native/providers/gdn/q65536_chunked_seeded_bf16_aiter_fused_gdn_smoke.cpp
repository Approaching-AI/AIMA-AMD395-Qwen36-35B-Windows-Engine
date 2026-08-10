#include <hip/hip_runtime.h>

#define NOMINMAX
#include <windows.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

constexpr int32_t kChunkTokens = 32768;
constexpr int32_t kFullTokens = 2 * kChunkTokens;
constexpr unsigned int kQkvRows = 8192u;
constexpr unsigned int kKeyHeads = 16u;
constexpr unsigned int kKeyDim = 128u;
constexpr unsigned int kKeyFeatures = kKeyHeads * kKeyDim;
constexpr unsigned int kValueHeads = 32u;
constexpr unsigned int kValueDim = 128u;
constexpr unsigned int kValueFeatures = kValueHeads * kValueDim;
constexpr unsigned int kGateRows = 32u;
constexpr unsigned int kGateOutputRows = 64u;
constexpr unsigned int kFillThreads = 256u;
constexpr unsigned int kGridValueTiles = 16u;
constexpr unsigned int kGridValueHeads = 32u;
constexpr unsigned int kRecurrentThreads = 32u;
constexpr unsigned int kZeroStateDynamicSharedBytes = 4096u;
constexpr float kQScale = 0.08838834764831845f;

using PrepareFunction = int (__cdecl *)(const char *);
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
    SeededLaunchFunction q32768_seeded_bf16_launch_async = nullptr;
    LastErrorFunction last_error = nullptr;
    ReleaseFunction release = nullptr;
};

struct Comparison {
    unsigned long long mismatches = 0u;
    unsigned long long nonfinite = 0u;
    unsigned long long first_mismatch = 0u;
    unsigned int max_abs_bits = 0u;
};

struct ModeResult {
    int gate_values_are_decay = 0;
    float zero_state_ms = 0.0f;
    float two_chunk_ms = 0.0f;
    Comparison output{};
    Comparison state{};
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

__global__ void fill_postconv_kernel(uint16_t *postconv, size_t elements) {
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
    float value = 0.0f;
    if (feature < kKeyFeatures) {
        value = bf16_round(unit * 0.125f) * kQScale;
    } else if (feature < 2u * kKeyFeatures) {
        value = bf16_round(unit * 0.125f);
    } else {
        value = bf16_round(unit * 0.0625f);
    }
    postconv[index] = float_to_bf16(value);
}

__global__ void fill_gate_kernel(float *gate, int gate_values_are_decay) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements =
        static_cast<size_t>(kFullTokens) * kGateOutputRows;
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

__device__ void record_comparison(
    float reference,
    float candidate,
    size_t index,
    Comparison *comparison,
    bool exact) {
    if (exact && __float_as_uint(reference) != __float_as_uint(candidate)) {
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
    const float difference = fabsf(reference - candidate);
    if (!exact && difference != 0.0f) {
        atomicAdd(&comparison->mismatches, 1ull);
        atomicMin(
            &comparison->first_mismatch,
            static_cast<unsigned long long>(index)
        );
    }
    atomicMax(&comparison->max_abs_bits, __float_as_uint(difference));
}

__global__ void compare_output_kernel(
    const uint16_t *reference,
    const uint16_t *candidate,
    size_t elements,
    Comparison *comparison) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        record_comparison(
            bf16_to_float(reference[index]),
            bf16_to_float(candidate[index]),
            index,
            comparison,
            true
        );
    }
}

__global__ void compare_state_kernel(
    const float *reference_row_major,
    const float *candidate_key_major,
    size_t elements,
    Comparison *comparison) {
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
        reference_row_major[row_index],
        candidate_key_major[key_index],
        row_index,
        comparison,
        true
    );
}

dim3 blocks_for(size_t elements) {
    return dim3(static_cast<unsigned int>(
        (elements + kFillThreads - 1u) / kFillThreads
    ));
}

bool load_provider(const char *path, ProviderApi *api) {
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) {
        return false;
    }
    api->prepare = reinterpret_cast<PrepareFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    api->q32768_seeded_bf16_launch_async =
        reinterpret_cast<SeededLaunchFunction>(GetProcAddress(
            api->module,
            "qrt_aiter_fused_gdn_q32768_seeded_bf16_launch_async"
        ));
    api->last_error = reinterpret_cast<LastErrorFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    api->release = reinterpret_cast<ReleaseFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_release")
    );
    return api->prepare != nullptr &&
        api->q32768_seeded_bf16_launch_async != nullptr &&
        api->last_error != nullptr && api->release != nullptr;
}

bool reset_comparison(Comparison *device) {
    Comparison initial{};
    initial.first_mismatch =
        (std::numeric_limits<unsigned long long>::max)();
    return hipMemcpy(
        device,
        &initial,
        sizeof(initial),
        hipMemcpyHostToDevice
    ) == hipSuccess;
}

bool launch_zero_state_bf16(
    hipFunction_t function,
    const uint16_t *postconv,
    const float *gate,
    uint16_t *output,
    float *final_state,
    int gate_values_are_decay) {
    const uint16_t *postconv_pointer = postconv;
    const float *gate_pointer = gate;
    uint16_t *output_pointer = output;
    float *final_state_pointer = final_state;
    int decay_flag = gate_values_are_decay;
    int32_t tokens = kFullTokens;
    void *global_scratch = nullptr;
    void *profile_scratch = nullptr;
    void *arguments[] = {
        &postconv_pointer,
        &gate_pointer,
        &output_pointer,
        &final_state_pointer,
        &decay_flag,
        &tokens,
        &global_scratch,
        &profile_scratch,
    };
    return hipModuleLaunchKernel(
        function,
        kGridValueTiles,
        kGridValueHeads,
        1u,
        kRecurrentThreads,
        1u,
        1u,
        kZeroStateDynamicSharedBytes,
        nullptr,
        arguments,
        nullptr
    ) == hipSuccess;
}

bool elapsed_ms(
    hipEvent_t start,
    hipEvent_t stop,
    float *milliseconds) {
    return hipEventRecord(stop) == hipSuccess &&
        hipEventSynchronize(stop) == hipSuccess &&
        hipEventElapsedTime(milliseconds, start, stop) == hipSuccess;
}

float bits_to_float(unsigned int bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int fail(const char *stage, hipError_t status) {
    std::cerr
        << "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke"
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
            << "usage: q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke "
               "KERNEL_DIR PROVIDER_DLL"
            << std::endl;
        return 2;
    }

    ProviderApi api;
    if (!load_provider(argv[2], &api)) {
        std::cerr
            << "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke"
            << " stage=load_provider windows_error=" << GetLastError()
            << std::endl;
        return 1;
    }
    if (api.prepare(argv[1]) == 0) {
        std::cerr
            << "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke"
            << " stage=prepare error=" << api.last_error()
            << std::endl;
        FreeLibrary(api.module);
        return 1;
    }

    char zero_state_path[1400];
    const int path_length = std::snprintf(
        zero_state_path,
        sizeof(zero_state_path),
        "%s\\q262144_aiter_fused_gdn_bf16.hsaco",
        argv[1]
    );
    hipModule_t zero_state_module = nullptr;
    hipFunction_t zero_state_function = nullptr;
    hipError_t status =
        path_length > 0 &&
                static_cast<size_t>(path_length) < sizeof(zero_state_path)
            ? hipModuleLoad(&zero_state_module, zero_state_path)
            : hipErrorInvalidValue;
    if (status == hipSuccess) {
        status = hipModuleGetFunction(
            &zero_state_function,
            zero_state_module,
            "_fixed_q8192_aiter_fused_gdn_kernel"
        );
    }
    if (status != hipSuccess) {
        api.release();
        FreeLibrary(api.module);
        return fail("load_zero_state_bf16", status);
    }

    const size_t postconv_elements =
        static_cast<size_t>(kFullTokens) * kQkvRows;
    const size_t gate_elements =
        static_cast<size_t>(kFullTokens) * kGateOutputRows;
    const size_t output_elements =
        static_cast<size_t>(kFullTokens) * kValueFeatures;
    const size_t state_elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;
    uint16_t *postconv = nullptr;
    float *gate = nullptr;
    uint16_t *zero_state_output = nullptr;
    uint16_t *chunked_output = nullptr;
    float *initial_state = nullptr;
    float *intermediate_state = nullptr;
    float *chunked_final_state = nullptr;
    float *zero_state_final_state = nullptr;
    Comparison *output_comparison = nullptr;
    Comparison *state_comparison = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;

    auto cleanup = [&]() {
        if (start != nullptr) {
            (void)hipEventDestroy(start);
        }
        if (stop != nullptr) {
            (void)hipEventDestroy(stop);
        }
        (void)hipFree(state_comparison);
        (void)hipFree(output_comparison);
        (void)hipFree(zero_state_final_state);
        (void)hipFree(chunked_final_state);
        (void)hipFree(intermediate_state);
        (void)hipFree(initial_state);
        (void)hipFree(chunked_output);
        (void)hipFree(zero_state_output);
        (void)hipFree(gate);
        (void)hipFree(postconv);
        (void)hipModuleUnload(zero_state_module);
        api.release();
        FreeLibrary(api.module);
    };
    auto allocate = [](auto **pointer, size_t bytes) {
        return hipMalloc(reinterpret_cast<void **>(pointer), bytes);
    };
    status = allocate(&postconv, postconv_elements * sizeof(uint16_t));
    if (status == hipSuccess) {
        status = allocate(&gate, gate_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(
            &zero_state_output,
            output_elements * sizeof(uint16_t)
        );
    }
    if (status == hipSuccess) {
        status = allocate(&chunked_output, output_elements * sizeof(uint16_t));
    }
    if (status == hipSuccess) {
        status = allocate(&initial_state, state_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(&intermediate_state, state_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(
            &chunked_final_state,
            state_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(
            &zero_state_final_state,
            state_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(&output_comparison, sizeof(Comparison));
    }
    if (status == hipSuccess) {
        status = allocate(&state_comparison, sizeof(Comparison));
    }
    if (status == hipSuccess) {
        status = hipEventCreate(&start);
    }
    if (status == hipSuccess) {
        status = hipEventCreate(&stop);
    }
    if (status != hipSuccess) {
        cleanup();
        return fail("allocate", status);
    }

    fill_postconv_kernel<<<blocks_for(postconv_elements), kFillThreads>>>(
        postconv,
        postconv_elements
    );
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        cleanup();
        return fail("fill_postconv", status);
    }

    ModeResult results[2];
    bool all_ok = true;
    for (int mode = 0; mode <= 1; ++mode) {
        ModeResult &result = results[mode];
        result.gate_values_are_decay = mode;
        fill_gate_kernel<<<blocks_for(gate_elements), kFillThreads>>>(
            gate,
            mode
        );
        status = hipMemset(
            initial_state,
            0,
            state_elements * sizeof(float)
        );
        if (status != hipSuccess || hipDeviceSynchronize() != hipSuccess) {
            all_ok = false;
            break;
        }

        if (hipEventRecord(start) != hipSuccess ||
            !launch_zero_state_bf16(
                zero_state_function,
                postconv,
                gate,
                zero_state_output,
                zero_state_final_state,
                mode
            ) ||
            !elapsed_ms(start, stop, &result.zero_state_ms)) {
            all_ok = false;
            break;
        }

        if (hipEventRecord(start) != hipSuccess ||
            api.q32768_seeded_bf16_launch_async(
                reinterpret_cast<const float *>(postconv),
                gate,
                initial_state,
                reinterpret_cast<float *>(chunked_output),
                intermediate_state,
                mode,
                nullptr
            ) == 0 ||
            api.q32768_seeded_bf16_launch_async(
                reinterpret_cast<const float *>(
                    postconv + static_cast<size_t>(kChunkTokens) * kQkvRows
                ),
                gate +
                    static_cast<size_t>(kChunkTokens) * kGateOutputRows,
                intermediate_state,
                reinterpret_cast<float *>(
                    chunked_output +
                    static_cast<size_t>(kChunkTokens) * kValueFeatures
                ),
                chunked_final_state,
                mode,
                nullptr
            ) == 0 ||
            !elapsed_ms(start, stop, &result.two_chunk_ms)) {
            std::cerr
                << "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke"
                << " stage=seeded mode=" << mode
                << " error=" << api.last_error() << std::endl;
            all_ok = false;
            break;
        }

        if (!reset_comparison(output_comparison) ||
            !reset_comparison(state_comparison)) {
            all_ok = false;
            break;
        }
        compare_output_kernel<<<blocks_for(output_elements), kFillThreads>>>(
            zero_state_output,
            chunked_output,
            output_elements,
            output_comparison
        );
        compare_state_kernel<<<blocks_for(state_elements), kFillThreads>>>(
            zero_state_final_state,
            chunked_final_state,
            state_elements,
            state_comparison
        );
        status = hipDeviceSynchronize();
        if (status != hipSuccess ||
            hipMemcpy(
                &result.output,
                output_comparison,
                sizeof(Comparison),
                hipMemcpyDeviceToHost
            ) != hipSuccess ||
            hipMemcpy(
                &result.state,
                state_comparison,
                sizeof(Comparison),
                hipMemcpyDeviceToHost
            ) != hipSuccess) {
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
            << "q65536_chunked_seeded_bf16_aiter_fused_gdn_smoke"
            << " gate_values_are_decay=" << result.gate_values_are_decay
            << " full_tokens=" << kFullTokens
            << " chunk_tokens=" << kChunkTokens
            << " chunks=2"
            << " zero_state_ms=" << result.zero_state_ms
            << " two_chunk_ms=" << result.two_chunk_ms
            << " output_elements=" << output_elements
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
            << " pass=" << (result.pass ? 1 : 0)
            << std::endl;
    }

    cleanup();
    return all_ok ? 0 : 1;
}
