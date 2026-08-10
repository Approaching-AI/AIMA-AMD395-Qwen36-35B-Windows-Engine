#include <hip/hip_runtime.h>

#define NOMINMAX
#include <windows.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

constexpr unsigned int kPrefixTokens = 16384u;
constexpr unsigned int kSuffixTokens = 1024u;
constexpr unsigned int kFullTokens = kPrefixTokens + kSuffixTokens;
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
constexpr float kQScale = 0.08838834764831845f;
constexpr float kSuffixMeanMsCeiling = 20.0f;

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
    ZeroStateLaunchFunction q16384_launch = nullptr;
    ZeroStateLaunchFunction q17408_launch = nullptr;
    SeededLaunchFunction q1024_seeded_launch = nullptr;
    SeededLaunchFunction q1024_seeded_launch_async = nullptr;
    LastErrorFunction last_error = nullptr;
    ReleaseFunction release = nullptr;
};

struct Comparison {
    unsigned long long mismatches;
    unsigned long long nonfinite;
    unsigned long long first_mismatch;
    unsigned int max_abs_bits;
};

struct ModeResult {
    int gate_values_are_decay = 0;
    float suffix_mean_ms = 0.0f;
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

__global__ void fill_nan_kernel(float *values, size_t elements) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        values[index] = __int_as_float(0x7fc00000);
    }
}

__global__ void row_to_key_major_kernel(
    const float *row_major,
    float *key_major) {
    const size_t row_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;
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
    key_major[key_index] = row_major[row_index];
}

__device__ void record_comparison(
    float reference,
    float candidate,
    size_t index,
    Comparison *comparison) {
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
    const float *full_tail,
    const float *suffix,
    size_t elements,
    Comparison *comparison) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        record_comparison(
            full_tail[index],
            suffix[index],
            index,
            comparison
        );
    }
}

__global__ void compare_state_kernel(
    const float *full_row_major,
    const float *suffix_key_major,
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
        full_row_major[row_index],
        suffix_key_major[key_index],
        row_index,
        comparison
    );
}

bool parse_repetitions(const char *text, unsigned int *value) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != '\0' || parsed < 11u || parsed > 100u) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

bool load_provider(const char *path, ProviderApi *api) {
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) {
        return false;
    }
    api->prepare = reinterpret_cast<PrepareFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    api->q16384_launch = reinterpret_cast<ZeroStateLaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q16384_launch")
    );
    api->q17408_launch = reinterpret_cast<ZeroStateLaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q17408_launch")
    );
    api->q1024_seeded_launch = reinterpret_cast<SeededLaunchFunction>(
        GetProcAddress(
            api->module,
            "qrt_aiter_fused_gdn_q1024_seeded_launch"
        )
    );
    api->q1024_seeded_launch_async =
        reinterpret_cast<SeededLaunchFunction>(
            GetProcAddress(
                api->module,
                "qrt_aiter_fused_gdn_q1024_seeded_launch_async"
            )
        );
    api->last_error = reinterpret_cast<LastErrorFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    api->release = reinterpret_cast<ReleaseFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_release")
    );
    return api->prepare != nullptr && api->q16384_launch != nullptr &&
        api->q17408_launch != nullptr &&
        api->q1024_seeded_launch != nullptr &&
        api->q1024_seeded_launch_async != nullptr &&
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

bool time_seeded(
    const ProviderApi &api,
    const float *postconv,
    const float *gate,
    const float *initial_state,
    float *output,
    float *final_state,
    int gate_values_are_decay,
    unsigned int repetitions,
    float *mean_ms) {
    if (api.q1024_seeded_launch_async(
            postconv,
            gate,
            initial_state,
            output,
            final_state,
            gate_values_are_decay,
            nullptr
        ) == 0 ||
        hipDeviceSynchronize() != hipSuccess) {
        return false;
    }

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    bool ok = hipEventCreate(&start) == hipSuccess &&
        hipEventCreate(&stop) == hipSuccess &&
        hipEventRecord(start) == hipSuccess;
    for (unsigned int repetition = 0u; ok && repetition < repetitions;
         ++repetition) {
        ok = api.q1024_seeded_launch_async(
                 postconv,
                 gate,
                 initial_state,
                 output,
                 final_state,
                 gate_values_are_decay,
                 nullptr
             ) != 0;
    }
    float elapsed_ms = 0.0f;
    ok = ok && hipEventRecord(stop) == hipSuccess &&
        hipEventSynchronize(stop) == hipSuccess &&
        hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess;
    if (start != nullptr) {
        (void)hipEventDestroy(start);
    }
    if (stop != nullptr) {
        (void)hipEventDestroy(stop);
    }
    if (ok) {
        *mean_ms = elapsed_ms / static_cast<float>(repetitions);
    }
    return ok;
}

float bits_to_float(unsigned int bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

int fail(const char *stage, hipError_t status) {
    std::cerr
        << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
        << " stage=" << stage
        << " hip_status=" << static_cast<int>(status)
        << " hip_error=" << hipGetErrorString(status)
        << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr
            << "usage: q16384_suffix1024_seeded_aiter_fused_gdn_smoke "
               "KERNEL_DIR PROVIDER_DLL REPETITIONS>=11"
            << std::endl;
        return 2;
    }
    unsigned int repetitions = 0u;
    if (!parse_repetitions(argv[3], &repetitions)) {
        return 2;
    }

    ProviderApi api;
    if (!load_provider(argv[2], &api)) {
        std::cerr
            << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
            << " stage=load_provider windows_error=" << GetLastError()
            << std::endl;
        return 1;
    }
    if (api.prepare(argv[1]) == 0) {
        std::cerr
            << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
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
    const size_t suffix_output_elements =
        static_cast<size_t>(kSuffixTokens) * kValueFeatures;
    const size_t state_elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;
    float *postconv = nullptr;
    float *gate = nullptr;
    float *full_output = nullptr;
    float *suffix_output = nullptr;
    float *prefix_state_row = nullptr;
    float *prefix_state_key = nullptr;
    float *full_state_row = nullptr;
    float *suffix_state_key = nullptr;
    Comparison *output_comparison = nullptr;
    Comparison *state_comparison = nullptr;

    auto allocate = [](auto **pointer, size_t bytes) {
        return hipMalloc(reinterpret_cast<void **>(pointer), bytes);
    };
    hipError_t status = allocate(
        &postconv,
        postconv_elements * sizeof(float)
    );
    if (status == hipSuccess) {
        status = allocate(&gate, gate_elements * sizeof(float));
    }
    if (status == hipSuccess) {
        status = allocate(
            &full_output,
            full_output_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(
            &suffix_output,
            suffix_output_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(
            &prefix_state_row,
            state_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(
            &prefix_state_key,
            state_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(
            &full_state_row,
            state_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(
            &suffix_state_key,
            state_elements * sizeof(float)
        );
    }
    if (status == hipSuccess) {
        status = allocate(&output_comparison, sizeof(Comparison));
    }
    if (status == hipSuccess) {
        status = allocate(&state_comparison, sizeof(Comparison));
    }
    if (status != hipSuccess) {
        (void)hipFree(state_comparison);
        (void)hipFree(output_comparison);
        (void)hipFree(suffix_state_key);
        (void)hipFree(full_state_row);
        (void)hipFree(prefix_state_key);
        (void)hipFree(prefix_state_row);
        (void)hipFree(suffix_output);
        (void)hipFree(full_output);
        (void)hipFree(gate);
        (void)hipFree(postconv);
        api.release();
        FreeLibrary(api.module);
        return fail("hipMalloc", status);
    }

    fill_postconv_kernel<<<
        dim3(static_cast<unsigned int>(
            (postconv_elements + kFillThreads - 1u) / kFillThreads
        )),
        dim3(kFillThreads)>>>(postconv, postconv_elements);
    status = hipDeviceSynchronize();
    if (status != hipSuccess) {
        (void)hipFree(state_comparison);
        (void)hipFree(output_comparison);
        (void)hipFree(suffix_state_key);
        (void)hipFree(full_state_row);
        (void)hipFree(prefix_state_key);
        (void)hipFree(prefix_state_row);
        (void)hipFree(suffix_output);
        (void)hipFree(full_output);
        (void)hipFree(gate);
        (void)hipFree(postconv);
        api.release();
        FreeLibrary(api.module);
        return fail("fill_postconv", status);
    }

    ModeResult results[2];
    bool all_ok = true;
    for (int mode = 0; mode <= 1; ++mode) {
        ModeResult &result = results[mode];
        result.gate_values_are_decay = mode;
        fill_gate_kernel<<<
            dim3(static_cast<unsigned int>(
                (gate_elements + kFillThreads - 1u) / kFillThreads
            )),
            dim3(kFillThreads)>>>(gate, mode);
        fill_nan_kernel<<<
            dim3(static_cast<unsigned int>(
                (full_output_elements + kFillThreads - 1u) / kFillThreads
            )),
            dim3(kFillThreads)>>>(full_output, full_output_elements);
        fill_nan_kernel<<<
            dim3(static_cast<unsigned int>(
                (suffix_output_elements + kFillThreads - 1u) / kFillThreads
            )),
            dim3(kFillThreads)>>>(suffix_output, suffix_output_elements);
        if (hipDeviceSynchronize() != hipSuccess) {
            all_ok = false;
            break;
        }

        if (api.q16384_launch(
                postconv,
                gate,
                full_output,
                prefix_state_row,
                mode,
                nullptr
            ) == 0) {
            std::cerr
                << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
                << " stage=q16384 mode=" << mode
                << " error=" << api.last_error() << std::endl;
            all_ok = false;
            break;
        }
        row_to_key_major_kernel<<<
            dim3(static_cast<unsigned int>(
                (state_elements + kFillThreads - 1u) / kFillThreads
            )),
            dim3(kFillThreads)>>>(prefix_state_row, prefix_state_key);
        if (hipDeviceSynchronize() != hipSuccess) {
            all_ok = false;
            break;
        }

        const float *suffix_postconv =
            postconv + static_cast<size_t>(kPrefixTokens) * kQkvRows;
        const float *suffix_gate =
            gate + static_cast<size_t>(kPrefixTokens) * kGateOutputRows;
        if (api.q1024_seeded_launch(
                suffix_postconv,
                suffix_gate,
                prefix_state_key,
                suffix_output,
                suffix_state_key,
                mode,
                nullptr
            ) == 0) {
            std::cerr
                << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
                << " stage=q1024_seeded mode=" << mode
                << " error=" << api.last_error() << std::endl;
            all_ok = false;
            break;
        }
        if (api.q17408_launch(
                postconv,
                gate,
                full_output,
                full_state_row,
                mode,
                nullptr
            ) == 0) {
            std::cerr
                << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
                << " stage=q17408 mode=" << mode
                << " error=" << api.last_error() << std::endl;
            all_ok = false;
            break;
        }

        if (!reset_comparison(output_comparison) ||
            !reset_comparison(state_comparison)) {
            all_ok = false;
            break;
        }
        compare_output_kernel<<<
            dim3(static_cast<unsigned int>(
                (suffix_output_elements + kFillThreads - 1u) / kFillThreads
            )),
            dim3(kFillThreads)>>>(
                full_output +
                    static_cast<size_t>(kPrefixTokens) * kValueFeatures,
                suffix_output,
                suffix_output_elements,
                output_comparison
            );
        compare_state_kernel<<<
            dim3(static_cast<unsigned int>(
                (state_elements + kFillThreads - 1u) / kFillThreads
            )),
            dim3(kFillThreads)>>>(
                full_state_row,
                suffix_state_key,
                state_elements,
                state_comparison
            );
        if (hipDeviceSynchronize() != hipSuccess ||
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
            ) != hipSuccess ||
            !time_seeded(
                api,
                suffix_postconv,
                suffix_gate,
                prefix_state_key,
                suffix_output,
                suffix_state_key,
                mode,
                repetitions,
                &result.suffix_mean_ms
            )) {
            all_ok = false;
            break;
        }
        const float thirty_layer_projected_ms =
            30.0f * result.suffix_mean_ms;
        result.pass =
            result.output.mismatches == 0u &&
            result.output.nonfinite == 0u &&
            result.output.max_abs_bits == 0u &&
            result.state.mismatches == 0u &&
            result.state.nonfinite == 0u &&
            result.state.max_abs_bits == 0u &&
            result.suffix_mean_ms <= kSuffixMeanMsCeiling &&
            thirty_layer_projected_ms <= 600.0f;
        all_ok = all_ok && result.pass;
    }

    for (const ModeResult &result : results) {
        const float thirty_layer_projected_ms =
            30.0f * result.suffix_mean_ms;
        std::cout
            << std::fixed << std::setprecision(6)
            << "q16384_suffix1024_seeded_aiter_fused_gdn_smoke"
            << " gate_values_are_decay=" << result.gate_values_are_decay
            << " prefix_tokens=" << kPrefixTokens
            << " suffix_tokens=" << kSuffixTokens
            << " full_tokens=" << kFullTokens
            << " repetitions=" << repetitions
            << " suffix_mean_ms=" << result.suffix_mean_ms
            << " suffix_ms_ceiling=" << kSuffixMeanMsCeiling
            << " thirty_layer_projected_ms="
            << thirty_layer_projected_ms
            << " output_elements=" << suffix_output_elements
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

    (void)hipFree(state_comparison);
    (void)hipFree(output_comparison);
    (void)hipFree(suffix_state_key);
    (void)hipFree(full_state_row);
    (void)hipFree(prefix_state_key);
    (void)hipFree(prefix_state_row);
    (void)hipFree(suffix_output);
    (void)hipFree(full_output);
    (void)hipFree(gate);
    (void)hipFree(postconv);
    api.release();
    FreeLibrary(api.module);
    return all_ok ? 0 : 1;
}
