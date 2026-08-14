#include <hip/hip_runtime.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kQ8191Tokens = 8191u;
constexpr unsigned int kQ8192Tokens = 8192u;
constexpr unsigned int kQ8193Tokens = 8193u;
constexpr unsigned int kQ16384Tokens = 16384u;
constexpr unsigned int kQkvRows = 8192u;
constexpr unsigned int kKeyHeads = 16u;
constexpr unsigned int kKeyDim = 128u;
constexpr unsigned int kKeyFeatures = kKeyHeads * kKeyDim;
constexpr unsigned int kValueHeads = 32u;
constexpr unsigned int kValueDim = 128u;
constexpr unsigned int kValueFeatures = kValueHeads * kValueDim;
constexpr unsigned int kValueHeadsPerKeyHead = kValueHeads / kKeyHeads;
constexpr unsigned int kGateRows = 32u;
constexpr unsigned int kGateOutputRows = 64u;
constexpr unsigned int kFillThreads = 256u;
constexpr float kQScale = 0.08838834764831845f;

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
using ScratchBytesFunction = uint64_t (*)();
using LastErrorFunction = const char *(*)();
using ReleaseFunction = void (*)();

struct ProviderApi {
#if defined(_WIN32)
    HMODULE module = nullptr;
#else
    void *module = nullptr;
#endif
    PrepareFunction prepare = nullptr;
    LaunchFunction launch = nullptr;
    LaunchFunction launch_async = nullptr;
    DynamicLaunchFunction dynamic_launch_async = nullptr;
    LaunchFunction q16384_launch = nullptr;
    LaunchFunction q16384_launch_async = nullptr;
    ScratchBytesFunction scratch_bytes = nullptr;
    LastErrorFunction last_error = nullptr;
    ReleaseFunction release = nullptr;
};

struct Metrics {
    size_t exact_mismatches = 0u;
    size_t reference_nonfinite = 0u;
    size_t candidate_nonfinite = 0u;
    float max_abs = 0.0f;
    double rmse = 0.0;
    uint64_t reference_hash = UINT64_C(1469598103934665603);
    uint64_t candidate_hash = UINT64_C(1469598103934665603);
};

int fail(const char *stage, hipError_t status) {
    std::cerr << "q8192_aiter_fused_gdn_smoke stage=" << stage
              << " hip_status=" << static_cast<int>(status)
              << " hip_error=" << hipGetErrorString(status) << std::endl;
    return 1;
}

bool parse_repetitions(const char *text, unsigned int *value) {
    if (text == nullptr || value == nullptr) return false;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 3u || parsed > 100u) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

bool parse_tokens(const char *text, unsigned int *value) {
    if (text == nullptr || value == nullptr) return false;
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0' ||
        (parsed != kQ8191Tokens && parsed != kQ8192Tokens &&
         parsed != kQ8193Tokens && parsed != kQ16384Tokens)) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

__device__ float bf16_to_float(uint16_t bits) {
    union {
        uint32_t u32;
        float f32;
    } value;
    value.u32 = static_cast<uint32_t>(bits) << 16u;
    return value.f32;
}

__device__ uint16_t float_to_bf16(float value) {
    union {
        float f32;
        uint32_t u32;
    } raw;
    raw.f32 = value;
    const uint32_t lsb = (raw.u32 >> 16u) & 1u;
    return static_cast<uint16_t>((raw.u32 + UINT32_C(0x7fff) + lsb) >> 16u);
}

__device__ float bf16_round(float value) {
    return bf16_to_float(float_to_bf16(value));
}

__device__ float mul_add_separate(float acc, float left, float right) {
    float product;
    float sum;
    asm("v_mul_f32 %0, %1, %2"
        : "=v"(product)
        : "v"(left), "v"(right));
    asm("v_add_f32 %0, %1, %2"
        : "=v"(sum)
        : "v"(acc), "v"(product));
    return sum;
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
    if (index >= elements) return;
    const unsigned int feature = static_cast<unsigned int>(index % kQkvRows);
    const uint32_t mixed = mix_index(static_cast<uint32_t>(index));
    const int centered = static_cast<int>(mixed & UINT32_C(0xffff)) - 32768;
    const float unit = static_cast<float>(centered) / 32768.0f;
    if (feature < kKeyFeatures) {
        // Postconv Q is a BF16-normalized value multiplied by the F32 scale.
        postconv[index] = bf16_round(unit * 0.125f) * kQScale;
    } else if (feature < 2u * kKeyFeatures) {
        postconv[index] = bf16_round(unit * 0.125f);
    } else {
        postconv[index] = bf16_round(unit * 0.0625f);
    }
}

template <bool kDecayDirect>
__device__ __forceinline__ float core_mul(float left, float right) {
    if constexpr (kDecayDirect) {
        return mul_add_separate(0.0f, left, right);
    }
    return left * right;
}

template <bool kDecayDirect>
__device__ __forceinline__ float core_mul_add(
    float acc,
    float left,
    float right
) {
    if constexpr (kDecayDirect) {
        return mul_add_separate(acc, left, right);
    }
    return acc + left * right;
}

// This mirrors the retained scalar sequential GPU recurrence.  Layers 0/1
// use precomputed decay plus explicit AMD multiply/add instructions; later
// layers consume negative log-g and use the compiler's expression path.
template <bool kDecayDirect>
__global__ void scalar_reference_kernel(
    const float *postconv,
    const float *gate,
    float *state_output,
    float *outputs,
    unsigned int tokens
) {
    const unsigned int value_dim = threadIdx.x;
    const unsigned int value_head = blockIdx.x;
    if (value_dim >= kValueDim || value_head >= kValueHeads) return;

    const unsigned int value_index = value_head * kValueDim + value_dim;
    const unsigned int key_head = value_head / kValueHeadsPerKeyHead;
    const unsigned int key_base = key_head * kKeyDim;
    const size_t state_base = static_cast<size_t>(value_index) * kKeyDim;
    float state[kKeyDim];
    for (unsigned int key = 0u; key < kKeyDim; ++key) state[key] = 0.0f;

    for (unsigned int token = 0u; token < tokens; ++token) {
        const size_t post_base = static_cast<size_t>(token) * kQkvRows;
        const size_t gate_base = static_cast<size_t>(token) * kGateOutputRows;
        const float value =
            postconv[post_base + 2u * kKeyFeatures + value_index];
        const float beta = gate[gate_base + kGateRows + value_head];
        const float decay = kDecayDirect
            ? gate[gate_base + value_head]
            : expf(gate[gate_base + value_head]);
        float projected = 0.0f;
        float output_acc = 0.0f;

        for (unsigned int key = 0u; key < kKeyDim; ++key) {
            state[key] = core_mul<kDecayDirect>(state[key], decay);
            const float k_value =
                postconv[post_base + kKeyFeatures + key_base + key];
            projected = core_mul_add<kDecayDirect>(
                projected,
                state[key],
                k_value
            );
        }
        const float update = core_mul<kDecayDirect>(value - projected, beta);
        for (unsigned int key = 0u; key < kKeyDim; ++key) {
            const float k_value =
                postconv[post_base + kKeyFeatures + key_base + key];
            const float q_value = postconv[post_base + key_base + key];
            state[key] = core_mul_add<kDecayDirect>(
                state[key],
                update,
                k_value
            );
            output_acc = core_mul_add<kDecayDirect>(
                output_acc,
                state[key],
                q_value
            );
        }
        outputs[static_cast<size_t>(token) * kValueFeatures + value_index] =
            bf16_round(output_acc);
    }
    for (unsigned int key = 0u; key < kKeyDim; ++key) {
        state_output[state_base + key] = state[key];
    }
}

template <bool kDecayDirect>
void launch_reference(
    const float *postconv,
    const float *gate,
    float *state,
    float *output,
    unsigned int tokens
) {
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(scalar_reference_kernel<kDecayDirect>),
        dim3(kValueHeads),
        dim3(kValueDim),
        0,
        0,
        postconv,
        gate,
        state,
        output,
        tokens
    );
}

template <typename Launch>
bool time_launch(
    Launch &&launch,
    unsigned int repetitions,
    float *mean_ms,
    hipStream_t stream = nullptr
) {
    launch();
    if (hipGetLastError() != hipSuccess ||
        hipStreamSynchronize(stream) != hipSuccess) {
        return false;
    }
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&stop) != hipSuccess ||
        hipEventRecord(start, stream) != hipSuccess) {
        return false;
    }
    for (unsigned int rep = 0u; rep < repetitions; ++rep) launch();
    if (hipGetLastError() != hipSuccess ||
        hipEventRecord(stop, stream) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess) {
        return false;
    }
    float elapsed = 0.0f;
    const hipError_t status = hipEventElapsedTime(&elapsed, start, stop);
    hipEventDestroy(stop);
    hipEventDestroy(start);
    if (status != hipSuccess) return false;
    *mean_ms = elapsed / static_cast<float>(repetitions);
    return true;
}

uint64_t fnv1a64(const std::vector<float> &values) {
    uint64_t hash = UINT64_C(1469598103934665603);
    const auto *bytes = reinterpret_cast<const uint8_t *>(values.data());
    const size_t byte_count = values.size() * sizeof(float);
    for (size_t index = 0u; index < byte_count; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

Metrics compare(
    const std::vector<float> &reference,
    const std::vector<float> &candidate
) {
    Metrics result;
    double squared_sum = 0.0;
    size_t finite_pairs = 0u;
    for (size_t index = 0u; index < reference.size(); ++index) {
        if (std::memcmp(&reference[index], &candidate[index], sizeof(float)) != 0) {
            ++result.exact_mismatches;
        }
        const bool reference_finite = std::isfinite(reference[index]);
        const bool candidate_finite = std::isfinite(candidate[index]);
        if (!reference_finite) ++result.reference_nonfinite;
        if (!candidate_finite) ++result.candidate_nonfinite;
        if (reference_finite && candidate_finite) {
            const float difference = std::fabs(reference[index] - candidate[index]);
            result.max_abs = std::max(result.max_abs, difference);
            squared_sum += static_cast<double>(difference) * difference;
            ++finite_pairs;
        }
    }
    result.rmse = finite_pairs == 0u
        ? std::numeric_limits<double>::infinity()
        : std::sqrt(squared_sum / static_cast<double>(finite_pairs));
    result.reference_hash = fnv1a64(reference);
    result.candidate_hash = fnv1a64(candidate);
    return result;
}

float host_bf16_round(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    bits &= UINT32_C(0xffff0000);
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<float> make_gate(
    unsigned int tokens,
    bool decay_direct,
    float *log_min,
    float *log_max
) {
    std::vector<float> gate(static_cast<size_t>(tokens) * kGateOutputRows);
    *log_min = std::numeric_limits<float>::infinity();
    *log_max = -std::numeric_limits<float>::infinity();
    for (unsigned int token = 0u; token < tokens; ++token) {
        const size_t base = static_cast<size_t>(token) * kGateOutputRows;
        for (unsigned int head = 0u; head < kGateRows; ++head) {
            const unsigned int phase = (37u * token + 101u * head) & 1023u;
            const float log_g = -(
                0.000125f +
                0.031125f * static_cast<float>(phase) / 1023.0f
            );
            *log_min = std::min(*log_min, log_g);
            *log_max = std::max(*log_max, log_g);
            gate[base + head] = decay_direct ? std::exp(log_g) : log_g;
            const float beta =
                0.05f +
                0.9f * static_cast<float>((13u * token + 17u * head) % 997u) /
                    996.0f;
            gate[base + kGateRows + head] = host_bf16_round(beta);
        }
    }
    return gate;
}

bool load_provider(const char *path, ProviderApi *api) {
#if defined(_WIN32)
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) return false;
    api->prepare = reinterpret_cast<PrepareFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    api->launch = reinterpret_cast<LaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_launch")
    );
    api->launch_async = reinterpret_cast<LaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_launch_async")
    );
    api->dynamic_launch_async = reinterpret_cast<DynamicLaunchFunction>(
        GetProcAddress(
            api->module,
            "qrt_aiter_fused_gdn_launch_async_dynamic"
        )
    );
    api->q16384_launch = reinterpret_cast<LaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q16384_launch")
    );
    api->q16384_launch_async = reinterpret_cast<LaunchFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q16384_launch_async")
    );
    api->scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_scratch_bytes")
    );
    api->last_error = reinterpret_cast<LastErrorFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    api->release = reinterpret_cast<ReleaseFunction>(
        GetProcAddress(api->module, "qrt_aiter_fused_gdn_q8192_release")
    );
    return api->prepare != nullptr && api->launch != nullptr &&
        api->launch_async != nullptr &&
        api->dynamic_launch_async != nullptr &&
        api->q16384_launch != nullptr &&
        api->q16384_launch_async != nullptr &&
        api->scratch_bytes != nullptr && api->last_error != nullptr &&
        api->release != nullptr;
#else
    api->module = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (api->module == nullptr) return false;
    api->prepare = reinterpret_cast<PrepareFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q8192_prepare")
    );
    api->launch = reinterpret_cast<LaunchFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q8192_launch")
    );
    api->launch_async = reinterpret_cast<LaunchFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q8192_launch_async")
    );
    api->dynamic_launch_async = reinterpret_cast<DynamicLaunchFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_launch_async_dynamic")
    );
    api->q16384_launch = reinterpret_cast<LaunchFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q16384_launch")
    );
    api->q16384_launch_async = reinterpret_cast<LaunchFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q16384_launch_async")
    );
    api->scratch_bytes = reinterpret_cast<ScratchBytesFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q8192_scratch_bytes")
    );
    api->last_error = reinterpret_cast<LastErrorFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q8192_last_error")
    );
    api->release = reinterpret_cast<ReleaseFunction>(
        dlsym(api->module, "qrt_aiter_fused_gdn_q8192_release")
    );
    return api->prepare != nullptr && api->launch != nullptr &&
        api->launch_async != nullptr &&
        api->dynamic_launch_async != nullptr &&
        api->q16384_launch != nullptr &&
        api->q16384_launch_async != nullptr &&
        api->scratch_bytes != nullptr && api->last_error != nullptr &&
        api->release != nullptr;
#endif
}

}  // namespace

int main(int argc, char **argv) {
    unsigned int repetitions = 3u;
    unsigned int tokens = kQ8192Tokens;
    if ((argc != 4 && argc != 5) ||
        !parse_repetitions(argv[3], &repetitions) ||
        (argc == 5 && !parse_tokens(argv[4], &tokens))) {
        std::cerr << "usage: q8192_aiter_fused_gdn_smoke KERNEL_DIR "
                     "PROVIDER_DLL REPETITIONS>=3 "
                     "[8191|8192|8193|16384]\n";
        return 2;
    }

    hipError_t status = hipInit(0);
    if (status == hipSuccess) status = hipSetDevice(0);
    if (status != hipSuccess) return fail("hip_init", status);

    ProviderApi api;
    if (!load_provider(argv[2], &api)) {
        std::cerr << "q8192_aiter_fused_gdn_smoke stage=load_provider\n";
        return 2;
    }
    if (api.prepare(argv[1]) == 0) {
        std::cerr << "q8192_aiter_fused_gdn_smoke stage=provider_prepare error="
                  << api.last_error() << std::endl;
        return 2;
    }
    const LaunchFunction provider_launch =
        tokens == kQ16384Tokens ? api.q16384_launch : api.launch;
    const LaunchFunction provider_launch_async =
        tokens == kQ16384Tokens
        ? api.q16384_launch_async
        : api.launch_async;
    const bool use_dynamic_launch =
        tokens == kQ8191Tokens || tokens == kQ8193Tokens;
    hipStream_t provider_stream = nullptr;
    status = hipStreamCreate(&provider_stream);
    if (status != hipSuccess) return fail("hipStreamCreate(provider)", status);

    const size_t postconv_elements = static_cast<size_t>(tokens) * kQkvRows;
    const size_t gate_elements = static_cast<size_t>(tokens) * kGateOutputRows;
    const size_t output_elements = static_cast<size_t>(tokens) * kValueFeatures;
    const size_t state_elements = static_cast<size_t>(kValueFeatures) * kKeyDim;
    float *postconv = nullptr;
    float *gate = nullptr;
    float *reference_output = nullptr;
    float *provider_output = nullptr;
    float *provider_async_output = nullptr;
    float *reference_state = nullptr;
    float *provider_state = nullptr;
    float *provider_async_state = nullptr;
    const auto allocate = [](float **pointer, size_t elements) {
        return hipMalloc(reinterpret_cast<void **>(pointer), elements * sizeof(float));
    };
    status = allocate(&postconv, postconv_elements);
    if (status == hipSuccess) status = allocate(&gate, gate_elements);
    if (status == hipSuccess) status = allocate(&reference_output, output_elements);
    if (status == hipSuccess) status = allocate(&provider_output, output_elements);
    if (status == hipSuccess) status = allocate(&provider_async_output, output_elements);
    if (status == hipSuccess) status = allocate(&reference_state, state_elements);
    if (status == hipSuccess) status = allocate(&provider_state, state_elements);
    if (status == hipSuccess) status = allocate(&provider_async_state, state_elements);
    if (status != hipSuccess) return fail("hipMalloc", status);

    hipLaunchKernelGGL(
        fill_postconv_kernel,
        dim3(static_cast<uint32_t>(
            (postconv_elements + kFillThreads - 1u) / kFillThreads
        )),
        dim3(kFillThreads),
        0,
        0,
        postconv,
        postconv_elements
    );
    status = hipGetLastError();
    if (status == hipSuccess) status = hipDeviceSynchronize();
    if (status != hipSuccess) return fail("fill_postconv", status);

    std::vector<float> host_reference_output(output_elements);
    std::vector<float> host_provider_output(output_elements);
    std::vector<float> host_provider_async_output(output_elements);
    std::vector<float> host_reference_state(state_elements);
    std::vector<float> host_provider_state(state_elements);
    std::vector<float> host_provider_async_state(state_elements);
    bool all_modes_close = api.scratch_bytes() == 0u;

    for (int mode = 0; mode < 2; ++mode) {
        const bool decay_direct = mode == 0;
        float log_min = 0.0f;
        float log_max = 0.0f;
        const std::vector<float> host_gate =
            make_gate(tokens, decay_direct, &log_min, &log_max);
        status = hipMemcpy(
            gate,
            host_gate.data(),
            gate_elements * sizeof(float),
            hipMemcpyHostToDevice
        );
        if (status != hipSuccess) return fail("hipMemcpy(gate)", status);

        float reference_ms = 0.0f;
        float provider_ms = 0.0f;
        float provider_async_ms = 0.0f;
        const bool reference_timed = decay_direct
            ? time_launch(
                  [&]() {
                      launch_reference<true>(
                          postconv,
                          gate,
                          reference_state,
                          reference_output,
                          tokens
                      );
                  },
                  repetitions,
                  &reference_ms
              )
            : time_launch(
                  [&]() {
                      launch_reference<false>(
                          postconv,
                          gate,
                          reference_state,
                          reference_output,
                          tokens
                      );
                  },
                  repetitions,
                  &reference_ms
              );
        if (!reference_timed) return fail("reference_timing", hipGetLastError());

        bool provider_launch_ok = true;
        const bool provider_timed = time_launch(
            [&]() {
                const int launch_status = use_dynamic_launch
                    ? api.dynamic_launch_async(
                          postconv,
                          gate,
                          provider_output,
                          provider_state,
                          decay_direct ? 1 : 0,
                          provider_stream,
                          static_cast<int32_t>(tokens)
                      )
                    : provider_launch(
                        postconv,
                        gate,
                        provider_output,
                        provider_state,
                        decay_direct ? 1 : 0,
                        provider_stream
                      );
                if (launch_status == 0) {
                    provider_launch_ok = false;
                }
            },
            repetitions,
            &provider_ms,
            provider_stream
        );
        if (!provider_timed || !provider_launch_ok) {
            std::cerr << "q8192_aiter_fused_gdn_smoke stage=provider_timing error="
                      << api.last_error() << std::endl;
            return 2;
        }

        bool provider_async_launch_ok = true;
        const bool provider_async_timed = time_launch(
            [&]() {
                const int launch_status = use_dynamic_launch
                    ? api.dynamic_launch_async(
                          postconv,
                          gate,
                          provider_async_output,
                          provider_async_state,
                          decay_direct ? 1 : 0,
                          provider_stream,
                          static_cast<int32_t>(tokens)
                      )
                    : provider_launch_async(
                        postconv,
                        gate,
                        provider_async_output,
                        provider_async_state,
                        decay_direct ? 1 : 0,
                        provider_stream
                      );
                if (launch_status == 0) {
                    provider_async_launch_ok = false;
                }
            },
            repetitions,
            &provider_async_ms,
            provider_stream
        );
        if (!provider_async_timed || !provider_async_launch_ok) {
            std::cerr
                << "q8192_aiter_fused_gdn_smoke stage=provider_async_timing error="
                << api.last_error() << std::endl;
            return 2;
        }

        status = hipMemcpy(
            host_reference_output.data(),
            reference_output,
            output_elements * sizeof(float),
            hipMemcpyDeviceToHost
        );
        if (status == hipSuccess) {
            status = hipMemcpy(
                host_provider_output.data(),
                provider_output,
                output_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                host_provider_async_output.data(),
                provider_async_output,
                output_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                host_reference_state.data(),
                reference_state,
                state_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                host_provider_state.data(),
                provider_state,
                state_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status == hipSuccess) {
            status = hipMemcpy(
                host_provider_async_state.data(),
                provider_async_state,
                state_elements * sizeof(float),
                hipMemcpyDeviceToHost
            );
        }
        if (status != hipSuccess) return fail("hipMemcpy(results)", status);

        const Metrics output_metrics =
            compare(host_reference_output, host_provider_output);
        const Metrics state_metrics =
            compare(host_reference_state, host_provider_state);
        const Metrics async_output_parity =
            compare(host_provider_output, host_provider_async_output);
        const Metrics async_state_parity =
            compare(host_provider_state, host_provider_async_state);
        const bool negative_log_range = log_min < 0.0f && log_max < 0.0f;
        const bool finite =
            output_metrics.reference_nonfinite == 0u &&
            output_metrics.candidate_nonfinite == 0u &&
            state_metrics.reference_nonfinite == 0u &&
            state_metrics.candidate_nonfinite == 0u;
        // This is a standalone numerical screen, not a product correctness
        // claim.  Exact mismatch counts and hashes remain visible below.
        const float output_tolerance =
            tokens == kQ16384Tokens ? 0.000020f : 0.000030517578125f;
        const float state_tolerance =
            tokens == kQ16384Tokens ? 0.000000020f : 0.000010f;
        const bool provider_within_bound = provider_ms <= 40.0f;
        const bool close =
            negative_log_range && finite &&
            output_metrics.max_abs <= output_tolerance &&
            state_metrics.max_abs <= state_tolerance &&
            provider_within_bound;
        const bool async_exact =
            async_output_parity.exact_mismatches == 0u &&
            async_state_parity.exact_mismatches == 0u &&
            async_output_parity.candidate_nonfinite == 0u &&
            async_state_parity.candidate_nonfinite == 0u;
        all_modes_close = all_modes_close && close && async_exact;

        std::cout << std::scientific << std::setprecision(9)
                  << "q" << tokens << "_aiter_fused_gdn_mode"
                  << " mode=" << (decay_direct ? "decay" : "log_g")
                  << " tokens=" << tokens
                  << " provider_surface="
                  << (use_dynamic_launch ? "dynamic" : "fixed")
                  << " repetitions=" << repetitions
                  << " log_g_min=" << log_min
                  << " log_g_max=" << log_max
                  << " negative_log_range=" << (negative_log_range ? 1 : 0)
                  << " output_mismatches=" << output_metrics.exact_mismatches
                  << " output_max_abs=" << output_metrics.max_abs
                  << " output_rmse=" << output_metrics.rmse
                  << " output_reference_nonfinite="
                  << output_metrics.reference_nonfinite
                  << " output_provider_nonfinite="
                  << output_metrics.candidate_nonfinite
                  << " state_mismatches=" << state_metrics.exact_mismatches
                  << " state_max_abs=" << state_metrics.max_abs
                  << " state_rmse=" << state_metrics.rmse
                  << " state_reference_nonfinite="
                  << state_metrics.reference_nonfinite
                  << " state_provider_nonfinite="
                  << state_metrics.candidate_nonfinite
                  << " output_reference_hash=" << std::hex
                  << output_metrics.reference_hash
                  << " output_provider_hash=" << output_metrics.candidate_hash
                  << " state_reference_hash=" << state_metrics.reference_hash
                  << " state_provider_hash=" << state_metrics.candidate_hash
                  << " async_output_mismatches=" << std::dec
                  << async_output_parity.exact_mismatches
                  << " async_state_mismatches="
                  << async_state_parity.exact_mismatches
                  << " async_output_hash=" << std::hex
                  << async_output_parity.candidate_hash
                  << " async_state_hash="
                  << async_state_parity.candidate_hash
                  << std::dec
                  << " output0_reference=" << host_reference_output.front()
                  << " output0_provider=" << host_provider_output.front()
                  << " state0_reference=" << host_reference_state.front()
                  << " state0_provider=" << host_provider_state.front()
                  << " reference_ms=" << reference_ms
                  << " provider_ms=" << provider_ms
                  << " provider_async_one_sync_ms=" << provider_async_ms
                  << " provider_le_40ms="
                  << (provider_within_bound ? 1 : 0)
                  << " output_tolerance=" << output_tolerance
                  << " state_tolerance=" << state_tolerance
                  << " async_exact=" << (async_exact ? 1 : 0)
                  << " close=" << (close ? 1 : 0)
                  << std::endl;
    }

    status = hipStreamDestroy(provider_stream);
    if (status != hipSuccess) return fail("hipStreamDestroy(provider)", status);
    api.release();
#if defined(_WIN32)
    FreeLibrary(api.module);
#else
    dlclose(api.module);
#endif
    hipFree(provider_state);
    hipFree(provider_async_state);
    hipFree(reference_state);
    hipFree(provider_output);
    hipFree(provider_async_output);
    hipFree(reference_output);
    hipFree(gate);
    hipFree(postconv);

    std::cout << "q" << tokens << "_aiter_fused_gdn_smoke status="
              << (all_modes_close ? "pass" : "fail")
              << " target_device=AMD395"
              << " scratch_bytes=0"
              << " state_layout=value_head_value_key"
              << std::endl;
    return all_modes_close ? 0 : 3;
}
