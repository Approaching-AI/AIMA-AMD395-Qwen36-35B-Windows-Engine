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
constexpr unsigned int kKvTokens = kPrefixTokens + kSuffixTokens;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kThreads = 256u;
constexpr float kSuffixMeanMsCeiling = 25.0f;

using Bf16LaunchFn = int (__cdecl *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *);

struct DeviceComparison {
    unsigned long long mismatches;
    unsigned long long nonfinite;
    unsigned long long first_mismatch;
    unsigned int max_abs_bits;
};

__device__ uint16_t f32_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        if ((bits & 0x007fffffu) != 0u) {
            upper |= 0x0040u;
        }
        return upper;
    }
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__global__ void initialize_bf16(
    uint16_t *values,
    size_t elements,
    uint32_t seed) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    uint32_t x = static_cast<uint32_t>(index) ^ seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    const int centered = static_cast<int>((x >> 8) & 0xffffu) - 32768;
    values[index] = f32_to_bf16(
        static_cast<float>(centered) * (0.25f / 32768.0f));
}

__global__ void initialize_nan(float *values, size_t elements) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        values[index] = __int_as_float(0x7fc00000);
    }
}

__global__ void compare_outputs(
    const float *full_tail,
    const float *suffix,
    size_t elements,
    DeviceComparison *comparison) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float full_value = full_tail[index];
    const float suffix_value = suffix[index];
    const unsigned int full_bits = __float_as_uint(full_value);
    const unsigned int suffix_bits = __float_as_uint(suffix_value);
    if (full_bits != suffix_bits) {
        atomicAdd(&comparison->mismatches, 1ull);
        atomicMin(
            &comparison->first_mismatch,
            static_cast<unsigned long long>(index));
    }
    if (!isfinite(full_value) || !isfinite(suffix_value)) {
        atomicAdd(&comparison->nonfinite, 1ull);
        return;
    }
    atomicMax(
        &comparison->max_abs_bits,
        __float_as_uint(fabsf(full_value - suffix_value)));
}

bool parse_positive(const char *text, unsigned int *value) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != '\0' || parsed == 0u || parsed > 100u) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

template <typename Launch>
bool time_launch(Launch launch, unsigned int repetitions, float *mean_ms) {
    if (launch() != 0 || hipDeviceSynchronize() != hipSuccess) {
        return false;
    }
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    bool ok = hipEventCreate(&start) == hipSuccess &&
        hipEventCreate(&stop) == hipSuccess &&
        hipEventRecord(start) == hipSuccess;
    if (ok) {
        for (unsigned int iteration = 0u;
             iteration < repetitions;
             ++iteration) {
            if (launch() != 0) {
                ok = false;
                break;
            }
        }
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

int fail(const char *stage, hipError_t status) {
    std::cerr << "ck_fmha_q16384_suffix1024_smoke stage=" << stage
              << " status=" << static_cast<int>(status)
              << " error=" << hipGetErrorString(status) << std::endl;
    return 1;
}

bool reset_comparison(DeviceComparison *device) {
    DeviceComparison initial{};
    initial.first_mismatch =
        (std::numeric_limits<unsigned long long>::max)();
    return hipMemcpy(
        device,
        &initial,
        sizeof(initial),
        hipMemcpyHostToDevice) == hipSuccess;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr
            << "usage: q16384_suffix1024_ck_fmha_smoke PROVIDER_DLL "
               "REPETITIONS"
            << std::endl;
        return 2;
    }
    unsigned int repetitions = 0u;
    if (!parse_positive(argv[2], &repetitions) || repetitions < 5u) {
        return 2;
    }

    HMODULE dll = LoadLibraryA(argv[1]);
    if (dll == nullptr) {
        std::cerr
            << "ck_fmha_q16384_suffix1024_smoke LoadLibrary failed error="
            << GetLastError() << std::endl;
        return 1;
    }
    const auto full_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q17408_bf16_launch"));
    const auto suffix_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(
            dll,
            "qrt_ck_fmha_q1024_kv17408_suffix_bf16_launch"));
    if (full_launch == nullptr || suffix_launch == nullptr) {
        std::cerr
            << "ck_fmha_q16384_suffix1024_smoke provider symbols missing"
            << std::endl;
        FreeLibrary(dll);
        return 1;
    }

    const size_t full_q_elements =
        static_cast<size_t>(kKvTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kKvTokens) * kKvFeatures;
    const size_t suffix_elements =
        static_cast<size_t>(kSuffixTokens) * kQueryFeatures;
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
    float *full_output = nullptr;
    float *suffix_output = nullptr;
    DeviceComparison *device_comparison = nullptr;
    hipError_t status = hipSuccess;

    if ((status = hipMalloc(
             reinterpret_cast<void **>(&q),
             full_q_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&k),
             kv_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&v),
             kv_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&full_output),
             full_q_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&suffix_output),
             suffix_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&device_comparison),
             sizeof(DeviceComparison))) != hipSuccess) {
        (void)hipFree(device_comparison);
        (void)hipFree(suffix_output);
        (void)hipFree(full_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        FreeLibrary(dll);
        return fail("hipMalloc", status);
    }

    initialize_bf16<<<
        dim3(static_cast<unsigned int>(
            (full_q_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(q, full_q_elements, 0x91e10da5u);
    initialize_bf16<<<
        dim3(static_cast<unsigned int>(
            (kv_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(k, kv_elements, 0xa511e9b3u);
    initialize_bf16<<<
        dim3(static_cast<unsigned int>(
            (kv_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(v, kv_elements, 0x63d83595u);
    initialize_nan<<<
        dim3(static_cast<unsigned int>(
            (full_q_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(full_output, full_q_elements);
    initialize_nan<<<
        dim3(static_cast<unsigned int>(
            (suffix_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(suffix_output, suffix_elements);
    if ((status = hipDeviceSynchronize()) != hipSuccess) {
        (void)hipFree(device_comparison);
        (void)hipFree(suffix_output);
        (void)hipFree(full_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        FreeLibrary(dll);
        return fail("initialize", status);
    }

    float full_ms = 0.0f;
    float suffix_ms = 0.0f;
    const uint16_t *suffix_q =
        q + static_cast<size_t>(kPrefixTokens) * kQueryFeatures;
    const bool timed = time_launch(
        [&]() {
            return full_launch(q, k, v, full_output, nullptr);
        },
        1u,
        &full_ms) && time_launch(
        [&]() {
            return suffix_launch(
                suffix_q,
                k,
                v,
                suffix_output,
                nullptr);
        },
        repetitions,
        &suffix_ms);
    if (!timed) {
        status = hipGetLastError();
        (void)hipFree(device_comparison);
        (void)hipFree(suffix_output);
        (void)hipFree(full_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        FreeLibrary(dll);
        return fail("timed_launch", status);
    }

    DeviceComparison comparison{};
    if (!reset_comparison(device_comparison)) {
        status = hipGetLastError();
    } else {
        compare_outputs<<<
            dim3(static_cast<unsigned int>(
                (suffix_elements + kThreads - 1u) / kThreads)),
            dim3(kThreads)>>>(
                full_output +
                    static_cast<size_t>(kPrefixTokens) * kQueryFeatures,
                suffix_output,
                suffix_elements,
                device_comparison);
        status = hipDeviceSynchronize();
    }
    if (status == hipSuccess) {
        status = hipMemcpy(
            &comparison,
            device_comparison,
            sizeof(comparison),
            hipMemcpyDeviceToHost);
    }

    (void)hipFree(device_comparison);
    (void)hipFree(suffix_output);
    (void)hipFree(full_output);
    (void)hipFree(v);
    (void)hipFree(k);
    (void)hipFree(q);
    FreeLibrary(dll);
    if (status != hipSuccess) {
        return fail("compare", status);
    }

    float max_abs = 0.0f;
    std::memcpy(&max_abs, &comparison.max_abs_bits, sizeof(max_abs));
    const float ten_layer_projected_ms = 10.0f * suffix_ms;
    const float remaining_budget_ms =
        2977.5396f - ten_layer_projected_ms;
    std::cout
        << std::fixed << std::setprecision(6)
        << "ck_fmha_q16384_suffix1024_smoke"
        << " prefix_tokens=" << kPrefixTokens
        << " suffix_tokens=" << kSuffixTokens
        << " kv_tokens=" << kKvTokens
        << " mask=bottom_right"
        << " dtype=bf16"
        << " full_ms=" << full_ms
        << " suffix_ms=" << suffix_ms
        << " repetitions=" << repetitions
        << " compared_elements=" << suffix_elements
        << " mismatches=" << comparison.mismatches
        << " nonfinite=" << comparison.nonfinite
        << " first_mismatch=" << comparison.first_mismatch
        << " max_abs=" << max_abs
        << " suffix_ms_ceiling=" << kSuffixMeanMsCeiling
        << " ten_layer_projected_ms=" << ten_layer_projected_ms
        << " remaining_budget_ms=" << remaining_budget_ms
        << std::endl;

    const bool pass =
        comparison.mismatches == 0u &&
        comparison.nonfinite == 0u &&
        comparison.max_abs_bits == 0u &&
        suffix_ms <= kSuffixMeanMsCeiling &&
        ten_layer_projected_ms <= 250.0f;
    return pass ? 0 : 1;
}
