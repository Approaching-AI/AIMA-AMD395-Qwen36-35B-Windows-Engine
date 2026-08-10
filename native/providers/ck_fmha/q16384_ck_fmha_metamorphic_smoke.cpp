#include <hip/hip_runtime.h>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

constexpr unsigned int kQ8192Tokens = 8192u;
constexpr unsigned int kQ16384Tokens = 16384u;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kPackedRows =
    2u * kQueryFeatures + 2u * kKvFeatures;
constexpr unsigned int kThreads = 256u;
constexpr float kQ16384DirectMeanMsCeiling = 250.0f;

using PrepareFn = int (__cdecl *)();
using PackedLaunchFn = int (__cdecl *)(const float *, float *, void *);
using Bf16LaunchFn = int (__cdecl *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *);
using ReleaseFn = int (__cdecl *)();

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

__global__ void initialize_packed(float *values, size_t elements) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    uint32_t x = static_cast<uint32_t>(index) ^ 0x91e10da5u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    const int centered = static_cast<int>((x >> 8) & 0xffffu) - 32768;
    values[index] = static_cast<float>(centered) * (0.5f / 32768.0f);
}

__global__ void pack_direct_qkv(
    const float *__restrict__ packed,
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    uint16_t *__restrict__ v,
    unsigned int tokens) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_elements =
        static_cast<size_t>(tokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(tokens) * kKvFeatures;
    if (index < q_elements) {
        const size_t token = index / kQueryFeatures;
        const size_t feature = index - token * kQueryFeatures;
        q[index] = f32_to_bf16(packed[token * kPackedRows + feature]);
        return;
    }
    const size_t kv_index = index - q_elements;
    if (kv_index >= 2u * kv_elements) {
        return;
    }
    const size_t tensor_index = kv_index % kv_elements;
    const size_t token = tensor_index / kKvFeatures;
    const size_t feature = tensor_index - token * kKvFeatures;
    if (kv_index < kv_elements) {
        k[tensor_index] = f32_to_bf16(
            packed[token * kPackedRows + 2u * kQueryFeatures + feature]);
    } else {
        v[tensor_index] = f32_to_bf16(
            packed[token * kPackedRows +
                   2u * kQueryFeatures + kKvFeatures + feature]);
    }
}

__global__ void compare_outputs(
    const float *lhs,
    const float *rhs,
    size_t elements,
    DeviceComparison *comparison) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float lhs_value = lhs[index];
    const float rhs_value = rhs[index];
    const unsigned int lhs_bits = __float_as_uint(lhs_value);
    const unsigned int rhs_bits = __float_as_uint(rhs_value);
    if (lhs_bits != rhs_bits) {
        atomicAdd(&comparison->mismatches, 1ull);
        atomicMin(
            &comparison->first_mismatch,
            static_cast<unsigned long long>(index));
    }
    if (!isfinite(lhs_value) || !isfinite(rhs_value)) {
        atomicAdd(&comparison->nonfinite, 1ull);
        return;
    }
    const float abs_diff = fabsf(lhs_value - rhs_value);
    atomicMax(&comparison->max_abs_bits, __float_as_uint(abs_diff));
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

int fail(const char *stage, hipError_t status) {
    std::cerr << "ck_fmha_q16384_metamorphic_smoke stage=" << stage
              << " status=" << static_cast<int>(status)
              << " error=" << hipGetErrorString(status) << std::endl;
    return 1;
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
        for (unsigned int i = 0u; i < repetitions; ++i) {
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
            << "usage: q16384_ck_fmha_metamorphic_smoke PROVIDER_DLL "
               "REPETITIONS"
            << std::endl;
        return 2;
    }
    unsigned int repetitions = 0u;
    if (!parse_positive(argv[2], &repetitions)) {
        return 2;
    }

    HMODULE dll = LoadLibraryA(argv[1]);
    if (dll == nullptr) {
        std::cerr
            << "ck_fmha_q16384_metamorphic_smoke LoadLibrary failed error="
            << GetLastError() << std::endl;
        return 1;
    }
    const auto prepare = reinterpret_cast<PrepareFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_prepare"));
    const auto q8192_direct = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_bf16_launch"));
    const auto q16384_packed = reinterpret_cast<PackedLaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q16384_f32_launch"));
    const auto q16384_direct = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q16384_bf16_launch"));
    const auto release = reinterpret_cast<ReleaseFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_release"));
    if (prepare == nullptr || q8192_direct == nullptr ||
        q16384_packed == nullptr || q16384_direct == nullptr ||
        release == nullptr || prepare() != 0) {
        std::cerr
            << "ck_fmha_q16384_metamorphic_smoke provider symbols/prepare failed"
            << std::endl;
        FreeLibrary(dll);
        return 1;
    }

    const size_t q16384_elements =
        static_cast<size_t>(kQ16384Tokens) * kQueryFeatures;
    const size_t q8192_elements =
        static_cast<size_t>(kQ8192Tokens) * kQueryFeatures;
    const size_t kv16384_elements =
        static_cast<size_t>(kQ16384Tokens) * kKvFeatures;
    const size_t packed_elements =
        static_cast<size_t>(kQ16384Tokens) * kPackedRows;
    float *packed = nullptr;
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
    float *q16384_packed_output = nullptr;
    float *q16384_direct_output = nullptr;
    float *q8192_direct_output = nullptr;
    DeviceComparison *comparison = nullptr;
    hipError_t status = hipSuccess;
    if ((status = hipMalloc(
             reinterpret_cast<void **>(&packed),
             packed_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q),
             q16384_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&k),
             kv16384_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&v),
             kv16384_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q16384_packed_output),
             q16384_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q16384_direct_output),
             q16384_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q8192_direct_output),
             q8192_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&comparison),
             sizeof(DeviceComparison))) != hipSuccess) {
        (void)release();
        (void)hipFree(comparison);
        (void)hipFree(q8192_direct_output);
        (void)hipFree(q16384_direct_output);
        (void)hipFree(q16384_packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        FreeLibrary(dll);
        return fail("hipMalloc", status);
    }

    initialize_packed<<<
        dim3(static_cast<unsigned int>(
            (packed_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(packed, packed_elements);
    pack_direct_qkv<<<
        dim3(static_cast<unsigned int>(
            (q16384_elements + 2u * kv16384_elements + kThreads - 1u) /
            kThreads)),
        dim3(kThreads)>>>(
            packed,
            q,
            k,
            v,
            kQ16384Tokens);
    if ((status = hipDeviceSynchronize()) != hipSuccess) {
        (void)release();
        (void)hipFree(comparison);
        (void)hipFree(q8192_direct_output);
        (void)hipFree(q16384_direct_output);
        (void)hipFree(q16384_packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        FreeLibrary(dll);
        return fail("initialize", status);
    }

    float q16384_packed_ms = 0.0f;
    float q16384_direct_ms = 0.0f;
    float q8192_direct_ms = 0.0f;
    const bool timed =
        time_launch(
            [&]() {
                return q16384_packed(
                    packed,
                    q16384_packed_output,
                    nullptr);
            },
            repetitions,
            &q16384_packed_ms) &&
        time_launch(
            [&]() {
                return q16384_direct(
                    q,
                    k,
                    v,
                    q16384_direct_output,
                    nullptr);
            },
            repetitions,
            &q16384_direct_ms) &&
        time_launch(
            [&]() {
                return q8192_direct(
                    q,
                    k,
                    v,
                    q8192_direct_output,
                    nullptr);
            },
            repetitions,
            &q8192_direct_ms);
    if (!timed) {
        status = hipGetLastError();
        (void)release();
        (void)hipFree(comparison);
        (void)hipFree(q8192_direct_output);
        (void)hipFree(q16384_direct_output);
        (void)hipFree(q16384_packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        FreeLibrary(dll);
        return fail("timed_launch", status);
    }

    DeviceComparison packed_vs_direct{};
    DeviceComparison prefix_vs_q8192{};
    if (!reset_comparison(comparison)) {
        status = hipGetLastError();
    } else {
        compare_outputs<<<
            dim3(static_cast<unsigned int>(
                (q16384_elements + kThreads - 1u) / kThreads)),
            dim3(kThreads)>>>(
                q16384_packed_output,
                q16384_direct_output,
                q16384_elements,
                comparison);
        status = hipDeviceSynchronize();
    }
    if (status == hipSuccess) {
        status = hipMemcpy(
            &packed_vs_direct,
            comparison,
            sizeof(packed_vs_direct),
            hipMemcpyDeviceToHost);
    }
    if (status == hipSuccess && reset_comparison(comparison)) {
        compare_outputs<<<
            dim3(static_cast<unsigned int>(
                (q8192_elements + kThreads - 1u) / kThreads)),
            dim3(kThreads)>>>(
                q16384_direct_output,
                q8192_direct_output,
                q8192_elements,
                comparison);
        status = hipDeviceSynchronize();
    }
    if (status == hipSuccess) {
        status = hipMemcpy(
            &prefix_vs_q8192,
            comparison,
            sizeof(prefix_vs_q8192),
            hipMemcpyDeviceToHost);
    }

    (void)release();
    (void)hipFree(comparison);
    (void)hipFree(q8192_direct_output);
    (void)hipFree(q16384_direct_output);
    (void)hipFree(q16384_packed_output);
    (void)hipFree(v);
    (void)hipFree(k);
    (void)hipFree(q);
    (void)hipFree(packed);
    FreeLibrary(dll);
    if (status != hipSuccess) {
        return fail("compare", status);
    }

    float packed_max_abs = 0.0f;
    float prefix_max_abs = 0.0f;
    std::memcpy(
        &packed_max_abs,
        &packed_vs_direct.max_abs_bits,
        sizeof(packed_max_abs));
    std::memcpy(
        &prefix_max_abs,
        &prefix_vs_q8192.max_abs_bits,
        sizeof(prefix_max_abs));
    std::cout
        << std::fixed << std::setprecision(6)
        << "ck_fmha_q16384_metamorphic_smoke"
        << " q16384_packed_ms=" << q16384_packed_ms
        << " q16384_direct_ms=" << q16384_direct_ms
        << " q8192_direct_ms=" << q8192_direct_ms
        << " repetitions=" << repetitions
        << " q16384_elements=" << q16384_elements
        << " packed_vs_direct_mismatches="
        << packed_vs_direct.mismatches
        << " packed_vs_direct_nonfinite="
        << packed_vs_direct.nonfinite
        << " packed_vs_direct_max_abs=" << packed_max_abs
        << " prefix_vs_q8192_mismatches="
        << prefix_vs_q8192.mismatches
        << " prefix_vs_q8192_nonfinite="
        << prefix_vs_q8192.nonfinite
        << " prefix_vs_q8192_max_abs=" << prefix_max_abs
        << " direct_ms_ceiling=" << kQ16384DirectMeanMsCeiling
        << std::endl;

    const bool pass =
        packed_vs_direct.mismatches == 0u &&
        packed_vs_direct.nonfinite == 0u &&
        packed_vs_direct.max_abs_bits == 0u &&
        prefix_vs_q8192.mismatches == 0u &&
        prefix_vs_q8192.nonfinite == 0u &&
        prefix_vs_q8192.max_abs_bits == 0u &&
        q16384_direct_ms <= kQ16384DirectMeanMsCeiling;
    return pass ? 0 : 1;
}
