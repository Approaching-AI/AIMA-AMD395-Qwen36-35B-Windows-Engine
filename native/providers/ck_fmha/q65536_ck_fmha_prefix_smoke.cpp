#include <hip/hip_runtime.h>

#define NOMINMAX
#include <windows.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>

namespace {

constexpr unsigned int kPrefixTokens = 32768u;
constexpr unsigned int kQ65536Tokens = 65536u;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kThreads = 256u;
constexpr unsigned int kBlocks = 4096u;
constexpr float kPrefixMaxAbsTolerance = 1.0e-4f;

using Bf16LaunchFn = int (__cdecl *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *);

struct DeviceCheck {
    unsigned long long mismatches;
    unsigned long long prefix_nonfinite;
    unsigned long long full_nonfinite;
    unsigned long long first_mismatch;
    unsigned int max_abs_bits;
    unsigned long long chunk_nonfinite;
    unsigned long long schedule_mismatches;
    unsigned long long first_schedule_mismatch;
    unsigned int schedule_max_abs_bits;
};

__device__ uint16_t f32_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t lsb = (bits >> 16) & 1u;
    return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

__global__ void initialize_bf16(
    uint16_t *values,
    size_t elements,
    uint32_t seed,
    float scale) {
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        uint32_t x = static_cast<uint32_t>(index) ^ seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        const int centered =
            static_cast<int>((x >> 8) & UINT32_C(0xffff)) - 32768;
        values[index] = f32_to_bf16(
            static_cast<float>(centered) * (scale / 32768.0f));
    }
}

__global__ void compare_prefix(
    const float *q32768,
    const float *q65536,
    size_t elements,
    DeviceCheck *check) {
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float lhs = q32768[index];
        const float rhs = q65536[index];
        if (!isfinite(lhs) || !isfinite(rhs)) {
            atomicAdd(&check->prefix_nonfinite, 1ull);
            continue;
        }
        if (__float_as_uint(lhs) != __float_as_uint(rhs)) {
            atomicAdd(&check->mismatches, 1ull);
            atomicMin(
                &check->first_mismatch,
                static_cast<unsigned long long>(index));
        }
        atomicMax(
            &check->max_abs_bits,
            __float_as_uint(fabsf(lhs - rhs)));
    }
}

__global__ void scan_finite(
    const float *values,
    size_t elements,
    DeviceCheck *check) {
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (!isfinite(values[index])) {
            atomicAdd(&check->full_nonfinite, 1ull);
        }
    }
}

__global__ void compare_schedule(
    const float *monolithic,
    const float *chunked,
    size_t elements,
    DeviceCheck *check) {
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float lhs = monolithic[index];
        const float rhs = chunked[index];
        if (!isfinite(rhs)) {
            atomicAdd(&check->chunk_nonfinite, 1ull);
            continue;
        }
        if (!isfinite(lhs)) {
            continue;
        }
        if (__float_as_uint(lhs) != __float_as_uint(rhs)) {
            atomicAdd(&check->schedule_mismatches, 1ull);
            atomicMin(
                &check->first_schedule_mismatch,
                static_cast<unsigned long long>(index));
        }
        atomicMax(
            &check->schedule_max_abs_bits,
            __float_as_uint(fabsf(lhs - rhs)));
    }
}

int fail(const char *stage, hipError_t status) {
    std::cerr << "ck_fmha_q65536_prefix_smoke stage=" << stage
              << " status=" << static_cast<int>(status)
              << " error=" << hipGetErrorString(status) << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr
            << "usage: q65536_ck_fmha_prefix_smoke PROVIDER_DLL"
            << std::endl;
        return 2;
    }

    HMODULE dll = LoadLibraryA(argv[1]);
    if (dll == nullptr) {
        std::cerr
            << "ck_fmha_q65536_prefix_smoke LoadLibrary failed error="
            << GetLastError() << std::endl;
        return 1;
    }
    const auto q32768_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q32768_bf16_launch"));
    const auto q65536_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q65536_bf16_launch"));
    const auto q65536_chunk8192_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(
            dll,
            "qrt_ck_fmha_q65536_chunk8192_bf16_launch"));
    if (q32768_launch == nullptr || q65536_launch == nullptr ||
        q65536_chunk8192_launch == nullptr) {
        std::cerr
            << "ck_fmha_q65536_prefix_smoke provider symbols missing"
            << std::endl;
        FreeLibrary(dll);
        return 1;
    }

    const size_t q_elements =
        static_cast<size_t>(kQ65536Tokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kQ65536Tokens) * kKvFeatures;
    const size_t prefix_elements =
        static_cast<size_t>(kPrefixTokens) * kQueryFeatures;
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
    float *q32768_output = nullptr;
    float *q65536_output = nullptr;
    float *q65536_chunk8192_output = nullptr;
    DeviceCheck *device_check = nullptr;
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    hipError_t status = hipSuccess;

    if ((status = hipMalloc(
             reinterpret_cast<void **>(&q),
             q_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&k),
             kv_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&v),
             kv_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q32768_output),
             prefix_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q65536_output),
             q_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q65536_chunk8192_output),
             q_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&device_check),
             sizeof(DeviceCheck))) != hipSuccess ||
        (status = hipEventCreate(&start)) != hipSuccess ||
        (status = hipEventCreate(&stop)) != hipSuccess) {
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        (void)hipFree(device_check);
        (void)hipFree(q65536_chunk8192_output);
        (void)hipFree(q65536_output);
        (void)hipFree(q32768_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        FreeLibrary(dll);
        return fail("allocate", status);
    }

    initialize_bf16<<<kBlocks, kThreads>>>(
        q, q_elements, UINT32_C(0x91e10da5), 0.125f);
    initialize_bf16<<<kBlocks, kThreads>>>(
        k, kv_elements, UINT32_C(0xa511e9b3), 0.125f);
    initialize_bf16<<<kBlocks, kThreads>>>(
        v, kv_elements, UINT32_C(0x7f4a7c15), 0.25f);
    if ((status = hipDeviceSynchronize()) != hipSuccess ||
        q32768_launch(q, k, v, q32768_output, nullptr) != 0 ||
        (status = hipDeviceSynchronize()) != hipSuccess ||
        (status = hipEventRecord(start)) != hipSuccess ||
        q65536_launch(q, k, v, q65536_output, nullptr) != 0 ||
        (status = hipEventRecord(stop)) != hipSuccess ||
        (status = hipEventSynchronize(stop)) != hipSuccess) {
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        (void)hipFree(device_check);
        (void)hipFree(q65536_chunk8192_output);
        (void)hipFree(q65536_output);
        (void)hipFree(q32768_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        FreeLibrary(dll);
        return fail("launch", status);
    }

    float q65536_ms = 0.0f;
    if ((status = hipEventElapsedTime(
             &q65536_ms, start, stop)) != hipSuccess ||
        (status = hipEventRecord(start)) != hipSuccess ||
        q65536_chunk8192_launch(
            q, k, v, q65536_chunk8192_output, nullptr) != 0 ||
        (status = hipEventRecord(stop)) != hipSuccess ||
        (status = hipEventSynchronize(stop)) != hipSuccess) {
        return fail("chunk8192_launch", status);
    }
    float q65536_chunk8192_ms = 0.0f;
    DeviceCheck initial{};
    initial.first_mismatch =
        (std::numeric_limits<unsigned long long>::max)();
    initial.first_schedule_mismatch =
        (std::numeric_limits<unsigned long long>::max)();
    if ((status = hipEventElapsedTime(
             &q65536_chunk8192_ms, start, stop)) != hipSuccess ||
        (status = hipMemcpy(
             device_check,
             &initial,
             sizeof(initial),
             hipMemcpyHostToDevice)) != hipSuccess) {
        return fail("check_initialize", status);
    }
    compare_prefix<<<kBlocks, kThreads>>>(
        q32768_output, q65536_output, prefix_elements, device_check);
    scan_finite<<<kBlocks, kThreads>>>(
        q65536_output, q_elements, device_check);
    compare_schedule<<<kBlocks, kThreads>>>(
        q65536_output,
        q65536_chunk8192_output,
        q_elements,
        device_check);
    DeviceCheck host_check{};
    if ((status = hipDeviceSynchronize()) != hipSuccess ||
        (status = hipMemcpy(
             &host_check,
             device_check,
             sizeof(host_check),
             hipMemcpyDeviceToHost)) != hipSuccess) {
        return fail("check", status);
    }

    float max_abs = 0.0f;
    float schedule_max_abs = 0.0f;
    static_assert(sizeof(max_abs) == sizeof(host_check.max_abs_bits));
    std::memcpy(&max_abs, &host_check.max_abs_bits, sizeof(max_abs));
    std::memcpy(
        &schedule_max_abs,
        &host_check.schedule_max_abs_bits,
        sizeof(schedule_max_abs));
    const bool pass =
        host_check.prefix_nonfinite == 0u &&
        host_check.full_nonfinite == 0u &&
        host_check.chunk_nonfinite == 0u &&
        max_abs <= kPrefixMaxAbsTolerance;
    std::cout << std::fixed << std::setprecision(6)
              << "ck_fmha_q65536_prefix_smoke q65536_ms=" << q65536_ms
              << " prefix_mismatches=" << host_check.mismatches
              << " prefix_nonfinite=" << host_check.prefix_nonfinite
              << " full_nonfinite=" << host_check.full_nonfinite
              << " chunk8192_ms=" << q65536_chunk8192_ms
              << " chunk_nonfinite=" << host_check.chunk_nonfinite
              << " schedule_mismatches="
              << host_check.schedule_mismatches
              << " first_schedule_mismatch="
              << host_check.first_schedule_mismatch
              << " schedule_max_abs=" << schedule_max_abs
              << " first_mismatch=" << host_check.first_mismatch
              << " max_abs=" << max_abs
              << " tolerance=" << kPrefixMaxAbsTolerance
              << " pass=" << (pass ? 1 : 0)
              << std::endl;

    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    (void)hipFree(device_check);
    (void)hipFree(q65536_chunk8192_output);
    (void)hipFree(q65536_output);
    (void)hipFree(q32768_output);
    (void)hipFree(v);
    (void)hipFree(k);
    (void)hipFree(q);
    FreeLibrary(dll);
    return pass ? 0 : 1;
}
