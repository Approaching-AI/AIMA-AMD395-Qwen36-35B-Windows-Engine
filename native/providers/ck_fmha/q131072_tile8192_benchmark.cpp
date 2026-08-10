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
#include <vector>

namespace {

constexpr unsigned int kTotalTokens = 131072u;
constexpr unsigned int kTileTokens = 8192u;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kTileCount = kTotalTokens / kTileTokens;
constexpr unsigned int kThreads = 256u;
constexpr unsigned int kBlocks = 4096u;

using TileLaunchFn = int(__cdecl *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    unsigned int,
    void *);

struct Provider {
    HMODULE module = nullptr;
    TileLaunchFn launch = nullptr;
};

struct DeviceCheck {
    unsigned long long mismatches;
    unsigned long long nonfinite;
    unsigned long long first_mismatch;
    unsigned int max_abs_bits;
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

__global__ void compare_outputs(
    const float *baseline,
    const float *candidate,
    size_t elements,
    DeviceCheck *check) {
    for (size_t index =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float lhs = baseline[index];
        const float rhs = candidate[index];
        if (!isfinite(lhs) || !isfinite(rhs)) {
            atomicAdd(&check->nonfinite, 1ull);
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

bool load_provider(const char *path, Provider *provider) {
    provider->module = LoadLibraryA(path);
    if (provider->module == nullptr) {
        return false;
    }
    provider->launch = reinterpret_cast<TileLaunchFn>(GetProcAddress(
        provider->module,
        "qrt_ck_fmha_q131072_tile8192_bf16_launch"));
    return provider->launch != nullptr;
}

void close_provider(Provider *provider) {
    if (provider->module != nullptr) {
        FreeLibrary(provider->module);
    }
    *provider = Provider{};
}

bool launch_schedule(
    Provider &provider,
    const uint16_t *q,
    const uint16_t *k,
    const uint16_t *v,
    float *output,
    unsigned int repetitions,
    float *mean_ms,
    std::vector<float> *tile_ms) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    if (hipEventCreate(&start) != hipSuccess ||
        hipEventCreate(&stop) != hipSuccess) {
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        return false;
    }
    if (provider.launch(q, k, v, output, 0u, nullptr) != 0 ||
        hipDeviceSynchronize() != hipSuccess ||
        hipEventRecord(start) != hipSuccess) {
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        return false;
    }
    bool ok = true;
    for (unsigned int repetition = 0u;
         ok && repetition < repetitions;
         ++repetition) {
        for (unsigned int tile = 0u; tile < kTileCount; ++tile) {
            ok = provider.launch(
                     q,
                     k,
                     v,
                     output,
                     tile * kTileTokens,
                     nullptr) == 0;
            if (!ok) {
                break;
            }
        }
    }
    float elapsed_ms = 0.0f;
    ok = ok && hipEventRecord(stop) == hipSuccess &&
        hipEventSynchronize(stop) == hipSuccess &&
        hipEventElapsedTime(&elapsed_ms, start, stop) == hipSuccess;
    if (ok) {
        *mean_ms = elapsed_ms / static_cast<float>(repetitions);
    }

    tile_ms->assign(kTileCount, 0.0f);
    for (unsigned int tile = 0u; ok && tile < kTileCount; ++tile) {
        ok = hipEventRecord(start) == hipSuccess &&
            provider.launch(
                q,
                k,
                v,
                output,
                tile * kTileTokens,
                nullptr) == 0 &&
            hipEventRecord(stop) == hipSuccess &&
            hipEventSynchronize(stop) == hipSuccess &&
            hipEventElapsedTime(&(*tile_ms)[tile], start, stop) == hipSuccess;
    }
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr
            << "usage: q131072_tile8192_benchmark BASELINE_DLL "
               "CANDIDATE_DLL REPETITIONS\n";
        return 2;
    }
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(argv[3], &end, 10);
    if (end == argv[3] || *end != '\0' || parsed == 0u || parsed > 20u) {
        return 2;
    }
    const unsigned int repetitions = static_cast<unsigned int>(parsed);

    Provider baseline{};
    Provider candidate{};
    if (!load_provider(argv[1], &baseline) ||
        !load_provider(argv[2], &candidate)) {
        std::cerr << "provider load/symbol failure error=" << GetLastError()
                  << '\n';
        close_provider(&candidate);
        close_provider(&baseline);
        return 1;
    }

    const size_t q_elements =
        static_cast<size_t>(kTileTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kTotalTokens) * kKvFeatures;
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
    float *baseline_output = nullptr;
    float *candidate_output = nullptr;
    DeviceCheck *device_check = nullptr;
    bool ok = hipMalloc(reinterpret_cast<void **>(&q),
                        q_elements * sizeof(uint16_t)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void **>(&k),
                  kv_elements * sizeof(uint16_t)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void **>(&v),
                  kv_elements * sizeof(uint16_t)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void **>(&baseline_output),
                  q_elements * sizeof(float)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void **>(&candidate_output),
                  q_elements * sizeof(float)) == hipSuccess &&
        hipMalloc(reinterpret_cast<void **>(&device_check),
                  sizeof(DeviceCheck)) == hipSuccess;
    if (ok) {
        initialize_bf16<<<kBlocks, kThreads>>>(
            q, q_elements, UINT32_C(0x91e10da5), 0.125f);
        initialize_bf16<<<kBlocks, kThreads>>>(
            k, kv_elements, UINT32_C(0xa511e9b3), 0.125f);
        initialize_bf16<<<kBlocks, kThreads>>>(
            v, kv_elements, UINT32_C(0x7f4a7c15), 0.25f);
        ok = hipDeviceSynchronize() == hipSuccess;
    }

    float baseline_ms = 0.0f;
    float candidate_ms = 0.0f;
    std::vector<float> baseline_tile_ms;
    std::vector<float> candidate_tile_ms;
    ok = ok && launch_schedule(
        baseline,
        q,
        k,
        v,
        baseline_output,
        repetitions,
        &baseline_ms,
        &baseline_tile_ms);
    ok = ok && launch_schedule(
        candidate,
        q,
        k,
        v,
        candidate_output,
        repetitions,
        &candidate_ms,
        &candidate_tile_ms);

    DeviceCheck initial{};
    initial.first_mismatch =
        (std::numeric_limits<unsigned long long>::max)();
    if (ok) {
        ok = hipMemcpy(
                 device_check,
                 &initial,
                 sizeof(initial),
                 hipMemcpyHostToDevice) == hipSuccess;
    }
    for (unsigned int tile = 0u; ok && tile < kTileCount; ++tile) {
        ok = baseline.launch(
                 q,
                 k,
                 v,
                 baseline_output,
                 tile * kTileTokens,
                 nullptr) == 0 &&
            candidate.launch(
                 q,
                 k,
                 v,
                 candidate_output,
                 tile * kTileTokens,
                 nullptr) == 0;
        if (ok) {
            compare_outputs<<<kBlocks, kThreads>>>(
                baseline_output,
                candidate_output,
                q_elements,
                device_check);
            ok = hipGetLastError() == hipSuccess;
        }
    }
    DeviceCheck host_check{};
    if (ok) {
        ok = hipDeviceSynchronize() == hipSuccess &&
            hipMemcpy(
                &host_check,
                device_check,
                sizeof(host_check),
                hipMemcpyDeviceToHost) == hipSuccess;
    }
    if (!ok) {
        std::cerr << "benchmark GPU failure status=" << hipGetLastError()
                  << '\n';
        return 1;
    }

    float max_abs = 0.0f;
    static_assert(sizeof(max_abs) == sizeof(host_check.max_abs_bits));
    std::memcpy(&max_abs, &host_check.max_abs_bits, sizeof(max_abs));
    std::cout << std::fixed << std::setprecision(6)
              << "baseline_ms=" << baseline_ms
              << " candidate_ms=" << candidate_ms
              << " speedup=" << baseline_ms / candidate_ms
              << " delta_ms=" << candidate_ms - baseline_ms
              << " repetitions=" << repetitions
              << " compared_elements=" << q_elements * kTileCount
              << " mismatches=" << host_check.mismatches
              << " nonfinite=" << host_check.nonfinite
              << " first_mismatch=";
    if (host_check.first_mismatch ==
        (std::numeric_limits<unsigned long long>::max)()) {
        std::cout << "none";
    } else {
        std::cout << host_check.first_mismatch;
    }
    std::cout << " max_abs=" << max_abs << '\n';
    for (unsigned int tile = 0u; tile < kTileCount; ++tile) {
        std::cout << "tile=" << tile
                  << " query_start=" << tile * kTileTokens
                  << " baseline_ms=" << baseline_tile_ms[tile]
                  << " candidate_ms=" << candidate_tile_ms[tile]
                  << " speedup="
                  << baseline_tile_ms[tile] / candidate_tile_ms[tile]
                  << '\n';
    }

    (void)hipFree(device_check);
    (void)hipFree(candidate_output);
    (void)hipFree(baseline_output);
    (void)hipFree(v);
    (void)hipFree(k);
    (void)hipFree(q);
    close_provider(&candidate);
    close_provider(&baseline);
    return 0;
}
