#include <hip/hip_runtime.h>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#define QRT_CALL __cdecl
#else
#include <dlfcn.h>
#define QRT_CALL
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

constexpr unsigned int kTokens = 8192u;
constexpr unsigned int kNeighborLowTokens = kTokens - 1u;
constexpr unsigned int kNeighborHighTokens = kTokens + 1u;
constexpr unsigned int kLongTokens = 2u * kTokens;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kPackedRows = 2u * kQueryFeatures + 2u * kKvFeatures;
constexpr unsigned int kThreads = 256u;
constexpr float kNeighborMaxAbsTolerance = 1.0e-5f;

using PrepareFn = int (QRT_CALL *)();
using PackedLaunchFn = int (QRT_CALL *)(const float *, float *, void *);
using Bf16LaunchFn = int (QRT_CALL *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *);
using DynamicBf16LaunchFn = int (QRT_CALL *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *,
    unsigned int);
using Q262144Tile8192LaunchFn = int (QRT_CALL *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    unsigned int,
    void *);
using ReleaseFn = int (QRT_CALL *)();

#if defined(_WIN32)
using QrtModule = HMODULE;
#else
using QrtModule = void *;
#endif

QrtModule load_module(const char *path) {
#if defined(_WIN32)
    return LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

template <typename Function>
Function load_symbol(QrtModule module, const char *name) {
#if defined(_WIN32)
    return reinterpret_cast<Function>(GetProcAddress(module, name));
#else
    return reinterpret_cast<Function>(dlsym(module, name));
#endif
}

std::string module_error() {
#if defined(_WIN32)
    return std::to_string(GetLastError());
#else
    const char *message = dlerror();
    return message == nullptr ? "unknown" : message;
#endif
}

void unload_module(QrtModule module) {
#if defined(_WIN32)
    (void)FreeLibrary(module);
#else
    (void)dlclose(module);
#endif
}

struct PrefixMetrics {
    size_t mismatches = 0u;
    size_t above_tolerance = 0u;
    size_t signed_zero_mismatches = 0u;
    size_t nonfinite = 0u;
    size_t first_mismatch = (std::numeric_limits<size_t>::max)();
    float max_abs = 0.0f;
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
    uint16_t *__restrict__ v) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t q_elements =
        static_cast<size_t>(kTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kTokens) * kKvFeatures;
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

__global__ void initialize_bf16_tail(
    uint16_t *values,
    size_t begin,
    size_t elements,
    uint32_t seed
) {
    const size_t local_index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (local_index >= elements) {
        return;
    }
    const size_t index = begin + local_index;
    uint32_t x = static_cast<uint32_t>(index) ^ seed;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    const int centered = static_cast<int>((x >> 8) & 0xffffu) - 32768;
    values[index] = f32_to_bf16(
        static_cast<float>(centered) * (0.5f / 32768.0f)
    );
}

bool parse_positive(const char *text, unsigned int *value) {
    char *end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (text == end || *end != '\0' || parsed == 0u || parsed > UINT32_MAX) {
        return false;
    }
    *value = static_cast<unsigned int>(parsed);
    return true;
}

int fail(const char *stage, hipError_t status) {
    std::cerr << "ck_fmha_direct_smoke stage=" << stage
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

uint64_t fnv1a64(const void *data, size_t bytes) {
    const auto *values = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < bytes; ++index) {
        hash ^= values[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

PrefixMetrics compare_prefix(
    const std::vector<float> &reference,
    const std::vector<float> &candidate,
    size_t compared_elements
) {
    PrefixMetrics metrics;
    for (size_t index = 0u; index < candidate.size(); ++index) {
        if (!std::isfinite(candidate[index])) {
            ++metrics.nonfinite;
        }
        if (index >= compared_elements) {
            continue;
        }
        uint32_t reference_bits = 0u;
        uint32_t candidate_bits = 0u;
        std::memcpy(
            &reference_bits,
            &reference[index],
            sizeof(reference_bits)
        );
        std::memcpy(
            &candidate_bits,
            &candidate[index],
            sizeof(candidate_bits)
        );
        if (reference_bits != candidate_bits) {
            if (metrics.first_mismatch ==
                (std::numeric_limits<size_t>::max)()) {
                metrics.first_mismatch = index;
            }
            ++metrics.mismatches;
            if (reference[index] == 0.0f && candidate[index] == 0.0f) {
                ++metrics.signed_zero_mismatches;
            }
        }
        if (std::isfinite(reference[index]) &&
            std::isfinite(candidate[index])) {
            const float absolute_difference =
                std::fabs(reference[index] - candidate[index]);
            metrics.max_abs = std::max(metrics.max_abs, absolute_difference);
            if (absolute_difference > kNeighborMaxAbsTolerance) {
                ++metrics.above_tolerance;
            }
        }
    }
    return metrics;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "usage: q8192_ck_fmha_direct_smoke PROVIDER_DLL "
                     "REPETITIONS" << std::endl;
        return 2;
    }
    unsigned int repetitions = 0u;
    if (!parse_positive(argv[2], &repetitions)) {
        return 2;
    }

    QrtModule dll = load_module(argv[1]);
    if (dll == nullptr) {
        std::cerr << "ck_fmha_direct_smoke module load failed error="
                  << module_error() << std::endl;
        return 1;
    }
    const auto prepare = load_symbol<PrepareFn>(
        dll, "qrt_ck_fmha_q8192_prepare");
    const auto packed_launch = load_symbol<PackedLaunchFn>(
        dll, "qrt_ck_fmha_q8192_f32_launch");
    const auto bf16_launch = load_symbol<Bf16LaunchFn>(
        dll, "qrt_ck_fmha_q8192_bf16_launch");
    const auto dynamic_bf16_launch = load_symbol<DynamicBf16LaunchFn>(
        dll, "qrt_ck_fmha_dynamic_bf16_launch");
    const auto q16384_bf16_launch = load_symbol<Bf16LaunchFn>(
        dll, "qrt_ck_fmha_q16384_bf16_launch");
    const auto q262144_tile8192_launch =
        load_symbol<Q262144Tile8192LaunchFn>(
            dll,
            "qrt_ck_fmha_q262144_tile8192_bf16_launch"
        );
    const auto release = load_symbol<ReleaseFn>(
        dll, "qrt_ck_fmha_q8192_release");
    if (prepare == nullptr || packed_launch == nullptr ||
        bf16_launch == nullptr || dynamic_bf16_launch == nullptr ||
        q16384_bf16_launch == nullptr ||
        q262144_tile8192_launch == nullptr ||
        release == nullptr || prepare() != 0) {
        std::cerr << "ck_fmha_direct_smoke provider symbols/prepare failed"
                  << std::endl;
        unload_module(dll);
        return 1;
    }

    const size_t q_elements =
        static_cast<size_t>(kTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kTokens) * kKvFeatures;
    const size_t packed_elements =
        static_cast<size_t>(kTokens) * kPackedRows;
    const size_t output_elements = q_elements;
    const size_t neighbor_low_output_elements =
        static_cast<size_t>(kNeighborLowTokens) * kQueryFeatures;
    const size_t neighbor_high_output_elements =
        static_cast<size_t>(kNeighborHighTokens) * kQueryFeatures;
    const size_t q16384_q_elements =
        static_cast<size_t>(kLongTokens) * kQueryFeatures;
    const size_t q16384_kv_elements =
        static_cast<size_t>(kLongTokens) * kKvFeatures;
    float *packed = nullptr;
    uint16_t *q = nullptr;
    uint16_t *k = nullptr;
    uint16_t *v = nullptr;
    float *packed_output = nullptr;
    float *direct_output = nullptr;
    float *neighbor_low_output = nullptr;
    float *neighbor_high_output = nullptr;
    float *q262144_tile8192_output = nullptr;
    float *q16384_output = nullptr;
    float *q262144_tile8192_suffix_output = nullptr;
    hipError_t status = hipSuccess;
    if ((status = hipMalloc(
             reinterpret_cast<void **>(&packed),
             packed_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q),
             q16384_q_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&k),
             q16384_kv_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&v),
             q16384_kv_elements * sizeof(uint16_t))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&packed_output),
             output_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&direct_output),
             output_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&neighbor_low_output),
             neighbor_low_output_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&neighbor_high_output),
             neighbor_high_output_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q262144_tile8192_output),
             output_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(&q16384_output),
             q16384_q_elements * sizeof(float))) != hipSuccess ||
        (status = hipMalloc(
             reinterpret_cast<void **>(
                 &q262144_tile8192_suffix_output),
             output_elements * sizeof(float))) != hipSuccess) {
        (void)release();
        (void)hipFree(q262144_tile8192_suffix_output);
        (void)hipFree(q16384_output);
        (void)hipFree(q262144_tile8192_output);
        (void)hipFree(neighbor_high_output);
        (void)hipFree(neighbor_low_output);
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        unload_module(dll);
        return fail("hipMalloc", status);
    }

    initialize_packed<<<
        dim3(static_cast<unsigned int>(
            (packed_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(packed, packed_elements);
    pack_direct_qkv<<<
        dim3(static_cast<unsigned int>(
            (q_elements + 2u * kv_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(packed, q, k, v);
    initialize_bf16_tail<<<
        dim3(static_cast<unsigned int>(
            (q_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(q, q_elements, q_elements, 0x6d2b79f5u);
    initialize_bf16_tail<<<
        dim3(static_cast<unsigned int>(
            (kv_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(k, kv_elements, kv_elements, 0xa5a5f00du);
    initialize_bf16_tail<<<
        dim3(static_cast<unsigned int>(
            (kv_elements + kThreads - 1u) / kThreads)),
        dim3(kThreads)>>>(v, kv_elements, kv_elements, 0x1b873593u);
    if ((status = hipDeviceSynchronize()) != hipSuccess) {
        (void)release();
        (void)hipFree(q262144_tile8192_suffix_output);
        (void)hipFree(q16384_output);
        (void)hipFree(q262144_tile8192_output);
        (void)hipFree(neighbor_high_output);
        (void)hipFree(neighbor_low_output);
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        unload_module(dll);
        return fail("initialize", status);
    }

    float packed_ms = 0.0f;
    float direct_ms = 0.0f;
    float neighbor_low_ms = 0.0f;
    float neighbor_high_ms = 0.0f;
    float q262144_tile8192_ms = 0.0f;
    float q16384_bf16_ms = 0.0f;
    float q262144_tile8192_suffix_ms = 0.0f;
    const bool packed_timed = time_launch(
        [&]() { return packed_launch(packed, packed_output, nullptr); },
        repetitions,
        &packed_ms);
    const bool direct_timed = time_launch(
        [&]() { return bf16_launch(q, k, v, direct_output, nullptr); },
        repetitions,
        &direct_ms);
    const bool neighbor_low_timed = time_launch(
        [&]() {
            return dynamic_bf16_launch(
                q,
                k,
                v,
                neighbor_low_output,
                nullptr,
                kNeighborLowTokens
            );
        },
        repetitions,
        &neighbor_low_ms);
    const bool neighbor_high_timed = time_launch(
        [&]() {
            return dynamic_bf16_launch(
                q,
                k,
                v,
                neighbor_high_output,
                nullptr,
                kNeighborHighTokens
            );
        },
        repetitions,
        &neighbor_high_ms);
    const bool q262144_tile8192_timed = time_launch(
        [&]() {
            return q262144_tile8192_launch(
                q,
                k,
                v,
                q262144_tile8192_output,
                0u,
                nullptr
            );
        },
        repetitions,
        &q262144_tile8192_ms);
    const bool q16384_bf16_timed = time_launch(
        [&]() {
            return q16384_bf16_launch(
                q,
                k,
                v,
                q16384_output,
                nullptr
            );
        },
        repetitions,
        &q16384_bf16_ms);
    const bool q262144_tile8192_suffix_timed = time_launch(
        [&]() {
            return q262144_tile8192_launch(
                q + q_elements,
                k,
                v,
                q262144_tile8192_suffix_output,
                kTokens,
                nullptr
            );
        },
        repetitions,
        &q262144_tile8192_suffix_ms);
    if (!packed_timed || !direct_timed || !neighbor_low_timed ||
        !neighbor_high_timed || !q262144_tile8192_timed ||
        !q16384_bf16_timed || !q262144_tile8192_suffix_timed) {
        status = hipGetLastError();
        (void)release();
        (void)hipFree(q262144_tile8192_suffix_output);
        (void)hipFree(q16384_output);
        (void)hipFree(q262144_tile8192_output);
        (void)hipFree(neighbor_high_output);
        (void)hipFree(neighbor_low_output);
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        unload_module(dll);
        return fail("timed_launch", status);
    }

    std::vector<float> host_packed(output_elements);
    std::vector<float> host_direct(output_elements);
    std::vector<float> host_neighbor_low(neighbor_low_output_elements);
    std::vector<float> host_neighbor_high(neighbor_high_output_elements);
    std::vector<float> host_q262144_tile8192(output_elements);
    std::vector<float> host_q16384_suffix(output_elements);
    std::vector<float> host_q262144_tile8192_suffix(output_elements);
    if ((status = hipMemcpy(
             host_packed.data(),
             packed_output,
             output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess ||
        (status = hipMemcpy(
             host_direct.data(),
             direct_output,
             output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess ||
        (status = hipMemcpy(
             host_neighbor_low.data(),
             neighbor_low_output,
             neighbor_low_output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess ||
        (status = hipMemcpy(
             host_neighbor_high.data(),
             neighbor_high_output,
             neighbor_high_output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess ||
        (status = hipMemcpy(
             host_q262144_tile8192.data(),
             q262144_tile8192_output,
             output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess ||
        (status = hipMemcpy(
             host_q16384_suffix.data(),
             q16384_output + q_elements,
             output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess ||
        (status = hipMemcpy(
             host_q262144_tile8192_suffix.data(),
             q262144_tile8192_suffix_output,
             output_elements * sizeof(float),
             hipMemcpyDeviceToHost)) != hipSuccess) {
        (void)release();
        (void)hipFree(q262144_tile8192_suffix_output);
        (void)hipFree(q16384_output);
        (void)hipFree(q262144_tile8192_output);
        (void)hipFree(neighbor_high_output);
        (void)hipFree(neighbor_low_output);
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        unload_module(dll);
        return fail("hipMemcpy(outputs)", status);
    }

    size_t mismatches = 0u;
    size_t nonfinite = 0u;
    size_t first_mismatch = (std::numeric_limits<size_t>::max)();
    float max_abs = 0.0f;
    size_t q262144_tile8192_mismatches = 0u;
    size_t q262144_tile8192_nonfinite = 0u;
    size_t q262144_tile8192_first_mismatch =
        (std::numeric_limits<size_t>::max)();
    float q262144_tile8192_max_abs = 0.0f;
    size_t q262144_tile8192_suffix_mismatches = 0u;
    size_t q262144_tile8192_suffix_nonfinite = 0u;
    size_t q262144_tile8192_suffix_first_mismatch =
        (std::numeric_limits<size_t>::max)();
    float q262144_tile8192_suffix_max_abs = 0.0f;
    for (size_t index = 0u; index < output_elements; ++index) {
        uint32_t packed_bits = 0u;
        uint32_t direct_bits = 0u;
        std::memcpy(&packed_bits, &host_packed[index], sizeof(packed_bits));
        std::memcpy(&direct_bits, &host_direct[index], sizeof(direct_bits));
        if (packed_bits != direct_bits) {
            if (first_mismatch == (std::numeric_limits<size_t>::max)()) {
                first_mismatch = index;
            }
            ++mismatches;
        }
        if (!std::isfinite(host_packed[index]) ||
            !std::isfinite(host_direct[index])) {
            ++nonfinite;
        } else {
            max_abs = std::max(
                max_abs,
                std::fabs(host_packed[index] - host_direct[index]));
        }
        uint32_t q262144_tile8192_bits = 0u;
        std::memcpy(
            &q262144_tile8192_bits,
            &host_q262144_tile8192[index],
            sizeof(q262144_tile8192_bits)
        );
        if (q262144_tile8192_bits != direct_bits) {
            if (q262144_tile8192_first_mismatch ==
                (std::numeric_limits<size_t>::max)()) {
                q262144_tile8192_first_mismatch = index;
            }
            ++q262144_tile8192_mismatches;
        }
        if (!std::isfinite(host_q262144_tile8192[index]) ||
            !std::isfinite(host_direct[index])) {
            ++q262144_tile8192_nonfinite;
        } else {
            q262144_tile8192_max_abs = std::max(
                q262144_tile8192_max_abs,
                std::fabs(
                    host_q262144_tile8192[index] - host_direct[index]
                )
            );
        }
        uint32_t q16384_suffix_bits = 0u;
        uint32_t q262144_tile8192_suffix_bits = 0u;
        std::memcpy(
            &q16384_suffix_bits,
            &host_q16384_suffix[index],
            sizeof(q16384_suffix_bits)
        );
        std::memcpy(
            &q262144_tile8192_suffix_bits,
            &host_q262144_tile8192_suffix[index],
            sizeof(q262144_tile8192_suffix_bits)
        );
        if (q262144_tile8192_suffix_bits != q16384_suffix_bits) {
            if (q262144_tile8192_suffix_first_mismatch ==
                (std::numeric_limits<size_t>::max)()) {
                q262144_tile8192_suffix_first_mismatch = index;
            }
            ++q262144_tile8192_suffix_mismatches;
        }
        if (!std::isfinite(host_q16384_suffix[index]) ||
            !std::isfinite(host_q262144_tile8192_suffix[index])) {
            ++q262144_tile8192_suffix_nonfinite;
        } else {
            q262144_tile8192_suffix_max_abs = std::max(
                q262144_tile8192_suffix_max_abs,
                std::fabs(
                    host_q262144_tile8192_suffix[index] -
                    host_q16384_suffix[index]
                )
            );
        }
    }
    const PrefixMetrics neighbor_low_metrics = compare_prefix(
        host_direct,
        host_neighbor_low,
        neighbor_low_output_elements
    );
    const PrefixMetrics neighbor_high_metrics = compare_prefix(
        host_direct,
        host_neighbor_high,
        output_elements
    );
    const uint64_t packed_hash = fnv1a64(
        host_packed.data(),
        host_packed.size() * sizeof(host_packed[0]));
    const uint64_t direct_hash = fnv1a64(
        host_direct.data(),
        host_direct.size() * sizeof(host_direct[0]));
    const uint64_t neighbor_low_hash = fnv1a64(
        host_neighbor_low.data(),
        host_neighbor_low.size() * sizeof(host_neighbor_low[0]));
    const uint64_t neighbor_high_hash = fnv1a64(
        host_neighbor_high.data(),
        host_neighbor_high.size() * sizeof(host_neighbor_high[0]));
    const uint64_t q262144_tile8192_hash = fnv1a64(
        host_q262144_tile8192.data(),
        host_q262144_tile8192.size() *
            sizeof(host_q262144_tile8192[0]));
    const uint64_t q16384_suffix_hash = fnv1a64(
        host_q16384_suffix.data(),
        host_q16384_suffix.size() * sizeof(host_q16384_suffix[0]));
    const uint64_t q262144_tile8192_suffix_hash = fnv1a64(
        host_q262144_tile8192_suffix.data(),
        host_q262144_tile8192_suffix.size() *
            sizeof(host_q262144_tile8192_suffix[0]));

    std::cout << std::fixed << std::setprecision(6)
              << "ck_fmha_direct_smoke packed_ms=" << packed_ms
              << " direct_ms=" << direct_ms
              << " q262144_tile8192_ms=" << q262144_tile8192_ms
              << " q16384_bf16_ms=" << q16384_bf16_ms
              << " q262144_tile8192_suffix_ms="
              << q262144_tile8192_suffix_ms
              << " pack_elided_ms=" << (packed_ms - direct_ms)
              << " speedup=" << (packed_ms / direct_ms)
              << " repetitions=" << repetitions
              << " elements=" << output_elements
              << " mismatches=" << mismatches
              << " first_mismatch=";
    if (first_mismatch == (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << first_mismatch;
    }
    std::cout << " nonfinite=" << nonfinite
              << " max_abs=" << max_abs
              << " packed_hash=" << std::hex << std::setw(16)
              << std::setfill('0') << packed_hash
              << " direct_hash=" << std::setw(16) << direct_hash
              << " q262144_tile8192_mismatches=" << std::dec
              << q262144_tile8192_mismatches
              << " q262144_tile8192_first_mismatch=";
    if (q262144_tile8192_first_mismatch ==
        (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << q262144_tile8192_first_mismatch;
    }
    std::cout << " q262144_tile8192_nonfinite="
              << q262144_tile8192_nonfinite
              << " q262144_tile8192_max_abs="
              << q262144_tile8192_max_abs
              << " q262144_tile8192_hash=" << std::hex << std::setw(16)
              << q262144_tile8192_hash
              << " q262144_tile8192_suffix_mismatches=" << std::dec
              << q262144_tile8192_suffix_mismatches
              << " q262144_tile8192_suffix_first_mismatch=";
    if (q262144_tile8192_suffix_first_mismatch ==
        (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << q262144_tile8192_suffix_first_mismatch;
    }
    std::cout << " q262144_tile8192_suffix_nonfinite="
              << q262144_tile8192_suffix_nonfinite
              << " q262144_tile8192_suffix_max_abs="
              << q262144_tile8192_suffix_max_abs
              << " q16384_suffix_hash=" << std::hex << std::setw(16)
              << q16384_suffix_hash
              << " q262144_tile8192_suffix_hash=" << std::setw(16)
              << q262144_tile8192_suffix_hash
              << std::dec << std::setfill(' ') << std::endl;

    std::cout << std::fixed << std::setprecision(6)
              << "ck_fmha_neighbor_smoke fixed_q8192_ms=" << direct_ms
              << " dynamic_q8191_ms=" << neighbor_low_ms
              << " dynamic_q8193_ms=" << neighbor_high_ms
              << " q8191_to_q8192_ratio="
              << (neighbor_low_ms / direct_ms)
              << " q8193_to_q8192_ratio="
              << (neighbor_high_ms / direct_ms)
              << " q8191_compared_elements="
              << neighbor_low_output_elements
              << " q8191_mismatches=" << neighbor_low_metrics.mismatches
              << " q8191_first_mismatch=";
    if (neighbor_low_metrics.first_mismatch ==
        (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << neighbor_low_metrics.first_mismatch;
    }
    std::cout << " q8191_nonfinite=" << neighbor_low_metrics.nonfinite
              << " q8191_above_tolerance="
              << neighbor_low_metrics.above_tolerance
              << " q8191_signed_zero_mismatches="
              << neighbor_low_metrics.signed_zero_mismatches
              << " q8191_first_mismatch_token=";
    if (neighbor_low_metrics.first_mismatch ==
        (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << neighbor_low_metrics.first_mismatch / kQueryFeatures;
    }
    std::cout << " q8191_max_abs=" << std::scientific
              << std::setprecision(9) << neighbor_low_metrics.max_abs
              << " q8191_tolerance=" << kNeighborMaxAbsTolerance
              << std::fixed << std::setprecision(6)
              << " q8191_hash=" << std::hex << std::setw(16)
              << std::setfill('0') << neighbor_low_hash
              << std::dec << std::setfill(' ')
              << " q8193_compared_elements=" << output_elements
              << " q8193_total_elements=" << neighbor_high_output_elements
              << " q8193_mismatches=" << neighbor_high_metrics.mismatches
              << " q8193_first_mismatch=";
    if (neighbor_high_metrics.first_mismatch ==
        (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << neighbor_high_metrics.first_mismatch;
    }
    std::cout << " q8193_nonfinite=" << neighbor_high_metrics.nonfinite
              << " q8193_above_tolerance="
              << neighbor_high_metrics.above_tolerance
              << " q8193_signed_zero_mismatches="
              << neighbor_high_metrics.signed_zero_mismatches
              << " q8193_first_mismatch_token=";
    if (neighbor_high_metrics.first_mismatch ==
        (std::numeric_limits<size_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << neighbor_high_metrics.first_mismatch / kQueryFeatures;
    }
    std::cout << " q8193_max_abs=" << std::scientific
              << std::setprecision(9) << neighbor_high_metrics.max_abs
              << " q8193_tolerance=" << kNeighborMaxAbsTolerance
              << std::fixed << std::setprecision(6)
              << " q8193_hash=" << std::hex << std::setw(16)
              << std::setfill('0') << neighbor_high_hash
              << std::dec << std::setfill(' ')
              << " target_device=AMD395"
              << std::endl;

    (void)release();
    (void)hipFree(q262144_tile8192_suffix_output);
    (void)hipFree(q16384_output);
    (void)hipFree(q262144_tile8192_output);
    (void)hipFree(neighbor_high_output);
    (void)hipFree(neighbor_low_output);
    (void)hipFree(direct_output);
    (void)hipFree(packed_output);
    (void)hipFree(v);
    (void)hipFree(k);
    (void)hipFree(q);
    (void)hipFree(packed);
    unload_module(dll);
    return mismatches == 0u && nonfinite == 0u &&
            q262144_tile8192_mismatches == 0u &&
            q262144_tile8192_nonfinite == 0u &&
            q262144_tile8192_suffix_mismatches == 0u &&
            q262144_tile8192_suffix_nonfinite == 0u &&
            neighbor_low_metrics.above_tolerance == 0u &&
            neighbor_low_metrics.nonfinite == 0u &&
            neighbor_high_metrics.above_tolerance == 0u &&
            neighbor_high_metrics.nonfinite == 0u
        ? 0
        : 1;
}
