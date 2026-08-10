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

constexpr unsigned int kTokens = 8192u;
constexpr unsigned int kLongTokens = 2u * kTokens;
constexpr unsigned int kQueryHeads = 16u;
constexpr unsigned int kKvHeads = 2u;
constexpr unsigned int kHeadDim = 256u;
constexpr unsigned int kQueryFeatures = kQueryHeads * kHeadDim;
constexpr unsigned int kKvFeatures = kKvHeads * kHeadDim;
constexpr unsigned int kPackedRows = 2u * kQueryFeatures + 2u * kKvFeatures;
constexpr unsigned int kThreads = 256u;

using PrepareFn = int (__cdecl *)();
using PackedLaunchFn = int (__cdecl *)(const float *, float *, void *);
using Bf16LaunchFn = int (__cdecl *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *);
using Q262144Tile8192LaunchFn = int (__cdecl *)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    unsigned int,
    void *);
using ReleaseFn = int (__cdecl *)();

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

    HMODULE dll = LoadLibraryA(argv[1]);
    if (dll == nullptr) {
        std::cerr << "ck_fmha_direct_smoke LoadLibrary failed error="
                  << GetLastError() << std::endl;
        return 1;
    }
    const auto prepare = reinterpret_cast<PrepareFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_prepare"));
    const auto packed_launch = reinterpret_cast<PackedLaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_f32_launch"));
    const auto bf16_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_bf16_launch"));
    const auto q16384_bf16_launch = reinterpret_cast<Bf16LaunchFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q16384_bf16_launch"));
    const auto q262144_tile8192_launch =
        reinterpret_cast<Q262144Tile8192LaunchFn>(GetProcAddress(
            dll,
            "qrt_ck_fmha_q262144_tile8192_bf16_launch"
        ));
    const auto release = reinterpret_cast<ReleaseFn>(
        GetProcAddress(dll, "qrt_ck_fmha_q8192_release"));
    if (prepare == nullptr || packed_launch == nullptr ||
        bf16_launch == nullptr || q16384_bf16_launch == nullptr ||
        q262144_tile8192_launch == nullptr ||
        release == nullptr || prepare() != 0) {
        std::cerr << "ck_fmha_direct_smoke provider symbols/prepare failed"
                  << std::endl;
        FreeLibrary(dll);
        return 1;
    }

    const size_t q_elements =
        static_cast<size_t>(kTokens) * kQueryFeatures;
    const size_t kv_elements =
        static_cast<size_t>(kTokens) * kKvFeatures;
    const size_t packed_elements =
        static_cast<size_t>(kTokens) * kPackedRows;
    const size_t output_elements = q_elements;
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
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
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
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        FreeLibrary(dll);
        return fail("initialize", status);
    }

    float packed_ms = 0.0f;
    float direct_ms = 0.0f;
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
    if (!packed_timed || !direct_timed || !q262144_tile8192_timed ||
        !q16384_bf16_timed || !q262144_tile8192_suffix_timed) {
        status = hipGetLastError();
        (void)release();
        (void)hipFree(q262144_tile8192_suffix_output);
        (void)hipFree(q16384_output);
        (void)hipFree(q262144_tile8192_output);
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        FreeLibrary(dll);
        return fail("timed_launch", status);
    }

    std::vector<float> host_packed(output_elements);
    std::vector<float> host_direct(output_elements);
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
        (void)hipFree(direct_output);
        (void)hipFree(packed_output);
        (void)hipFree(v);
        (void)hipFree(k);
        (void)hipFree(q);
        (void)hipFree(packed);
        FreeLibrary(dll);
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
    const uint64_t packed_hash = fnv1a64(
        host_packed.data(),
        host_packed.size() * sizeof(host_packed[0]));
    const uint64_t direct_hash = fnv1a64(
        host_direct.data(),
        host_direct.size() * sizeof(host_direct[0]));
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

    (void)release();
    (void)hipFree(q262144_tile8192_suffix_output);
    (void)hipFree(q16384_output);
    (void)hipFree(q262144_tile8192_output);
    (void)hipFree(direct_output);
    (void)hipFree(packed_output);
    (void)hipFree(v);
    (void)hipFree(k);
    (void)hipFree(q);
    (void)hipFree(packed);
    FreeLibrary(dll);
    return mismatches == 0u && nonfinite == 0u &&
            q262144_tile8192_mismatches == 0u &&
            q262144_tile8192_nonfinite == 0u &&
            q262144_tile8192_suffix_mismatches == 0u &&
            q262144_tile8192_suffix_nonfinite == 0u
        ? 0
        : 1;
}
