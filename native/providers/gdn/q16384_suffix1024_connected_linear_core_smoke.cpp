#include <hip/hip_runtime.h>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#ifndef QRT_QWEN36_Q1024_OWNER_PREFIX_TOKENS
#define QRT_QWEN36_Q1024_OWNER_PREFIX_TOKENS 16384u
#endif

constexpr unsigned int kPrefixTokens =
    QRT_QWEN36_Q1024_OWNER_PREFIX_TOKENS;
constexpr unsigned int kSuffixTokens = 1024u;
constexpr unsigned int kFullTokens = kPrefixTokens + kSuffixTokens;
constexpr unsigned int kQkvRows = 8192u;
constexpr unsigned int kConvTaps = 4u;
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
constexpr double kCaseMeanCeilingMs = 20.0;
constexpr double kNativeProjectionCeilingMs = 600.0;

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
    SeededLaunchFunction q1024_seeded_launch_async = nullptr;
    LastErrorFunction last_error = nullptr;
    ReleaseFunction release = nullptr;
};

void check_hip(hipError_t status, const char *stage) {
    if (status != hipSuccess) {
        throw std::runtime_error(
            std::string(stage) + ": " + hipGetErrorString(status)
        );
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        check_hip(
            hipMalloc(reinterpret_cast<void **>(&data_), count * sizeof(T)),
            "hipMalloc"
        );
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)hipFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() {
        return data_;
    }

    const T *get() const {
        return data_;
    }

private:
    T *data_ = nullptr;
    size_t count_ = 0u;
};

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
    return api->prepare != nullptr &&
        api->q16384_launch != nullptr &&
        api->q17408_launch != nullptr &&
        api->q1024_seeded_launch_async != nullptr &&
        api->last_error != nullptr &&
        api->release != nullptr;
}

__device__ __forceinline__ uint16_t device_float_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000)) {
        uint16_t upper = static_cast<uint16_t>(bits >> 16u);
        if ((bits & UINT32_C(0x007fffff)) != 0u) {
            upper |= UINT16_C(0x0040);
        }
        return upper;
    }
    const uint32_t rounding_bias =
        UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16u);
}

__device__ __forceinline__ float device_bf16_to_float(uint16_t value) {
    return __uint_as_float(static_cast<uint32_t>(value) << 16u);
}

__device__ __forceinline__ float device_bf16_round_to_float(float value) {
    return device_bf16_to_float(device_float_to_bf16(value));
}

template <typename T>
__device__ __forceinline__ T device_native_from_float(float value);

template <>
__device__ __forceinline__ float device_native_from_float<float>(float value) {
    return value;
}

template <>
__device__ __forceinline__ uint16_t
device_native_from_float<uint16_t>(float value) {
    return device_float_to_bf16(value);
}

__device__ __forceinline__ float device_native_to_float(float value) {
    return value;
}

__device__ __forceinline__ float device_native_to_float(uint16_t value) {
    return device_bf16_to_float(value);
}

__device__ __forceinline__ float device_conv_endpoint(
    float source0,
    float source1,
    float source2,
    float source3,
    const uint16_t *weights
) {
    const float sources[kConvTaps] = {
        source0,
        source1,
        source2,
        source3,
    };
    float accumulator = 0.0f;
#pragma unroll
    for (unsigned int tap = 0u; tap < kConvTaps; ++tap) {
        accumulator +=
            device_bf16_round_to_float(sources[tap]) *
            device_bf16_to_float(weights[tap]);
    }
    const float rounded_accumulator =
        device_bf16_round_to_float(accumulator);
    const float silu =
        rounded_accumulator / (1.0f + expf(-rounded_accumulator));
    return device_bf16_round_to_float(silu);
}

template <typename T>
__global__ void fill_qkv_kernel(T *values, size_t count) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    uint32_t state =
        static_cast<uint32_t>(index) * UINT32_C(747796405) +
        UINT32_C(2891336453);
    state ^= state >> 16u;
    state *= UINT32_C(2246822519);
    state ^= state >> 13u;
    const int32_t centered =
        static_cast<int32_t>(state & UINT32_C(0xffff)) - 32768;
    const float value = static_cast<float>(centered) * 0.000244140625f;
    values[index] = device_native_from_float<T>(value);
}

__global__ void fill_weights_kernel(uint16_t *weights, size_t count) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    uint32_t state =
        static_cast<uint32_t>(index) * UINT32_C(277803737) +
        UINT32_C(1013904223);
    state ^= state >> 15u;
    const int32_t centered =
        static_cast<int32_t>(state & UINT32_C(0xff)) - 128;
    weights[index] = device_float_to_bf16(
        static_cast<float>(centered) * 0.0009765625f
    );
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
        const float log_gate = -(
            0.000125f +
            0.031125f * static_cast<float>(phase) / 1023.0f
        );
        gate[index] =
            gate_values_are_decay != 0 ? expf(log_gate) : log_gate;
    } else {
        const float beta =
            0.05f +
            0.9f *
                static_cast<float>((13u * token + 17u * head) % 997u) /
                996.0f;
        gate[index] = device_bf16_round_to_float(beta);
    }
}

template <typename T>
__global__ void capture_four_token_ring_kernel(
    const T *full_qkv,
    T *ring,
    unsigned int boundary_token
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total =
        static_cast<size_t>(kConvTaps) * kQkvRows;
    if (index >= total) {
        return;
    }
    const unsigned int relative_token =
        static_cast<unsigned int>(index / kQkvRows);
    const unsigned int feature =
        static_cast<unsigned int>(index % kQkvRows);
    const unsigned int absolute_token =
        boundary_token - kConvTaps + relative_token;
    const unsigned int slot = absolute_token % kConvTaps;
    ring[static_cast<size_t>(slot) * kQkvRows + feature] =
        full_qkv[static_cast<size_t>(absolute_token) * kQkvRows + feature];
}

template <typename T>
__global__ void full_conv_kernel(
    const T *full_qkv,
    const uint16_t *weights,
    float *outputs,
    unsigned int tokens
) {
    const unsigned int feature =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int token = blockIdx.y;
    if (feature >= kQkvRows || token >= tokens) {
        return;
    }
    float sources[kConvTaps] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (unsigned int tap = 0u; tap < kConvTaps; ++tap) {
        if (token + tap >= kConvTaps - 1u) {
            const unsigned int source_token =
                token + tap - (kConvTaps - 1u);
            sources[tap] = device_native_to_float(
                full_qkv[
                    static_cast<size_t>(source_token) * kQkvRows + feature
                ]
            );
        }
    }
    const uint16_t *row_weights =
        weights + static_cast<size_t>(feature) * kConvTaps;
    outputs[static_cast<size_t>(token) * kQkvRows + feature] =
        device_conv_endpoint(
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            row_weights
        );
}

template <typename T>
__global__ void suffix_halo_conv_kernel(
    const T *prefix_ring,
    const T *suffix_qkv,
    const uint16_t *weights,
    float *outputs
) {
    const unsigned int feature =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int suffix_token = blockIdx.y;
    if (feature >= kQkvRows || suffix_token >= kSuffixTokens) {
        return;
    }
    const unsigned int absolute_token = kPrefixTokens + suffix_token;
    float sources[kConvTaps];
#pragma unroll
    for (unsigned int tap = 0u; tap < kConvTaps; ++tap) {
        const unsigned int source_token =
            absolute_token + tap - (kConvTaps - 1u);
        if (source_token < kPrefixTokens) {
            sources[tap] = device_native_to_float(
                prefix_ring[
                    static_cast<size_t>(source_token % kConvTaps) *
                        kQkvRows +
                    feature
                ]
            );
        } else {
            sources[tap] = device_native_to_float(
                suffix_qkv[
                    static_cast<size_t>(source_token - kPrefixTokens) *
                        kQkvRows +
                    feature
                ]
            );
        }
    }
    const uint16_t *row_weights =
        weights + static_cast<size_t>(feature) * kConvTaps;
    outputs[static_cast<size_t>(suffix_token) * kQkvRows + feature] =
        device_conv_endpoint(
            sources[0],
            sources[1],
            sources[2],
            sources[3],
            row_weights
        );
}

template <typename T>
__global__ void publish_final_ring_kernel(
    const T *suffix_qkv,
    T *private_ring
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total =
        static_cast<size_t>(kConvTaps) * kQkvRows;
    if (index >= total) {
        return;
    }
    const unsigned int relative_token =
        static_cast<unsigned int>(index / kQkvRows);
    const unsigned int feature =
        static_cast<unsigned int>(index % kQkvRows);
    const unsigned int suffix_token =
        kSuffixTokens - kConvTaps + relative_token;
    const unsigned int absolute_token = kPrefixTokens + suffix_token;
    private_ring[
        static_cast<size_t>(absolute_token % kConvTaps) * kQkvRows + feature
    ] =
        suffix_qkv[static_cast<size_t>(suffix_token) * kQkvRows + feature];
}

__global__ void postconv_qk_inplace_kernel(
    float *values,
    unsigned int tokens
) {
    __shared__ float q_partial[kKeyDim];
    __shared__ float k_partial[kKeyDim];
    __shared__ float q_inv;
    __shared__ float k_inv;

    const unsigned int token = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (token >= tokens || head >= kKeyHeads || dim >= kKeyDim) {
        return;
    }
    const unsigned int index = head * kKeyDim + dim;
    const size_t token_base = static_cast<size_t>(token) * kQkvRows;
    const float q_value = values[token_base + index];
    const float k_value = values[token_base + kKeyFeatures + index];
    q_partial[dim] =
        device_bf16_round_to_float(q_value * q_value);
    k_partial[dim] =
        device_bf16_round_to_float(k_value * k_value);
    __syncthreads();

    for (unsigned int stride = kKeyDim / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (dim < stride) {
            q_partial[dim] += q_partial[dim + stride];
            k_partial[dim] += k_partial[dim + stride];
        }
        __syncthreads();
    }
    if (dim == 0u) {
        const float q_sumsq =
            device_bf16_round_to_float(q_partial[0]);
        const float k_sumsq =
            device_bf16_round_to_float(k_partial[0]);
        q_inv = device_bf16_round_to_float(
            1.0f /
            sqrtf(device_bf16_round_to_float(q_sumsq + 1.0e-6f))
        );
        k_inv = device_bf16_round_to_float(
            1.0f /
            sqrtf(device_bf16_round_to_float(k_sumsq + 1.0e-6f))
        );
    }
    __syncthreads();

    const float q_norm =
        device_bf16_round_to_float(q_value * q_inv);
    values[token_base + index] = q_norm * kQScale;
    values[token_base + kKeyFeatures + index] =
        device_bf16_round_to_float(k_value * k_inv);
}

__global__ void postconv_value_inplace_kernel(
    float *values,
    unsigned int tokens
) {
    const unsigned int feature =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int token = blockIdx.y;
    if (feature >= kValueFeatures || token >= tokens) {
        return;
    }
    const size_t index =
        static_cast<size_t>(token) * kQkvRows +
        2u * kKeyFeatures +
        feature;
    values[index] = device_bf16_round_to_float(values[index]);
}

__global__ void row_to_key_major_kernel(
    const float *row_major,
    float *key_major
) {
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
        static_cast<size_t>(key) * kValueDim +
        value;
    key_major[key_index] = row_major[row_index];
}

uint32_t host_float_bits(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float host_bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16u;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

template <typename T>
float host_native_to_float(T value);

template <>
float host_native_to_float<float>(float value) {
    return value;
}

template <>
float host_native_to_float<uint16_t>(uint16_t value) {
    return host_bf16_to_float(value);
}

uint64_t fnv1a64(const void *data, size_t bytes) {
    const auto *input = static_cast<const uint8_t *>(data);
    uint64_t hash = UINT64_C(1469598103934665603);
    for (size_t index = 0u; index < bytes; ++index) {
        hash ^= input[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

struct CompareStats {
    uint64_t elements = 0u;
    uint64_t mismatches = 0u;
    uint64_t nonfinite = 0u;
    uint64_t first_mismatch = (std::numeric_limits<uint64_t>::max)();
    double max_abs = 0.0;
};

CompareStats compare_f32(
    const std::vector<float> &reference,
    const std::vector<float> &candidate
) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("F32 compare size mismatch");
    }
    CompareStats stats;
    stats.elements = static_cast<uint64_t>(reference.size());
    for (size_t index = 0u; index < reference.size(); ++index) {
        const float left = reference[index];
        const float right = candidate[index];
        if (!std::isfinite(left) || !std::isfinite(right)) {
            ++stats.nonfinite;
        }
        if (host_float_bits(left) != host_float_bits(right)) {
            if (stats.first_mismatch ==
                (std::numeric_limits<uint64_t>::max)()) {
                stats.first_mismatch = static_cast<uint64_t>(index);
            }
            ++stats.mismatches;
        }
        if (std::isfinite(left) && std::isfinite(right)) {
            stats.max_abs = std::max(
                stats.max_abs,
                std::abs(
                    static_cast<double>(left) -
                    static_cast<double>(right)
                )
            );
        }
    }
    return stats;
}

template <typename T>
CompareStats compare_native(
    const std::vector<T> &reference,
    const std::vector<T> &candidate
) {
    if (reference.size() != candidate.size()) {
        throw std::runtime_error("native compare size mismatch");
    }
    CompareStats stats;
    stats.elements = static_cast<uint64_t>(reference.size());
    for (size_t index = 0u; index < reference.size(); ++index) {
        const float left = host_native_to_float(reference[index]);
        const float right = host_native_to_float(candidate[index]);
        if (!std::isfinite(left) || !std::isfinite(right)) {
            ++stats.nonfinite;
        }
        if (std::memcmp(
                &reference[index],
                &candidate[index],
                sizeof(T)
            ) != 0) {
            if (stats.first_mismatch ==
                (std::numeric_limits<uint64_t>::max)()) {
                stats.first_mismatch = static_cast<uint64_t>(index);
            }
            ++stats.mismatches;
        }
        if (std::isfinite(left) && std::isfinite(right)) {
            stats.max_abs = std::max(
                stats.max_abs,
                std::abs(
                    static_cast<double>(left) -
                    static_cast<double>(right)
                )
            );
        }
    }
    return stats;
}

CompareStats compare_state_row_to_key(
    const std::vector<float> &row_major,
    const std::vector<float> &key_major
) {
    const size_t elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;
    if (row_major.size() != elements || key_major.size() != elements) {
        throw std::runtime_error("state compare size mismatch");
    }
    std::vector<float> mapped(elements);
    for (size_t row_index = 0u; row_index < elements; ++row_index) {
        const unsigned int key =
            static_cast<unsigned int>(row_index % kKeyDim);
        const unsigned int value_index =
            static_cast<unsigned int>(row_index / kKeyDim);
        const unsigned int value_head = value_index / kValueDim;
        const unsigned int value = value_index % kValueDim;
        const size_t key_index =
            static_cast<size_t>(value_head) * kKeyDim * kValueDim +
            static_cast<size_t>(key) * kValueDim +
            value;
        mapped[row_index] = key_major[key_index];
    }
    return compare_f32(row_major, mapped);
}

template <typename T>
std::vector<T> copy_device(const T *device, size_t count) {
    std::vector<T> host(count);
    check_hip(
        hipMemcpy(
            host.data(),
            device,
            count * sizeof(T),
            hipMemcpyDeviceToHost
        ),
        "hipMemcpy(device-to-host)"
    );
    return host;
}

struct CaseResult {
    std::string ring_mode;
    int gate_values_are_decay = 0;
    double mean_ms = 0.0;
    bool pass = false;
};

template <typename T>
std::vector<CaseResult> run_ring_mode(
    const char *ring_mode,
    const ProviderApi &api,
    unsigned int repetitions
) {
    const size_t full_qkv_elements =
        static_cast<size_t>(kFullTokens) * kQkvRows;
    const size_t suffix_qkv_elements =
        static_cast<size_t>(kSuffixTokens) * kQkvRows;
    const size_t full_output_elements =
        static_cast<size_t>(kFullTokens) * kValueFeatures;
    const size_t suffix_output_elements =
        static_cast<size_t>(kSuffixTokens) * kValueFeatures;
    const size_t gate_elements =
        static_cast<size_t>(kFullTokens) * kGateOutputRows;
    const size_t state_elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;
    const size_t ring_elements =
        static_cast<size_t>(kConvTaps) * kQkvRows;
    const size_t weight_elements =
        static_cast<size_t>(kQkvRows) * kConvTaps;

    DeviceBuffer<T> full_qkv(full_qkv_elements);
    DeviceBuffer<uint16_t> weights(weight_elements);
    DeviceBuffer<T> prefix_ring(ring_elements);
    DeviceBuffer<T> private_final_ring(ring_elements);
    DeviceBuffer<float> full_postconv(full_qkv_elements);
    DeviceBuffer<float> suffix_postconv(suffix_qkv_elements);
    DeviceBuffer<float> gate(gate_elements);
    DeviceBuffer<float> full_output(full_output_elements);
    DeviceBuffer<float> suffix_output(suffix_output_elements);
    DeviceBuffer<float> prefix_state_row(state_elements);
    DeviceBuffer<float> prefix_state_key(state_elements);
    DeviceBuffer<float> full_state_row(state_elements);
    DeviceBuffer<float> suffix_state_key(state_elements);

    const dim3 block(kThreads);
    const dim3 fill_qkv_grid(
        static_cast<unsigned int>(
            (full_qkv_elements + kThreads - 1u) / kThreads
        )
    );
    const dim3 weight_grid(
        static_cast<unsigned int>(
            (weight_elements + kThreads - 1u) / kThreads
        )
    );
    const dim3 ring_grid(
        static_cast<unsigned int>(
            (ring_elements + kThreads - 1u) / kThreads
        )
    );
    const dim3 full_conv_grid(
        (kQkvRows + kThreads - 1u) / kThreads,
        kFullTokens
    );
    const dim3 suffix_conv_grid(
        (kQkvRows + kThreads - 1u) / kThreads,
        kSuffixTokens
    );
    const dim3 full_value_grid(
        (kValueFeatures + kThreads - 1u) / kThreads,
        kFullTokens
    );
    const dim3 suffix_value_grid(
        (kValueFeatures + kThreads - 1u) / kThreads,
        kSuffixTokens
    );
    const dim3 state_grid(
        static_cast<unsigned int>(
            (state_elements + kThreads - 1u) / kThreads
        )
    );
    const dim3 gate_grid(
        static_cast<unsigned int>(
            (gate_elements + kThreads - 1u) / kThreads
        )
    );

    hipLaunchKernelGGL(
        fill_qkv_kernel<T>,
        fill_qkv_grid,
        block,
        0u,
        nullptr,
        full_qkv.get(),
        full_qkv_elements
    );
    hipLaunchKernelGGL(
        fill_weights_kernel,
        weight_grid,
        block,
        0u,
        nullptr,
        weights.get(),
        weight_elements
    );
    hipLaunchKernelGGL(
        capture_four_token_ring_kernel<T>,
        ring_grid,
        block,
        0u,
        nullptr,
        full_qkv.get(),
        prefix_ring.get(),
        kPrefixTokens
    );
    hipLaunchKernelGGL(
        full_conv_kernel<T>,
        full_conv_grid,
        block,
        0u,
        nullptr,
        full_qkv.get(),
        weights.get(),
        full_postconv.get(),
        kFullTokens
    );
    hipLaunchKernelGGL(
        postconv_qk_inplace_kernel,
        dim3(kFullTokens, kKeyHeads),
        dim3(kKeyDim),
        0u,
        nullptr,
        full_postconv.get(),
        kFullTokens
    );
    hipLaunchKernelGGL(
        postconv_value_inplace_kernel,
        full_value_grid,
        block,
        0u,
        nullptr,
        full_postconv.get(),
        kFullTokens
    );
    check_hip(hipGetLastError(), "full fixture launch");
    check_hip(hipDeviceSynchronize(), "full fixture synchronize");

    const std::vector<T> host_prefix_before =
        copy_device(prefix_ring.get(), ring_elements);
    const std::vector<T> host_prefix_direct = copy_device(
        full_qkv.get() +
            static_cast<size_t>(kPrefixTokens - kConvTaps) * kQkvRows,
        ring_elements
    );
    const CompareStats prefix_mapping =
        compare_native(host_prefix_direct, host_prefix_before);
    const T *suffix_qkv =
        full_qkv.get() + static_cast<size_t>(kPrefixTokens) * kQkvRows;

    std::vector<CaseResult> results;
    for (int gate_values_are_decay = 0;
         gate_values_are_decay <= 1;
         ++gate_values_are_decay) {
        hipLaunchKernelGGL(
            fill_gate_kernel,
            gate_grid,
            block,
            0u,
            nullptr,
            gate.get(),
            gate_values_are_decay
        );
        check_hip(hipGetLastError(), "fill gate launch");
        check_hip(hipDeviceSynchronize(), "fill gate synchronize");

        if (api.q16384_launch(
                full_postconv.get(),
                gate.get(),
                full_output.get(),
                prefix_state_row.get(),
                gate_values_are_decay,
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("q16384 provider failed: ") + api.last_error()
            );
        }
        hipLaunchKernelGGL(
            row_to_key_major_kernel,
            state_grid,
            block,
            0u,
            nullptr,
            prefix_state_row.get(),
            prefix_state_key.get()
        );
        check_hip(hipGetLastError(), "state transpose launch");
        check_hip(hipDeviceSynchronize(), "state transpose synchronize");

        const std::vector<float> host_state_before =
            copy_device(prefix_state_key.get(), state_elements);

        auto launch_candidate = [&]() {
            hipLaunchKernelGGL(
                suffix_halo_conv_kernel<T>,
                suffix_conv_grid,
                block,
                0u,
                nullptr,
                prefix_ring.get(),
                suffix_qkv,
                weights.get(),
                suffix_postconv.get()
            );
            hipLaunchKernelGGL(
                postconv_qk_inplace_kernel,
                dim3(kSuffixTokens, kKeyHeads),
                dim3(kKeyDim),
                0u,
                nullptr,
                suffix_postconv.get(),
                kSuffixTokens
            );
            hipLaunchKernelGGL(
                postconv_value_inplace_kernel,
                suffix_value_grid,
                block,
                0u,
                nullptr,
                suffix_postconv.get(),
                kSuffixTokens
            );
            if (api.q1024_seeded_launch_async(
                    suffix_postconv.get(),
                    gate.get() +
                        static_cast<size_t>(kPrefixTokens) *
                            kGateOutputRows,
                    prefix_state_key.get(),
                    suffix_output.get(),
                    suffix_state_key.get(),
                    gate_values_are_decay,
                    nullptr
                ) == 0) {
                throw std::runtime_error(
                    std::string("q1024 seeded provider failed: ") +
                    api.last_error()
                );
            }
            hipLaunchKernelGGL(
                publish_final_ring_kernel<T>,
                ring_grid,
                block,
                0u,
                nullptr,
                suffix_qkv,
                private_final_ring.get()
            );
        };

        for (unsigned int warmup = 0u; warmup < 2u; ++warmup) {
            launch_candidate();
        }
        check_hip(hipGetLastError(), "candidate warmup launch");
        check_hip(
            hipDeviceSynchronize(),
            "candidate warmup synchronize"
        );

        hipEvent_t start = nullptr;
        hipEvent_t stop = nullptr;
        check_hip(hipEventCreate(&start), "hipEventCreate(start)");
        check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
        check_hip(hipEventRecord(start, nullptr), "hipEventRecord(start)");
        for (unsigned int repetition = 0u;
             repetition < repetitions;
             ++repetition) {
            launch_candidate();
        }
        check_hip(hipGetLastError(), "candidate timed launch");
        check_hip(hipEventRecord(stop, nullptr), "hipEventRecord(stop)");
        check_hip(
            hipEventSynchronize(stop),
            "hipEventSynchronize(stop)"
        );
        float total_ms = 0.0f;
        check_hip(
            hipEventElapsedTime(&total_ms, start, stop),
            "hipEventElapsedTime"
        );
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        const double mean_ms =
            static_cast<double>(total_ms) /
            static_cast<double>(repetitions);

        if (api.q17408_launch(
                full_postconv.get(),
                gate.get(),
                full_output.get(),
                full_state_row.get(),
                gate_values_are_decay,
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("q17408 provider failed: ") + api.last_error()
            );
        }

        const std::vector<float> host_postconv_reference = copy_device(
            full_postconv.get() +
                static_cast<size_t>(kPrefixTokens) * kQkvRows,
            suffix_qkv_elements
        );
        const std::vector<float> host_postconv_candidate =
            copy_device(suffix_postconv.get(), suffix_qkv_elements);
        const std::vector<float> host_output_reference = copy_device(
            full_output.get() +
                static_cast<size_t>(kPrefixTokens) * kValueFeatures,
            suffix_output_elements
        );
        const std::vector<float> host_output_candidate =
            copy_device(suffix_output.get(), suffix_output_elements);
        const std::vector<float> host_state_reference =
            copy_device(full_state_row.get(), state_elements);
        const std::vector<float> host_state_candidate =
            copy_device(suffix_state_key.get(), state_elements);
        const std::vector<float> host_state_after =
            copy_device(prefix_state_key.get(), state_elements);
        const std::vector<T> host_prefix_after =
            copy_device(prefix_ring.get(), ring_elements);
        const std::vector<T> host_final_candidate =
            copy_device(private_final_ring.get(), ring_elements);
        const std::vector<T> host_final_reference = copy_device(
            full_qkv.get() +
                static_cast<size_t>(kFullTokens - kConvTaps) * kQkvRows,
            ring_elements
        );

        const CompareStats postconv_stats = compare_f32(
            host_postconv_reference,
            host_postconv_candidate
        );
        const CompareStats output_stats = compare_f32(
            host_output_reference,
            host_output_candidate
        );
        const CompareStats state_stats = compare_state_row_to_key(
            host_state_reference,
            host_state_candidate
        );
        const CompareStats prefix_state_mutation = compare_f32(
            host_state_before,
            host_state_after
        );
        const CompareStats prefix_ring_mutation = compare_native(
            host_prefix_before,
            host_prefix_after
        );
        const CompareStats final_ring_stats = compare_native(
            host_final_reference,
            host_final_candidate
        );
        const bool pass =
            prefix_mapping.mismatches == 0u &&
            prefix_mapping.nonfinite == 0u &&
            postconv_stats.mismatches == 0u &&
            postconv_stats.nonfinite == 0u &&
            postconv_stats.max_abs == 0.0 &&
            output_stats.mismatches == 0u &&
            output_stats.nonfinite == 0u &&
            output_stats.max_abs == 0.0 &&
            state_stats.mismatches == 0u &&
            state_stats.nonfinite == 0u &&
            state_stats.max_abs == 0.0 &&
            prefix_state_mutation.mismatches == 0u &&
            prefix_ring_mutation.mismatches == 0u &&
            final_ring_stats.mismatches == 0u &&
            final_ring_stats.nonfinite == 0u &&
            final_ring_stats.max_abs == 0.0 &&
            mean_ms <= kCaseMeanCeilingMs;

        std::cout
            << std::fixed << std::setprecision(6)
            << "q16384_suffix1024_connected_linear_core_smoke"
            << " ring_mode=" << ring_mode
            << " gate_values_are_decay=" << gate_values_are_decay
            << " prefix_tokens=" << kPrefixTokens
            << " suffix_tokens=" << kSuffixTokens
            << " full_tokens=" << kFullTokens
            << " repetitions=" << repetitions
            << " connected_mean_ms=" << mean_ms
            << " connected_ms_ceiling=" << kCaseMeanCeilingMs
            << " postconv_elements=" << postconv_stats.elements
            << " postconv_mismatches=" << postconv_stats.mismatches
            << " postconv_nonfinite=" << postconv_stats.nonfinite
            << " postconv_first_mismatch="
            << postconv_stats.first_mismatch
            << " postconv_max_abs=" << postconv_stats.max_abs
            << " recurrent_output_elements=" << output_stats.elements
            << " recurrent_output_mismatches="
            << output_stats.mismatches
            << " recurrent_output_nonfinite=" << output_stats.nonfinite
            << " recurrent_output_first_mismatch="
            << output_stats.first_mismatch
            << " recurrent_output_max_abs=" << output_stats.max_abs
            << " final_state_elements=" << state_stats.elements
            << " final_state_mismatches=" << state_stats.mismatches
            << " final_state_nonfinite=" << state_stats.nonfinite
            << " final_state_first_mismatch="
            << state_stats.first_mismatch
            << " final_state_max_abs=" << state_stats.max_abs
            << " prefix_ring_mapping_mismatches="
            << prefix_mapping.mismatches
            << " prefix_ring_mutation_mismatches="
            << prefix_ring_mutation.mismatches
            << " prefix_state_mutation_mismatches="
            << prefix_state_mutation.mismatches
            << " final_ring_elements=" << final_ring_stats.elements
            << " final_ring_mismatches="
            << final_ring_stats.mismatches
            << " final_ring_nonfinite=" << final_ring_stats.nonfinite
            << " final_ring_first_mismatch="
            << final_ring_stats.first_mismatch
            << " final_ring_max_abs=" << final_ring_stats.max_abs
            << " postconv_reference_hash=" << std::hex
            << fnv1a64(
                   host_postconv_reference.data(),
                   host_postconv_reference.size() * sizeof(float)
               )
            << " postconv_candidate_hash="
            << fnv1a64(
                   host_postconv_candidate.data(),
                   host_postconv_candidate.size() * sizeof(float)
               )
            << " recurrent_output_reference_hash="
            << fnv1a64(
                   host_output_reference.data(),
                   host_output_reference.size() * sizeof(float)
               )
            << " recurrent_output_candidate_hash="
            << fnv1a64(
                   host_output_candidate.data(),
                   host_output_candidate.size() * sizeof(float)
               )
            << std::dec
            << " quantized=0"
            << " mtp_active=0"
            << " dflash_active=0"
            << " speculative_decode=0"
            << " shared_prefix_mutated="
            << (
                prefix_ring_mutation.mismatches == 0u &&
                prefix_state_mutation.mismatches == 0u
                    ? 0
                    : 1
            )
            << " pass=" << (pass ? 1 : 0)
            << "\n";

        results.push_back(CaseResult{
            ring_mode,
            gate_values_are_decay,
            mean_ms,
            pass,
        });
    }
    return results;
}

}  // namespace

int main(int argc, char **argv) {
    ProviderApi api;
    try {
        if (argc != 4) {
            std::cerr
                << "usage: q16384_suffix1024_connected_linear_core_smoke "
                << "KERNEL_DIR PROVIDER_DLL REPETITIONS>=11\n";
            return 2;
        }
        const int parsed_repetitions = std::stoi(argv[3]);
        if (parsed_repetitions < 11 || parsed_repetitions > 100) {
            std::cerr << "repetitions must be in [11,100]\n";
            return 2;
        }
        const unsigned int repetitions =
            static_cast<unsigned int>(parsed_repetitions);
        int device_count = 0;
        check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("no HIP device is available");
        }
        check_hip(hipSetDevice(0), "hipSetDevice");
        if (!load_provider(argv[2], &api)) {
            throw std::runtime_error(
                "could not load all required provider exports"
            );
        }
        if (api.prepare(argv[1]) == 0) {
            throw std::runtime_error(
                std::string("provider prepare failed: ") + api.last_error()
            );
        }

        const std::vector<CaseResult> early_results =
            run_ring_mode<float>("early_f32", api, repetitions);
        const std::vector<CaseResult> retained_results =
            run_ring_mode<uint16_t>("retained_bf16", api, repetitions);
        if (early_results.size() != 2u || retained_results.size() != 2u) {
            throw std::runtime_error("connected case count mismatch");
        }
        const double native_projection_ms =
            2.0 * early_results[1].mean_ms +
            28.0 * retained_results[0].mean_ms;
        const bool all_cases_pass =
            early_results[0].pass &&
            early_results[1].pass &&
            retained_results[0].pass &&
            retained_results[1].pass;
        const bool summary_pass =
            all_cases_pass &&
            native_projection_ms <= kNativeProjectionCeilingMs;
        std::cout
            << std::fixed << std::setprecision(6)
            << "q16384_suffix1024_connected_linear_core_summary"
            << " product_early_gate_values_are_decay=1"
            << " product_retained_gate_values_are_decay=0"
            << " early_f32_layers=2"
            << " retained_bf16_layers=28"
            << " native_thirty_layer_projected_ms="
            << native_projection_ms
            << " native_projection_ceiling_ms="
            << kNativeProjectionCeilingMs
            << " cases_pass=" << (all_cases_pass ? 4 : 0)
            << " pass=" << (summary_pass ? 1 : 0)
            << "\n";

        api.release();
        FreeLibrary(api.module);
        return summary_pass ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr
            << "q16384_suffix1024_connected_linear_core_smoke error: "
            << error.what() << "\n";
        if (api.release != nullptr) {
            api.release();
        }
        if (api.module != nullptr) {
            FreeLibrary(api.module);
        }
        return 1;
    }
}
