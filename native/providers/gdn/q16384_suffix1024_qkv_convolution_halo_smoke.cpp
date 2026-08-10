#include <hip/hip_runtime.h>

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
#include <type_traits>
#include <vector>

namespace {

constexpr unsigned int kPrefixTokens = 16384u;
constexpr unsigned int kSuffixTokens = 1024u;
constexpr unsigned int kFullTokens = kPrefixTokens + kSuffixTokens;
constexpr unsigned int kQkvRows = 8192u;
constexpr unsigned int kConvTaps = 4u;
constexpr unsigned int kThreads = 256u;
constexpr double kSingleLayerCeilingMs = 10.0;
constexpr double kThirtyLayerCeilingMs = 300.0;

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

    size_t count() const {
        return count_;
    }

private:
    T *data_ = nullptr;
    size_t count_ = 0u;
};

__device__ __forceinline__ uint16_t device_float_to_bf16(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t rounding_bias = UINT32_C(0x7fff) + ((bits >> 16u) & 1u);
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
        const float source = device_bf16_round_to_float(sources[tap]);
        const float weight = device_bf16_to_float(weights[tap]);
        accumulator += source * weight;
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
    const float value = static_cast<float>(centered) * 0.0009765625f;
    weights[index] = device_float_to_bf16(value);
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
__global__ void uninterrupted_tail_conv_kernel(
    const T *full_qkv,
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
    const unsigned int source_base = absolute_token - (kConvTaps - 1u);
    const size_t row0 =
        static_cast<size_t>(source_base) * kQkvRows + feature;
    const size_t row1 = row0 + kQkvRows;
    const size_t row2 = row1 + kQkvRows;
    const size_t row3 = row2 + kQkvRows;
    const uint16_t *row_weights =
        weights + static_cast<size_t>(feature) * kConvTaps;
    outputs[static_cast<size_t>(suffix_token) * kQkvRows + feature] =
        device_conv_endpoint(
            device_native_to_float(full_qkv[row0]),
            device_native_to_float(full_qkv[row1]),
            device_native_to_float(full_qkv[row2]),
            device_native_to_float(full_qkv[row3]),
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
            const unsigned int slot = source_token % kConvTaps;
            sources[tap] = device_native_to_float(
                prefix_ring[
                    static_cast<size_t>(slot) * kQkvRows + feature
                ]
            );
        } else {
            const unsigned int private_token =
                source_token - kPrefixTokens;
            sources[tap] = device_native_to_float(
                suffix_qkv[
                    static_cast<size_t>(private_token) * kQkvRows + feature
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
    const unsigned int slot = absolute_token % kConvTaps;
    private_ring[static_cast<size_t>(slot) * kQkvRows + feature] =
        suffix_qkv[static_cast<size_t>(suffix_token) * kQkvRows + feature];
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
float host_native_to_float(T value);

template <>
float host_native_to_float<float>(float value) {
    return value;
}

template <>
float host_native_to_float<uint16_t>(uint16_t value) {
    return host_bf16_to_float(value);
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

struct ModeResult {
    std::string name;
    double mean_ms = 0.0;
    double thirty_layer_ms = 0.0;
    bool pass = false;
};

template <typename T>
ModeResult run_mode(const char *mode_name, unsigned int repetitions) {
    const size_t full_elements =
        static_cast<size_t>(kFullTokens) * kQkvRows;
    const size_t suffix_elements =
        static_cast<size_t>(kSuffixTokens) * kQkvRows;
    const size_t ring_elements =
        static_cast<size_t>(kConvTaps) * kQkvRows;
    const size_t weight_elements =
        static_cast<size_t>(kQkvRows) * kConvTaps;

    DeviceBuffer<T> full_qkv(full_elements);
    DeviceBuffer<uint16_t> weights(weight_elements);
    DeviceBuffer<T> prefix_ring(ring_elements);
    DeviceBuffer<T> private_final_ring(ring_elements);
    DeviceBuffer<float> reference_output(suffix_elements);
    DeviceBuffer<float> candidate_output(suffix_elements);

    const dim3 linear_block(kThreads);
    const dim3 full_grid(
        static_cast<unsigned int>(
            (full_elements + kThreads - 1u) / kThreads
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
    const dim3 conv_grid(
        (kQkvRows + kThreads - 1u) / kThreads,
        kSuffixTokens
    );

    hipLaunchKernelGGL(
        fill_qkv_kernel<T>,
        full_grid,
        linear_block,
        0u,
        nullptr,
        full_qkv.get(),
        full_elements
    );
    hipLaunchKernelGGL(
        fill_weights_kernel,
        weight_grid,
        linear_block,
        0u,
        nullptr,
        weights.get(),
        weight_elements
    );
    hipLaunchKernelGGL(
        capture_four_token_ring_kernel<T>,
        ring_grid,
        linear_block,
        0u,
        nullptr,
        full_qkv.get(),
        prefix_ring.get(),
        kPrefixTokens
    );
    hipLaunchKernelGGL(
        uninterrupted_tail_conv_kernel<T>,
        conv_grid,
        linear_block,
        0u,
        nullptr,
        full_qkv.get(),
        weights.get(),
        reference_output.get()
    );
    check_hip(hipGetLastError(), "reference launch");
    check_hip(hipDeviceSynchronize(), "reference synchronize");

    const std::vector<T> host_prefix_before =
        copy_device(prefix_ring.get(), ring_elements);
    const std::vector<T> host_prefix_direct = copy_device(
        full_qkv.get() +
            static_cast<size_t>(kPrefixTokens - kConvTaps) * kQkvRows,
        ring_elements
    );
    const T *suffix_qkv =
        full_qkv.get() + static_cast<size_t>(kPrefixTokens) * kQkvRows;
    for (unsigned int warmup = 0u; warmup < 2u; ++warmup) {
        hipLaunchKernelGGL(
            suffix_halo_conv_kernel<T>,
            conv_grid,
            linear_block,
            0u,
            nullptr,
            prefix_ring.get(),
            suffix_qkv,
            weights.get(),
            candidate_output.get()
        );
        hipLaunchKernelGGL(
            publish_final_ring_kernel<T>,
            ring_grid,
            linear_block,
            0u,
            nullptr,
            suffix_qkv,
            private_final_ring.get()
        );
    }
    check_hip(hipGetLastError(), "warmup launch");
    check_hip(hipDeviceSynchronize(), "warmup synchronize");

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "hipEventCreate(start)");
    check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
    double elapsed_total_ms = 0.0;
    for (unsigned int repetition = 0u;
         repetition < repetitions;
         ++repetition) {
        check_hip(hipEventRecord(start, nullptr), "hipEventRecord(start)");
        hipLaunchKernelGGL(
            suffix_halo_conv_kernel<T>,
            conv_grid,
            linear_block,
            0u,
            nullptr,
            prefix_ring.get(),
            suffix_qkv,
            weights.get(),
            candidate_output.get()
        );
        hipLaunchKernelGGL(
            publish_final_ring_kernel<T>,
            ring_grid,
            linear_block,
            0u,
            nullptr,
            suffix_qkv,
            private_final_ring.get()
        );
        check_hip(hipGetLastError(), "timed launch");
        check_hip(hipEventRecord(stop, nullptr), "hipEventRecord(stop)");
        check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
        float elapsed_ms = 0.0f;
        check_hip(
            hipEventElapsedTime(&elapsed_ms, start, stop),
            "hipEventElapsedTime"
        );
        elapsed_total_ms += static_cast<double>(elapsed_ms);
    }
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);

    const std::vector<float> host_reference =
        copy_device(reference_output.get(), suffix_elements);
    const std::vector<float> host_candidate =
        copy_device(candidate_output.get(), suffix_elements);
    const std::vector<T> host_prefix_after =
        copy_device(prefix_ring.get(), ring_elements);
    const std::vector<T> host_final_ring =
        copy_device(private_final_ring.get(), ring_elements);
    const std::vector<T> host_final_direct = copy_device(
        full_qkv.get() +
            static_cast<size_t>(kFullTokens - kConvTaps) * kQkvRows,
        ring_elements
    );

    const CompareStats output_stats =
        compare_f32(host_reference, host_candidate);
    const CompareStats prefix_mapping_stats =
        compare_native(host_prefix_direct, host_prefix_before);
    const CompareStats prefix_mutation_stats =
        compare_native(host_prefix_before, host_prefix_after);
    const CompareStats final_ring_stats =
        compare_native(host_final_direct, host_final_ring);
    const double mean_ms =
        elapsed_total_ms / static_cast<double>(repetitions);
    const double thirty_layer_ms = mean_ms * 30.0;
    const bool pass =
        output_stats.mismatches == 0u &&
        output_stats.nonfinite == 0u &&
        output_stats.max_abs == 0.0 &&
        prefix_mapping_stats.mismatches == 0u &&
        prefix_mapping_stats.nonfinite == 0u &&
        prefix_mutation_stats.mismatches == 0u &&
        final_ring_stats.mismatches == 0u &&
        final_ring_stats.nonfinite == 0u &&
        final_ring_stats.max_abs == 0.0 &&
        mean_ms <= kSingleLayerCeilingMs &&
        thirty_layer_ms <= kThirtyLayerCeilingMs;

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_qkv_convolution_halo_smoke"
        << " ring_mode=" << mode_name
        << " prefix_tokens=" << kPrefixTokens
        << " suffix_tokens=" << kSuffixTokens
        << " full_tokens=" << kFullTokens
        << " qkv_rows=" << kQkvRows
        << " conv_taps=" << kConvTaps
        << " repetitions=" << repetitions
        << " suffix_mean_ms=" << mean_ms
        << " suffix_ms_ceiling=" << kSingleLayerCeilingMs
        << " thirty_layer_projected_ms=" << thirty_layer_ms
        << " output_elements=" << output_stats.elements
        << " output_mismatches=" << output_stats.mismatches
        << " output_nonfinite=" << output_stats.nonfinite
        << " output_first_mismatch=" << output_stats.first_mismatch
        << " output_max_abs=" << output_stats.max_abs
        << " prefix_ring_elements=" << prefix_mapping_stats.elements
        << " prefix_ring_mapping_mismatches="
        << prefix_mapping_stats.mismatches
        << " prefix_ring_mutation_mismatches="
        << prefix_mutation_stats.mismatches
        << " final_ring_elements=" << final_ring_stats.elements
        << " final_ring_mismatches=" << final_ring_stats.mismatches
        << " final_ring_nonfinite=" << final_ring_stats.nonfinite
        << " final_ring_first_mismatch="
        << final_ring_stats.first_mismatch
        << " final_ring_max_abs=" << final_ring_stats.max_abs
        << " output_reference_hash=" << std::hex
        << fnv1a64(
               host_reference.data(),
               host_reference.size() * sizeof(float)
           )
        << " output_candidate_hash="
        << fnv1a64(
               host_candidate.data(),
               host_candidate.size() * sizeof(float)
           )
        << " final_ring_reference_hash="
        << fnv1a64(
               host_final_direct.data(),
               host_final_direct.size() * sizeof(T)
           )
        << " final_ring_candidate_hash="
        << fnv1a64(
               host_final_ring.data(),
               host_final_ring.size() * sizeof(T)
           )
        << std::dec
        << " quantized=0"
        << " mtp_active=0"
        << " dflash_active=0"
        << " speculative_decode=0"
        << " shared_prefix_ring_mutated="
        << (prefix_mutation_stats.mismatches == 0u ? 0 : 1)
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    return ModeResult{
        mode_name,
        mean_ms,
        thirty_layer_ms,
        pass,
    };
}

}  // namespace

int main(int argc, char **argv) {
    try {
        if (argc != 2) {
            std::cerr
                << "usage: q16384_suffix1024_qkv_convolution_halo_smoke "
                << "REPETITIONS>=11\n";
            return 2;
        }
        const int parsed_repetitions = std::stoi(argv[1]);
        if (parsed_repetitions < 11) {
            std::cerr << "repetitions must be at least 11\n";
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

        const ModeResult early_f32 =
            run_mode<float>("early_f32", repetitions);
        const ModeResult retained_bf16 =
            run_mode<uint16_t>("retained_bf16", repetitions);
        const double native_mixed_projection_ms =
            2.0 * early_f32.mean_ms + 28.0 * retained_bf16.mean_ms;
        std::cout
            << std::fixed << std::setprecision(6)
            << "q16384_suffix1024_qkv_convolution_halo_summary"
            << " early_f32_layers=2"
            << " retained_bf16_layers=28"
            << " native_mixed_thirty_layer_projected_ms="
            << native_mixed_projection_ms
            << " native_mixed_ceiling_ms=" << kThirtyLayerCeilingMs
            << " pass="
            << (
                early_f32.pass &&
                retained_bf16.pass &&
                native_mixed_projection_ms <= kThirtyLayerCeilingMs
                    ? 1
                    : 0
            )
            << "\n";
        return (
            early_f32.pass &&
            retained_bf16.pass &&
            native_mixed_projection_ms <= kThirtyLayerCeilingMs
        ) ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr
            << "q16384_suffix1024_qkv_convolution_halo_smoke error: "
            << error.what() << "\n";
        return 1;
    }
}
