#define main qrt_d228_reference_main
#include "q16384_suffix1024_connected_linear_core_smoke.cpp"
#undef main

#define ROCBLAS_BETA_FEATURES_API
#include <rocblas/rocblas.h>

#include <array>
#include <type_traits>

namespace {

constexpr unsigned int kLayerHidden = 2048u;
constexpr unsigned int kZRows = 4096u;
constexpr unsigned int kAbRows = 32u;
constexpr float kRmsNormEpsilon = 1.0e-6f;
constexpr double kProjectionSampleMaxAbs = 0.125;
constexpr double kSublayerMeanCeilingMs = 60.0;
constexpr double kSublayerNativeProjectionCeilingMs = 1800.0;

void check_rocblas(rocblas_status status, const char *stage) {
    if (status != rocblas_status_success) {
        throw std::runtime_error(
            std::string(stage) + ": rocBLAS status " +
            std::to_string(static_cast<int>(status))
        );
    }
}

class RocblasContext {
public:
    RocblasContext() {
        check_rocblas(
            rocblas_create_handle(&handle_),
            "rocblas_create_handle"
        );
        check_rocblas(
            rocblas_set_stream(handle_, nullptr),
            "rocblas_set_stream"
        );
        check_rocblas(
            rocblas_set_pointer_mode(handle_, rocblas_pointer_mode_host),
            "rocblas_set_pointer_mode"
        );
    }

    ~RocblasContext() {
        if (handle_ != nullptr) {
            (void)rocblas_destroy_handle(handle_);
        }
    }

    RocblasContext(const RocblasContext &) = delete;
    RocblasContext &operator=(const RocblasContext &) = delete;

    template <typename Output>
    void matmul(
        const uint16_t *weights,
        const uint16_t *inputs,
        Output *outputs,
        unsigned int output_features,
        unsigned int input_features
    ) const {
        static_assert(
            std::is_same<Output, float>::value ||
            std::is_same<Output, uint16_t>::value,
            "unsupported projection endpoint"
        );
        const float alpha = 1.0f;
        const float beta = 0.0f;
        const rocblas_datatype output_type =
            std::is_same<Output, float>::value
                ? rocblas_datatype_f32_r
                : rocblas_datatype_bf16_r;
        check_rocblas(
            rocblas_gemm_ex(
                handle_,
                rocblas_operation_transpose,
                rocblas_operation_none,
                static_cast<rocblas_int>(output_features),
                static_cast<rocblas_int>(kSuffixTokens),
                static_cast<rocblas_int>(input_features),
                &alpha,
                weights,
                rocblas_datatype_bf16_r,
                static_cast<rocblas_int>(input_features),
                inputs,
                rocblas_datatype_bf16_r,
                static_cast<rocblas_int>(input_features),
                &beta,
                outputs,
                output_type,
                static_cast<rocblas_int>(output_features),
                outputs,
                output_type,
                static_cast<rocblas_int>(output_features),
                rocblas_datatype_f32_r,
                rocblas_gemm_algo_standard,
                0,
                rocblas_gemm_flags_none
            ),
            "rocblas_gemm_ex"
        );
    }

private:
    rocblas_handle handle_ = nullptr;
};

__global__ void fill_hidden_kernel(float *values, size_t count) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    uint32_t state =
        static_cast<uint32_t>(index) * UINT32_C(2246822519) +
        UINT32_C(3266489917);
    state ^= state >> 15u;
    state *= UINT32_C(668265263);
    state ^= state >> 13u;
    const int32_t centered =
        static_cast<int32_t>(state & UINT32_C(0xff)) - 128;
    values[index] = static_cast<float>(centered) * 0.0078125f;
}

__global__ void fill_dense_weights_kernel(
    uint16_t *values,
    size_t count,
    uint32_t seed,
    float scale
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    uint32_t state =
        static_cast<uint32_t>(index) * UINT32_C(747796405) + seed;
    state ^= state >> 16u;
    state *= UINT32_C(2246822519);
    state ^= state >> 13u;
    const int32_t centered =
        static_cast<int32_t>(state & UINT32_C(0xff)) - 128;
    values[index] =
        device_float_to_bf16(static_cast<float>(centered) * scale);
}

__global__ void fill_norm_weights_kernel(
    uint16_t *values,
    size_t count,
    int direct_scale
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const int32_t centered =
        static_cast<int32_t>((index * 29u + 17u) & 255u) - 128;
    const float delta = static_cast<float>(centered) * 0.0009765625f;
    values[index] = device_float_to_bf16(
        direct_scale != 0 ? 1.0f + delta : delta
    );
}

__global__ void fill_gate_parameters_kernel(
    uint16_t *a_log,
    uint16_t *dt_bias
) {
    const unsigned int head =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (head >= kGateRows) {
        return;
    }
    a_log[head] = device_float_to_bf16(
        -1.75f - 0.0078125f * static_cast<float>(head)
    );
    dt_bias[head] = device_float_to_bf16(
        (static_cast<int>(head) - 16) * 0.015625f
    );
}

__global__ void input_rmsnorm_kernel(
    const float *inputs,
    const uint16_t *weights,
    float *outputs
) {
    __shared__ double partial[kThreads];
    __shared__ float inv_rms;

    const unsigned int token = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (token >= kSuffixTokens) {
        return;
    }
    const size_t base = static_cast<size_t>(token) * kLayerHidden;
    double sumsq = 0.0;
    for (unsigned int col = lane; col < kLayerHidden; col += blockDim.x) {
        const float value = inputs[base + col];
        sumsq += static_cast<double>(value) * static_cast<double>(value);
    }
    partial[lane] = sumsq;
    __syncthreads();
    for (unsigned int stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        inv_rms = 1.0f / sqrtf(
            static_cast<float>(
                partial[0] / static_cast<double>(kLayerHidden)
            ) + kRmsNormEpsilon
        );
    }
    __syncthreads();
    for (unsigned int col = lane; col < kLayerHidden; col += blockDim.x) {
        outputs[base + col] = device_bf16_round_to_float(
            inputs[base + col] * inv_rms *
            (1.0f + device_bf16_to_float(weights[col]))
        );
    }
}

__global__ void f32_to_bf16_sublayer_kernel(
    const float *inputs,
    uint16_t *outputs,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        outputs[index] = device_float_to_bf16(inputs[index]);
    }
}

template <typename T>
__global__ void native_to_f32_sublayer_kernel(
    const T *inputs,
    float *outputs,
    size_t count
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        outputs[index] = device_native_to_float(inputs[index]);
    }
}

__device__ float sublayer_softplus(float value) {
    float result = value > 0.0f
        ? value + logf(1.0f + expf(-value))
        : logf(1.0f + expf(value));
    if (!(value <= 20.0f)) {
        result = value;
    }
    return result;
}

template <bool GateValuesAreDecay>
__global__ void gate_from_ab_kernel(
    const float *a_values,
    const float *b_values,
    const uint16_t *a_log,
    const uint16_t *dt_bias,
    float *outputs
) {
    const unsigned int head =
        static_cast<unsigned int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned int token = blockIdx.y;
    if (head >= kGateRows || token >= kSuffixTokens) {
        return;
    }
    const size_t input_index =
        static_cast<size_t>(token) * kGateRows + head;
    const float a =
        a_values[input_index] + device_bf16_to_float(dt_bias[head]);
    const float log_gate =
        -expf(device_bf16_to_float(a_log[head])) * sublayer_softplus(a);
    const float beta =
        1.0f / (1.0f + expf(-b_values[input_index]));
    const size_t output_base =
        static_cast<size_t>(token) * kGateOutputRows;
    outputs[output_base + head] =
        GateValuesAreDecay ? expf(log_gate) : log_gate;
    outputs[output_base + kGateRows + head] =
        device_bf16_round_to_float(beta);
}

__global__ void gated_rmsnorm_sublayer_kernel(
    const float *core_values,
    const float *z_values,
    const uint16_t *weights,
    uint16_t *outputs
) {
    __shared__ double partial[kValueDim];
    __shared__ float inv_rms;

    const unsigned int value_dim = threadIdx.x;
    const unsigned int value_head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    if (value_dim >= kValueDim || value_head >= kValueHeads ||
        token >= kSuffixTokens) {
        return;
    }
    const unsigned int value_index =
        value_head * kValueDim + value_dim;
    const size_t base = static_cast<size_t>(token) * kValueFeatures;
    const float core = core_values[base + value_index];
    partial[value_dim] =
        static_cast<double>(core) * static_cast<double>(core);
    __syncthreads();
    for (unsigned int stride = kValueDim / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (value_dim < stride) {
            partial[value_dim] += partial[value_dim + stride];
        }
        __syncthreads();
    }
    if (value_dim == 0u) {
        inv_rms = 1.0f / sqrtf(
            static_cast<float>(
                partial[0] / static_cast<double>(kValueDim)
            ) + kRmsNormEpsilon
        );
    }
    __syncthreads();
    const float normalized = device_bf16_round_to_float(
        core * inv_rms * device_bf16_to_float(weights[value_dim])
    );
    const float gated = device_bf16_round_to_float(
        normalized *
        (z_values[base + value_index] /
         (1.0f + expf(-z_values[base + value_index])))
    );
    outputs[base + value_index] = device_float_to_bf16(gated);
}

template <typename T>
__global__ void residual_postnorm_sublayer_kernel(
    const float *residual_inputs,
    const T *attention_updates,
    const uint16_t *weights,
    float *residual_outputs,
    float *postnorm_outputs
) {
    __shared__ double partial[kThreads];
    __shared__ float inv_rms;
    constexpr unsigned int kValuesPerLane = kLayerHidden / kThreads;
    float values[kValuesPerLane];

    const unsigned int token = blockIdx.x;
    const unsigned int lane = threadIdx.x;
    if (token >= kSuffixTokens) {
        return;
    }
    const size_t base = static_cast<size_t>(token) * kLayerHidden;
    double sumsq = 0.0;
#pragma unroll
    for (unsigned int item = 0u; item < kValuesPerLane; ++item) {
        const unsigned int col = lane + item * kThreads;
        const float value =
            residual_inputs[base + col] +
            device_native_to_float(attention_updates[base + col]);
        values[item] = value;
        residual_outputs[base + col] = value;
        sumsq += static_cast<double>(value) * static_cast<double>(value);
    }
    partial[lane] = sumsq;
    __syncthreads();
    for (unsigned int stride = kThreads / 2u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0u) {
        inv_rms = 1.0f / sqrtf(
            static_cast<float>(
                partial[0] / static_cast<double>(kLayerHidden)
            ) + kRmsNormEpsilon
        );
    }
    __syncthreads();
#pragma unroll
    for (unsigned int item = 0u; item < kValuesPerLane; ++item) {
        const unsigned int col = lane + item * kThreads;
        postnorm_outputs[base + col] = device_bf16_round_to_float(
            values[item] * inv_rms *
            (1.0f + device_bf16_to_float(weights[col]))
        );
    }
}

struct ProjectionSampleStats {
    uint64_t samples = 0u;
    uint64_t nonfinite = 0u;
    double max_abs = 0.0;
};

template <typename T>
ProjectionSampleStats validate_projection_samples(
    const uint16_t *device_weights,
    const uint16_t *device_inputs,
    const T *device_outputs,
    unsigned int output_features,
    unsigned int input_features
) {
    const std::array<unsigned int, 4> tokens = {
        0u,
        kSuffixTokens / 3u,
        (2u * kSuffixTokens) / 3u,
        kSuffixTokens - 1u,
    };
    const std::array<unsigned int, 4> rows = {
        0u,
        output_features / 3u,
        (2u * output_features) / 3u,
        output_features - 1u,
    };
    ProjectionSampleStats stats;
    for (unsigned int token : tokens) {
        std::vector<uint16_t> input(input_features);
        check_hip(
            hipMemcpy(
                input.data(),
                device_inputs +
                    static_cast<size_t>(token) * input_features,
                static_cast<size_t>(input_features) * sizeof(uint16_t),
                hipMemcpyDeviceToHost
            ),
            "projection sample input copy"
        );
        for (unsigned int row : rows) {
            std::vector<uint16_t> weight(input_features);
            check_hip(
                hipMemcpy(
                    weight.data(),
                    device_weights +
                        static_cast<size_t>(row) * input_features,
                    static_cast<size_t>(input_features) * sizeof(uint16_t),
                    hipMemcpyDeviceToHost
                ),
                "projection sample weight copy"
            );
            T candidate_native{};
            check_hip(
                hipMemcpy(
                    &candidate_native,
                    device_outputs +
                        static_cast<size_t>(token) * output_features + row,
                    sizeof(T),
                    hipMemcpyDeviceToHost
                ),
                "projection sample output copy"
            );
            float reference = 0.0f;
            for (unsigned int col = 0u; col < input_features; ++col) {
                reference +=
                    host_bf16_to_float(input[col]) *
                    host_bf16_to_float(weight[col]);
            }
            const float candidate =
                host_native_to_float(candidate_native);
            if (!std::isfinite(reference) || !std::isfinite(candidate)) {
                ++stats.nonfinite;
            } else {
                stats.max_abs = std::max(
                    stats.max_abs,
                    std::abs(
                        static_cast<double>(reference) -
                        static_cast<double>(candidate)
                    )
                );
            }
            ++stats.samples;
        }
    }
    return stats;
}

struct SublayerResult {
    std::string mode;
    double mean_ms = 0.0;
    bool pass = false;
};

struct SublayerSurfaceExport {
    using TailLaunchFunction = void (*)(
        void *,
        const float *,
        const float *,
        hipStream_t
    );

    float *candidate_residual = nullptr;
    float *candidate_postnorm = nullptr;
    float *reference_residual = nullptr;
    float *reference_postnorm = nullptr;
    size_t elements = 0u;
    TailLaunchFunction tail_launch = nullptr;
    void *tail_context = nullptr;
    double mean_ms_ceiling = 0.0;
    bool tail_is_moe = false;
};

template <typename T, bool GateValuesAreDecay>
SublayerResult run_sublayer_mode(
    const char *mode,
    const ProviderApi &api,
    unsigned int repetitions,
    const SublayerSurfaceExport *surface_export = nullptr
) {
    const size_t hidden_elements =
        static_cast<size_t>(kSuffixTokens) * kLayerHidden;
    const size_t qkv_elements =
        static_cast<size_t>(kSuffixTokens) * kQkvRows;
    const size_t z_elements =
        static_cast<size_t>(kSuffixTokens) * kZRows;
    const size_t ab_elements =
        static_cast<size_t>(kSuffixTokens) * kAbRows;
    const size_t core_elements =
        static_cast<size_t>(kSuffixTokens) * kValueFeatures;
    const size_t gate_elements =
        static_cast<size_t>(kFullTokens) * kGateOutputRows;
    const size_t state_elements =
        static_cast<size_t>(kValueFeatures) * kKeyDim;
    const size_t ring_elements =
        static_cast<size_t>(kConvTaps) * kQkvRows;
    const size_t conv_weight_elements =
        static_cast<size_t>(kQkvRows) * kConvTaps;
    const size_t full_qkv_elements =
        static_cast<size_t>(kFullTokens) * kQkvRows;
    const size_t full_core_elements =
        static_cast<size_t>(kFullTokens) * kValueFeatures;

    DeviceBuffer<float> hidden(hidden_elements);
    DeviceBuffer<uint16_t> input_norm_weights(kLayerHidden);
    DeviceBuffer<float> input_norm_f32(hidden_elements);
    DeviceBuffer<uint16_t> input_norm_bf16(hidden_elements);

    DeviceBuffer<uint16_t> qkv_weights(
        static_cast<size_t>(kQkvRows) * kLayerHidden
    );
    DeviceBuffer<uint16_t> z_weights(
        static_cast<size_t>(kZRows) * kLayerHidden
    );
    DeviceBuffer<uint16_t> a_weights(
        static_cast<size_t>(kAbRows) * kLayerHidden
    );
    DeviceBuffer<uint16_t> b_weights(
        static_cast<size_t>(kAbRows) * kLayerHidden
    );
    DeviceBuffer<uint16_t> out_weights(
        static_cast<size_t>(kLayerHidden) * kValueFeatures
    );
    DeviceBuffer<T> suffix_qkv(qkv_elements);
    DeviceBuffer<T> z_native(z_elements);
    DeviceBuffer<T> a_native(ab_elements);
    DeviceBuffer<T> b_native(ab_elements);
    DeviceBuffer<float> z_f32(z_elements);
    DeviceBuffer<float> a_f32(ab_elements);
    DeviceBuffer<float> b_f32(ab_elements);

    DeviceBuffer<uint16_t> conv_weights(conv_weight_elements);
    DeviceBuffer<uint16_t> a_log(kGateRows);
    DeviceBuffer<uint16_t> dt_bias(kGateRows);
    DeviceBuffer<uint16_t> gated_norm_weights(kValueDim);
    DeviceBuffer<uint16_t> postnorm_weights(kLayerHidden);
    DeviceBuffer<float> suffix_postconv(qkv_elements);
    DeviceBuffer<float> suffix_core(core_elements);
    DeviceBuffer<float> suffix_state_key(state_elements);
    DeviceBuffer<uint16_t> candidate_gated(core_elements);
    DeviceBuffer<T> candidate_out(hidden_elements);
    DeviceBuffer<float> candidate_residual(hidden_elements);
    DeviceBuffer<float> candidate_postnorm(hidden_elements);

    DeviceBuffer<T> full_qkv(full_qkv_elements);
    DeviceBuffer<T> prefix_ring(ring_elements);
    DeviceBuffer<T> private_final_ring(ring_elements);
    DeviceBuffer<float> full_postconv(full_qkv_elements);
    DeviceBuffer<float> full_gate(gate_elements);
    DeviceBuffer<float> full_core(full_core_elements);
    DeviceBuffer<float> prefix_state_row(state_elements);
    DeviceBuffer<float> prefix_state_key(state_elements);
    DeviceBuffer<float> full_state_row(state_elements);
    DeviceBuffer<uint16_t> reference_gated(core_elements);
    DeviceBuffer<T> reference_out(hidden_elements);
    DeviceBuffer<float> reference_residual(hidden_elements);
    DeviceBuffer<float> reference_postnorm(hidden_elements);

    const dim3 block(kThreads);
    auto linear_grid = [&](size_t count) {
        return dim3(
            static_cast<unsigned int>((count + kThreads - 1u) / kThreads)
        );
    };
    hipLaunchKernelGGL(
        fill_hidden_kernel,
        linear_grid(hidden_elements),
        block,
        0u,
        nullptr,
        hidden.get(),
        hidden_elements
    );
    hipLaunchKernelGGL(
        fill_norm_weights_kernel,
        linear_grid(kLayerHidden),
        block,
        0u,
        nullptr,
        input_norm_weights.get(),
        static_cast<size_t>(kLayerHidden),
        0
    );
    hipLaunchKernelGGL(
        fill_norm_weights_kernel,
        linear_grid(kValueDim),
        block,
        0u,
        nullptr,
        gated_norm_weights.get(),
        static_cast<size_t>(kValueDim),
        1
    );
    hipLaunchKernelGGL(
        fill_norm_weights_kernel,
        linear_grid(kLayerHidden),
        block,
        0u,
        nullptr,
        postnorm_weights.get(),
        static_cast<size_t>(kLayerHidden),
        0
    );
    hipLaunchKernelGGL(
        fill_gate_parameters_kernel,
        dim3(1u),
        block,
        0u,
        nullptr,
        a_log.get(),
        dt_bias.get()
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(static_cast<size_t>(kQkvRows) * kLayerHidden),
        block,
        0u,
        nullptr,
        qkv_weights.get(),
        static_cast<size_t>(kQkvRows) * kLayerHidden,
        UINT32_C(0x51564b31),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(static_cast<size_t>(kZRows) * kLayerHidden),
        block,
        0u,
        nullptr,
        z_weights.get(),
        static_cast<size_t>(kZRows) * kLayerHidden,
        UINT32_C(0x5a50524f),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(static_cast<size_t>(kAbRows) * kLayerHidden),
        block,
        0u,
        nullptr,
        a_weights.get(),
        static_cast<size_t>(kAbRows) * kLayerHidden,
        UINT32_C(0x4150524f),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(static_cast<size_t>(kAbRows) * kLayerHidden),
        block,
        0u,
        nullptr,
        b_weights.get(),
        static_cast<size_t>(kAbRows) * kLayerHidden,
        UINT32_C(0x4250524f),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(static_cast<size_t>(kLayerHidden) * kValueFeatures),
        block,
        0u,
        nullptr,
        out_weights.get(),
        static_cast<size_t>(kLayerHidden) * kValueFeatures,
        UINT32_C(0x4f555450),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_weights_kernel,
        linear_grid(conv_weight_elements),
        block,
        0u,
        nullptr,
        conv_weights.get(),
        conv_weight_elements
    );
    hipLaunchKernelGGL(
        fill_qkv_kernel<T>,
        linear_grid(full_qkv_elements),
        block,
        0u,
        nullptr,
        full_qkv.get(),
        full_qkv_elements
    );
    hipLaunchKernelGGL(
        fill_gate_kernel,
        linear_grid(gate_elements),
        block,
        0u,
        nullptr,
        full_gate.get(),
        GateValuesAreDecay ? 1 : 0
    );
    check_hip(hipGetLastError(), "D229 fixture launch");
    check_hip(hipDeviceSynchronize(), "D229 fixture synchronize");

    RocblasContext blas;
    auto launch_projection_bundle = [&]() {
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden.get(),
            input_norm_weights.get(),
            input_norm_f32.get()
        );
        hipLaunchKernelGGL(
            f32_to_bf16_sublayer_kernel,
            linear_grid(hidden_elements),
            block,
            0u,
            nullptr,
            input_norm_f32.get(),
            input_norm_bf16.get(),
            hidden_elements
        );
        blas.matmul(
            qkv_weights.get(),
            input_norm_bf16.get(),
            suffix_qkv.get(),
            kQkvRows,
            kLayerHidden
        );
        blas.matmul(
            z_weights.get(),
            input_norm_bf16.get(),
            z_native.get(),
            kZRows,
            kLayerHidden
        );
        blas.matmul(
            a_weights.get(),
            input_norm_bf16.get(),
            a_native.get(),
            kAbRows,
            kLayerHidden
        );
        blas.matmul(
            b_weights.get(),
            input_norm_bf16.get(),
            b_native.get(),
            kAbRows,
            kLayerHidden
        );
        hipLaunchKernelGGL(
            native_to_f32_sublayer_kernel<T>,
            linear_grid(z_elements),
            block,
            0u,
            nullptr,
            z_native.get(),
            z_f32.get(),
            z_elements
        );
        hipLaunchKernelGGL(
            native_to_f32_sublayer_kernel<T>,
            linear_grid(ab_elements),
            block,
            0u,
            nullptr,
            a_native.get(),
            a_f32.get(),
            ab_elements
        );
        hipLaunchKernelGGL(
            native_to_f32_sublayer_kernel<T>,
            linear_grid(ab_elements),
            block,
            0u,
            nullptr,
            b_native.get(),
            b_f32.get(),
            ab_elements
        );
        hipLaunchKernelGGL(
            gate_from_ab_kernel<GateValuesAreDecay>,
            dim3(1u, kSuffixTokens),
            block,
            0u,
            nullptr,
            a_f32.get(),
            b_f32.get(),
            a_log.get(),
            dt_bias.get(),
            full_gate.get() +
                static_cast<size_t>(kPrefixTokens) * kGateOutputRows
        );
    };

    launch_projection_bundle();
    check_hip(hipGetLastError(), "D229 projection fixture launch");
    check_hip(
        hipDeviceSynchronize(),
        "D229 projection fixture synchronize"
    );
    check_hip(
        hipMemcpy(
            full_qkv.get() +
                static_cast<size_t>(kPrefixTokens) * kQkvRows,
            suffix_qkv.get(),
            qkv_elements * sizeof(T),
            hipMemcpyDeviceToDevice
        ),
        "D229 suffix QKV reference publication"
    );
    hipLaunchKernelGGL(
        capture_four_token_ring_kernel<T>,
        linear_grid(ring_elements),
        block,
        0u,
        nullptr,
        full_qkv.get(),
        prefix_ring.get(),
        kPrefixTokens
    );
    hipLaunchKernelGGL(
        full_conv_kernel<T>,
        dim3((kQkvRows + kThreads - 1u) / kThreads, kFullTokens),
        block,
        0u,
        nullptr,
        full_qkv.get(),
        conv_weights.get(),
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
        dim3(
            (kValueFeatures + kThreads - 1u) / kThreads,
            kFullTokens
        ),
        block,
        0u,
        nullptr,
        full_postconv.get(),
        kFullTokens
    );
    check_hip(hipGetLastError(), "D229 uninterrupted fixture launch");
    check_hip(
        hipDeviceSynchronize(),
        "D229 uninterrupted fixture synchronize"
    );

    if (api.q16384_launch(
            full_postconv.get(),
            full_gate.get(),
            full_core.get(),
            prefix_state_row.get(),
            GateValuesAreDecay ? 1 : 0,
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D229 q16384 provider failed: ") + api.last_error()
        );
    }
    hipLaunchKernelGGL(
        row_to_key_major_kernel,
        linear_grid(state_elements),
        block,
        0u,
        nullptr,
        prefix_state_row.get(),
        prefix_state_key.get()
    );
    check_hip(hipGetLastError(), "D229 prefix state transpose launch");
    check_hip(
        hipDeviceSynchronize(),
        "D229 prefix state transpose synchronize"
    );
    if (api.q17408_launch(
            full_postconv.get(),
            full_gate.get(),
            full_core.get(),
            full_state_row.get(),
            GateValuesAreDecay ? 1 : 0,
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D229 q17408 provider failed: ") + api.last_error()
        );
    }

    const float *reference_core_tail =
        full_core.get() +
        static_cast<size_t>(kPrefixTokens) * kValueFeatures;
    hipLaunchKernelGGL(
        gated_rmsnorm_sublayer_kernel,
        dim3(kValueHeads, kSuffixTokens),
        dim3(kValueDim),
        0u,
        nullptr,
        reference_core_tail,
        z_f32.get(),
        gated_norm_weights.get(),
        reference_gated.get()
    );
    blas.matmul(
        out_weights.get(),
        reference_gated.get(),
        reference_out.get(),
        kLayerHidden,
        kValueFeatures
    );
    hipLaunchKernelGGL(
        residual_postnorm_sublayer_kernel<T>,
        dim3(kSuffixTokens),
        block,
        0u,
        nullptr,
        hidden.get(),
        reference_out.get(),
        postnorm_weights.get(),
        reference_residual.get(),
        reference_postnorm.get()
    );
    check_hip(hipGetLastError(), "D229 reference tail launch");
    check_hip(
        hipDeviceSynchronize(),
        "D229 reference tail synchronize"
    );

    const std::vector<T> host_prefix_before =
        copy_device(prefix_ring.get(), ring_elements);
    const std::vector<float> host_state_before =
        copy_device(prefix_state_key.get(), state_elements);

    auto launch_candidate = [&]() {
        launch_projection_bundle();
        hipLaunchKernelGGL(
            suffix_halo_conv_kernel<T>,
            dim3(
                (kQkvRows + kThreads - 1u) / kThreads,
                kSuffixTokens
            ),
            block,
            0u,
            nullptr,
            prefix_ring.get(),
            suffix_qkv.get(),
            conv_weights.get(),
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
            dim3(
                (kValueFeatures + kThreads - 1u) / kThreads,
                kSuffixTokens
            ),
            block,
            0u,
            nullptr,
            suffix_postconv.get(),
            kSuffixTokens
        );
        if (api.q1024_seeded_launch_async(
                suffix_postconv.get(),
                full_gate.get() +
                    static_cast<size_t>(kPrefixTokens) * kGateOutputRows,
                prefix_state_key.get(),
                suffix_core.get(),
                suffix_state_key.get(),
                GateValuesAreDecay ? 1 : 0,
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("D229 q1024 seeded provider failed: ") +
                api.last_error()
            );
        }
        hipLaunchKernelGGL(
            gated_rmsnorm_sublayer_kernel,
            dim3(kValueHeads, kSuffixTokens),
            dim3(kValueDim),
            0u,
            nullptr,
            suffix_core.get(),
            z_f32.get(),
            gated_norm_weights.get(),
            candidate_gated.get()
        );
        blas.matmul(
            out_weights.get(),
            candidate_gated.get(),
            candidate_out.get(),
            kLayerHidden,
            kValueFeatures
        );
        hipLaunchKernelGGL(
            residual_postnorm_sublayer_kernel<T>,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden.get(),
            candidate_out.get(),
            postnorm_weights.get(),
            candidate_residual.get(),
            candidate_postnorm.get()
        );
        hipLaunchKernelGGL(
            publish_final_ring_kernel<T>,
            linear_grid(ring_elements),
            block,
            0u,
            nullptr,
            suffix_qkv.get(),
            private_final_ring.get()
        );
        if (surface_export != nullptr &&
            surface_export->tail_launch != nullptr) {
            surface_export->tail_launch(
                surface_export->tail_context,
                candidate_residual.get(),
                candidate_postnorm.get(),
                nullptr
            );
        }
    };

    for (unsigned int warmup = 0u; warmup < 2u; ++warmup) {
        launch_candidate();
    }
    check_hip(hipGetLastError(), "D229 candidate warmup launch");
    check_hip(
        hipDeviceSynchronize(),
        "D229 candidate warmup synchronize"
    );

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "D229 event create(start)");
    check_hip(hipEventCreate(&stop), "D229 event create(stop)");
    check_hip(hipEventRecord(start, nullptr), "D229 event record(start)");
    for (unsigned int repetition = 0u;
         repetition < repetitions;
         ++repetition) {
        launch_candidate();
    }
    check_hip(hipGetLastError(), "D229 candidate timed launch");
    check_hip(hipEventRecord(stop, nullptr), "D229 event record(stop)");
    check_hip(hipEventSynchronize(stop), "D229 event synchronize(stop)");
    float total_ms = 0.0f;
    check_hip(
        hipEventElapsedTime(&total_ms, start, stop),
        "D229 event elapsed"
    );
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    const double mean_ms =
        static_cast<double>(total_ms) /
        static_cast<double>(repetitions);

    if (surface_export != nullptr) {
        if (surface_export->candidate_residual == nullptr ||
            surface_export->candidate_postnorm == nullptr ||
            surface_export->reference_residual == nullptr ||
            surface_export->reference_postnorm == nullptr ||
            surface_export->elements < hidden_elements) {
            throw std::runtime_error(
                "D229 surface export received an invalid destination"
            );
        }
        const size_t bytes = hidden_elements * sizeof(float);
        check_hip(
            hipMemcpy(
                surface_export->candidate_residual,
                candidate_residual.get(),
                bytes,
                hipMemcpyDeviceToDevice
            ),
            "D229 candidate residual export"
        );
        check_hip(
            hipMemcpy(
                surface_export->candidate_postnorm,
                candidate_postnorm.get(),
                bytes,
                hipMemcpyDeviceToDevice
            ),
            "D229 candidate postnorm export"
        );
        check_hip(
            hipMemcpy(
                surface_export->reference_residual,
                reference_residual.get(),
                bytes,
                hipMemcpyDeviceToDevice
            ),
            "D229 reference residual export"
        );
        check_hip(
            hipMemcpy(
                surface_export->reference_postnorm,
                reference_postnorm.get(),
                bytes,
                hipMemcpyDeviceToDevice
            ),
            "D229 reference postnorm export"
        );
    }

    const ProjectionSampleStats qkv_projection =
        validate_projection_samples(
            qkv_weights.get(),
            input_norm_bf16.get(),
            suffix_qkv.get(),
            kQkvRows,
            kLayerHidden
        );
    const ProjectionSampleStats z_projection =
        validate_projection_samples(
            z_weights.get(),
            input_norm_bf16.get(),
            z_native.get(),
            kZRows,
            kLayerHidden
        );
    const ProjectionSampleStats a_projection =
        validate_projection_samples(
            a_weights.get(),
            input_norm_bf16.get(),
            a_native.get(),
            kAbRows,
            kLayerHidden
        );
    const ProjectionSampleStats b_projection =
        validate_projection_samples(
            b_weights.get(),
            input_norm_bf16.get(),
            b_native.get(),
            kAbRows,
            kLayerHidden
        );
    const ProjectionSampleStats out_projection =
        validate_projection_samples(
            out_weights.get(),
            candidate_gated.get(),
            candidate_out.get(),
            kLayerHidden,
            kValueFeatures
        );
    const ProjectionSampleStats projections[] = {
        qkv_projection,
        z_projection,
        a_projection,
        b_projection,
        out_projection,
    };
    bool projection_pass = true;
    for (const ProjectionSampleStats &stats : projections) {
        projection_pass =
            projection_pass &&
            stats.samples >= 16u &&
            stats.nonfinite == 0u &&
            stats.max_abs <= kProjectionSampleMaxAbs;
    }

    const std::vector<float> postconv_reference = copy_device(
        full_postconv.get() +
            static_cast<size_t>(kPrefixTokens) * kQkvRows,
        qkv_elements
    );
    const std::vector<float> postconv_candidate =
        copy_device(suffix_postconv.get(), qkv_elements);
    const std::vector<float> core_reference =
        copy_device(reference_core_tail, core_elements);
    const std::vector<float> core_candidate =
        copy_device(suffix_core.get(), core_elements);
    const std::vector<float> state_reference =
        copy_device(full_state_row.get(), state_elements);
    const std::vector<float> state_candidate =
        copy_device(suffix_state_key.get(), state_elements);
    const std::vector<float> residual_reference =
        copy_device(reference_residual.get(), hidden_elements);
    const std::vector<float> residual_candidate =
        copy_device(candidate_residual.get(), hidden_elements);
    const std::vector<float> postnorm_reference =
        copy_device(reference_postnorm.get(), hidden_elements);
    const std::vector<float> postnorm_candidate =
        copy_device(candidate_postnorm.get(), hidden_elements);
    const std::vector<T> final_ring_reference = copy_device(
        full_qkv.get() +
            static_cast<size_t>(kFullTokens - kConvTaps) * kQkvRows,
        ring_elements
    );
    const std::vector<T> final_ring_candidate =
        copy_device(private_final_ring.get(), ring_elements);
    const std::vector<T> host_prefix_after =
        copy_device(prefix_ring.get(), ring_elements);
    const std::vector<float> host_state_after =
        copy_device(prefix_state_key.get(), state_elements);

    const CompareStats postconv_stats =
        compare_f32(postconv_reference, postconv_candidate);
    const CompareStats core_stats =
        compare_f32(core_reference, core_candidate);
    const CompareStats state_stats =
        compare_state_row_to_key(state_reference, state_candidate);
    const CompareStats residual_stats =
        compare_f32(residual_reference, residual_candidate);
    const CompareStats postnorm_stats =
        compare_f32(postnorm_reference, postnorm_candidate);
    const CompareStats ring_stats =
        compare_native(final_ring_reference, final_ring_candidate);
    const CompareStats prefix_ring_mutation =
        compare_native(host_prefix_before, host_prefix_after);
    const CompareStats prefix_state_mutation =
        compare_f32(host_state_before, host_state_after);

    const auto exact = [](const CompareStats &stats) {
        return stats.mismatches == 0u &&
            stats.nonfinite == 0u &&
            stats.max_abs == 0.0;
    };
    const bool tail_active =
        surface_export != nullptr &&
        surface_export->tail_launch != nullptr;
    const double mean_ms_ceiling =
        surface_export != nullptr &&
        surface_export->mean_ms_ceiling > 0.0
            ? surface_export->mean_ms_ceiling
            : kSublayerMeanCeilingMs;
    const bool pass =
        projection_pass &&
        exact(postconv_stats) &&
        exact(core_stats) &&
        exact(state_stats) &&
        exact(residual_stats) &&
        exact(postnorm_stats) &&
        exact(ring_stats) &&
        prefix_ring_mutation.mismatches == 0u &&
        prefix_state_mutation.mismatches == 0u &&
        mean_ms <= mean_ms_ceiling;

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_full_linear_attention_sublayer_smoke"
        << " mode=" << mode
        << " gate_values_are_decay="
        << (GateValuesAreDecay ? 1 : 0)
        << " repetitions=" << repetitions
        << " mean_ms=" << mean_ms
        << " mean_ms_ceiling=" << mean_ms_ceiling
        << " qkv_projection_samples=" << qkv_projection.samples
        << " qkv_projection_nonfinite=" << qkv_projection.nonfinite
        << " qkv_projection_max_abs=" << qkv_projection.max_abs
        << " z_projection_samples=" << z_projection.samples
        << " z_projection_nonfinite=" << z_projection.nonfinite
        << " z_projection_max_abs=" << z_projection.max_abs
        << " a_projection_samples=" << a_projection.samples
        << " a_projection_nonfinite=" << a_projection.nonfinite
        << " a_projection_max_abs=" << a_projection.max_abs
        << " b_projection_samples=" << b_projection.samples
        << " b_projection_nonfinite=" << b_projection.nonfinite
        << " b_projection_max_abs=" << b_projection.max_abs
        << " out_projection_samples=" << out_projection.samples
        << " out_projection_nonfinite=" << out_projection.nonfinite
        << " out_projection_max_abs=" << out_projection.max_abs
        << " postconv_elements=" << postconv_stats.elements
        << " postconv_mismatches=" << postconv_stats.mismatches
        << " postconv_nonfinite=" << postconv_stats.nonfinite
        << " postconv_max_abs=" << postconv_stats.max_abs
        << " recurrent_output_elements=" << core_stats.elements
        << " recurrent_output_mismatches=" << core_stats.mismatches
        << " recurrent_output_nonfinite=" << core_stats.nonfinite
        << " recurrent_output_max_abs=" << core_stats.max_abs
        << " final_state_elements=" << state_stats.elements
        << " final_state_mismatches=" << state_stats.mismatches
        << " final_state_nonfinite=" << state_stats.nonfinite
        << " final_state_max_abs=" << state_stats.max_abs
        << " residual_elements=" << residual_stats.elements
        << " residual_mismatches=" << residual_stats.mismatches
        << " residual_nonfinite=" << residual_stats.nonfinite
        << " residual_max_abs=" << residual_stats.max_abs
        << " postnorm_elements=" << postnorm_stats.elements
        << " postnorm_mismatches=" << postnorm_stats.mismatches
        << " postnorm_nonfinite=" << postnorm_stats.nonfinite
        << " postnorm_max_abs=" << postnorm_stats.max_abs
        << " final_ring_elements=" << ring_stats.elements
        << " final_ring_mismatches=" << ring_stats.mismatches
        << " final_ring_nonfinite=" << ring_stats.nonfinite
        << " final_ring_max_abs=" << ring_stats.max_abs
        << " prefix_ring_mutation_mismatches="
        << prefix_ring_mutation.mismatches
        << " prefix_state_mutation_mismatches="
        << prefix_state_mutation.mismatches
        << " residual_reference_hash=" << std::hex
        << fnv1a64(
               residual_reference.data(),
               residual_reference.size() * sizeof(float)
           )
        << " residual_candidate_hash="
        << fnv1a64(
               residual_candidate.data(),
               residual_candidate.size() * sizeof(float)
           )
        << " postnorm_reference_hash="
        << fnv1a64(
               postnorm_reference.data(),
               postnorm_reference.size() * sizeof(float)
           )
        << " postnorm_candidate_hash="
        << fnv1a64(
               postnorm_candidate.data(),
               postnorm_candidate.size() * sizeof(float)
           )
        << std::dec
        << " quantized=0"
        << " mtp_active=0"
        << " dflash_active=0"
        << " speculative_decode=0"
        << " moe_in_scope="
        << (
            tail_active && surface_export->tail_is_moe
                ? 1
                : 0
        )
        << " complete_transformer_layer_claimed="
        << (
            tail_active && surface_export->tail_is_moe
                ? 1
                : 0
        )
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    return SublayerResult{mode, mean_ms, pass};
}

}  // namespace

#ifndef QRT_D229_DISABLE_MAIN
int main(int argc, char **argv) {
    ProviderApi api;
    try {
        if (argc != 4) {
            std::cerr
                << "usage: "
                << "q16384_suffix1024_full_linear_attention_sublayer_smoke "
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

        const SublayerResult early =
            run_sublayer_mode<float, true>(
                "early_f32",
                api,
                repetitions
            );
        const SublayerResult retained =
            run_sublayer_mode<uint16_t, false>(
                "retained_bf16",
                api,
                repetitions
            );
        const double native_projection_ms =
            2.0 * early.mean_ms + 28.0 * retained.mean_ms;
        const bool pass =
            early.pass &&
            retained.pass &&
            native_projection_ms <=
                kSublayerNativeProjectionCeilingMs;
        std::cout
            << std::fixed << std::setprecision(6)
            << "q16384_suffix1024_full_linear_attention_sublayer_summary"
            << " early_f32_layers=2"
            << " retained_bf16_layers=28"
            << " native_thirty_layer_projected_ms="
            << native_projection_ms
            << " native_projection_ceiling_ms="
            << kSublayerNativeProjectionCeilingMs
            << " cases_pass="
            << ((early.pass ? 1 : 0) + (retained.pass ? 1 : 0))
            << " moe_in_scope=0"
            << " complete_transformer_layer_claimed=0"
            << " pass=" << (pass ? 1 : 0)
            << "\n";

        api.release();
        FreeLibrary(api.module);
        return pass ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr
            << "q16384_suffix1024_full_linear_attention_sublayer_smoke "
            << "error: " << error.what() << "\n";
        if (api.release != nullptr) {
            api.release();
        }
        if (api.module != nullptr) {
            FreeLibrary(api.module);
        }
        return 1;
    }
}
#endif
