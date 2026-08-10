#define QRT_D229_DISABLE_MAIN
#include "q16384_suffix1024_full_linear_attention_sublayer_smoke.cpp"
#undef QRT_D229_DISABLE_MAIN

#include <array>
#include <limits>

namespace {

constexpr unsigned int kD230Experts = 256u;
constexpr unsigned int kD230TopK = 8u;
constexpr unsigned int kD230ActiveExperts = 8u;
constexpr unsigned int kD230Intermediate = 512u;
constexpr unsigned int kD230GateUpRows = 2u * kD230Intermediate;
constexpr size_t kD230HiddenElements =
    static_cast<size_t>(kSuffixTokens) * kLayerHidden;
constexpr size_t kD230Routes =
    static_cast<size_t>(kSuffixTokens) * kD230TopK;
constexpr double kD230ReferenceCeiling = 0.125;
constexpr double kD230MoeMeanCeilingMs = 30.0;
constexpr double kD230CompleteMeanCeilingMs = 90.0;
constexpr double kD230NativeProjectionCeilingMs = 2700.0;

using MoePrepareFunction = int (*)(const char *);
using MoeLaunchFunction = int (*)(
    const float *,
    const float *,
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *
);
using MoeRouterDebugFunction = int (*)(
    const float *,
    const uint16_t *,
    void *
);
using MoeCopyTopkFunction = int (*)(uint32_t *, float *);
using MoeCopyToken0StageFunction = int (*)(float *, uint16_t *, float *);
using MoeCopyEarlyF32ActivationFunction = int (*)(float *);
using MoeLastErrorFunction = const char *(*)();
using MoeScratchBytesFunction = uint64_t (*)();
using MoeReleaseFunction = void (*)();

struct MoeApi {
    HMODULE module = nullptr;
    MoePrepareFunction prepare = nullptr;
    MoeLaunchFunction launch_v2 = nullptr;
    MoeLaunchFunction launch_q1024_early_f32_v1 = nullptr;
    MoeLaunchFunction launch_v3_async = nullptr;
    MoeRouterDebugFunction launch_router_debug = nullptr;
    MoeCopyTopkFunction copy_topk_debug = nullptr;
    MoeCopyToken0StageFunction copy_token0_stage_debug = nullptr;
    MoeCopyEarlyF32ActivationFunction
        copy_q1024_early_f32_token0_activation_debug = nullptr;
    MoeLastErrorFunction last_error = nullptr;
    MoeScratchBytesFunction scratch_bytes = nullptr;
    MoeReleaseFunction release = nullptr;
};

bool load_moe_provider(const char *path, MoeApi *api) {
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) {
        return false;
    }
    api->prepare = reinterpret_cast<MoePrepareFunction>(
        GetProcAddress(api->module, "qrt_triton_moe_q8192_prepare")
    );
    api->launch_v2 = reinterpret_cast<MoeLaunchFunction>(
        GetProcAddress(api->module, "qrt_triton_moe_q8192_launch_full_v2")
    );
    api->launch_q1024_early_f32_v1 =
        reinterpret_cast<MoeLaunchFunction>(
            GetProcAddress(
                api->module,
                "qrt_triton_moe_q8192_launch_full_q1024_early_f32_v1"
            )
        );
    api->launch_v3_async = reinterpret_cast<MoeLaunchFunction>(
        GetProcAddress(
            api->module,
            "qrt_triton_moe_q8192_launch_full_v3_async"
        )
    );
    api->launch_router_debug = reinterpret_cast<MoeRouterDebugFunction>(
        GetProcAddress(
            api->module,
            "qrt_triton_moe_q8192_launch_router_debug"
        )
    );
    api->copy_topk_debug = reinterpret_cast<MoeCopyTopkFunction>(
        GetProcAddress(
            api->module,
            "qrt_triton_moe_q8192_copy_topk_debug"
        )
    );
    api->copy_token0_stage_debug =
        reinterpret_cast<MoeCopyToken0StageFunction>(
            GetProcAddress(
                api->module,
                "qrt_triton_moe_q8192_copy_token0_stage_debug"
            )
        );
    api->copy_q1024_early_f32_token0_activation_debug =
        reinterpret_cast<MoeCopyEarlyF32ActivationFunction>(
            GetProcAddress(
                api->module,
                "qrt_triton_moe_q8192_copy_q1024_early_f32_token0_activation_debug"
            )
        );
    api->last_error = reinterpret_cast<MoeLastErrorFunction>(
        GetProcAddress(api->module, "qrt_triton_moe_q8192_last_error")
    );
    api->scratch_bytes = reinterpret_cast<MoeScratchBytesFunction>(
        GetProcAddress(api->module, "qrt_triton_moe_q8192_scratch_bytes")
    );
    api->release = reinterpret_cast<MoeReleaseFunction>(
        GetProcAddress(api->module, "qrt_triton_moe_q8192_release")
    );
    return api->prepare != nullptr &&
        api->launch_v2 != nullptr &&
        api->launch_v3_async != nullptr &&
        api->launch_router_debug != nullptr &&
        api->copy_topk_debug != nullptr &&
        api->last_error != nullptr &&
        api->scratch_bytes != nullptr &&
        api->release != nullptr;
}

uint16_t d230_float_to_bf16(float value) {
    union {
        float f;
        uint32_t u;
    } bits{value};
    bits.u += UINT32_C(0x7fff) + ((bits.u >> 16u) & 1u);
    return static_cast<uint16_t>(bits.u >> 16u);
}

float d230_bf16_round(float value) {
    return host_bf16_to_float(d230_float_to_bf16(value));
}

__global__ void d230_fill_routed_gate_up_kernel(
    uint16_t *weights,
    size_t elements
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    constexpr size_t kPerExpert =
        static_cast<size_t>(kD230GateUpRows) * kLayerHidden;
    const unsigned int expert =
        static_cast<unsigned int>(index / kPerExpert);
    const size_t within = index - static_cast<size_t>(expert) * kPerExpert;
    const unsigned int row =
        static_cast<unsigned int>(within / kLayerHidden);
    const float numerator =
        row < kD230Intermediate
            ? 0.75f * static_cast<float>(expert + 1u)
            : 0.375f * static_cast<float>(expert + 1u);
    weights[index] = device_float_to_bf16(
        numerator / static_cast<float>(kLayerHidden)
    );
}

__global__ void d230_fill_routed_down_kernel(
    uint16_t *weights,
    size_t elements
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    constexpr size_t kPerExpert =
        static_cast<size_t>(kLayerHidden) * kD230Intermediate;
    const unsigned int expert =
        static_cast<unsigned int>(index / kPerExpert);
    const float numerator =
        0.5f * static_cast<float>(expert % 3u + 1u);
    weights[index] = device_float_to_bf16(
        numerator / static_cast<float>(kD230Intermediate)
    );
}

__global__ void d230_fill_constant_bf16_kernel(
    uint16_t *values,
    size_t elements,
    float value
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        values[index] = device_float_to_bf16(value);
    }
}

struct MoeWeights {
    DeviceBuffer<uint16_t> router;
    DeviceBuffer<uint16_t> routed_gate_up;
    DeviceBuffer<uint16_t> routed_down;
    DeviceBuffer<uint16_t> shared_gate;
    DeviceBuffer<uint16_t> shared_gate_projection;
    DeviceBuffer<uint16_t> shared_up_projection;
    DeviceBuffer<uint16_t> shared_down;

    MoeWeights()
        : router(static_cast<size_t>(kD230Experts) * kLayerHidden),
          routed_gate_up(
              static_cast<size_t>(kD230ActiveExperts) *
              kD230GateUpRows *
              kLayerHidden
          ),
          routed_down(
              static_cast<size_t>(kD230ActiveExperts) *
              kLayerHidden *
              kD230Intermediate
          ),
          shared_gate(kLayerHidden),
          shared_gate_projection(
              static_cast<size_t>(kD230Intermediate) * kLayerHidden
          ),
          shared_up_projection(
              static_cast<size_t>(kD230Intermediate) * kLayerHidden
          ),
          shared_down(
              static_cast<size_t>(kLayerHidden) * kD230Intermediate
          ) {}
};

void initialize_moe_weights(MoeWeights *weights) {
    const size_t routed_gate_up_elements =
        static_cast<size_t>(kD230ActiveExperts) *
        kD230GateUpRows *
        kLayerHidden;
    const size_t routed_down_elements =
        static_cast<size_t>(kD230ActiveExperts) *
        kLayerHidden *
        kD230Intermediate;
    const size_t shared_projection_elements =
        static_cast<size_t>(kD230Intermediate) * kLayerHidden;
    const size_t shared_down_elements =
        static_cast<size_t>(kLayerHidden) * kD230Intermediate;
    const auto grid_for = [](size_t elements) {
        return dim3(
            static_cast<unsigned int>((elements + kThreads - 1u) / kThreads)
        );
    };
    check_hip(
        hipMemset(
            weights->router.get(),
            0,
            static_cast<size_t>(kD230Experts) *
                kLayerHidden *
                sizeof(uint16_t)
        ),
        "D230 zero router weights"
    );
    check_hip(
        hipMemset(
            weights->shared_gate.get(),
            0,
            static_cast<size_t>(kLayerHidden) * sizeof(uint16_t)
        ),
        "D230 zero shared gate"
    );
    hipLaunchKernelGGL(
        d230_fill_routed_gate_up_kernel,
        grid_for(routed_gate_up_elements),
        dim3(kThreads),
        0u,
        nullptr,
        weights->routed_gate_up.get(),
        routed_gate_up_elements
    );
    hipLaunchKernelGGL(
        d230_fill_routed_down_kernel,
        grid_for(routed_down_elements),
        dim3(kThreads),
        0u,
        nullptr,
        weights->routed_down.get(),
        routed_down_elements
    );
    hipLaunchKernelGGL(
        d230_fill_constant_bf16_kernel,
        grid_for(shared_projection_elements),
        dim3(kThreads),
        0u,
        nullptr,
        weights->shared_gate_projection.get(),
        shared_projection_elements,
        0.5f / static_cast<float>(kLayerHidden)
    );
    hipLaunchKernelGGL(
        d230_fill_constant_bf16_kernel,
        grid_for(shared_projection_elements),
        dim3(kThreads),
        0u,
        nullptr,
        weights->shared_up_projection.get(),
        shared_projection_elements,
        0.25f / static_cast<float>(kLayerHidden)
    );
    hipLaunchKernelGGL(
        d230_fill_constant_bf16_kernel,
        grid_for(shared_down_elements),
        dim3(kThreads),
        0u,
        nullptr,
        weights->shared_down.get(),
        shared_down_elements,
        0.5f / static_cast<float>(kD230Intermediate)
    );
    check_hip(hipGetLastError(), "D230 initialize weight kernels");
    check_hip(hipDeviceSynchronize(), "D230 initialize weight synchronize");
}

struct MoeTailContext {
    const MoeApi *api = nullptr;
    const MoeWeights *weights = nullptr;
    float *output = nullptr;
};

void launch_moe_tail(
    void *opaque,
    const float *residual,
    const float *postnorm,
    hipStream_t stream
) {
    auto *context = static_cast<MoeTailContext *>(opaque);
    if (context == nullptr ||
        context->api == nullptr ||
        context->weights == nullptr ||
        context->output == nullptr) {
        throw std::runtime_error("D230 tail context is invalid");
    }
    const MoeWeights &weights = *context->weights;
    if (context->api->launch_v3_async(
            postnorm,
            residual,
            weights.router.get(),
            weights.routed_gate_up.get(),
            weights.routed_down.get(),
            weights.shared_gate.get(),
            weights.shared_gate_projection.get(),
            weights.shared_up_projection.get(),
            weights.shared_down.get(),
            context->output,
            stream
        ) == 0) {
        throw std::runtime_error(
            std::string("D230 MoE tail launch failed: ") +
            context->api->last_error()
        );
    }
}

float independent_output_reference(
    const float *postnorm,
    float residual
) {
    std::array<float, kLayerHidden> input{};
    for (unsigned int column = 0u; column < kLayerHidden; ++column) {
        input[column] = d230_bf16_round(postnorm[column]);
    }

    float routed = 0.0f;
    for (unsigned int expert = 0u;
         expert < kD230ActiveExperts;
         ++expert) {
        const float gate_weight = host_bf16_to_float(
            d230_float_to_bf16(
                0.75f * static_cast<float>(expert + 1u) /
                static_cast<float>(kLayerHidden)
            )
        );
        const float up_weight = host_bf16_to_float(
            d230_float_to_bf16(
                0.375f * static_cast<float>(expert + 1u) /
                static_cast<float>(kLayerHidden)
            )
        );
        float gate_accumulator = 0.0f;
        float up_accumulator = 0.0f;
        for (unsigned int column = 0u;
             column < kLayerHidden;
             ++column) {
            gate_accumulator += input[column] * gate_weight;
            up_accumulator += input[column] * up_weight;
        }
        const float gate = d230_bf16_round(gate_accumulator);
        const float up = d230_bf16_round(up_accumulator);
        const float activated = d230_bf16_round(
            (gate / (1.0f + std::exp(-gate))) * up
        );
        const float down_weight = host_bf16_to_float(
            d230_float_to_bf16(
                0.5f * static_cast<float>(expert % 3u + 1u) /
                static_cast<float>(kD230Intermediate)
            )
        );
        float down = 0.0f;
        for (unsigned int intermediate = 0u;
             intermediate < kD230Intermediate;
             ++intermediate) {
            down += activated * down_weight;
        }
        routed += 0.125f * down;
    }

    const float shared_gate_weight = host_bf16_to_float(
        d230_float_to_bf16(
            0.5f / static_cast<float>(kLayerHidden)
        )
    );
    const float shared_up_weight = host_bf16_to_float(
        d230_float_to_bf16(
            0.25f / static_cast<float>(kLayerHidden)
        )
    );
    float shared_gate_projection = 0.0f;
    float shared_up_projection = 0.0f;
    for (unsigned int column = 0u; column < kLayerHidden; ++column) {
        shared_gate_projection += input[column] * shared_gate_weight;
        shared_up_projection += input[column] * shared_up_weight;
    }
    shared_gate_projection = d230_bf16_round(shared_gate_projection);
    shared_up_projection = d230_bf16_round(shared_up_projection);
    const float shared_activated = d230_bf16_round(
        (
            shared_gate_projection /
            (1.0f + std::exp(-shared_gate_projection))
        ) * shared_up_projection
    );
    const float shared_down_weight = host_bf16_to_float(
        d230_float_to_bf16(
            0.5f / static_cast<float>(kD230Intermediate)
        )
    );
    float shared_down = 0.0f;
    for (unsigned int intermediate = 0u;
         intermediate < kD230Intermediate;
         ++intermediate) {
        shared_down += shared_activated * shared_down_weight;
    }
    shared_down = d230_bf16_round(shared_down);
    const float shared = 0.5f * shared_down;
    return residual + (routed + shared);
}

struct D230ModeResult {
    std::string mode;
    double complete_mean_ms = 0.0;
    double moe_mean_ms = 0.0;
    bool pass = false;
};

template <typename T, bool GateValuesAreDecay>
D230ModeResult run_d230_mode(
    const char *mode,
    const ProviderApi &gdn_api,
    const MoeApi &moe_api,
    const MoeWeights &weights,
    unsigned int repetitions
) {
    DeviceBuffer<float> candidate_residual(kD230HiddenElements);
    DeviceBuffer<float> candidate_postnorm(kD230HiddenElements);
    DeviceBuffer<float> reference_residual(kD230HiddenElements);
    DeviceBuffer<float> reference_postnorm(kD230HiddenElements);
    DeviceBuffer<float> callback_output(kD230HiddenElements);
    DeviceBuffer<float> v2_output(kD230HiddenElements);
    DeviceBuffer<float> v3_output(kD230HiddenElements);

    MoeTailContext tail_context{
        &moe_api,
        &weights,
        callback_output.get(),
    };
    SublayerSurfaceExport surface_export{
        candidate_residual.get(),
        candidate_postnorm.get(),
        reference_residual.get(),
        reference_postnorm.get(),
        kD230HiddenElements,
        launch_moe_tail,
        &tail_context,
        kD230CompleteMeanCeilingMs,
        true,
    };
    const SublayerResult complete =
        run_sublayer_mode<T, GateValuesAreDecay>(
            mode,
            gdn_api,
            repetitions,
            &surface_export
        );

    const std::vector<float> residual_before =
        copy_device(candidate_residual.get(), kD230HiddenElements);
    const std::vector<float> postnorm_before =
        copy_device(candidate_postnorm.get(), kD230HiddenElements);
    const std::vector<float> reference_residual_host =
        copy_device(reference_residual.get(), kD230HiddenElements);
    const std::vector<float> reference_postnorm_host =
        copy_device(reference_postnorm.get(), kD230HiddenElements);
    const CompareStats d229_residual_export =
        compare_f32(reference_residual_host, residual_before);
    const CompareStats d229_postnorm_export =
        compare_f32(reference_postnorm_host, postnorm_before);

    if (moe_api.launch_v2(
            reference_postnorm.get(),
            reference_residual.get(),
            weights.router.get(),
            weights.routed_gate_up.get(),
            weights.routed_down.get(),
            weights.shared_gate.get(),
            weights.shared_gate_projection.get(),
            weights.shared_up_projection.get(),
            weights.shared_down.get(),
            v2_output.get(),
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D230 v2 launch failed: ") + moe_api.last_error()
        );
    }
    if (moe_api.launch_v3_async(
            candidate_postnorm.get(),
            candidate_residual.get(),
            weights.router.get(),
            weights.routed_gate_up.get(),
            weights.routed_down.get(),
            weights.shared_gate.get(),
            weights.shared_gate_projection.get(),
            weights.shared_up_projection.get(),
            weights.shared_down.get(),
            v3_output.get(),
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D230 v3 launch failed: ") + moe_api.last_error()
        );
    }
    check_hip(hipDeviceSynchronize(), "D230 v3 comparison synchronize");

    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "D230 event create(start)");
    check_hip(hipEventCreate(&stop), "D230 event create(stop)");
    check_hip(hipEventRecord(start, nullptr), "D230 event record(start)");
    for (unsigned int repetition = 0u;
         repetition < repetitions;
         ++repetition) {
        if (moe_api.launch_v3_async(
                candidate_postnorm.get(),
                candidate_residual.get(),
                weights.router.get(),
                weights.routed_gate_up.get(),
                weights.routed_down.get(),
                weights.shared_gate.get(),
                weights.shared_gate_projection.get(),
                weights.shared_up_projection.get(),
                weights.shared_down.get(),
                v3_output.get(),
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("D230 timed v3 launch failed: ") +
                moe_api.last_error()
            );
        }
    }
    check_hip(hipEventRecord(stop, nullptr), "D230 event record(stop)");
    check_hip(hipEventSynchronize(stop), "D230 event synchronize(stop)");
    float total_moe_ms = 0.0f;
    check_hip(
        hipEventElapsedTime(&total_moe_ms, start, stop),
        "D230 event elapsed"
    );
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    const double moe_mean_ms =
        static_cast<double>(total_moe_ms) /
        static_cast<double>(repetitions);

    if (moe_api.launch_router_debug(
            candidate_postnorm.get(),
            weights.router.get(),
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D230 router debug failed: ") +
            moe_api.last_error()
        );
    }
    check_hip(hipDeviceSynchronize(), "D230 router debug synchronize");
    std::vector<uint32_t> topk_ids(kD230Routes);
    std::vector<float> topk_weights(kD230Routes);
    if (moe_api.copy_topk_debug(
            topk_ids.data(),
            topk_weights.data()
        ) == 0) {
        throw std::runtime_error("D230 top-k debug copy failed");
    }
    size_t router_id_mismatches = 0u;
    size_t router_weight_mismatches = 0u;
    for (unsigned int token = 0u;
         token < kSuffixTokens;
         ++token) {
        for (unsigned int route = 0u;
             route < kD230TopK;
             ++route) {
            const size_t index =
                static_cast<size_t>(token) * kD230TopK + route;
            router_id_mismatches +=
                topk_ids[index] == route ? 0u : 1u;
            router_weight_mismatches +=
                topk_weights[index] == 0.125f ? 0u : 1u;
        }
    }

    const std::vector<float> callback_output_host =
        copy_device(callback_output.get(), kD230HiddenElements);
    const std::vector<float> v2_output_host =
        copy_device(v2_output.get(), kD230HiddenElements);
    const std::vector<float> v3_output_host =
        copy_device(v3_output.get(), kD230HiddenElements);
    const std::vector<float> residual_after =
        copy_device(candidate_residual.get(), kD230HiddenElements);
    const std::vector<float> postnorm_after =
        copy_device(candidate_postnorm.get(), kD230HiddenElements);
    const CompareStats v2_v3 =
        compare_f32(v2_output_host, v3_output_host);
    const CompareStats callback_v3 =
        compare_f32(callback_output_host, v3_output_host);
    const CompareStats residual_mutation =
        compare_f32(residual_before, residual_after);
    const CompareStats postnorm_mutation =
        compare_f32(postnorm_before, postnorm_after);

    const std::array<unsigned int, 4> sample_tokens = {
        0u,
        kSuffixTokens / 3u,
        (2u * kSuffixTokens) / 3u,
        kSuffixTokens - 1u,
    };
    const std::array<unsigned int, 4> sample_columns = {
        0u,
        kLayerHidden / 3u,
        (2u * kLayerHidden) / 3u,
        kLayerHidden - 1u,
    };
    size_t reference_samples = 0u;
    size_t reference_nonfinite = 0u;
    double reference_max_abs = 0.0;
    for (unsigned int token : sample_tokens) {
        const float *token_postnorm =
            postnorm_before.data() +
            static_cast<size_t>(token) * kLayerHidden;
        for (unsigned int column : sample_columns) {
            const size_t index =
                static_cast<size_t>(token) * kLayerHidden + column;
            const float reference = independent_output_reference(
                token_postnorm,
                residual_before[index]
            );
            const float candidate = v3_output_host[index];
            if (!std::isfinite(reference) || !std::isfinite(candidate)) {
                ++reference_nonfinite;
            } else {
                reference_max_abs = std::max(
                    reference_max_abs,
                    std::abs(
                        static_cast<double>(reference) -
                        static_cast<double>(candidate)
                    )
                );
            }
            ++reference_samples;
        }
    }
    size_t output_nonfinite = 0u;
    for (float value : v3_output_host) {
        output_nonfinite += std::isfinite(value) ? 0u : 1u;
    }

    const auto exact = [](const CompareStats &stats) {
        return stats.mismatches == 0u &&
            stats.nonfinite == 0u &&
            stats.max_abs == 0.0;
    };
    const bool pass =
        complete.pass &&
        complete.mean_ms <= kD230CompleteMeanCeilingMs &&
        moe_mean_ms <= kD230MoeMeanCeilingMs &&
        exact(d229_residual_export) &&
        exact(d229_postnorm_export) &&
        exact(v2_v3) &&
        exact(callback_v3) &&
        exact(residual_mutation) &&
        exact(postnorm_mutation) &&
        router_id_mismatches == 0u &&
        router_weight_mismatches == 0u &&
        reference_samples >= 16u &&
        reference_nonfinite == 0u &&
        reference_max_abs <= kD230ReferenceCeiling &&
        output_nonfinite == 0u;

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_complete_linear_transformer_layer_smoke"
        << " mode=" << mode
        << " gate_values_are_decay="
        << (GateValuesAreDecay ? 1 : 0)
        << " suffix_tokens=" << kSuffixTokens
        << " experts=" << kD230Experts
        << " top_k=" << kD230TopK
        << " materialized_active_experts=" << kD230ActiveExperts
        << " repetitions=" << repetitions
        << " complete_layer_mean_ms=" << complete.mean_ms
        << " complete_layer_mean_ceiling_ms="
        << kD230CompleteMeanCeilingMs
        << " moe_mean_ms=" << moe_mean_ms
        << " moe_mean_ceiling_ms=" << kD230MoeMeanCeilingMs
        << " router_id_mismatches=" << router_id_mismatches
        << " router_weight_mismatches=" << router_weight_mismatches
        << " d229_residual_export_mismatches="
        << d229_residual_export.mismatches
        << " d229_postnorm_export_mismatches="
        << d229_postnorm_export.mismatches
        << " v2_v3_elements=" << v2_v3.elements
        << " v2_v3_mismatches=" << v2_v3.mismatches
        << " v2_v3_nonfinite=" << v2_v3.nonfinite
        << " v2_v3_max_abs=" << v2_v3.max_abs
        << " callback_v3_mismatches=" << callback_v3.mismatches
        << " residual_input_mutations=" << residual_mutation.mismatches
        << " postnorm_input_mutations=" << postnorm_mutation.mismatches
        << " reference_samples=" << reference_samples
        << " reference_nonfinite=" << reference_nonfinite
        << " reference_max_abs=" << reference_max_abs
        << " output_elements=" << v3_output_host.size()
        << " output_nonfinite=" << output_nonfinite
        << " output_hash=" << std::hex
        << fnv1a64(
               v3_output_host.data(),
               v3_output_host.size() * sizeof(float)
           )
        << std::dec
        << " provider_scratch_bytes=" << moe_api.scratch_bytes()
        << " weight_bits=16"
        << " activation_bits=16"
        << " accumulation_bits=32"
        << " quantized=0"
        << " mtp_active=0"
        << " dflash_active=0"
        << " speculative_decode=0"
        << " complete_transformer_layer_claimed=1"
        << " real_model_loaded=0"
        << " product_metric_valid=0"
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    return D230ModeResult{
        mode,
        complete.mean_ms,
        moe_mean_ms,
        pass,
    };
}

}  // namespace

#ifndef QRT_D230_DISABLE_MAIN
int main(int argc, char **argv) {
    ProviderApi gdn_api;
    MoeApi moe_api;
    try {
        if (argc != 6) {
            std::cerr
                << "usage: "
                << "q16384_suffix1024_complete_linear_transformer_layer_smoke "
                << "GDN_KERNEL_DIR MOE_KERNEL_DIR "
                << "GDN_PROVIDER_DLL MOE_PROVIDER_DLL "
                << "REPETITIONS>=11\n";
            return 2;
        }
        const int parsed_repetitions = std::stoi(argv[5]);
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
        if (!load_provider(argv[3], &gdn_api)) {
            throw std::runtime_error(
                "could not load all required GDN provider exports"
            );
        }
        if (gdn_api.prepare(argv[1]) == 0) {
            throw std::runtime_error(
                std::string("GDN provider prepare failed: ") +
                gdn_api.last_error()
            );
        }
        if (!load_moe_provider(argv[4], &moe_api)) {
            throw std::runtime_error(
                "could not load all required MoE provider exports"
            );
        }
        if (moe_api.prepare(argv[2]) == 0) {
            throw std::runtime_error(
                std::string("MoE provider prepare failed: ") +
                moe_api.last_error()
            );
        }

        MoeWeights weights;
        initialize_moe_weights(&weights);
        const D230ModeResult early =
            run_d230_mode<float, true>(
                "early_f32",
                gdn_api,
                moe_api,
                weights,
                repetitions
            );
        const D230ModeResult retained =
            run_d230_mode<uint16_t, false>(
                "retained_bf16",
                gdn_api,
                moe_api,
                weights,
                repetitions
            );
        const double native_projection_ms =
            2.0 * early.complete_mean_ms +
            28.0 * retained.complete_mean_ms;
        const bool pass =
            early.pass &&
            retained.pass &&
            native_projection_ms <=
                kD230NativeProjectionCeilingMs;
        std::cout
            << std::fixed << std::setprecision(6)
            << "q16384_suffix1024_complete_linear_transformer_layer_summary"
            << " early_f32_layers=2"
            << " retained_bf16_layers=28"
            << " native_thirty_linear_layer_projected_ms="
            << native_projection_ms
            << " native_projection_ceiling_ms="
            << kD230NativeProjectionCeilingMs
            << " cases_pass="
            << ((early.pass ? 1 : 0) + (retained.pass ? 1 : 0))
            << " quantized=0"
            << " mtp_active=0"
            << " dflash_active=0"
            << " speculative_decode=0"
            << " full_attention_layers_in_scope=0"
            << " all_forty_layers_claimed=0"
            << " real_model_loaded=0"
            << " product_metric_valid=0"
            << " pass=" << (pass ? 1 : 0)
            << "\n";

        moe_api.release();
        FreeLibrary(moe_api.module);
        gdn_api.release();
        FreeLibrary(gdn_api.module);
        return pass ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr
            << "q16384_suffix1024_complete_linear_transformer_layer_smoke "
            << "error: " << error.what() << "\n";
        if (moe_api.release != nullptr) {
            moe_api.release();
        }
        if (moe_api.module != nullptr) {
            FreeLibrary(moe_api.module);
        }
        if (gdn_api.release != nullptr) {
            gdn_api.release();
        }
        if (gdn_api.module != nullptr) {
            FreeLibrary(gdn_api.module);
        }
        return 1;
    }
}
#endif
