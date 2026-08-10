#define QRT_D230_DISABLE_MAIN
#include "../gdn/q16384_suffix1024_complete_linear_transformer_layer_smoke.cpp"
#undef QRT_D230_DISABLE_MAIN

#include <array>

namespace {

constexpr unsigned int kD231QueryHeads = 16u;
constexpr unsigned int kD231KvHeads = 2u;
constexpr unsigned int kD231HeadDim = 256u;
constexpr unsigned int kD231RotaryDim = 64u;
constexpr unsigned int kD231QFeatures =
    kD231QueryHeads * kD231HeadDim;
constexpr unsigned int kD231KvFeatures =
    kD231KvHeads * kD231HeadDim;
constexpr unsigned int kD231QProjectionRows =
    2u * kD231QFeatures;
constexpr unsigned int kD231KProjectionRows = kD231KvFeatures;
constexpr unsigned int kD231VProjectionRows = kD231KvFeatures;
constexpr unsigned int kD231QkvRows =
    kD231QProjectionRows +
    kD231KProjectionRows +
    kD231VProjectionRows;
constexpr double kD231RopeTheta = 10000000.0;
constexpr double kD231SampleCeiling = 0.125;
constexpr double kD231CompleteMeanCeilingMs = 150.0;
constexpr double kD231MoeMeanCeilingMs = 30.0;
constexpr double kD231TenLayerCeilingMs = 1500.0;
constexpr double kD231CombinedCeilingMs = 2700.0;
constexpr double kD231DefaultD230LinearFixtureMs = 415.311629;

static_assert(
    kPrefixTokens == 16384u ||
    kPrefixTokens == 32768u ||
    kPrefixTokens == 65536u ||
    kPrefixTokens == 129536u ||
    kPrefixTokens == 131072u ||
    kPrefixTokens == 262144u
);
static_assert(kSuffixTokens == 1024u);
static_assert(kFullTokens == kPrefixTokens + kSuffixTokens);
static_assert(kLayerHidden == 2048u);
static_assert(kD231QFeatures == 4096u);
static_assert(kD231KvFeatures == 512u);
static_assert(kD231QProjectionRows == 8192u);
static_assert(kD231QkvRows == 9216u);
static_assert(kD231RotaryDim == 64u);

using CkPrepareFunction = int (*)();
using CkLaunchFunction = int (*)(
    const uint16_t *,
    const uint16_t *,
    const uint16_t *,
    float *,
    void *
);
using CkReleaseFunction = int (*)();

struct CkApi {
    HMODULE module = nullptr;
    CkPrepareFunction prepare = nullptr;
    CkLaunchFunction full_launch = nullptr;
    CkLaunchFunction suffix_launch = nullptr;
    CkReleaseFunction release = nullptr;
};

bool load_ck_provider(const char *path, CkApi *api) {
    api->module = LoadLibraryA(path);
    if (api->module == nullptr) {
        return false;
    }
    api->prepare = reinterpret_cast<CkPrepareFunction>(
        GetProcAddress(api->module, "qrt_ck_fmha_q8192_prepare")
    );
    api->full_launch = reinterpret_cast<CkLaunchFunction>(
        GetProcAddress(api->module, "qrt_ck_fmha_q17408_bf16_launch")
    );
    api->suffix_launch = reinterpret_cast<CkLaunchFunction>(
        GetProcAddress(
            api->module,
            "qrt_ck_fmha_q1024_kv17408_suffix_bf16_launch"
        )
    );
    api->release = reinterpret_cast<CkReleaseFunction>(
        GetProcAddress(api->module, "qrt_ck_fmha_q8192_release")
    );
    return api->prepare != nullptr &&
        api->full_launch != nullptr &&
        api->suffix_launch != nullptr &&
        api->release != nullptr;
}

__global__ void d231_prepare_q_gate_kernel(
    const uint16_t *qkv,
    const uint16_t *q_norm_weight,
    uint16_t *q,
    uint16_t *gate
) {
    __shared__ double partial[kD231HeadDim];
    __shared__ float inv_rms;
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (head >= kD231QueryHeads ||
        token >= kSuffixTokens ||
        dim >= kD231HeadDim) {
        return;
    }
    const size_t projection_base =
        static_cast<size_t>(token) * kD231QkvRows +
        static_cast<size_t>(head) * 2u * kD231HeadDim;
    const float value =
        device_bf16_to_float(qkv[projection_base + dim]);
    partial[dim] =
        static_cast<double>(value) * static_cast<double>(value);
    __syncthreads();
    for (unsigned int stride = kD231HeadDim / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (dim < stride) {
            partial[dim] += partial[dim + stride];
        }
        __syncthreads();
    }
    if (dim == 0u) {
        inv_rms = 1.0f / sqrtf(
            static_cast<float>(
                partial[0] / static_cast<double>(kD231HeadDim)
            ) + kRmsNormEpsilon
        );
    }
    __syncthreads();
    const size_t output_index =
        static_cast<size_t>(token) * kD231QFeatures +
        static_cast<size_t>(head) * kD231HeadDim +
        dim;
    q[output_index] = device_float_to_bf16(
        value * inv_rms *
        (1.0f + device_bf16_to_float(q_norm_weight[dim]))
    );
    gate[output_index] =
        qkv[projection_base + kD231HeadDim + dim];
}

__global__ void d231_prepare_k_v_kernel(
    const uint16_t *qkv,
    const uint16_t *k_norm_weight,
    uint16_t *k,
    uint16_t *v
) {
    __shared__ double partial[kD231HeadDim];
    __shared__ float inv_rms;
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int dim = threadIdx.x;
    if (head >= kD231KvHeads ||
        token >= kSuffixTokens ||
        dim >= kD231HeadDim) {
        return;
    }
    const size_t token_base =
        static_cast<size_t>(token) * kD231QkvRows;
    const size_t head_offset =
        static_cast<size_t>(head) * kD231HeadDim + dim;
    const size_t k_index =
        token_base + kD231QProjectionRows + head_offset;
    const size_t v_index =
        token_base +
        kD231QProjectionRows +
        kD231KProjectionRows +
        head_offset;
    const float value = device_bf16_to_float(qkv[k_index]);
    partial[dim] =
        static_cast<double>(value) * static_cast<double>(value);
    __syncthreads();
    for (unsigned int stride = kD231HeadDim / 2u;
         stride > 0u;
         stride >>= 1u) {
        if (dim < stride) {
            partial[dim] += partial[dim + stride];
        }
        __syncthreads();
    }
    if (dim == 0u) {
        inv_rms = 1.0f / sqrtf(
            static_cast<float>(
                partial[0] / static_cast<double>(kD231HeadDim)
            ) + kRmsNormEpsilon
        );
    }
    __syncthreads();
    const size_t output_index =
        static_cast<size_t>(token) * kD231KvFeatures + head_offset;
    k[output_index] = device_float_to_bf16(
        value * inv_rms *
        (1.0f + device_bf16_to_float(k_norm_weight[dim]))
    );
    v[output_index] = qkv[v_index];
}

__global__ void d231_rope_bf16_kernel(
    uint16_t *values,
    unsigned int heads,
    unsigned int features
) {
    const unsigned int head = blockIdx.x;
    const unsigned int token = blockIdx.y;
    const unsigned int pair = threadIdx.x;
    constexpr unsigned int kHalfRotary = kD231RotaryDim / 2u;
    if (head >= heads ||
        token >= kSuffixTokens ||
        pair >= kHalfRotary) {
        return;
    }
    const size_t base =
        static_cast<size_t>(token) * features +
        static_cast<size_t>(head) * kD231HeadDim;
    const unsigned int second_dim = pair + kHalfRotary;
    const float first =
        device_bf16_to_float(values[base + pair]);
    const float second =
        device_bf16_to_float(values[base + second_dim]);
    const double inv_freq =
        1.0 /
        pow(
            kD231RopeTheta,
            (2.0 * static_cast<double>(pair)) /
                static_cast<double>(kD231RotaryDim)
        );
    const double angle =
        static_cast<double>(kPrefixTokens + token) * inv_freq;
    const float cosine = static_cast<float>(cos(angle));
    const float sine = static_cast<float>(sin(angle));
    values[base + pair] =
        device_float_to_bf16(first * cosine - second * sine);
    values[base + second_dim] =
        device_float_to_bf16(second * cosine + first * sine);
}

__global__ void d231_attention_gate_kernel(
    const float *context,
    const uint16_t *gate,
    uint16_t *gated,
    size_t elements
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const float gate_value = device_bf16_to_float(gate[index]);
    const float sigmoid = 1.0f / (1.0f + expf(-gate_value));
    gated[index] =
        device_float_to_bf16(context[index] * sigmoid);
}

template <typename Launch>
double d231_time_launch(
    Launch launch,
    unsigned int repetitions,
    const char *stage
) {
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "D231 event create(start)");
    try {
        check_hip(hipEventCreate(&stop), "D231 event create(stop)");
        check_hip(
            hipEventRecord(start, nullptr),
            "D231 event record(start)"
        );
        for (unsigned int iteration = 0u;
             iteration < repetitions;
             ++iteration) {
            launch();
        }
        check_hip(hipGetLastError(), stage);
        check_hip(
            hipEventRecord(stop, nullptr),
            "D231 event record(stop)"
        );
        check_hip(
            hipEventSynchronize(stop),
            "D231 event synchronize(stop)"
        );
        float elapsed_ms = 0.0f;
        check_hip(
            hipEventElapsedTime(&elapsed_ms, start, stop),
            "D231 event elapsed"
        );
        (void)hipEventDestroy(stop);
        (void)hipEventDestroy(start);
        return static_cast<double>(elapsed_ms) /
            static_cast<double>(repetitions);
    } catch (...) {
        if (stop != nullptr) {
            (void)hipEventDestroy(stop);
        }
        if (start != nullptr) {
            (void)hipEventDestroy(start);
        }
        throw;
    }
}

struct D231SampleStats {
    uint64_t samples = 0u;
    uint64_t nonfinite = 0u;
    double max_abs = 0.0;
};

void d231_add_sample(
    float reference,
    float candidate,
    D231SampleStats *stats
) {
    ++stats->samples;
    if (!std::isfinite(reference) || !std::isfinite(candidate)) {
        ++stats->nonfinite;
        return;
    }
    stats->max_abs = std::max(
        stats->max_abs,
        std::abs(
            static_cast<double>(reference) -
            static_cast<double>(candidate)
        )
    );
}

float d231_host_norm_value(
    const uint16_t *input,
    const uint16_t *weight,
    unsigned int dim
) {
    double sumsq = 0.0;
    for (unsigned int index = 0u;
         index < kD231HeadDim;
         ++index) {
        const float value = host_bf16_to_float(input[index]);
        sumsq += static_cast<double>(value) *
            static_cast<double>(value);
    }
    const float inv_rms = 1.0f / std::sqrt(
        static_cast<float>(
            sumsq / static_cast<double>(kD231HeadDim)
        ) + kRmsNormEpsilon
    );
    return d230_bf16_round(
        host_bf16_to_float(input[dim]) *
        inv_rms *
        (1.0f + host_bf16_to_float(weight[dim]))
    );
}

float d231_host_rope_value(
    const uint16_t *input,
    const uint16_t *weight,
    unsigned int dim,
    unsigned int absolute_token
) {
    if (dim >= kD231RotaryDim) {
        return d231_host_norm_value(input, weight, dim);
    }
    constexpr unsigned int kHalfRotary = kD231RotaryDim / 2u;
    const unsigned int pair =
        dim < kHalfRotary ? dim : dim - kHalfRotary;
    const float first =
        d231_host_norm_value(input, weight, pair);
    const float second =
        d231_host_norm_value(
            input,
            weight,
            pair + kHalfRotary
        );
    const double inv_freq =
        1.0 /
        std::pow(
            kD231RopeTheta,
            (2.0 * static_cast<double>(pair)) /
                static_cast<double>(kD231RotaryDim)
        );
    const double angle =
        static_cast<double>(absolute_token) * inv_freq;
    const float cosine = static_cast<float>(std::cos(angle));
    const float sine = static_cast<float>(std::sin(angle));
    return d230_bf16_round(
        dim < kHalfRotary
            ? first * cosine - second * sine
            : second * cosine + first * sine
    );
}

D231SampleStats d231_validate_qk_rope_gate_samples(
    const std::vector<uint16_t> &qkv,
    const std::vector<uint16_t> &q,
    const std::vector<uint16_t> &k,
    const std::vector<uint16_t> &gate,
    const std::vector<uint16_t> &q_norm_weight,
    const std::vector<uint16_t> &k_norm_weight
) {
    const std::array<unsigned int, 4> tokens = {
        0u,
        kSuffixTokens / 3u,
        (2u * kSuffixTokens) / 3u,
        kSuffixTokens - 1u,
    };
    const std::array<unsigned int, 4> dims = {
        0u,
        31u,
        63u,
        kD231HeadDim - 1u,
    };
    D231SampleStats stats;
    for (size_t sample = 0u; sample < tokens.size(); ++sample) {
        const unsigned int token = tokens[sample];
        const unsigned int q_head =
            static_cast<unsigned int>((sample * 5u + 3u) %
                                      kD231QueryHeads);
        const unsigned int k_head =
            static_cast<unsigned int>(sample % kD231KvHeads);
        const uint16_t *row =
            qkv.data() +
            static_cast<size_t>(token) * kD231QkvRows;
        const uint16_t *q_input =
            row + static_cast<size_t>(q_head) *
                2u * kD231HeadDim;
        const uint16_t *k_input =
            row +
            kD231QProjectionRows +
            static_cast<size_t>(k_head) * kD231HeadDim;
        for (unsigned int dim : dims) {
            const unsigned int absolute_token =
                kPrefixTokens + token;
            const size_t q_index =
                static_cast<size_t>(token) * kD231QFeatures +
                static_cast<size_t>(q_head) * kD231HeadDim +
                dim;
            const size_t k_index =
                static_cast<size_t>(token) * kD231KvFeatures +
                static_cast<size_t>(k_head) * kD231HeadDim +
                dim;
            d231_add_sample(
                d231_host_rope_value(
                    q_input,
                    q_norm_weight.data(),
                    dim,
                    absolute_token
                ),
                host_bf16_to_float(q[q_index]),
                &stats
            );
            d231_add_sample(
                host_bf16_to_float(
                    q_input[kD231HeadDim + dim]
                ),
                host_bf16_to_float(gate[q_index]),
                &stats
            );
            d231_add_sample(
                d231_host_rope_value(
                    k_input,
                    k_norm_weight.data(),
                    dim,
                    absolute_token
                ),
                host_bf16_to_float(k[k_index]),
                &stats
            );
        }
    }
    return stats;
}

bool d231_exact(const CompareStats &stats) {
    return stats.mismatches == 0u &&
        stats.nonfinite == 0u &&
        stats.max_abs == 0.0;
}

struct D231Result {
    double complete_mean_ms = 0.0;
    double moe_mean_ms = 0.0;
    bool pass = false;
};

D231Result run_d231(
    const CkApi &ck,
    const MoeApi &moe,
    const MoeWeights &moe_weights,
    unsigned int repetitions,
    double d230_linear_fixture_ms
) {
    const size_t hidden_elements =
        static_cast<size_t>(kSuffixTokens) * kLayerHidden;
    const size_t qkv_elements =
        static_cast<size_t>(kSuffixTokens) * kD231QkvRows;
    const size_t suffix_q_elements =
        static_cast<size_t>(kSuffixTokens) * kD231QFeatures;
    const size_t suffix_kv_elements =
        static_cast<size_t>(kSuffixTokens) * kD231KvFeatures;
    const size_t full_q_elements =
        static_cast<size_t>(kFullTokens) * kD231QFeatures;
    const size_t full_kv_elements =
        static_cast<size_t>(kFullTokens) * kD231KvFeatures;
    const size_t prefix_kv_elements =
        static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;

    DeviceBuffer<float> hidden(hidden_elements);
    DeviceBuffer<uint16_t> input_norm_weight(kLayerHidden);
    DeviceBuffer<float> input_norm_f32(hidden_elements);
    DeviceBuffer<uint16_t> input_norm_bf16(hidden_elements);
    DeviceBuffer<uint16_t> qkv_weight(
        static_cast<size_t>(kD231QkvRows) * kLayerHidden
    );
    DeviceBuffer<uint16_t> qkv(qkv_elements);
    DeviceBuffer<uint16_t> q_norm_weight(kD231HeadDim);
    DeviceBuffer<uint16_t> k_norm_weight(kD231HeadDim);
    DeviceBuffer<uint16_t> full_q(full_q_elements);
    DeviceBuffer<uint16_t> full_k(full_kv_elements);
    DeviceBuffer<uint16_t> full_v(full_kv_elements);
    DeviceBuffer<uint16_t> gate(suffix_q_elements);
    DeviceBuffer<float> full_context(full_q_elements);
    DeviceBuffer<float> suffix_context(suffix_q_elements);
    DeviceBuffer<uint16_t> reference_gated(suffix_q_elements);
    DeviceBuffer<uint16_t> candidate_gated(suffix_q_elements);
    DeviceBuffer<uint16_t> output_weight(
        static_cast<size_t>(kLayerHidden) * kD231QFeatures
    );
    DeviceBuffer<uint16_t> reference_output(hidden_elements);
    DeviceBuffer<uint16_t> candidate_output(hidden_elements);
    DeviceBuffer<uint16_t> postnorm_weight(kLayerHidden);
    DeviceBuffer<float> reference_residual(hidden_elements);
    DeviceBuffer<float> candidate_residual(hidden_elements);
    DeviceBuffer<float> reference_postnorm(hidden_elements);
    DeviceBuffer<float> candidate_postnorm(hidden_elements);
    DeviceBuffer<float> reference_final(hidden_elements);
    DeviceBuffer<float> candidate_final(hidden_elements);
    DeviceBuffer<float> callback_final(hidden_elements);

    const dim3 block(kThreads);
    const auto linear_grid = [](size_t count) {
        return dim3(
            static_cast<unsigned int>(
                (count + kThreads - 1u) / kThreads
            )
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
        input_norm_weight.get(),
        static_cast<size_t>(kLayerHidden),
        0
    );
    hipLaunchKernelGGL(
        fill_norm_weights_kernel,
        linear_grid(kLayerHidden),
        block,
        0u,
        nullptr,
        postnorm_weight.get(),
        static_cast<size_t>(kLayerHidden),
        0
    );
    hipLaunchKernelGGL(
        fill_norm_weights_kernel,
        dim3(1u),
        block,
        0u,
        nullptr,
        q_norm_weight.get(),
        static_cast<size_t>(kD231HeadDim),
        0
    );
    hipLaunchKernelGGL(
        fill_norm_weights_kernel,
        dim3(1u),
        block,
        0u,
        nullptr,
        k_norm_weight.get(),
        static_cast<size_t>(kD231HeadDim),
        0
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(
            static_cast<size_t>(kD231QkvRows) * kLayerHidden
        ),
        block,
        0u,
        nullptr,
        qkv_weight.get(),
        static_cast<size_t>(kD231QkvRows) * kLayerHidden,
        UINT32_C(0x44323331),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(
            static_cast<size_t>(kLayerHidden) * kD231QFeatures
        ),
        block,
        0u,
        nullptr,
        output_weight.get(),
        static_cast<size_t>(kLayerHidden) * kD231QFeatures,
        UINT32_C(0x4f323331),
        0.000244140625f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(full_q_elements),
        block,
        0u,
        nullptr,
        full_q.get(),
        full_q_elements,
        UINT32_C(0x51323331),
        0.001953125f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(full_kv_elements),
        block,
        0u,
        nullptr,
        full_k.get(),
        full_kv_elements,
        UINT32_C(0x4b323331),
        0.001953125f
    );
    hipLaunchKernelGGL(
        fill_dense_weights_kernel,
        linear_grid(full_kv_elements),
        block,
        0u,
        nullptr,
        full_v.get(),
        full_kv_elements,
        UINT32_C(0x56323331),
        0.001953125f
    );
    check_hip(hipGetLastError(), "D231 fixture initialization");
    check_hip(
        hipDeviceSynchronize(),
        "D231 fixture initialization synchronize"
    );

    const std::vector<float> hidden_before =
        copy_device(hidden.get(), hidden_elements);
    const std::vector<uint16_t> prefix_k_before =
        copy_device(full_k.get(), prefix_kv_elements);
    const std::vector<uint16_t> prefix_v_before =
        copy_device(full_v.get(), prefix_kv_elements);

    RocblasContext blas;
    uint16_t *suffix_q =
        full_q.get() +
        static_cast<size_t>(kPrefixTokens) * kD231QFeatures;
    uint16_t *suffix_k =
        full_k.get() +
        static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
    uint16_t *suffix_v =
        full_v.get() +
        static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
    float *reference_context =
        full_context.get() +
        static_cast<size_t>(kPrefixTokens) * kD231QFeatures;

    auto prepare_qkv = [&]() {
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden.get(),
            input_norm_weight.get(),
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
            qkv_weight.get(),
            input_norm_bf16.get(),
            qkv.get(),
            kD231QkvRows,
            kLayerHidden
        );
        hipLaunchKernelGGL(
            d231_prepare_q_gate_kernel,
            dim3(kD231QueryHeads, kSuffixTokens),
            dim3(kD231HeadDim),
            0u,
            nullptr,
            qkv.get(),
            q_norm_weight.get(),
            suffix_q,
            gate.get()
        );
        hipLaunchKernelGGL(
            d231_prepare_k_v_kernel,
            dim3(kD231KvHeads, kSuffixTokens),
            dim3(kD231HeadDim),
            0u,
            nullptr,
            qkv.get(),
            k_norm_weight.get(),
            suffix_k,
            suffix_v
        );
        hipLaunchKernelGGL(
            d231_rope_bf16_kernel,
            dim3(kD231QueryHeads, kSuffixTokens),
            dim3(kD231RotaryDim / 2u),
            0u,
            nullptr,
            suffix_q,
            kD231QueryHeads,
            kD231QFeatures
        );
        hipLaunchKernelGGL(
            d231_rope_bf16_kernel,
            dim3(kD231KvHeads, kSuffixTokens),
            dim3(kD231RotaryDim / 2u),
            0u,
            nullptr,
            suffix_k,
            kD231KvHeads,
            kD231KvFeatures
        );
    };

    auto launch_attention_tail = [&]() {
        prepare_qkv();
        if (ck.suffix_launch(
                suffix_q,
                full_k.get(),
                full_v.get(),
                suffix_context.get(),
                nullptr
            ) != 0) {
            throw std::runtime_error(
                "D231 CK suffix attention launch failed"
            );
        }
        hipLaunchKernelGGL(
            d231_attention_gate_kernel,
            linear_grid(suffix_q_elements),
            block,
            0u,
            nullptr,
            suffix_context.get(),
            gate.get(),
            candidate_gated.get(),
            suffix_q_elements
        );
        blas.matmul(
            output_weight.get(),
            candidate_gated.get(),
            candidate_output.get(),
            kLayerHidden,
            kD231QFeatures
        );
        hipLaunchKernelGGL(
            residual_postnorm_sublayer_kernel<uint16_t>,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden.get(),
            candidate_output.get(),
            postnorm_weight.get(),
            candidate_residual.get(),
            candidate_postnorm.get()
        );
        if (moe.launch_v3_async(
                candidate_postnorm.get(),
                candidate_residual.get(),
                moe_weights.router.get(),
                moe_weights.routed_gate_up.get(),
                moe_weights.routed_down.get(),
                moe_weights.shared_gate.get(),
                moe_weights.shared_gate_projection.get(),
                moe_weights.shared_up_projection.get(),
                moe_weights.shared_down.get(),
                candidate_final.get(),
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("D231 MoE launch failed: ") +
                moe.last_error()
            );
        }
    };

    prepare_qkv();
    check_hip(hipGetLastError(), "D231 reference prepare");
    check_hip(
        hipDeviceSynchronize(),
        "D231 reference prepare synchronize"
    );
    if (ck.full_launch(
            full_q.get(),
            full_k.get(),
            full_v.get(),
            full_context.get(),
            nullptr
        ) != 0) {
        throw std::runtime_error(
            "D231 CK full attention reference launch failed"
        );
    }
    hipLaunchKernelGGL(
        d231_attention_gate_kernel,
        linear_grid(suffix_q_elements),
        block,
        0u,
        nullptr,
        reference_context,
        gate.get(),
        reference_gated.get(),
        suffix_q_elements
    );
    blas.matmul(
        output_weight.get(),
        reference_gated.get(),
        reference_output.get(),
        kLayerHidden,
        kD231QFeatures
    );
    hipLaunchKernelGGL(
        residual_postnorm_sublayer_kernel<uint16_t>,
        dim3(kSuffixTokens),
        block,
        0u,
        nullptr,
        hidden.get(),
        reference_output.get(),
        postnorm_weight.get(),
        reference_residual.get(),
        reference_postnorm.get()
    );
    if (moe.launch_v2(
            reference_postnorm.get(),
            reference_residual.get(),
            moe_weights.router.get(),
            moe_weights.routed_gate_up.get(),
            moe_weights.routed_down.get(),
            moe_weights.shared_gate.get(),
            moe_weights.shared_gate_projection.get(),
            moe_weights.shared_up_projection.get(),
            moe_weights.shared_down.get(),
            reference_final.get(),
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D231 reference MoE launch failed: ") +
            moe.last_error()
        );
    }
    check_hip(hipGetLastError(), "D231 reference tail");
    check_hip(
        hipDeviceSynchronize(),
        "D231 reference tail synchronize"
    );

    launch_attention_tail();
    check_hip(hipGetLastError(), "D231 candidate warmup");
    check_hip(
        hipDeviceSynchronize(),
        "D231 candidate warmup synchronize"
    );
    const double complete_mean_ms = d231_time_launch(
        launch_attention_tail,
        repetitions,
        "D231 candidate timed launch"
    );

    const std::vector<float> moe_residual_before =
        copy_device(candidate_residual.get(), hidden_elements);
    const std::vector<float> moe_postnorm_before =
        copy_device(candidate_postnorm.get(), hidden_elements);
    MoeTailContext callback_context{
        &moe,
        &moe_weights,
        callback_final.get(),
    };
    launch_moe_tail(
        &callback_context,
        candidate_residual.get(),
        candidate_postnorm.get(),
        nullptr
    );
    check_hip(hipGetLastError(), "D231 callback launch");
    check_hip(
        hipDeviceSynchronize(),
        "D231 callback synchronize"
    );
    const double moe_mean_ms = d231_time_launch(
        [&]() {
            if (moe.launch_v3_async(
                    candidate_postnorm.get(),
                    candidate_residual.get(),
                    moe_weights.router.get(),
                    moe_weights.routed_gate_up.get(),
                    moe_weights.routed_down.get(),
                    moe_weights.shared_gate.get(),
                    moe_weights.shared_gate_projection.get(),
                    moe_weights.shared_up_projection.get(),
                    moe_weights.shared_down.get(),
                    candidate_final.get(),
                    nullptr
                ) == 0) {
                throw std::runtime_error(
                    std::string("D231 timed MoE launch failed: ") +
                    moe.last_error()
                );
            }
        },
        repetitions,
        "D231 timed MoE"
    );

    if (moe.launch_router_debug(
            candidate_postnorm.get(),
            moe_weights.router.get(),
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D231 router debug failed: ") +
            moe.last_error()
        );
    }
    check_hip(
        hipDeviceSynchronize(),
        "D231 router debug synchronize"
    );
    std::vector<uint32_t> topk_ids(kD230Routes);
    std::vector<float> topk_weights(kD230Routes);
    if (moe.copy_topk_debug(
            topk_ids.data(),
            topk_weights.data()
        ) == 0) {
        throw std::runtime_error(
            "D231 router debug copy failed"
        );
    }
    uint64_t router_id_mismatches = 0u;
    uint64_t router_weight_mismatches = 0u;
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

    const std::vector<float> attention_reference =
        copy_device(reference_context, suffix_q_elements);
    const std::vector<float> attention_candidate =
        copy_device(suffix_context.get(), suffix_q_elements);
    const std::vector<float> residual_reference =
        copy_device(reference_residual.get(), hidden_elements);
    const std::vector<float> residual_candidate =
        copy_device(candidate_residual.get(), hidden_elements);
    const std::vector<float> postnorm_reference =
        copy_device(reference_postnorm.get(), hidden_elements);
    const std::vector<float> postnorm_candidate =
        copy_device(candidate_postnorm.get(), hidden_elements);
    const std::vector<float> final_reference =
        copy_device(reference_final.get(), hidden_elements);
    const std::vector<float> final_candidate =
        copy_device(candidate_final.get(), hidden_elements);
    const std::vector<float> final_callback =
        copy_device(callback_final.get(), hidden_elements);
    const std::vector<float> hidden_after =
        copy_device(hidden.get(), hidden_elements);
    const std::vector<uint16_t> prefix_k_after =
        copy_device(full_k.get(), prefix_kv_elements);
    const std::vector<uint16_t> prefix_v_after =
        copy_device(full_v.get(), prefix_kv_elements);
    const std::vector<float> moe_residual_after =
        copy_device(candidate_residual.get(), hidden_elements);
    const std::vector<float> moe_postnorm_after =
        copy_device(candidate_postnorm.get(), hidden_elements);

    const CompareStats attention_stats =
        compare_f32(attention_reference, attention_candidate);
    const CompareStats residual_stats =
        compare_f32(residual_reference, residual_candidate);
    const CompareStats postnorm_stats =
        compare_f32(postnorm_reference, postnorm_candidate);
    const CompareStats final_stats =
        compare_f32(final_reference, final_candidate);
    const CompareStats callback_stats =
        compare_f32(final_candidate, final_callback);
    const CompareStats hidden_mutation =
        compare_f32(hidden_before, hidden_after);
    const CompareStats prefix_k_mutation =
        compare_native(prefix_k_before, prefix_k_after);
    const CompareStats prefix_v_mutation =
        compare_native(prefix_v_before, prefix_v_after);
    const CompareStats residual_input_mutation =
        compare_f32(moe_residual_before, moe_residual_after);
    const CompareStats postnorm_input_mutation =
        compare_f32(moe_postnorm_before, moe_postnorm_after);

    const ProjectionSampleStats qkv_projection_samples =
        validate_projection_samples<uint16_t>(
            qkv_weight.get(),
            input_norm_bf16.get(),
            qkv.get(),
            kD231QkvRows,
            kLayerHidden
        );
    const ProjectionSampleStats output_projection_samples =
        validate_projection_samples<uint16_t>(
            output_weight.get(),
            candidate_gated.get(),
            candidate_output.get(),
            kLayerHidden,
            kD231QFeatures
        );
    const std::vector<uint16_t> qkv_host =
        copy_device(qkv.get(), qkv_elements);
    const std::vector<uint16_t> suffix_q_host =
        copy_device(suffix_q, suffix_q_elements);
    const std::vector<uint16_t> suffix_k_host =
        copy_device(suffix_k, suffix_kv_elements);
    const std::vector<uint16_t> gate_host =
        copy_device(gate.get(), suffix_q_elements);
    const std::vector<uint16_t> q_norm_weight_host =
        copy_device(q_norm_weight.get(), kD231HeadDim);
    const std::vector<uint16_t> k_norm_weight_host =
        copy_device(k_norm_weight.get(), kD231HeadDim);
    const D231SampleStats prep_samples =
        d231_validate_qk_rope_gate_samples(
            qkv_host,
            suffix_q_host,
            suffix_k_host,
            gate_host,
            q_norm_weight_host,
            k_norm_weight_host
        );

    D231SampleStats final_samples;
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
    for (unsigned int token : sample_tokens) {
        const float *postnorm =
            postnorm_candidate.data() +
            static_cast<size_t>(token) * kLayerHidden;
        for (unsigned int column : sample_columns) {
            const size_t index =
                static_cast<size_t>(token) * kLayerHidden + column;
            d231_add_sample(
                independent_output_reference(
                    postnorm,
                    residual_candidate[index]
                ),
                final_candidate[index],
                &final_samples
            );
        }
    }

    const double ten_layer_projected_ms =
        10.0 * complete_mean_ms;
    const double combined_fixture_ms =
        d230_linear_fixture_ms + ten_layer_projected_ms;
    const bool pass =
        d231_exact(attention_stats) &&
        d231_exact(residual_stats) &&
        d231_exact(postnorm_stats) &&
        d231_exact(final_stats) &&
        d231_exact(callback_stats) &&
        d231_exact(hidden_mutation) &&
        d231_exact(prefix_k_mutation) &&
        d231_exact(prefix_v_mutation) &&
        d231_exact(residual_input_mutation) &&
        d231_exact(postnorm_input_mutation) &&
        qkv_projection_samples.samples >= 16u &&
        qkv_projection_samples.nonfinite == 0u &&
        qkv_projection_samples.max_abs <= kD231SampleCeiling &&
        output_projection_samples.samples >= 16u &&
        output_projection_samples.nonfinite == 0u &&
        output_projection_samples.max_abs <= kD231SampleCeiling &&
        prep_samples.samples >= 16u &&
        prep_samples.nonfinite == 0u &&
        prep_samples.max_abs <= kD231SampleCeiling &&
        router_id_mismatches == 0u &&
        router_weight_mismatches == 0u &&
        final_samples.samples >= 16u &&
        final_samples.nonfinite == 0u &&
        final_samples.max_abs <= kD231SampleCeiling &&
        complete_mean_ms <= kD231CompleteMeanCeilingMs &&
        moe_mean_ms <= kD231MoeMeanCeilingMs &&
        ten_layer_projected_ms <= kD231TenLayerCeilingMs &&
        combined_fixture_ms <= kD231CombinedCeilingMs;

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_complete_full_attention_transformer_layer_smoke"
        << " prefix_tokens=" << kPrefixTokens
        << " suffix_tokens=" << kSuffixTokens
        << " kv_tokens=" << kFullTokens
        << " q_projection_rows=" << kD231QProjectionRows
        << " k_projection_rows=" << kD231KProjectionRows
        << " v_projection_rows=" << kD231VProjectionRows
        << " query_heads=" << kD231QueryHeads
        << " kv_heads=" << kD231KvHeads
        << " head_dim=" << kD231HeadDim
        << " rotary_dim=" << kD231RotaryDim
        << " complete_layer_mean_ms=" << complete_mean_ms
        << " complete_layer_mean_ceiling_ms="
        << kD231CompleteMeanCeilingMs
        << " moe_mean_ms=" << moe_mean_ms
        << " moe_mean_ceiling_ms=" << kD231MoeMeanCeilingMs
        << " attention_elements=" << attention_stats.elements
        << " attention_mismatches=" << attention_stats.mismatches
        << " attention_nonfinite=" << attention_stats.nonfinite
        << " attention_max_abs=" << attention_stats.max_abs
        << " qkv_projection_samples="
        << qkv_projection_samples.samples
        << " qkv_projection_nonfinite="
        << qkv_projection_samples.nonfinite
        << " qkv_projection_max_abs="
        << qkv_projection_samples.max_abs
        << " output_projection_samples="
        << output_projection_samples.samples
        << " output_projection_nonfinite="
        << output_projection_samples.nonfinite
        << " output_projection_max_abs="
        << output_projection_samples.max_abs
        << " prep_samples=" << prep_samples.samples
        << " prep_nonfinite=" << prep_samples.nonfinite
        << " prep_max_abs=" << prep_samples.max_abs
        << " residual_mismatches=" << residual_stats.mismatches
        << " postnorm_mismatches=" << postnorm_stats.mismatches
        << " final_mismatches=" << final_stats.mismatches
        << " callback_mismatches=" << callback_stats.mismatches
        << " router_id_mismatches=" << router_id_mismatches
        << " router_weight_mismatches="
        << router_weight_mismatches
        << " final_reference_samples=" << final_samples.samples
        << " final_reference_nonfinite=" << final_samples.nonfinite
        << " final_reference_max_abs=" << final_samples.max_abs
        << " prefix_k_mutations=" << prefix_k_mutation.mismatches
        << " prefix_v_mutations=" << prefix_v_mutation.mismatches
        << " input_hidden_mutations=" << hidden_mutation.mismatches
        << " residual_input_mutations="
        << residual_input_mutation.mismatches
        << " postnorm_input_mutations="
        << postnorm_input_mutation.mismatches
        << " output_hash=" << std::hex
        << fnv1a64(
               final_candidate.data(),
               final_candidate.size() * sizeof(float)
           )
        << std::dec
        << " provider_scratch_bytes=" << moe.scratch_bytes()
        << " materialized_active_experts=" << kD230ActiveExperts
        << " all_256_expert_weights_resident=0"
        << " repetitions=" << repetitions
        << " weight_bits=16"
        << " activation_bits=16"
        << " accumulation_bits=32"
        << " quantized=0"
        << " mtp_active=0"
        << " dflash_active=0"
        << " speculative_decode=0"
        << " complete_full_attention_layer_claimed=1"
        << " real_model_loaded=0"
        << " correctness_boundary_attached=0"
        << " product_metric_valid=0"
        << " product_performance_accepted=0"
        << " inference_success_claimed=0"
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_complete_full_attention_transformer_layer_summary"
        << " full_attention_layers=10"
        << " native_ten_full_attention_layer_projected_ms="
        << ten_layer_projected_ms
        << " ten_layer_projection_ceiling_ms="
        << kD231TenLayerCeilingMs
        << " d230_thirty_linear_layer_fixture_ms="
        << d230_linear_fixture_ms
        << " combined_forty_layer_fixture_ms="
        << combined_fixture_ms
        << " combined_fixture_ceiling_ms="
        << kD231CombinedCeilingMs
        << " complete_full_attention_layer_claimed=1"
        << " all_forty_layers_claimed=0"
        << " real_model_loaded=0"
        << " correctness_boundary_attached=0"
        << " product_metric_valid=0"
        << " product_performance_accepted=0"
        << " inference_success_claimed=0"
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    return D231Result{
        complete_mean_ms,
        moe_mean_ms,
        pass,
    };
}

}  // namespace

#ifndef QRT_D231_DISABLE_MAIN
int main(int argc, char **argv) {
    CkApi ck;
    MoeApi moe;
    try {
        if (argc != 5 && argc != 6) {
            std::cerr
                << "usage: "
                << "q16384_suffix1024_complete_full_attention_"
                   "transformer_layer_smoke "
                << "CK_PROVIDER_DLL MOE_KERNEL_DIR "
                   "MOE_PROVIDER_DLL REPETITIONS>=11 "
                   "[D230_LINEAR_FIXTURE_MS]\n";
            return 2;
        }
        const int parsed_repetitions = std::stoi(argv[4]);
        if (parsed_repetitions < 11 || parsed_repetitions > 100) {
            std::cerr << "repetitions must be in [11,100]\n";
            return 2;
        }
        const double d230_linear_fixture_ms =
            argc == 6
                ? std::stod(argv[5])
                : kD231DefaultD230LinearFixtureMs;
        if (!std::isfinite(d230_linear_fixture_ms) ||
            d230_linear_fixture_ms <= 0.0) {
            std::cerr
                << "D230_LINEAR_FIXTURE_MS must be finite and positive\n";
            return 2;
        }
        int device_count = 0;
        check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("no HIP device is available");
        }
        check_hip(hipSetDevice(0), "hipSetDevice");
        if (!load_ck_provider(argv[1], &ck)) {
            throw std::runtime_error(
                "could not load all required CK provider exports"
            );
        }
        if (ck.prepare() != 0) {
            throw std::runtime_error("CK provider prepare failed");
        }
        if (!load_moe_provider(argv[3], &moe)) {
            throw std::runtime_error(
                "could not load all required MoE provider exports"
            );
        }
        if (moe.prepare(argv[2]) == 0) {
            throw std::runtime_error(
                std::string("MoE provider prepare failed: ") +
                moe.last_error()
            );
        }
        MoeWeights moe_weights;
        initialize_moe_weights(&moe_weights);
        const D231Result result = run_d231(
            ck,
            moe,
            moe_weights,
            static_cast<unsigned int>(parsed_repetitions),
            d230_linear_fixture_ms
        );
        moe.release();
        FreeLibrary(moe.module);
        (void)ck.release();
        FreeLibrary(ck.module);
        return result.pass ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr
            << "q16384_suffix1024_complete_full_attention_"
               "transformer_layer_smoke error: "
            << error.what() << "\n";
        if (moe.release != nullptr) {
            moe.release();
        }
        if (moe.module != nullptr) {
            FreeLibrary(moe.module);
        }
        if (ck.release != nullptr) {
            (void)ck.release();
        }
        if (ck.module != nullptr) {
            FreeLibrary(ck.module);
        }
        return 1;
    }
}
#endif
