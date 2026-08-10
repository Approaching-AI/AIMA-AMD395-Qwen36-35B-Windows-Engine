#define QRT_D231_DISABLE_MAIN
#include "q16384_suffix1024_complete_full_attention_transformer_layer_smoke.cpp"
#undef QRT_D231_DISABLE_MAIN

#include <array>
#include <type_traits>

namespace {

constexpr unsigned int kD232Layers = 40u;
constexpr unsigned int kD232LinearLayers = 30u;
constexpr unsigned int kD232FullLayers = 10u;
constexpr unsigned int kD232EarlyLinearLayers = 2u;
constexpr unsigned int kD232RetainedLinearLayers = 28u;
constexpr unsigned int kD232MinimumRepetitions = 5u;
constexpr double kD232MeanCeilingMs = 2700.0;
constexpr double kD232BaselineRatioCeiling = 1.25;
constexpr double kD232SampleCeiling = 0.125;

constexpr size_t kD232HiddenElements =
    static_cast<size_t>(kSuffixTokens) * kLayerHidden;
constexpr size_t kD232LinearQkvElements =
    static_cast<size_t>(kSuffixTokens) * kQkvRows;
constexpr size_t kD232LinearZElements =
    static_cast<size_t>(kSuffixTokens) * kZRows;
constexpr size_t kD232LinearAbElements =
    static_cast<size_t>(kSuffixTokens) * kAbRows;
constexpr size_t kD232LinearCoreElements =
    static_cast<size_t>(kSuffixTokens) * kValueFeatures;
constexpr size_t kD232LinearGateElements =
    static_cast<size_t>(kSuffixTokens) * kGateOutputRows;
constexpr size_t kD232LinearStateElements =
    static_cast<size_t>(kValueFeatures) * kKeyDim;
constexpr size_t kD232LinearRingElements =
    static_cast<size_t>(kConvTaps) * kQkvRows;
constexpr size_t kD232LinearConvWeightElements =
    static_cast<size_t>(kQkvRows) * kConvTaps;
constexpr size_t kD232FullQkvElements =
    static_cast<size_t>(kSuffixTokens) * kD231QkvRows;
constexpr size_t kD232FullQElements =
    static_cast<size_t>(kSuffixTokens) * kD231QFeatures;
constexpr size_t kD232FullSuffixKvElements =
    static_cast<size_t>(kSuffixTokens) * kD231KvFeatures;
constexpr size_t kD232FullKvElements =
    static_cast<size_t>(kFullTokens) * kD231KvFeatures;
constexpr size_t kD232FullPrefixKvElements =
    static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
constexpr size_t kD232CheckpointElements =
    static_cast<size_t>(kD232Layers) * kD232HiddenElements;

static_assert(kD232Layers == 40u);
static_assert(kD232LinearLayers == 30u);
static_assert(kD232FullLayers == 10u);
static_assert(kD232CheckpointElements == UINT64_C(83886080));

dim3 d232_linear_grid(size_t count) {
    return dim3(
        static_cast<unsigned int>((count + kThreads - 1u) / kThreads)
    );
}

__global__ void d232_fill_norm_banks_kernel(
    uint16_t *values,
    size_t count,
    unsigned int features,
    uint32_t seed,
    int direct_scale
) {
    const size_t index =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    const unsigned int layer =
        static_cast<unsigned int>(index / features);
    const unsigned int column =
        static_cast<unsigned int>(index % features);
    const int32_t centered = static_cast<int32_t>(
        (
            static_cast<uint32_t>(column) * 29u +
            static_cast<uint32_t>(layer) * 43u +
            seed
        ) & UINT32_C(0xff)
    ) - 128;
    const float delta = static_cast<float>(centered) * 0.0009765625f;
    values[index] = device_float_to_bf16(
        direct_scale != 0 ? 1.0f + delta : delta
    );
}

__global__ void d232_fill_float_kernel(
    float *values,
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
        static_cast<uint32_t>(index) * UINT32_C(2246822519) + seed;
    state ^= state >> 15u;
    state *= UINT32_C(668265263);
    state ^= state >> 13u;
    const int32_t centered =
        static_cast<int32_t>(state & UINT32_C(0xff)) - 128;
    values[index] = static_cast<float>(centered) * scale;
}

template <typename T>
__global__ void d232_fill_native_kernel(
    T *values,
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
        device_native_from_float<T>(static_cast<float>(centered) * scale);
}

bool d232_exact(const CompareStats &stats) {
    return stats.mismatches == 0u &&
        stats.nonfinite == 0u &&
        stats.max_abs == 0.0;
}

void d232_accumulate(
    CompareStats *aggregate,
    const CompareStats &part,
    size_t element_offset
) {
    if (aggregate == nullptr) {
        throw std::runtime_error("D232 aggregate destination is null");
    }
    if (
        aggregate->first_mismatch ==
            (std::numeric_limits<uint64_t>::max)() &&
        part.first_mismatch !=
            (std::numeric_limits<uint64_t>::max)()
    ) {
        aggregate->first_mismatch =
            static_cast<uint64_t>(element_offset) +
            part.first_mismatch;
    }
    aggregate->elements += part.elements;
    aggregate->mismatches += part.mismatches;
    aggregate->nonfinite += part.nonfinite;
    aggregate->max_abs =
        std::max(aggregate->max_abs, part.max_abs);
}

class D232LinearOwner {
public:
    D232LinearOwner(
        const ProviderApi &gdn,
        const MoeApi &moe,
        const MoeWeights &moe_weights,
        const RocblasContext &blas
    )
        : gdn_(gdn),
          moe_(moe),
          moe_weights_(moe_weights),
          blas_(blas),
          input_norm_weights_(
              static_cast<size_t>(kD232LinearLayers) * kLayerHidden
          ),
          gated_norm_weights_(
              static_cast<size_t>(kD232LinearLayers) * kValueDim
          ),
          postnorm_weights_(
              static_cast<size_t>(kD232LinearLayers) * kLayerHidden
          ),
          input_norm_f32_(kD232HiddenElements),
          input_norm_bf16_(kD232HiddenElements),
          qkv_weights_(
              static_cast<size_t>(kQkvRows) * kLayerHidden
          ),
          z_weights_(
              static_cast<size_t>(kZRows) * kLayerHidden
          ),
          a_weights_(
              static_cast<size_t>(kAbRows) * kLayerHidden
          ),
          b_weights_(
              static_cast<size_t>(kAbRows) * kLayerHidden
          ),
          out_weights_(
              static_cast<size_t>(kLayerHidden) * kValueFeatures
          ),
          conv_weights_(kD232LinearConvWeightElements),
          a_log_(kGateRows),
          dt_bias_(kGateRows),
          early_qkv_(kD232LinearQkvElements),
          early_z_(kD232LinearZElements),
          early_a_(kD232LinearAbElements),
          early_b_(kD232LinearAbElements),
          early_out_(kD232HiddenElements),
          retained_qkv_(kD232LinearQkvElements),
          retained_z_(kD232LinearZElements),
          retained_a_(kD232LinearAbElements),
          retained_b_(kD232LinearAbElements),
          retained_out_(kD232HiddenElements),
          z_f32_(kD232LinearZElements),
          a_f32_(kD232LinearAbElements),
          b_f32_(kD232LinearAbElements),
          gate_(kD232LinearGateElements),
          postconv_(kD232LinearQkvElements),
          core_(kD232LinearCoreElements),
          gated_(kD232LinearCoreElements),
          residual_(kD232HiddenElements),
          postnorm_(kD232HiddenElements),
          prefix_state_(
              static_cast<size_t>(kD232LinearLayers) *
              kD232LinearStateElements
          ),
          candidate_final_state_(
              static_cast<size_t>(kD232LinearLayers) *
              kD232LinearStateElements
          ),
          reference_final_state_(
              static_cast<size_t>(kD232LinearLayers) *
              kD232LinearStateElements
          ),
          early_prefix_ring_(
              static_cast<size_t>(kD232EarlyLinearLayers) *
              kD232LinearRingElements
          ),
          early_candidate_ring_(
              static_cast<size_t>(kD232EarlyLinearLayers) *
              kD232LinearRingElements
          ),
          early_reference_ring_(
              static_cast<size_t>(kD232EarlyLinearLayers) *
              kD232LinearRingElements
          ),
          retained_prefix_ring_(
              static_cast<size_t>(kD232RetainedLinearLayers) *
              kD232LinearRingElements
          ),
          retained_candidate_ring_(
              static_cast<size_t>(kD232RetainedLinearLayers) *
              kD232LinearRingElements
          ),
          retained_reference_ring_(
              static_cast<size_t>(kD232RetainedLinearLayers) *
              kD232LinearRingElements
          ) {}

    void initialize() {
        const dim3 block(kThreads);
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232LinearLayers) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            input_norm_weights_.get(),
            static_cast<size_t>(kD232LinearLayers) * kLayerHidden,
            kLayerHidden,
            UINT32_C(0x32),
            0
        );
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232LinearLayers) * kValueDim
            ),
            block,
            0u,
            nullptr,
            gated_norm_weights_.get(),
            static_cast<size_t>(kD232LinearLayers) * kValueDim,
            kValueDim,
            UINT32_C(0x47),
            1
        );
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232LinearLayers) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            postnorm_weights_.get(),
            static_cast<size_t>(kD232LinearLayers) * kLayerHidden,
            kLayerHidden,
            UINT32_C(0x59),
            0
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kQkvRows) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            qkv_weights_.get(),
            static_cast<size_t>(kQkvRows) * kLayerHidden,
            UINT32_C(0x51564b31),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kZRows) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            z_weights_.get(),
            static_cast<size_t>(kZRows) * kLayerHidden,
            UINT32_C(0x5a50524f),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kAbRows) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            a_weights_.get(),
            static_cast<size_t>(kAbRows) * kLayerHidden,
            UINT32_C(0x4150524f),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kAbRows) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            b_weights_.get(),
            static_cast<size_t>(kAbRows) * kLayerHidden,
            UINT32_C(0x4250524f),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kLayerHidden) * kValueFeatures
            ),
            block,
            0u,
            nullptr,
            out_weights_.get(),
            static_cast<size_t>(kLayerHidden) * kValueFeatures,
            UINT32_C(0x4f555450),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            fill_weights_kernel,
            d232_linear_grid(kD232LinearConvWeightElements),
            block,
            0u,
            nullptr,
            conv_weights_.get(),
            kD232LinearConvWeightElements
        );
        hipLaunchKernelGGL(
            fill_gate_parameters_kernel,
            dim3(1u),
            block,
            0u,
            nullptr,
            a_log_.get(),
            dt_bias_.get()
        );
        hipLaunchKernelGGL(
            d232_fill_float_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements
            ),
            block,
            0u,
            nullptr,
            prefix_state_.get(),
            static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements,
            UINT32_C(0x53544154),
            0.0000152587890625f
        );
        hipLaunchKernelGGL(
            d232_fill_native_kernel<float>,
            d232_linear_grid(
                static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements
            ),
            block,
            0u,
            nullptr,
            early_prefix_ring_.get(),
            static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements,
            UINT32_C(0x4541524c),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            d232_fill_native_kernel<uint16_t>,
            d232_linear_grid(
                static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements
            ),
            block,
            0u,
            nullptr,
            retained_prefix_ring_.get(),
            static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements,
            UINT32_C(0x52455441),
            0.000244140625f
        );
    }

    template <typename T, bool GateValuesAreDecay>
    void launch(
        unsigned int layer_index,
        unsigned int linear_ordinal,
        const float *hidden_input,
        float *hidden_output,
        bool reference
    ) {
        if (linear_ordinal >= kD232LinearLayers) {
            throw std::runtime_error("D232 linear ordinal is out of range");
        }
        if constexpr (std::is_same<T, float>::value) {
            if (layer_index >= kD232EarlyLinearLayers ||
                linear_ordinal >= kD232EarlyLinearLayers) {
                throw std::runtime_error(
                    "D232 F32 linear mode escaped layers zero and one"
                );
            }
        } else {
            if (layer_index < kD232EarlyLinearLayers ||
                linear_ordinal < kD232EarlyLinearLayers) {
                throw std::runtime_error(
                    "D232 BF16 linear mode entered an early layer"
                );
            }
        }

        T *qkv = nullptr;
        T *z_native = nullptr;
        T *a_native = nullptr;
        T *b_native = nullptr;
        T *out_native = nullptr;
        const T *prefix_ring = nullptr;
        T *final_ring = nullptr;
        if constexpr (std::is_same<T, float>::value) {
            qkv = early_qkv_.get();
            z_native = early_z_.get();
            a_native = early_a_.get();
            b_native = early_b_.get();
            out_native = early_out_.get();
            prefix_ring =
                early_prefix_ring_.get() +
                static_cast<size_t>(linear_ordinal) *
                    kD232LinearRingElements;
            final_ring =
                (reference
                    ? early_reference_ring_.get()
                    : early_candidate_ring_.get()) +
                static_cast<size_t>(linear_ordinal) *
                    kD232LinearRingElements;
        } else {
            qkv = retained_qkv_.get();
            z_native = retained_z_.get();
            a_native = retained_a_.get();
            b_native = retained_b_.get();
            out_native = retained_out_.get();
            const unsigned int retained_ordinal =
                linear_ordinal - kD232EarlyLinearLayers;
            prefix_ring =
                retained_prefix_ring_.get() +
                static_cast<size_t>(retained_ordinal) *
                    kD232LinearRingElements;
            final_ring =
                (reference
                    ? retained_reference_ring_.get()
                    : retained_candidate_ring_.get()) +
                static_cast<size_t>(retained_ordinal) *
                    kD232LinearRingElements;
        }

        const dim3 block(kThreads);
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            input_norm_weights_.get() +
                static_cast<size_t>(linear_ordinal) * kLayerHidden,
            input_norm_f32_.get()
        );
        hipLaunchKernelGGL(
            f32_to_bf16_sublayer_kernel,
            d232_linear_grid(kD232HiddenElements),
            block,
            0u,
            nullptr,
            input_norm_f32_.get(),
            input_norm_bf16_.get(),
            kD232HiddenElements
        );
        blas_.matmul(
            qkv_weights_.get(),
            input_norm_bf16_.get(),
            qkv,
            kQkvRows,
            kLayerHidden
        );
        blas_.matmul(
            z_weights_.get(),
            input_norm_bf16_.get(),
            z_native,
            kZRows,
            kLayerHidden
        );
        blas_.matmul(
            a_weights_.get(),
            input_norm_bf16_.get(),
            a_native,
            kAbRows,
            kLayerHidden
        );
        blas_.matmul(
            b_weights_.get(),
            input_norm_bf16_.get(),
            b_native,
            kAbRows,
            kLayerHidden
        );
        hipLaunchKernelGGL(
            native_to_f32_sublayer_kernel<T>,
            d232_linear_grid(kD232LinearZElements),
            block,
            0u,
            nullptr,
            z_native,
            z_f32_.get(),
            kD232LinearZElements
        );
        hipLaunchKernelGGL(
            native_to_f32_sublayer_kernel<T>,
            d232_linear_grid(kD232LinearAbElements),
            block,
            0u,
            nullptr,
            a_native,
            a_f32_.get(),
            kD232LinearAbElements
        );
        hipLaunchKernelGGL(
            native_to_f32_sublayer_kernel<T>,
            d232_linear_grid(kD232LinearAbElements),
            block,
            0u,
            nullptr,
            b_native,
            b_f32_.get(),
            kD232LinearAbElements
        );
        hipLaunchKernelGGL(
            gate_from_ab_kernel<GateValuesAreDecay>,
            dim3(1u, kSuffixTokens),
            block,
            0u,
            nullptr,
            a_f32_.get(),
            b_f32_.get(),
            a_log_.get(),
            dt_bias_.get(),
            gate_.get()
        );
        hipLaunchKernelGGL(
            suffix_halo_conv_kernel<T>,
            dim3(
                (kQkvRows + kThreads - 1u) / kThreads,
                kSuffixTokens
            ),
            block,
            0u,
            nullptr,
            prefix_ring,
            qkv,
            conv_weights_.get(),
            postconv_.get()
        );
        hipLaunchKernelGGL(
            postconv_qk_inplace_kernel,
            dim3(kSuffixTokens, kKeyHeads),
            dim3(kKeyDim),
            0u,
            nullptr,
            postconv_.get(),
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
            postconv_.get(),
            kSuffixTokens
        );
        float *final_state =
            (reference
                ? reference_final_state_.get()
                : candidate_final_state_.get()) +
            static_cast<size_t>(linear_ordinal) *
                kD232LinearStateElements;
        if (gdn_.q1024_seeded_launch_async(
                postconv_.get(),
                gate_.get(),
                prefix_state_.get() +
                    static_cast<size_t>(linear_ordinal) *
                        kD232LinearStateElements,
                core_.get(),
                final_state,
                GateValuesAreDecay ? 1 : 0,
                nullptr
            ) == 0) {
            throw std::runtime_error(
                std::string("D232 seeded GDN launch failed: ") +
                gdn_.last_error()
            );
        }
        hipLaunchKernelGGL(
            gated_rmsnorm_sublayer_kernel,
            dim3(kValueHeads, kSuffixTokens),
            dim3(kValueDim),
            0u,
            nullptr,
            core_.get(),
            z_f32_.get(),
            gated_norm_weights_.get() +
                static_cast<size_t>(linear_ordinal) * kValueDim,
            gated_.get()
        );
        blas_.matmul(
            out_weights_.get(),
            gated_.get(),
            out_native,
            kLayerHidden,
            kValueFeatures
        );
        hipLaunchKernelGGL(
            residual_postnorm_sublayer_kernel<T>,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            out_native,
            postnorm_weights_.get() +
                static_cast<size_t>(linear_ordinal) * kLayerHidden,
            residual_.get(),
            postnorm_.get()
        );
        hipLaunchKernelGGL(
            publish_final_ring_kernel<T>,
            d232_linear_grid(kD232LinearRingElements),
            block,
            0u,
            nullptr,
            qkv,
            final_ring
        );
        launch_moe(postnorm_.get(), residual_.get(), hidden_output, reference);
    }

    const float *prefix_state() const {
        return prefix_state_.get();
    }

    const float *candidate_final_state() const {
        return candidate_final_state_.get();
    }

    const float *reference_final_state() const {
        return reference_final_state_.get();
    }

    const float *early_prefix_ring() const {
        return early_prefix_ring_.get();
    }

    const float *early_candidate_ring() const {
        return early_candidate_ring_.get();
    }

    const float *early_reference_ring() const {
        return early_reference_ring_.get();
    }

    const uint16_t *retained_prefix_ring() const {
        return retained_prefix_ring_.get();
    }

    const uint16_t *retained_candidate_ring() const {
        return retained_candidate_ring_.get();
    }

    const uint16_t *retained_reference_ring() const {
        return retained_reference_ring_.get();
    }

private:
    void launch_moe(
        const float *postnorm,
        const float *residual,
        float *output,
        bool reference
    ) {
        const int launched =
            reference
                ? moe_.launch_v2(
                      postnorm,
                      residual,
                      moe_weights_.router.get(),
                      moe_weights_.routed_gate_up.get(),
                      moe_weights_.routed_down.get(),
                      moe_weights_.shared_gate.get(),
                      moe_weights_.shared_gate_projection.get(),
                      moe_weights_.shared_up_projection.get(),
                      moe_weights_.shared_down.get(),
                      output,
                      nullptr
                  )
                : moe_.launch_v3_async(
                      postnorm,
                      residual,
                      moe_weights_.router.get(),
                      moe_weights_.routed_gate_up.get(),
                      moe_weights_.routed_down.get(),
                      moe_weights_.shared_gate.get(),
                      moe_weights_.shared_gate_projection.get(),
                      moe_weights_.shared_up_projection.get(),
                      moe_weights_.shared_down.get(),
                      output,
                      nullptr
                  );
        if (launched == 0) {
            throw std::runtime_error(
                std::string("D232 linear MoE launch failed: ") +
                moe_.last_error()
            );
        }
    }

    const ProviderApi &gdn_;
    const MoeApi &moe_;
    const MoeWeights &moe_weights_;
    const RocblasContext &blas_;
    DeviceBuffer<uint16_t> input_norm_weights_;
    DeviceBuffer<uint16_t> gated_norm_weights_;
    DeviceBuffer<uint16_t> postnorm_weights_;
    DeviceBuffer<float> input_norm_f32_;
    DeviceBuffer<uint16_t> input_norm_bf16_;
    DeviceBuffer<uint16_t> qkv_weights_;
    DeviceBuffer<uint16_t> z_weights_;
    DeviceBuffer<uint16_t> a_weights_;
    DeviceBuffer<uint16_t> b_weights_;
    DeviceBuffer<uint16_t> out_weights_;
    DeviceBuffer<uint16_t> conv_weights_;
    DeviceBuffer<uint16_t> a_log_;
    DeviceBuffer<uint16_t> dt_bias_;
    DeviceBuffer<float> early_qkv_;
    DeviceBuffer<float> early_z_;
    DeviceBuffer<float> early_a_;
    DeviceBuffer<float> early_b_;
    DeviceBuffer<float> early_out_;
    DeviceBuffer<uint16_t> retained_qkv_;
    DeviceBuffer<uint16_t> retained_z_;
    DeviceBuffer<uint16_t> retained_a_;
    DeviceBuffer<uint16_t> retained_b_;
    DeviceBuffer<uint16_t> retained_out_;
    DeviceBuffer<float> z_f32_;
    DeviceBuffer<float> a_f32_;
    DeviceBuffer<float> b_f32_;
    DeviceBuffer<float> gate_;
    DeviceBuffer<float> postconv_;
    DeviceBuffer<float> core_;
    DeviceBuffer<uint16_t> gated_;
    DeviceBuffer<float> residual_;
    DeviceBuffer<float> postnorm_;
    DeviceBuffer<float> prefix_state_;
    DeviceBuffer<float> candidate_final_state_;
    DeviceBuffer<float> reference_final_state_;
    DeviceBuffer<float> early_prefix_ring_;
    DeviceBuffer<float> early_candidate_ring_;
    DeviceBuffer<float> early_reference_ring_;
    DeviceBuffer<uint16_t> retained_prefix_ring_;
    DeviceBuffer<uint16_t> retained_candidate_ring_;
    DeviceBuffer<uint16_t> retained_reference_ring_;
};

class D232FullOwner {
public:
    D232FullOwner(
        const CkApi &ck,
        const MoeApi &moe,
        const MoeWeights &moe_weights,
        const RocblasContext &blas
    )
        : ck_(ck),
          moe_(moe),
          moe_weights_(moe_weights),
          blas_(blas),
          input_norm_weights_(
              static_cast<size_t>(kD232FullLayers) * kLayerHidden
          ),
          postnorm_weights_(
              static_cast<size_t>(kD232FullLayers) * kLayerHidden
          ),
          q_norm_weights_(
              static_cast<size_t>(kD232FullLayers) * kD231HeadDim
          ),
          k_norm_weights_(
              static_cast<size_t>(kD232FullLayers) * kD231HeadDim
          ),
          input_norm_f32_(kD232HiddenElements),
          input_norm_bf16_(kD232HiddenElements),
          qkv_weights_(
              static_cast<size_t>(kD231QkvRows) * kLayerHidden
          ),
          qkv_(kD232FullQkvElements),
          q_(kD232FullQElements),
          gate_(kD232FullQElements),
          full_k_(
              static_cast<size_t>(kD232FullLayers) *
              kD232FullKvElements
          ),
          full_v_(
              static_cast<size_t>(kD232FullLayers) *
              kD232FullKvElements
          ),
          context_(kD232FullQElements),
          gated_(kD232FullQElements),
          out_weights_(
              static_cast<size_t>(kLayerHidden) * kD231QFeatures
          ),
          out_(kD232HiddenElements),
          residual_(kD232HiddenElements),
          postnorm_(kD232HiddenElements) {}

    void initialize() {
        const dim3 block(kThreads);
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232FullLayers) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            input_norm_weights_.get(),
            static_cast<size_t>(kD232FullLayers) * kLayerHidden,
            kLayerHidden,
            UINT32_C(0x63),
            0
        );
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232FullLayers) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            postnorm_weights_.get(),
            static_cast<size_t>(kD232FullLayers) * kLayerHidden,
            kLayerHidden,
            UINT32_C(0x75),
            0
        );
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232FullLayers) * kD231HeadDim
            ),
            block,
            0u,
            nullptr,
            q_norm_weights_.get(),
            static_cast<size_t>(kD232FullLayers) * kD231HeadDim,
            kD231HeadDim,
            UINT32_C(0x87),
            0
        );
        hipLaunchKernelGGL(
            d232_fill_norm_banks_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD232FullLayers) * kD231HeadDim
            ),
            block,
            0u,
            nullptr,
            k_norm_weights_.get(),
            static_cast<size_t>(kD232FullLayers) * kD231HeadDim,
            kD231HeadDim,
            UINT32_C(0x99),
            0
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kD231QkvRows) * kLayerHidden
            ),
            block,
            0u,
            nullptr,
            qkv_weights_.get(),
            static_cast<size_t>(kD231QkvRows) * kLayerHidden,
            UINT32_C(0x44323331),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            fill_dense_weights_kernel,
            d232_linear_grid(
                static_cast<size_t>(kLayerHidden) * kD231QFeatures
            ),
            block,
            0u,
            nullptr,
            out_weights_.get(),
            static_cast<size_t>(kLayerHidden) * kD231QFeatures,
            UINT32_C(0x4f323331),
            0.000244140625f
        );
        hipLaunchKernelGGL(
            d232_fill_native_kernel<uint16_t>,
            d232_linear_grid(
                static_cast<size_t>(kD232FullLayers) *
                kD232FullKvElements
            ),
            block,
            0u,
            nullptr,
            full_k_.get(),
            static_cast<size_t>(kD232FullLayers) *
                kD232FullKvElements,
            UINT32_C(0x4b323332),
            0.001953125f
        );
        hipLaunchKernelGGL(
            d232_fill_native_kernel<uint16_t>,
            d232_linear_grid(
                static_cast<size_t>(kD232FullLayers) *
                kD232FullKvElements
            ),
            block,
            0u,
            nullptr,
            full_v_.get(),
            static_cast<size_t>(kD232FullLayers) *
                kD232FullKvElements,
            UINT32_C(0x56323332),
            0.001953125f
        );
    }

    void launch(
        unsigned int full_ordinal,
        const float *hidden_input,
        float *hidden_output,
        bool reference
    ) {
        if (full_ordinal >= kD232FullLayers) {
            throw std::runtime_error("D232 full ordinal is out of range");
        }
        const dim3 block(kThreads);
        hipLaunchKernelGGL(
            input_rmsnorm_kernel,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            input_norm_weights_.get() +
                static_cast<size_t>(full_ordinal) * kLayerHidden,
            input_norm_f32_.get()
        );
        hipLaunchKernelGGL(
            f32_to_bf16_sublayer_kernel,
            d232_linear_grid(kD232HiddenElements),
            block,
            0u,
            nullptr,
            input_norm_f32_.get(),
            input_norm_bf16_.get(),
            kD232HiddenElements
        );
        blas_.matmul(
            qkv_weights_.get(),
            input_norm_bf16_.get(),
            qkv_.get(),
            kD231QkvRows,
            kLayerHidden
        );
        uint16_t *layer_k =
            full_k_.get() +
            static_cast<size_t>(full_ordinal) * kD232FullKvElements;
        uint16_t *layer_v =
            full_v_.get() +
            static_cast<size_t>(full_ordinal) * kD232FullKvElements;
        uint16_t *suffix_k =
            layer_k +
            static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
        uint16_t *suffix_v =
            layer_v +
            static_cast<size_t>(kPrefixTokens) * kD231KvFeatures;
        hipLaunchKernelGGL(
            d231_prepare_q_gate_kernel,
            dim3(kD231QueryHeads, kSuffixTokens),
            dim3(kD231HeadDim),
            0u,
            nullptr,
            qkv_.get(),
            q_norm_weights_.get() +
                static_cast<size_t>(full_ordinal) * kD231HeadDim,
            q_.get(),
            gate_.get()
        );
        hipLaunchKernelGGL(
            d231_prepare_k_v_kernel,
            dim3(kD231KvHeads, kSuffixTokens),
            dim3(kD231HeadDim),
            0u,
            nullptr,
            qkv_.get(),
            k_norm_weights_.get() +
                static_cast<size_t>(full_ordinal) * kD231HeadDim,
            suffix_k,
            suffix_v
        );
        hipLaunchKernelGGL(
            d231_rope_bf16_kernel,
            dim3(kD231QueryHeads, kSuffixTokens),
            dim3(kD231RotaryDim / 2u),
            0u,
            nullptr,
            q_.get(),
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
        if (ck_.suffix_launch(
                q_.get(),
                layer_k,
                layer_v,
                context_.get(),
                nullptr
            ) != 0) {
            throw std::runtime_error(
                "D232 CK suffix attention launch failed"
            );
        }
        hipLaunchKernelGGL(
            d231_attention_gate_kernel,
            d232_linear_grid(kD232FullQElements),
            block,
            0u,
            nullptr,
            context_.get(),
            gate_.get(),
            gated_.get(),
            kD232FullQElements
        );
        blas_.matmul(
            out_weights_.get(),
            gated_.get(),
            out_.get(),
            kLayerHidden,
            kD231QFeatures
        );
        hipLaunchKernelGGL(
            residual_postnorm_sublayer_kernel<uint16_t>,
            dim3(kSuffixTokens),
            block,
            0u,
            nullptr,
            hidden_input,
            out_.get(),
            postnorm_weights_.get() +
                static_cast<size_t>(full_ordinal) * kLayerHidden,
            residual_.get(),
            postnorm_.get()
        );
        launch_moe(postnorm_.get(), residual_.get(), hidden_output, reference);
    }

    const uint16_t *layer_k(unsigned int ordinal) const {
        return full_k_.get() +
            static_cast<size_t>(ordinal) * kD232FullKvElements;
    }

    const uint16_t *layer_v(unsigned int ordinal) const {
        return full_v_.get() +
            static_cast<size_t>(ordinal) * kD232FullKvElements;
    }

    const float *residual() const {
        return residual_.get();
    }

    const float *postnorm() const {
        return postnorm_.get();
    }

private:
    void launch_moe(
        const float *postnorm,
        const float *residual,
        float *output,
        bool reference
    ) {
        const int launched =
            reference
                ? moe_.launch_v2(
                      postnorm,
                      residual,
                      moe_weights_.router.get(),
                      moe_weights_.routed_gate_up.get(),
                      moe_weights_.routed_down.get(),
                      moe_weights_.shared_gate.get(),
                      moe_weights_.shared_gate_projection.get(),
                      moe_weights_.shared_up_projection.get(),
                      moe_weights_.shared_down.get(),
                      output,
                      nullptr
                  )
                : moe_.launch_v3_async(
                      postnorm,
                      residual,
                      moe_weights_.router.get(),
                      moe_weights_.routed_gate_up.get(),
                      moe_weights_.routed_down.get(),
                      moe_weights_.shared_gate.get(),
                      moe_weights_.shared_gate_projection.get(),
                      moe_weights_.shared_up_projection.get(),
                      moe_weights_.shared_down.get(),
                      output,
                      nullptr
                  );
        if (launched == 0) {
            throw std::runtime_error(
                std::string("D232 full MoE launch failed: ") +
                moe_.last_error()
            );
        }
    }

    const CkApi &ck_;
    const MoeApi &moe_;
    const MoeWeights &moe_weights_;
    const RocblasContext &blas_;
    DeviceBuffer<uint16_t> input_norm_weights_;
    DeviceBuffer<uint16_t> postnorm_weights_;
    DeviceBuffer<uint16_t> q_norm_weights_;
    DeviceBuffer<uint16_t> k_norm_weights_;
    DeviceBuffer<float> input_norm_f32_;
    DeviceBuffer<uint16_t> input_norm_bf16_;
    DeviceBuffer<uint16_t> qkv_weights_;
    DeviceBuffer<uint16_t> qkv_;
    DeviceBuffer<uint16_t> q_;
    DeviceBuffer<uint16_t> gate_;
    DeviceBuffer<uint16_t> full_k_;
    DeviceBuffer<uint16_t> full_v_;
    DeviceBuffer<float> context_;
    DeviceBuffer<uint16_t> gated_;
    DeviceBuffer<uint16_t> out_weights_;
    DeviceBuffer<uint16_t> out_;
    DeviceBuffer<float> residual_;
    DeviceBuffer<float> postnorm_;
};

std::vector<uint16_t> d232_copy_full_prefix(
    const D232FullOwner &owner,
    bool copy_k
) {
    std::vector<uint16_t> result(
        static_cast<size_t>(kD232FullLayers) *
        kD232FullPrefixKvElements
    );
    for (unsigned int ordinal = 0u;
         ordinal < kD232FullLayers;
         ++ordinal) {
        const uint16_t *source =
            copy_k ? owner.layer_k(ordinal) : owner.layer_v(ordinal);
        check_hip(
            hipMemcpy(
                result.data() +
                    static_cast<size_t>(ordinal) *
                        kD232FullPrefixKvElements,
                source,
                kD232FullPrefixKvElements * sizeof(uint16_t),
                hipMemcpyDeviceToHost
            ),
            "D232 full prefix copy"
        );
    }
    return result;
}

std::vector<uint16_t> d232_copy_full_suffix(
    const D232FullOwner &owner,
    bool copy_k
) {
    std::vector<uint16_t> result(
        static_cast<size_t>(kD232FullLayers) *
        kD232FullSuffixKvElements
    );
    for (unsigned int ordinal = 0u;
         ordinal < kD232FullLayers;
         ++ordinal) {
        const uint16_t *layer =
            copy_k ? owner.layer_k(ordinal) : owner.layer_v(ordinal);
        check_hip(
            hipMemcpy(
                result.data() +
                    static_cast<size_t>(ordinal) *
                        kD232FullSuffixKvElements,
                layer +
                    static_cast<size_t>(kPrefixTokens) *
                        kD231KvFeatures,
                kD232FullSuffixKvElements * sizeof(uint16_t),
                hipMemcpyDeviceToHost
            ),
            "D232 full suffix copy"
        );
    }
    return result;
}

struct D232Result {
    double ordered_mean_ms = 0.0;
    double baseline_ratio = 0.0;
    bool pass = false;
};

D232Result run_d232(
    const ProviderApi &gdn,
    const CkApi &ck,
    const MoeApi &moe,
    const MoeWeights &moe_weights,
    unsigned int repetitions,
    double same_run_d231_combined_ms
) {
    DeviceBuffer<float> initial_hidden(kD232HiddenElements);
    DeviceBuffer<float> work_a(kD232HiddenElements);
    DeviceBuffer<float> work_b(kD232HiddenElements);
    DeviceBuffer<float> candidate_checkpoints(kD232CheckpointElements);
    DeviceBuffer<float> reference_checkpoints(kD232CheckpointElements);

    hipLaunchKernelGGL(
        fill_hidden_kernel,
        d232_linear_grid(kD232HiddenElements),
        dim3(kThreads),
        0u,
        nullptr,
        initial_hidden.get(),
        kD232HiddenElements
    );

    RocblasContext blas;
    D232LinearOwner linear(gdn, moe, moe_weights, blas);
    D232FullOwner full(ck, moe, moe_weights, blas);
    linear.initialize();
    full.initialize();
    check_hip(hipGetLastError(), "D232 fixture initialization");
    check_hip(
        hipDeviceSynchronize(),
        "D232 fixture initialization synchronize"
    );

    const std::vector<float> initial_hidden_before =
        copy_device(initial_hidden.get(), kD232HiddenElements);
    const std::vector<float> linear_state_before =
        copy_device(
            linear.prefix_state(),
            static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements
        );
    const std::vector<float> early_ring_before =
        copy_device(
            linear.early_prefix_ring(),
            static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements
        );
    const std::vector<uint16_t> retained_ring_before =
        copy_device(
            linear.retained_prefix_ring(),
            static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements
        );
    const std::vector<uint16_t> full_k_prefix_before =
        d232_copy_full_prefix(full, true);
    const std::vector<uint16_t> full_v_prefix_before =
        d232_copy_full_prefix(full, false);

    auto launch_sequence = [&](
        bool reference,
        float *checkpoints
    ) {
        check_hip(
            hipMemcpyAsync(
                work_a.get(),
                initial_hidden.get(),
                kD232HiddenElements * sizeof(float),
                hipMemcpyDeviceToDevice,
                nullptr
            ),
            "D232 initial hidden reset"
        );
        float *current = work_a.get();
        float *next = work_b.get();
        unsigned int linear_ordinal = 0u;
        unsigned int full_ordinal = 0u;
        for (unsigned int layer = 0u; layer < kD232Layers; ++layer) {
            if ((layer % 4u) == 3u) {
                full.launch(
                    full_ordinal,
                    current,
                    next,
                    reference
                );
                ++full_ordinal;
            } else {
                if (layer < kD232EarlyLinearLayers) {
                    linear.launch<float, true>(
                        layer,
                        linear_ordinal,
                        current,
                        next,
                        reference
                    );
                } else {
                    linear.launch<uint16_t, false>(
                        layer,
                        linear_ordinal,
                        current,
                        next,
                        reference
                    );
                }
                ++linear_ordinal;
            }
            if (checkpoints != nullptr) {
                check_hip(
                    hipMemcpyAsync(
                        checkpoints +
                            static_cast<size_t>(layer) *
                                kD232HiddenElements,
                        next,
                        kD232HiddenElements * sizeof(float),
                        hipMemcpyDeviceToDevice,
                        nullptr
                    ),
                    "D232 layer checkpoint"
                );
            }
            std::swap(current, next);
        }
        if (linear_ordinal != kD232LinearLayers ||
            full_ordinal != kD232FullLayers ||
            current != work_a.get()) {
            throw std::runtime_error(
                "D232 ordered layer accounting failed"
            );
        }
    };

    launch_sequence(false, candidate_checkpoints.get());
    check_hip(hipGetLastError(), "D232 candidate replay launch");
    check_hip(
        hipDeviceSynchronize(),
        "D232 candidate replay synchronize"
    );
    const std::vector<uint16_t> candidate_suffix_k =
        d232_copy_full_suffix(full, true);
    const std::vector<uint16_t> candidate_suffix_v =
        d232_copy_full_suffix(full, false);

    launch_sequence(true, reference_checkpoints.get());
    check_hip(hipGetLastError(), "D232 reference replay launch");
    check_hip(
        hipDeviceSynchronize(),
        "D232 reference replay synchronize"
    );

    CompareStats checkpoint_stats;
    uint64_t first_bad_layer =
        (std::numeric_limits<uint64_t>::max)();
    for (unsigned int layer = 0u; layer < kD232Layers; ++layer) {
        const std::vector<float> candidate =
            copy_device(
                candidate_checkpoints.get() +
                    static_cast<size_t>(layer) *
                        kD232HiddenElements,
                kD232HiddenElements
            );
        const std::vector<float> reference =
            copy_device(
                reference_checkpoints.get() +
                    static_cast<size_t>(layer) *
                        kD232HiddenElements,
                kD232HiddenElements
            );
        const CompareStats layer_stats =
            compare_f32(reference, candidate);
        if (!d232_exact(layer_stats) &&
            first_bad_layer ==
                (std::numeric_limits<uint64_t>::max)()) {
            first_bad_layer = layer;
        }
        d232_accumulate(
            &checkpoint_stats,
            layer_stats,
            static_cast<size_t>(layer) * kD232HiddenElements
        );
    }

    const CompareStats final_state_stats = compare_f32(
        copy_device(
            linear.reference_final_state(),
            static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements
        ),
        copy_device(
            linear.candidate_final_state(),
            static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements
        )
    );
    const CompareStats early_ring_stats = compare_f32(
        copy_device(
            linear.early_reference_ring(),
            static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements
        ),
        copy_device(
            linear.early_candidate_ring(),
            static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements
        )
    );
    const CompareStats retained_ring_stats = compare_native(
        copy_device(
            linear.retained_reference_ring(),
            static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements
        ),
        copy_device(
            linear.retained_candidate_ring(),
            static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements
        )
    );
    CompareStats final_ring_stats;
    d232_accumulate(&final_ring_stats, early_ring_stats, 0u);
    d232_accumulate(
        &final_ring_stats,
        retained_ring_stats,
        static_cast<size_t>(kD232EarlyLinearLayers) *
            kD232LinearRingElements
    );
    const CompareStats suffix_k_stats = compare_native(
        d232_copy_full_suffix(full, true),
        candidate_suffix_k
    );
    const CompareStats suffix_v_stats = compare_native(
        d232_copy_full_suffix(full, false),
        candidate_suffix_v
    );

    launch_sequence(false, nullptr);
    check_hip(hipGetLastError(), "D232 warmup launch");
    check_hip(
        hipDeviceSynchronize(),
        "D232 warmup synchronize"
    );
    hipEvent_t start = nullptr;
    hipEvent_t stop = nullptr;
    check_hip(hipEventCreate(&start), "D232 event create(start)");
    check_hip(hipEventCreate(&stop), "D232 event create(stop)");
    double total_measured_ms = 0.0;
    for (unsigned int repetition = 0u;
         repetition < repetitions;
         ++repetition) {
        check_hip(
            hipEventRecord(start, nullptr),
            "D232 event record(start)"
        );
        launch_sequence(false, nullptr);
        check_hip(
            hipEventRecord(stop, nullptr),
            "D232 event record(stop)"
        );
        check_hip(
            hipEventSynchronize(stop),
            "D232 event synchronize(stop)"
        );
        float repetition_ms = 0.0f;
        check_hip(
            hipEventElapsedTime(&repetition_ms, start, stop),
            "D232 event elapsed"
        );
        total_measured_ms +=
            static_cast<double>(repetition_ms);
    }
    check_hip(hipGetLastError(), "D232 timed sequence launch");
    (void)hipEventDestroy(stop);
    (void)hipEventDestroy(start);
    const double ordered_mean_ms =
        total_measured_ms /
        static_cast<double>(repetitions);
    const double baseline_ratio =
        ordered_mean_ms / same_run_d231_combined_ms;

    const CompareStats initial_hidden_mutation = compare_f32(
        initial_hidden_before,
        copy_device(initial_hidden.get(), kD232HiddenElements)
    );
    const CompareStats linear_state_mutation = compare_f32(
        linear_state_before,
        copy_device(
            linear.prefix_state(),
            static_cast<size_t>(kD232LinearLayers) *
                kD232LinearStateElements
        )
    );
    const CompareStats early_ring_mutation = compare_f32(
        early_ring_before,
        copy_device(
            linear.early_prefix_ring(),
            static_cast<size_t>(kD232EarlyLinearLayers) *
                kD232LinearRingElements
        )
    );
    const CompareStats retained_ring_mutation = compare_native(
        retained_ring_before,
        copy_device(
            linear.retained_prefix_ring(),
            static_cast<size_t>(kD232RetainedLinearLayers) *
                kD232LinearRingElements
        )
    );
    CompareStats linear_ring_mutation;
    d232_accumulate(&linear_ring_mutation, early_ring_mutation, 0u);
    d232_accumulate(
        &linear_ring_mutation,
        retained_ring_mutation,
        static_cast<size_t>(kD232EarlyLinearLayers) *
            kD232LinearRingElements
    );
    const CompareStats full_k_mutation = compare_native(
        full_k_prefix_before,
        d232_copy_full_prefix(full, true)
    );
    const CompareStats full_v_mutation = compare_native(
        full_v_prefix_before,
        d232_copy_full_prefix(full, false)
    );

    if (moe.launch_router_debug(
            full.postnorm(),
            moe_weights.router.get(),
            nullptr
        ) == 0) {
        throw std::runtime_error(
            std::string("D232 router debug failed: ") +
            moe.last_error()
        );
    }
    check_hip(
        hipDeviceSynchronize(),
        "D232 router debug synchronize"
    );
    std::vector<uint32_t> topk_ids(kD230Routes);
    std::vector<float> topk_weights(kD230Routes);
    if (moe.copy_topk_debug(
            topk_ids.data(),
            topk_weights.data()
        ) == 0) {
        throw std::runtime_error("D232 router debug copy failed");
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

    const std::vector<float> final_output =
        copy_device(work_a.get(), kD232HiddenElements);
    const std::vector<float> final_residual =
        copy_device(full.residual(), kD232HiddenElements);
    const std::vector<float> final_postnorm =
        copy_device(full.postnorm(), kD232HiddenElements);
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
    uint64_t final_samples = 0u;
    uint64_t final_sample_nonfinite = 0u;
    double final_sample_max_abs = 0.0;
    for (unsigned int token : sample_tokens) {
        const float *token_postnorm =
            final_postnorm.data() +
            static_cast<size_t>(token) * kLayerHidden;
        for (unsigned int column : sample_columns) {
            const size_t index =
                static_cast<size_t>(token) * kLayerHidden + column;
            const float reference = independent_output_reference(
                token_postnorm,
                final_residual[index]
            );
            const float candidate = final_output[index];
            if (!std::isfinite(reference) ||
                !std::isfinite(candidate)) {
                ++final_sample_nonfinite;
            } else {
                final_sample_max_abs = std::max(
                    final_sample_max_abs,
                    std::abs(
                        static_cast<double>(reference) -
                        static_cast<double>(candidate)
                    )
                );
            }
            ++final_samples;
        }
    }
    uint64_t final_output_nonfinite = 0u;
    for (float value : final_output) {
        final_output_nonfinite += std::isfinite(value) ? 0u : 1u;
    }

    const bool pass =
        checkpoint_stats.elements == kD232CheckpointElements &&
        d232_exact(checkpoint_stats) &&
        d232_exact(final_state_stats) &&
        d232_exact(final_ring_stats) &&
        d232_exact(suffix_k_stats) &&
        d232_exact(suffix_v_stats) &&
        d232_exact(initial_hidden_mutation) &&
        d232_exact(linear_state_mutation) &&
        d232_exact(linear_ring_mutation) &&
        d232_exact(full_k_mutation) &&
        d232_exact(full_v_mutation) &&
        router_id_mismatches == 0u &&
        router_weight_mismatches == 0u &&
        final_samples >= 16u &&
        final_sample_nonfinite == 0u &&
        final_sample_max_abs <= kD232SampleCeiling &&
        final_output_nonfinite == 0u &&
        ordered_mean_ms <= kD232MeanCeilingMs &&
        baseline_ratio <= kD232BaselineRatioCeiling;

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_ordered_forty_layer_owner_smoke"
        << " layers=" << kD232Layers
        << " linear_layers=" << kD232LinearLayers
        << " full_attention_layers=" << kD232FullLayers
        << " first_full_layer=3"
        << " last_full_layer=39"
        << " early_f32_linear_layers=" << kD232EarlyLinearLayers
        << " retained_bf16_linear_layers="
        << kD232RetainedLinearLayers
        << " device_handoffs=39"
        << " host_staging_calls=0"
        << " inter_layer_host_synchronizations=0"
        << " moe_launches_per_transaction=40"
        << " checkpoint_elements=" << checkpoint_stats.elements
        << " checkpoint_mismatches=" << checkpoint_stats.mismatches
        << " checkpoint_nonfinite=" << checkpoint_stats.nonfinite
        << " checkpoint_max_abs=" << checkpoint_stats.max_abs
        << " first_bad_layer=";
    if (first_bad_layer ==
        (std::numeric_limits<uint64_t>::max)()) {
        std::cout << "none";
    } else {
        std::cout << first_bad_layer;
    }
    std::cout
        << " linear_final_state_elements=" << final_state_stats.elements
        << " linear_final_state_mismatches="
        << final_state_stats.mismatches
        << " linear_final_state_nonfinite="
        << final_state_stats.nonfinite
        << " linear_final_state_max_abs="
        << final_state_stats.max_abs
        << " linear_final_ring_elements=" << final_ring_stats.elements
        << " linear_final_ring_mismatches="
        << final_ring_stats.mismatches
        << " linear_final_ring_nonfinite="
        << final_ring_stats.nonfinite
        << " linear_final_ring_max_abs="
        << final_ring_stats.max_abs
        << " full_suffix_k_elements=" << suffix_k_stats.elements
        << " full_suffix_k_mismatches=" << suffix_k_stats.mismatches
        << " full_suffix_v_elements=" << suffix_v_stats.elements
        << " full_suffix_v_mismatches=" << suffix_v_stats.mismatches
        << " initial_hidden_mutations="
        << initial_hidden_mutation.mismatches
        << " linear_prefix_state_mutations="
        << linear_state_mutation.mismatches
        << " linear_prefix_ring_mutations="
        << linear_ring_mutation.mismatches
        << " full_prefix_k_mutations=" << full_k_mutation.mismatches
        << " full_prefix_v_mutations=" << full_v_mutation.mismatches
        << " router_id_mismatches=" << router_id_mismatches
        << " router_weight_mismatches="
        << router_weight_mismatches
        << " final_output_elements=" << final_output.size()
        << " final_output_nonfinite=" << final_output_nonfinite
        << " final_reference_samples=" << final_samples
        << " final_reference_nonfinite=" << final_sample_nonfinite
        << " final_reference_max_abs=" << final_sample_max_abs
        << " output_hash=" << std::hex
        << fnv1a64(
               final_output.data(),
               final_output.size() * sizeof(float)
           )
        << std::dec
        << " repetitions=" << repetitions
        << " ordered_forty_layer_mean_ms=" << ordered_mean_ms
        << " ordered_forty_layer_mean_ceiling_ms="
        << kD232MeanCeilingMs
        << " same_run_d231_combined_fixture_ms="
        << same_run_d231_combined_ms
        << " same_run_d231_ratio=" << baseline_ratio
        << " same_run_d231_ratio_ceiling="
        << kD232BaselineRatioCeiling
        << " materialized_active_experts=" << kD230ActiveExperts
        << " all_model_layer_weights_resident=0"
        << " all_256_expert_weights_resident=0"
        << " weight_bits=16"
        << " activation_bits=16"
        << " accumulation_bits=32"
        << " quantized=0"
        << " mtp_active=0"
        << " dflash_active=0"
        << " speculative_decode=0"
        << " ordered_forty_layer_owner_claimed=1"
        << " real_model_loaded=0"
        << " correctness_boundary_attached=0"
        << " product_metric_valid=0"
        << " product_performance_accepted=0"
        << " inference_success_claimed=0"
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    std::cout
        << std::fixed << std::setprecision(6)
        << "q16384_suffix1024_ordered_forty_layer_owner_summary"
        << " ordered_layers=40"
        << " ordered_forty_layer_mean_ms=" << ordered_mean_ms
        << " same_run_d231_combined_fixture_ms="
        << same_run_d231_combined_ms
        << " same_run_d231_ratio=" << baseline_ratio
        << " checkpoint_layers=40"
        << " checkpoint_elements=" << checkpoint_stats.elements
        << " device_handoffs=39"
        << " host_staging_calls=0"
        << " inter_layer_host_synchronizations=0"
        << " all_forty_layers_claimed=1"
        << " all_model_layer_weights_resident=0"
        << " real_model_loaded=0"
        << " correctness_boundary_attached=0"
        << " product_metric_valid=0"
        << " product_performance_accepted=0"
        << " inference_success_claimed=0"
        << " pass=" << (pass ? 1 : 0)
        << "\n";

    return D232Result{
        ordered_mean_ms,
        baseline_ratio,
        pass,
    };
}

}  // namespace

#ifndef QRT_D232_DISABLE_MAIN
int main(int argc, char **argv) {
    ProviderApi gdn;
    CkApi ck;
    MoeApi moe;
    try {
        if (argc != 8) {
            std::cerr
                << "usage: "
                << "q16384_suffix1024_ordered_forty_layer_owner_smoke "
                << "GDN_KERNEL_DIR GDN_PROVIDER_DLL CK_PROVIDER_DLL "
                << "MOE_KERNEL_DIR MOE_PROVIDER_DLL REPETITIONS>=5 "
                << "SAME_RUN_D231_COMBINED_FIXTURE_MS\n";
            return 2;
        }
        const int parsed_repetitions = std::stoi(argv[6]);
        if (parsed_repetitions <
                static_cast<int>(kD232MinimumRepetitions) ||
            parsed_repetitions > 20) {
            std::cerr << "repetitions must be in [5,20]\n";
            return 2;
        }
        const double same_run_d231_combined_ms = std::stod(argv[7]);
        if (!std::isfinite(same_run_d231_combined_ms) ||
            same_run_d231_combined_ms <= 0.0) {
            std::cerr
                << "SAME_RUN_D231_COMBINED_FIXTURE_MS must be finite "
                   "and positive\n";
            return 2;
        }
        int device_count = 0;
        check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
        if (device_count <= 0) {
            throw std::runtime_error("no HIP device is available");
        }
        check_hip(hipSetDevice(0), "hipSetDevice");
        if (!load_provider(argv[2], &gdn)) {
            throw std::runtime_error(
                "could not load all required GDN provider exports"
            );
        }
        if (gdn.prepare(argv[1]) == 0) {
            throw std::runtime_error(
                std::string("GDN provider prepare failed: ") +
                gdn.last_error()
            );
        }
        if (!load_ck_provider(argv[3], &ck)) {
            throw std::runtime_error(
                "could not load all required CK provider exports"
            );
        }
        if (ck.prepare() != 0) {
            throw std::runtime_error("CK provider prepare failed");
        }
        if (!load_moe_provider(argv[5], &moe)) {
            throw std::runtime_error(
                "could not load all required MoE provider exports"
            );
        }
        if (moe.prepare(argv[4]) == 0) {
            throw std::runtime_error(
                std::string("MoE provider prepare failed: ") +
                moe.last_error()
            );
        }

        MoeWeights moe_weights;
        initialize_moe_weights(&moe_weights);
        const D232Result result = run_d232(
            gdn,
            ck,
            moe,
            moe_weights,
            static_cast<unsigned int>(parsed_repetitions),
            same_run_d231_combined_ms
        );

        moe.release();
        FreeLibrary(moe.module);
        (void)ck.release();
        FreeLibrary(ck.module);
        gdn.release();
        FreeLibrary(gdn.module);
        return result.pass ? 0 : 1;
    } catch (const std::exception &error) {
        std::cerr
            << "q16384_suffix1024_ordered_forty_layer_owner_smoke "
               "error: "
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
        if (gdn.release != nullptr) {
            gdn.release();
        }
        if (gdn.module != nullptr) {
            FreeLibrary(gdn.module);
        }
        return 1;
    }
}
#endif
