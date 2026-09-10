// Compile the real provider kernels and launch guards in the same translation
// unit. Synthetic modes use no model files. The optional real-QKV mode reads
// fingerprinted captured tensors; its wrapper binds the reference provenance.
#define main qrt_legacy_provider_diagnostic_main
#include "../../native/providers/whole_provider.cpp"
#undef main

#include <stdexcept>

namespace projection_safety_test {
constexpr size_t kGuard = 128u;
constexpr float kF32Guard = 12345.25f;
constexpr uint16_t kBf16Guard = UINT16_C(0x5a5a);

void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}

void hip_ok(hipError_t status, const char *stage) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string(stage) + ": " + hipGetErrorString(status));
    }
}

uint16_t bf16(float value) {
    uint32_t bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
}

float weight_value(unsigned int row, unsigned int k) {
    return static_cast<float>(static_cast<int>((row * 7u + k * 3u) % 11u) - 5) / 8.0f;
}

float input_value(unsigned int token, unsigned int k) {
    return static_cast<float>(static_cast<int>((token * 3u + k) % 7u) - 3) / 16.0f;
}

template<class T> struct DeviceBuffer {
    T *base = nullptr;
    explicit DeviceBuffer(const std::vector<T> &host) {
        hip_ok(hipMalloc(reinterpret_cast<void **>(&base), host.size() * sizeof(T)), "allocate");
        hip_ok(hipMemcpy(base, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice), "upload");
    }
    ~DeviceBuffer() { if (base != nullptr) hipFree(base); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    T *data() { return base + kGuard; }
    void read(std::vector<T> &host) {
        hip_ok(hipMemcpy(host.data(), base, host.size() * sizeof(T), hipMemcpyDeviceToHost), "download");
    }
};

// These calls return before touching HIP, despite the deliberately invalid
// output/shape. Never submit an invalid kernel to reproduce the original fault.
void host_contract() {
    float f32 = 0.0f;
    uint16_t value = 0u;
    require(qrt_projection_output::needs_f32_buffer(false, true), "fused consumer suppressed F32 producer allocation");
    require(!qrt_projection_output::valid_buffers(&f32, nullptr, true), "missing BF16 consumer accepted");
    for (unsigned int which = 0; which < 6; ++which) {
        const auto status = launch_selected_bf16_projection_wmma_checked(
            which == 0 ? nullptr : &value, which == 1 ? nullptr : &value,
            which == 2 ? nullptr : &f32, which == 3 ? 0u : 128u,
            which == 4 ? 0u : (which == 5 ? UINT32_MAX : 64u), 0u, 0u, nullptr
        );
        require(status == hipErrorInvalidValue, "invalid WMMA contract reached dispatch");
    }
    require(launch_projection_f32_to_bf16_checked(nullptr, &value, 1u, nullptr) == hipErrorInvalidValue, "null conversion input accepted");
    require(launch_projection_f32_to_bf16_checked(&f32, nullptr, 1u, nullptr) == hipErrorInvalidValue, "null conversion output accepted");
    require(launch_projection_f32_to_bf16_checked(&f32, &value, 0u, nullptr) == hipErrorInvalidValue, "empty conversion accepted");
    require(launch_projection_f32_to_bf16_checked(&f32, &value, SIZE_MAX, nullptr) == hipErrorInvalidValue, "overflow conversion accepted");
}

void run_case(unsigned int rows, unsigned int tokens, bool consumer, bool full_shape) {
    const auto start = std::chrono::steady_clock::now();
    const size_t k = QRT_QWEN36_HIDDEN_SIZE;
    const size_t elements = static_cast<size_t>(rows) * tokens;
    std::vector<uint16_t> weights(static_cast<size_t>(rows) * k + 2u * kGuard, kBf16Guard);
    std::vector<uint16_t> inputs(static_cast<size_t>(tokens) * k + 2u * kGuard, kBf16Guard);
    std::vector<float> output(elements + 2u * kGuard, kF32Guard);
    std::vector<uint16_t> converted(elements + 2u * kGuard, kBf16Guard);
    for (unsigned int row = 0u; row < rows; ++row) {
        for (unsigned int column = 0u; column < k; ++column) {
            weights[kGuard + static_cast<size_t>(row) * k + column] = bf16(weight_value(row, column));
        }
    }
    for (unsigned int token = 0u; token < tokens; ++token) {
        for (unsigned int column = 0u; column < k; ++column) {
            inputs[kGuard + static_cast<size_t>(token) * k + column] = bf16(input_value(token, column));
        }
    }
    DeviceBuffer<uint16_t> dw(weights), di(inputs), dc(converted);
    DeviceBuffer<float> df(output);
    hipStream_t stream = nullptr;
    hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream_create");
    require(qrt_projection_output::valid_buffers(df.data(), consumer ? dc.data() : nullptr, consumer), "allocated contract rejected");
    hip_ok(launch_selected_bf16_projection_wmma_checked(dw.data(), di.data(), df.data(), rows, tokens, 0u, 0u, stream), "wmma_launch");
    // Synchronize the producer separately so an error cannot be attributed to
    // correction, conversion, or buffer cleanup later in the command stream.
    hip_ok(hipStreamSynchronize(stream), "wmma_sync");
    if (consumer) {
        hip_ok(launch_projection_f32_to_bf16_checked(df.data(), dc.data(), elements, stream), "convert_launch");
        hip_ok(hipStreamSynchronize(stream), "convert_sync");
    }
    hip_ok(hipStreamDestroy(stream), "stream_destroy");
    df.read(output);
    dc.read(converted);
    for (size_t i = 0u; i < kGuard; ++i) {
        require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard, "F32 output redzone modified");
        require(converted[i] == kBf16Guard && converted[kGuard + elements + i] == kBf16Guard, "BF16 output redzone modified");
    }
    for (size_t i = 0u; i < elements; ++i) {
        const float value = output[kGuard + i];
        require(std::isfinite(value) && value != kF32Guard, "F32 output cell not materialized");
        require(converted[kGuard + i] == (consumer ? bf16(value) : kBf16Guard), "BF16 consumer mismatch");
    }
    const size_t samples = full_shape ? 512u : elements;
    size_t reference_mismatches = 0u;
    size_t raw_f32_mismatches = 0u;
    double raw_f32_max_abs_diff = 0.0;
    for (size_t sample = 0u; sample < samples; ++sample) {
        const size_t index = full_shape ? sample * (elements - 1u) / (samples - 1u) : sample;
        const unsigned int row = static_cast<unsigned int>(index % rows);
        const unsigned int token = static_cast<unsigned int>(index / rows);
        float reference = 0.0f;
        for (unsigned int column = 0u; column < k; ++column) {
            reference += weight_value(row, column) * input_value(token, column);
        }
        // The downstream contract is BF16 RNE, not bitwise equality between
        // unrounded WMMA and scalar-host accumulators. Preserve raw differences
        // as diagnostics and require exact BF16 endpoint equality; this does
        // not relax the separately required real-model GB10 boundary.
        if (output[kGuard + index] != reference) {
            raw_f32_max_abs_diff = (std::max)(raw_f32_max_abs_diff,
                std::abs(static_cast<double>(output[kGuard + index]) - reference));
            ++raw_f32_mismatches;
        }
        if (bf16(output[kGuard + index]) != bf16(reference)) {
            if (reference_mismatches < 8u) {
                std::cerr << "projection_bf16_reference_mismatch rows=" << rows
                          << " tokens=" << tokens << " row=" << row << " token=" << token
                          << " actual=" << std::setprecision(12) << output[kGuard + index]
                          << " expected=" << reference << std::endl;
            }
            ++reference_mismatches;
        }
    }
    if (reference_mismatches != 0u) {
        std::cerr << "projection_reference_mismatches=" << reference_mismatches << '/' << samples << std::endl;
    }
    require(reference_mismatches == 0u, "synthetic BF16 projection reference mismatch");
    const auto expected_weights = weights;
    const auto expected_inputs = inputs;
    dw.read(weights);
    di.read(inputs);
    require(weights == expected_weights && inputs == expected_inputs, "read-only input or its redzone modified");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "{\"type\":\"projection_case\",\"rows\":" << rows
              << ",\"tokens\":" << tokens << ",\"bf16_consumer\":" << (consumer ? "true" : "false")
              << ",\"reference_cells\":" << samples << ",\"bf16_reference_mismatches\":" << reference_mismatches
              << ",\"raw_f32_mismatches\":" << raw_f32_mismatches
              << ",\"raw_f32_max_abs_diff\":" << std::setprecision(12) << raw_f32_max_abs_diff
              << ",\"redzones_pass\":true,\"wall_ms\":" << ms << "}" << std::endl;
}
// Seed sparse accumulator midpoints to exercise the actual selector, compact
// index transport and correction launcher. Exact dots vary with both token and
// row; other cells must still receive BF16 rounding. A product-sized case has
// more candidates than the former whole-tensor admission quota.
void run_correction_case(unsigned int rows, unsigned int tokens, unsigned int k, bool dense = false) {
    const size_t elements = static_cast<size_t>(rows) * tokens;
    std::vector<uint16_t> weights(static_cast<size_t>(rows) * k + 2u * kGuard, kBf16Guard);
    std::vector<uint16_t> inputs(static_cast<size_t>(tokens) * k + 2u * kGuard, kBf16Guard);
    std::vector<float> output(elements + 2u * kGuard, kF32Guard);
    std::fill(weights.begin() + kGuard, weights.end() - kGuard, uint16_t{0});
    std::fill(inputs.begin() + kGuard, inputs.end() - kGuard, uint16_t{0});
    for (unsigned int row = 0u; row < rows; ++row) {
        weights[kGuard + static_cast<size_t>(row) * k + k - 1u] =
            bf16(static_cast<float>(static_cast<int>(row % 13u) - 6) / 8.0f);
    }
    for (unsigned int token = 0u; token < tokens; ++token) {
        inputs[kGuard + static_cast<size_t>(token) * k + k - 1u] =
            bf16(static_cast<float>(static_cast<int>(token % 17u) - 8) / 16.0f);
    }
    for (size_t i = 0u; i < elements; ++i) {
        output[kGuard + i] = (dense || i % 64u == 0u || i + 1u == elements)
            ? 1.00390625f : 1.001f;
    }
    const auto expected_weights = weights;
    const auto expected_inputs = inputs;
    DeviceBuffer<uint16_t> dw(weights), di(inputs);
    DeviceBuffer<float> df(output);
    const auto start = std::chrono::steady_clock::now();
    hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(
        dw.data(), di.data(), nullptr, nullptr, nullptr, df.data(), rows, tokens,
        k, 512u, dense ? tokens : 0u, 0u, dense ? 64u : 8u, nullptr), "streamed_correction");
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    df.read(output);
    for (size_t i = 0u; i < kGuard; ++i) {
        require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard,
                "streamed correction wrote output redzone");
    }
    size_t candidates = 0u;
    for (size_t i = 0u; i < elements; ++i) {
        float expected = 1.0f;
        if (dense || i % 64u == 0u || i + 1u == elements) {
            ++candidates;
            const unsigned int row = static_cast<unsigned int>(i % rows);
            const unsigned int token = static_cast<unsigned int>(i / rows);
            expected = (static_cast<float>(static_cast<int>(row % 13u) - 6) / 8.0f) *
                       (static_cast<float>(static_cast<int>(token % 17u) - 8) / 16.0f);
            // The dot begins with positive zero and includes K-1 zero
            // products. A negative-zero final product still sums to +0.
            if (expected == 0.0f) expected = 0.0f;
        }
        if (bf16(output[kGuard + i]) != bf16(expected)) {
            std::cerr << "streamed_correction_mismatch index=" << i
                      << " row=" << i % rows << " token=" << i / rows
                      << " actual_bits=" << bf16(output[kGuard + i])
                      << " expected_bits=" << bf16(expected) << std::endl;
        }
        require(bf16(output[kGuard + i]) == bf16(expected), "streamed correction endpoint mismatch");
        uint32_t bits = 0u;
        std::memcpy(&bits, &output[kGuard + i], sizeof(bits));
        require((bits & 0xffffu) == 0u, "streamed correction left unrounded cell");
    }
    dw.read(weights); di.read(inputs);
    require(weights == expected_weights && inputs == expected_inputs,
            "streamed correction modified read-only input or redzone");
    std::cout << "{\"type\":\"correction_case\",\"rows\":" << rows
              << ",\"tokens\":" << tokens << ",\"k\":" << k
              << ",\"candidates\":" << candidates << ",\"reference_cells\":" << elements
              << ",\"bf16_reference_mismatches\":0,\"redzones_pass\":true,\"wall_ms\":"
              << ms << "}" << std::endl;
}
} // namespace projection_safety_test

#include "projection_real_replay.h"
#include "convolution_real_replay.h"
#include "final_norm_real_replay.h"

int main(int argc, char **argv) {
    using namespace projection_safety_test;
    try {
        require(argc >= 2, "select a synthetic or real-tensor mode");
        const std::string mode = argv[1];
        require(argc == ((mode == "--real-qkv" || mode == "--real-conv" || mode == "--real-finalnorm") ? 6 : 2), "select a synthetic mode, --real-qkv INPUT WEIGHT REFERENCE PPB, --real-conv INPUT WEIGHT REFERENCE_DIR TABLE, or --real-finalnorm INPUT WEIGHT REFERENCE CORRECTION");
        require(mode == "--host-only" || mode == "--small" || mode == "--full-shape" || mode == "--correction" || mode == "--real-qkv" || mode == "--real-conv" || mode == "--real-finalnorm", "unknown safety mode");
        host_contract();
        unsigned int cases = 0u;
        if (mode != "--host-only") {
            hip_ok(hipInit(0), "hip_init");
            hipDeviceProp_t properties{};
            hip_ok(hipGetDeviceProperties(&properties, 0), "device_properties");
            require(std::string(properties.gcnArchName).find("gfx1151") == 0u, "expected gfx1151 before any kernel dispatch");
            if (mode == "--real-qkv") {
                const auto ppb = std::stoul(argv[5]);
                require(ppb > 0u && ppb <= 1000000u, "invalid real-QKV selector bound");
                run_real_qkv(argv[2], argv[3], argv[4], static_cast<unsigned int>(ppb));
                ++cases;
            } else if (mode == "--real-conv") {
                run_real_convolution(argv[2], argv[3], argv[4], argv[5]);
                ++cases;
            } else if (mode == "--real-finalnorm") {
                run_real_final_norm(argv[2], argv[3], argv[4], argv[5]);
                ++cases;
            } else if (mode == "--small") {
                const unsigned int shapes[][2] = {{1u, 1u}, {127u, 63u}, {128u, 64u}, {129u, 65u}};
                for (const auto &shape : shapes) {
                    for (bool consumer : {false, true}) {
                        run_case(shape[0], shape[1], consumer, false);
                        ++cases;
                    }
                }
            } else if (mode == "--correction") {
                run_correction_case(129u, 1031u, 2048u);
                run_correction_case(8192u, 7169u, 16u);
                run_correction_case(32u, 7169u, 2048u, true);
                cases += 3u;
            } else {
                run_case(8192u, 7169u, true, true);
                ++cases;
            }
            hip_ok(hipDeviceSynchronize(), "final_sync");
        }
        std::cout << "{\"type\":\"summary\",\"status\":\"pass\",\"mode\":\"" << mode
                  << "\",\"gpu_cases\":" << cases
                  << ",\"inference_success_claimed\":false,\"numerical_scope\":\""
                  << (mode == "--real-qkv" ? "real_bf16_qkv_projection" : mode == "--real-conv" ? "real_bf16_convolution" : mode == "--real-finalnorm" ? "real_final_norm_bf16_endpoint" : "synthetic_bf16_projection_endpoint") << "\"}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "projection_safety_failure: " << error.what() << std::endl;
        return 1;
    }
}
