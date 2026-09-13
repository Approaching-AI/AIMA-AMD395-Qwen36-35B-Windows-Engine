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
void run_staging_case(unsigned int rows, unsigned int tokens) {
    const size_t k = QRT_QWEN36_HIDDEN_SIZE, cells = static_cast<size_t>(rows) * tokens;
    std::vector<uint16_t> weights(static_cast<size_t>(rows) * k + 2u * kGuard, kBf16Guard);
    std::vector<uint16_t> inputs(static_cast<size_t>(tokens) * k + 2u * kGuard, kBf16Guard);
    std::vector<float> original(cells + 2u * kGuard, kF32Guard), staged = original;
    uint32_t seed = UINT32_C(0x8191395);
    for (auto* operand : {&weights, &inputs}) {
        for (size_t i = kGuard; i + kGuard < operand->size(); ++i) {
            seed = seed * 1664525u + 1013904223u;
            (*operand)[i] = static_cast<uint16_t>(
                ((seed >> 16u) & 0x8000u) | (0x3a00u + (seed & 0x07ffu)));
        }
    }
    const auto saved_weights = weights, saved_inputs = inputs;
    DeviceBuffer<uint16_t> dw(weights), di(inputs);
    DeviceBuffer<float> old_output(original), new_output(staged);
    hipStream_t stream = nullptr;
    hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "staging_stream_create");
    double original_ms = 0.0, staged_ms = 0.0;
    for (unsigned int mode = 0; mode < 2u; ++mode) {
        const auto begin = std::chrono::steady_clock::now();
        if (mode == 0u) {
            hipLaunchKernelGGL(selected_bf16_projection_wmma_k16_m64_kernel,
                dim3((rows - 1u) / 128u + 1u, (tokens - 1u) / 64u + 1u),
                dim3(256u), 0, stream, dw.data(), di.data(), old_output.data(),
                rows, tokens, 0u, 0u);
        } else {
            hipLaunchKernelGGL(selected_bf16_projection_wmma_k16_m64_lds_kernel,
                dim3((rows - 1u) / 128u + 1u, (tokens - 1u) / 64u + 1u),
                dim3(256u), 0, stream, dw.data(), di.data(), new_output.data(),
                rows, tokens, 0u, 0u);
        }
        hip_ok(hipGetLastError(), "staging_kernel_launch");
        hip_ok(hipStreamSynchronize(stream), "staging_kernel_completion");
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count();
        (mode ? staged_ms : original_ms) = ms;
    }
    hip_ok(hipStreamDestroy(stream), "staging_stream_destroy");
    old_output.read(original); new_output.read(staged);
    size_t mismatches = 0u;
    for (size_t i = 0; i < original.size(); ++i) {
        if (i < kGuard || i >= kGuard + cells) {
            require(original[i] == kF32Guard && staged[i] == kF32Guard, "staging output redzone modified");
        } else {
            require(std::isfinite(original[i]) && std::isfinite(staged[i]), "staging output nonfinite");
            if (std::memcmp(&original[i], &staged[i], sizeof(float))) ++mismatches;
        }
    }
    dw.read(weights); di.read(inputs);
    require(weights == saved_weights && inputs == saved_inputs, "staging modified an operand");
    std::cout << "{\"type\":\"wmma_staging_case\",\"rows\":" << rows
              << ",\"tokens\":" << tokens << ",\"raw_f32_elements\":" << cells
              << ",\"raw_f32_bit_mismatches\":" << mismatches
              << ",\"original_completed_wall_ms\":" << original_ms
              << ",\"staged_completed_wall_ms\":" << staged_ms
              << ",\"redzones_pass\":true,\"input_immutable\":true,\"inference_acceptance\":false}"
              << std::endl;
    require(mismatches == 0u, "staged WMMA differs from the original ordered K16 kernel");
}

// Seed sparse accumulator midpoints to exercise the actual selector, compact
// index transport and correction launcher. Exact dots vary with both token and
// row; other cells must still receive BF16 rounding. A product-sized case has
// more candidates than the former whole-tensor admission quota.
void run_boundary_correction_case() {
    constexpr unsigned rows = 4u, k = 16u;
    std::vector<uint16_t> weights(rows * k + 2u * kGuard, kBf16Guard), inputs(k + 2u * kGuard, kBf16Guard);
    std::vector<float> output(rows + 2u * kGuard, kF32Guard), input_norm(1u + 2u * kGuard, kF32Guard), weight_norm(output);
    std::fill(weights.begin() + kGuard, weights.end() - kGuard, uint16_t{0});
    std::fill(inputs.begin() + kGuard, inputs.end() - kGuard, uint16_t{0});
    inputs[kGuard] = bf16(1.75f); inputs[kGuard + 1u] = bf16(std::ldexp(1.0f, -10));
    input_norm[kGuard] = 2.0f;
    for (unsigned row = 0; row < rows; ++row) {
        const float scale = (row & 1u ? -1.0f : 1.0f) * (row < 2u ? 1.0f : 2.0f);
        output[kGuard + row] = scale;
        weights[kGuard + row * k] = bf16(0.5703125f * scale);
        weights[kGuard + row * k + 1u] = bf16(-std::ldexp(scale, -14));
        weight_norm[kGuard + row] = 1000.0f * std::fabs(scale);
    }
    const auto original_weights = weights, original_inputs = inputs;
    DeviceBuffer<uint16_t> dw(weights), di(inputs);
    DeviceBuffer<float> df(output), dn(input_norm), wn(weight_norm);
    hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw.data(), di.data(), nullptr,
        dn.data(), wn.data(), df.data(), rows, 1u, k, 0u, 0u, 1000u, 8u, nullptr), "boundary_correction");
    df.read(output);
    for (unsigned row = 0; row < rows; ++row) {
        const float scale = (row & 1u ? -1.0f : 1.0f) * (row < 2u ? 1.0f : 2.0f);
        const float exact = scale * (0.998046875f - std::ldexp(1.0f, -24));
        require(bf16(output[kGuard + row]) == bf16(exact), "lower neighboring midpoint was not corrected");
        require(bf16(output[kGuard + row]) != bf16(scale), "boundary control did not change endpoint");
    }
    for (size_t i = 0; i < kGuard; ++i)
        require(output[i] == kF32Guard && output[kGuard + rows + i] == kF32Guard, "boundary correction redzone");
    dw.read(weights); di.read(inputs);
    require(weights == original_weights && inputs == original_inputs, "boundary correction modified operands");
    std::cout << "{\"type\":\"correction_boundary_case\",\"cells\":4,\"bf16_reference_mismatches\":0,\"both_signs\":true,\"redzones_pass\":true}" << std::endl;
}

void run_correction_case(unsigned int rows, unsigned int tokens, unsigned int k, bool dense = false,
                         bool mixed_fallback = false, bool absolute_bounds = false) {
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
    if (mixed_fallback) {
        // Each excluded value multiplies a zero in the opposite operand.
        // The independent closed-form endpoint remains unchanged while
        // adjacent candidate subgroups take prepared and original paths.
        for (unsigned row = 0u; row < rows; row += 17u)
            weights[kGuard + size_t(row) * k] = 0x0001u;
        for (unsigned token = 0u; token < tokens; token += 19u)
            inputs[kGuard + size_t(token) * k + 1u] = 0x8001u;
    }
    for (size_t i = 0u; i < elements; ++i) {
        output[kGuard + i] = (dense || i % 64u == 0u || i + 1u == elements)
            ? 1.00390625f : 1.001f;
    }
    const auto expected_weights = weights;
    const auto expected_inputs = inputs;
    DeviceBuffer<uint16_t> dw(weights), di(inputs);
    DeviceBuffer<float> df(output);
    // Conservative but deliberately loose Cauchy inputs make a useful control:
    // admitted matrix bounds must address their own window and retain every
    // midpoint/fallback cell without admitting all otherwise distant cells.
    std::vector<float> input_norm(tokens + 2u * kGuard, kF32Guard), weight_norm(rows + 2u * kGuard, kF32Guard);
    std::fill(input_norm.begin()+kGuard,input_norm.end()-kGuard,1.0e6f);
    std::fill(weight_norm.begin()+kGuard,weight_norm.end()-kGuard,1.0e6f);
    DeviceBuffer<float> din(input_norm), dwn(weight_norm);
    const auto start = std::chrono::steady_clock::now();
    hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(
        dw.data(), di.data(), nullptr, absolute_bounds ? din.data() : nullptr,
        absolute_bounds ? dwn.data() : nullptr, df.data(), rows, tokens,
        k, 512u, dense ? tokens : 0u, absolute_bounds ? 1000u : 0u, dense ? 64u : 8u, nullptr,
        absolute_bounds && rows == 1025u ? 65537u : qrt_hawkeye_dispatch::maximum_window_elements), "streamed_correction");
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
        if (dense || i % 64u == 0u || i + 1u == elements ||
            (absolute_bounds && mixed_fallback && (i % rows % 17u == 0u || i / rows % 19u == 0u))) {
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
    auto actual_input_norm=input_norm, actual_weight_norm=weight_norm;
    din.read(actual_input_norm); dwn.read(actual_weight_norm);
    require(actual_input_norm==input_norm && actual_weight_norm==weight_norm, "streamed correction modified bounds or redzones");
    std::cout << "{\"type\":\"correction_case\",\"rows\":" << rows
              << ",\"tokens\":" << tokens << ",\"k\":" << k
              << ",\"mixed_row_fallback\":" << (mixed_fallback ? "true" : "false")
              << ",\"absolute_product_bound\":" << (absolute_bounds ? "true" : "false")
              << ",\"candidates\":" << candidates << ",\"reference_cells\":" << elements
              << ",\"bf16_reference_mismatches\":0,\"redzones_pass\":true,\"wall_ms\":"
              << ms << "}" << std::endl;
}
} // namespace projection_safety_test

#include "projection_real_replay.h"
#include "absolute_product_hipblaslt_selftest.h"
#include "convolution_real_replay.h"
#include "final_norm_real_replay.h"
#include "hawkeye_device_replay_selftest.h"

int main(int argc, char **argv) {
    using namespace projection_safety_test;
    try {
        require(argc >= 2, "select a synthetic or real-tensor mode");
        const std::string mode = argv[1];
        require(argc == ((mode == "--real-qkv" || mode == "--real-conv" || mode == "--real-finalnorm") ? 6 : 2), "select a synthetic mode, --real-qkv INPUT WEIGHT REFERENCE PPB, --real-conv INPUT WEIGHT REFERENCE_DIR TABLE, or --real-finalnorm INPUT WEIGHT REFERENCE CORRECTION");
        require(mode == "--host-only" || mode == "--small" || mode == "--full-shape" || mode == "--correction" || mode == "--real-qkv" || mode == "--real-conv" || mode == "--real-finalnorm" || mode == "--wmma-staging" || mode == "--device-replay" || mode == "--prepared-correction" || mode == "--absolute-bound-correction" || mode == "--absolute-product-hipblaslt", "unknown safety mode");
        host_contract();
        unsigned int cases = 0u;
        if (mode != "--host-only") {
            hip_ok(hipInit(0), "hip_init");
            hipDeviceProp_t properties{};
            hip_ok(hipGetDeviceProperties(&properties, 0), "device_properties");
            require(std::string(properties.gcnArchName).find("gfx1151") == 0u, "expected gfx1151 before any kernel dispatch");
            if (mode == "--absolute-product-hipblaslt") {
                cases += run_absolute_product_hipblaslt_suite();
            } else if (mode == "--device-replay") {
                cases += run_device_replay_suite();
            } else if (mode == "--prepared-correction" || mode == "--absolute-bound-correction") {
                require(std::getenv("QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS") &&
                    !std::strcmp(std::getenv("QRT_QWEN36_HAWKEYE_PREPARED_OPERANDS"), "1"), "prepared correction mode requires prepared operands");
                const bool bound=mode=="--absolute-bound-correction";
                if(bound) require(std::getenv("QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_BOUND") &&
                    !std::strcmp(std::getenv("QRT_QWEN36_HAWKEYE_ABSOLUTE_PRODUCT_BOUND"),"1"), "absolute bound mode requires matrix bounds");
                run_correction_case(1025u, 1031u, 16u, false, true, bound);
                run_correction_case(1024u, 1024u, 512u, true, true, bound);
                run_correction_case(2048u, 1024u, 4096u, false, true, bound);
                run_correction_case(8192u, 7169u, 16u, false, true, bound);
                cases += 4u;
            } else if (mode == "--wmma-staging") {
                for (const auto shape : {std::pair{17u, 7u}, std::pair{129u, 65u}, std::pair{8192u, 8192u}}) {
                    run_staging_case(shape.first, shape.second);
                    ++cases;
                }
            } else if (mode == "--real-qkv") {
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
                run_boundary_correction_case();
                run_correction_case(129u, 1031u, 2048u);
                run_correction_case(8192u, 7169u, 16u);
                run_correction_case(32u, 7169u, 2048u, true);
                cases += 4u;
            } else {
                run_case(8192u, 7169u, true, true);
                ++cases;
            }
            hip_ok(hipDeviceSynchronize(), "final_sync");
        }
        std::cout << "{\"type\":\"summary\",\"status\":\"pass\",\"mode\":\"" << mode
                  << "\",\"gpu_cases\":" << cases
                  << ",\"inference_success_claimed\":false,\"numerical_scope\":\""
                  << (mode == "--wmma-staging" ? "synthetic_full_f32_staging_vs_original_wmma" : mode == "--real-qkv" ? "real_bf16_qkv_projection" : mode == "--real-conv" ? "real_bf16_convolution" : mode == "--real-finalnorm" ? "real_final_norm_bf16_endpoint" : "synthetic_bf16_projection_endpoint") << "\"}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "projection_safety_failure: " << error.what() << std::endl;
        return 1;
    }
}
