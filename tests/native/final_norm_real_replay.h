#pragma once

namespace projection_safety_test {
void run_real_final_norm(const char *input_path, const char *weight_path,
                         const char *reference_path, const char *correction_path) {
    constexpr size_t elements = QRT_QWEN36_HIDDEN_SIZE;
    auto inputs = read_replay_tensor<float>(input_path, elements, kF32Guard);
    auto weights = read_replay_tensor<uint16_t>(weight_path, elements, kBf16Guard);
    const auto reference = read_replay_tensor<uint16_t>(reference_path, elements, kBf16Guard);
    auto correction = read_replay_tensor<uint8_t>(correction_path, size_t{1} << 22, uint8_t{0x5a});
    std::vector<float> output(elements + 2u * kGuard, kF32Guard);
    DeviceBuffer<float> di(inputs), dout(output);
    DeviceBuffer<uint16_t> dw(weights);
    DeviceBuffer<uint8_t> dc(correction);
    hipLaunchKernelGGL(layer1_input_rmsnorm_kernel, dim3(1), dim3(kThreads),
                      0, nullptr, di.data(), dw.data(), dout.data(), 1u);
    hip_ok(hipGetLastError(), "final_norm_old_launch");
    dout.read(output);
    size_t prior_differences = 0;
    for (size_t i = 0; i < elements; ++i) {
        prior_differences += bf16(output[kGuard + i]) != reference[kGuard + i];
    }
    const auto start = std::chrono::steady_clock::now();
    hipLaunchKernelGGL(final_norm_unrounded_vllm_kernel, dim3(1), dim3(kThreads),
                      0, nullptr, di.data(), dw.data(), dout.data(), 1u, dc.data());
    hip_ok(hipGetLastError(), "final_norm_corrected_launch");
    dout.read(output);
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    size_t differences = 0;
    for (size_t i = 0; i < elements; ++i) {
        require(std::isfinite(output[kGuard + i]), "nonfinite final norm output");
        differences += bf16(output[kGuard + i]) != reference[kGuard + i];
    }
    for (size_t i = 0; i < kGuard; ++i) {
        require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard,
                "final norm output redzone modified");
    }
    auto after_inputs = inputs;
    auto after_weights = weights;
    auto after_correction = correction;
    di.read(after_inputs); dw.read(after_weights); dc.read(after_correction);
    require(after_inputs == inputs && after_weights == weights && after_correction == correction,
            "final norm modified its input, weights, correction or redzones");
    std::cout << "{\"type\":\"real_final_norm_result\",\"elements\":" << elements
              << ",\"prior_bf16_mismatches\":" << prior_differences
              << ",\"bf16_mismatches\":" << differences
              << ",\"redzones_pass\":true,\"inputs_immutable\":true,\"wall_ms\":" << ms
              << ",\"inference_acceptance\":false}" << std::endl;
    require(differences == 0, "final norm does not match original GB10 output");
}
}  // namespace projection_safety_test
