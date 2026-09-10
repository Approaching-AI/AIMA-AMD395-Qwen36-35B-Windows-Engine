#pragma once

namespace projection_safety_test {
void run_real_convolution(const char *input_path, const char *weight_path,
                          const char *reference_dir, const char *table_path) {
    constexpr unsigned int rows = 8192u, tokens = 7169u;
    constexpr size_t elements = static_cast<size_t>(rows) * tokens;
    const auto captured = read_replay_tensor<uint16_t>(input_path, elements, kBf16Guard);
    const auto weights = read_replay_tensor<uint16_t>(weight_path, rows * 4u, kBf16Guard);
    std::vector<float> input(elements + 2u * kGuard, kF32Guard);
    for (size_t i = 0; i < elements; ++i) input[kGuard + i] = qrt_bf16_to_float(captured[kGuard + i]);
    const std::string base = std::string(reference_dir) + "/full-";
    const auto q = read_replay_tensor<uint16_t>((base + "q-bf16.bin").c_str(), size_t(tokens) * 2048u, kBf16Guard);
    const auto k = read_replay_tensor<uint16_t>((base + "k-bf16.bin").c_str(), size_t(tokens) * 2048u, kBf16Guard);
    const auto v = read_replay_tensor<uint16_t>((base + "v-bf16.bin").c_str(), size_t(tokens) * 4096u, kBf16Guard);
    std::vector<float> output(elements + 2u * kGuard, kF32Guard);
    DeviceBuffer<float> di(input), dout(output);
    DeviceBuffer<uint16_t> dw(weights);
    const unsigned char *table = nullptr;
    hip_ok(qrt_sm121_silu_runtime::prepare(table_path, &table), "real_conv_table");
    for (bool compatible : {false, true}) {
        const auto start = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(selected_conv_qkv_window_kernel, dim3(rows / 256u, tokens),
            dim3(256u), 0u, nullptr, di.data(), dw.data(), nullptr, dout.data(),
            tokens, 3u, nullptr, nullptr, 0u, compatible ? table : nullptr);
        hip_ok(hipGetLastError(), "real_convolution");
        hip_ok(hipDeviceSynchronize(), "real_convolution_sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        require(ms < 100.0, "real convolution completed-dispatch deadline");
        dout.read(output);
        size_t differences[3]{};
        for (size_t i = 0; i < elements; ++i) {
            const size_t token = i / rows, row = i % rows;
            const unsigned int surface = row < 2048u ? 0u : row < 4096u ? 1u : 2u;
            const uint16_t reference = surface == 0u ? q[kGuard + token * 2048u + row]
                : surface == 1u ? k[kGuard + token * 2048u + row - 2048u]
                : v[kGuard + token * 4096u + row - 4096u];
            require(std::isfinite(output[kGuard + i]), "nonfinite real convolution result");
            differences[surface] += bf16(output[kGuard + i]) != reference;
        }
        for (size_t i = 0; i < kGuard; ++i)
            require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard, "real convolution output redzone modified");
        std::cout << "{\"type\":\"real_convolution_result\",\"elements\":" << elements
                  << ",\"sm121_table\":" << (compatible ? "true" : "false")
                  << ",\"q_bf16_mismatches\":" << differences[0]
                  << ",\"k_bf16_mismatches\":" << differences[1]
                  << ",\"v_bf16_mismatches\":" << differences[2]
                  << ",\"dispatch_wall_ms\":" << ms << ",\"redzones_pass\":true,\"inference_acceptance\":false}" << std::endl;
        if (compatible) require(differences[0] + differences[1] + differences[2] == 0u, "SM121 convolution differs from reference");
    }
    auto after_input = input;
    auto after_weights = weights;
    di.read(after_input); dw.read(after_weights);
    require(std::memcmp(after_input.data(), input.data(), input.size() * sizeof(float)) == 0 &&
        after_weights == weights, "real convolution inputs modified");
}
} // namespace projection_safety_test
