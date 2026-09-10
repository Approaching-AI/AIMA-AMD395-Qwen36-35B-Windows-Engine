#pragma once

namespace projection_safety_test {
template<class T> std::vector<T> read_replay_tensor(const char *path, size_t elements, T guard) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    require(file && file.tellg() == static_cast<std::streamoff>(elements * sizeof(T)), "real projection tensor span mismatch");
    std::vector<T> result(elements + 2u * kGuard, guard);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data() + kGuard), elements * sizeof(T));
    require(static_cast<bool>(file), "short real projection tensor read");
    return result;
}

void run_real_qkv(const char *input_path, const char *weight_path, const char *reference_path, unsigned int ppb) {
    constexpr unsigned int rows = 8192u, tokens = 7169u, k = 2048u;
    constexpr size_t elements = static_cast<size_t>(rows) * tokens;
    auto inputs = read_replay_tensor<uint16_t>(input_path, static_cast<size_t>(tokens) * k, kBf16Guard);
    auto weights = read_replay_tensor<uint16_t>(weight_path, static_cast<size_t>(rows) * k, kBf16Guard);
    const auto reference = read_replay_tensor<uint16_t>(reference_path, elements, kBf16Guard);
    std::vector<float> output(elements + 2u * kGuard, kF32Guard);
    std::vector<float> input_bounds(tokens + 2u * kGuard, kF32Guard), weight_bounds(rows + 2u * kGuard, kF32Guard);
    DeviceBuffer<uint16_t> di(inputs), dw(weights);
    DeviceBuffer<float> dout(output), dix(input_bounds), dwx(weight_bounds);
    hip_ok(launch_selected_bf16_projection_wmma_checked(dw.data(), di.data(), dout.data(), rows, tokens, 0u, 0u, nullptr), "real_wmma");
    hip_ok(hipDeviceSynchronize(), "real_wmma_sync");
    hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel, dim3(tokens), dim3(256u), 0u, nullptr, di.data(), dix.data(), tokens, k);
    hipLaunchKernelGGL(bf16_row_l2_upper_bound_kernel, dim3(rows), dim3(256u), 0u, nullptr, dw.data(), dwx.data(), rows, k);
    hip_ok(hipGetLastError(), "real_l2_bounds");
    hip_ok(hipDeviceSynchronize(), "real_l2_bounds_sync");
    dout.read(output); dix.read(input_bounds); dwx.read(weight_bounds);
    size_t initial_mismatches = 0u, midpoint_misses = 0u, bound_misses = 0u, candidates = 0u;
    double required_ppb = 0.0;
    for (size_t i = 0u; i < elements; ++i) {
        const float value = output[kGuard + i];
        require(std::isfinite(value), "nonfinite real WMMA output");
        uint32_t bits = 0u;
        std::memcpy(&bits, &value, 4u);
        const unsigned int low = bits & 65535u;
        const unsigned int distance = low >= 32768u ? low - 32768u : 32768u - low;
        uint32_t midpoint_bits = (bits & 0xffff0000u) | 0x8000u;
        float midpoint = 0.0f;
        std::memcpy(&midpoint, &midpoint_bits, 4u);
        const float upper = input_bounds[kGuard + i / rows] * weight_bounds[kGuard + i % rows];
        const float margin = std::fabs(value - midpoint);
        const bool tiny = ((bits >> 23u) & 255u) < 32u;
        const bool selected = distance <= 512u || tiny || margin <= upper * (static_cast<float>(ppb) * 1e-9f);
        candidates += selected;
        if (bf16(value) != reference[kGuard + i]) {
            ++initial_mismatches;
            if (distance > 512u) {
                ++midpoint_misses;
                if (!tiny && upper > 0.0f) required_ppb = (std::max)(required_ppb, std::ceil(static_cast<double>(margin) / (static_cast<double>(upper) * 1e-9)));
            }
            bound_misses += !selected;
        }
    }
    const unsigned int blocks = selected_hawkeye_correction_maximum_blocks_per_launch();
    std::cout << "{\"type\":\"real_qkv_selector\",\"elements\":" << elements
              << ",\"initial_bf16_mismatches\":" << initial_mismatches
              << ",\"midpoint_misses\":" << midpoint_misses << ",\"bound_misses\":" << bound_misses
              << ",\"required_ppb_observed\":" << required_ppb << ",\"configured_ppb\":" << ppb
              << ",\"prospective_candidates\":" << candidates << ",\"maximum_blocks\":" << blocks
              << ",\"inference_acceptance\":false}" << std::endl;
    const auto start = std::chrono::steady_clock::now();
    hip_ok(launch_selected_bf16_projection_hawkeye_midpoint_correction(dw.data(), di.data(), nullptr,
        dix.data(), dwx.data(), dout.data(), rows, tokens, k, 512u, 0u, ppb, blocks, nullptr), "real_qkv_correction");
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    dout.read(output);
    size_t differences = 0u, first = elements;
    for (size_t i = 0u; i < elements; ++i) {
        require(std::isfinite(output[kGuard + i]), "nonfinite corrected real output");
        if (bf16(output[kGuard + i]) != reference[kGuard + i]) {
            ++differences;
            if (first == elements) first = i;
        }
    }
    for (size_t i = 0u; i < kGuard; ++i) {
        require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard, "real output redzone modified");
        require(input_bounds[i] == kF32Guard && input_bounds[kGuard + tokens + i] == kF32Guard, "real input-bound redzone modified");
        require(weight_bounds[i] == kF32Guard && weight_bounds[kGuard + rows + i] == kF32Guard, "real weight-bound redzone modified");
    }
    auto after_inputs = inputs, after_weights = weights;
    di.read(after_inputs); dw.read(after_weights);
    require(after_inputs == inputs && after_weights == weights, "real projection inputs modified");
    std::cout << "{\"type\":\"real_qkv_result\",\"elements\":" << elements
              << ",\"bf16_mismatches\":" << differences << ",\"first_mismatch\":" << first
              << ",\"correction_wall_ms\":" << ms << ",\"redzones_pass\":true,\"inputs_immutable\":true,\"inference_acceptance\":false}" << std::endl;
    require(differences == 0u, "corrected real QKV differs from reference");
}
}  // namespace projection_safety_test
