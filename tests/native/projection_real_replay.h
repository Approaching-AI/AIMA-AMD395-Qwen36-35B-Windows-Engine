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

void run_real_qkv(const char *input_path, const char *weight_path, const char *reference_path,
                  unsigned int ppb, bool extend_q8192 = false) {
    constexpr unsigned int rows = 8192u, source_tokens = 7169u, k = 2048u;
    const unsigned tokens = extend_q8192 ? 8192u : source_tokens;
    const size_t elements = static_cast<size_t>(rows) * tokens;
    auto inputs = read_replay_tensor<uint16_t>(input_path, static_cast<size_t>(source_tokens) * k, kBf16Guard);
    auto weights = read_replay_tensor<uint16_t>(weight_path, static_cast<size_t>(rows) * k, kBf16Guard);
    auto reference = read_replay_tensor<uint16_t>(reference_path, size_t(rows) * source_tokens, kBf16Guard);
    if (extend_q8192) {
        // Projection rows are independent. Keep all original 7169 tokens and
        // repeat the first 1023 input rows to exercise the exact product shape.
        // The corresponding GB10 outputs are used only after GPU computation.
        auto extend = [&](std::vector<uint16_t>& data, unsigned width) {
            data.resize(size_t(tokens) * width + 2u * kGuard, kBf16Guard);
            std::copy_n(data.data() + kGuard, size_t(tokens - source_tokens) * width,
                data.data() + kGuard + size_t(source_tokens) * width);
        };
        extend(inputs, k); extend(reference, rows);
    }
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
    const char* scalar_option = std::getenv("QRT_PROJECTION_SAFETY_SCALAR_REPLAY");
    const bool scalar_replay = scalar_option && std::strcmp(scalar_option, "1") == 0;
    const char* tiled_option = std::getenv("QRT_PROJECTION_SAFETY_TILED_REPLAY");
    const bool tiled_replay = tiled_option && std::strcmp(tiled_option,"1")==0;
    const char* bounded_option = std::getenv("QRT_PROJECTION_SAFETY_BOUNDED_REPLAY");
    const bool bounded_replay = bounded_option && std::strcmp(bounded_option,"1")==0;
    const char* scaled_option = std::getenv("QRT_PROJECTION_SAFETY_SCALED_REPLAY");
    const bool scaled_replay = scaled_option && std::strcmp(scaled_option,"1")==0;
    const char* row_max_option = std::getenv("QRT_PROJECTION_SAFETY_ROW_MAX_REPLAY");
    const bool row_max_replay = row_max_option && std::strcmp(row_max_option,"1")==0;
    const char* f32_carry_option = std::getenv("QRT_PROJECTION_SAFETY_F32_CARRY_REPLAY");
    const bool f32_carry_replay = f32_carry_option && std::strcmp(f32_carry_option,"1")==0;
    const char* range_option = std::getenv("QRT_PROJECTION_SAFETY_RANGE_REPLAY");
    const bool range_replay = range_option && std::strcmp(range_option,"1")==0;
    const char* dual_lane_option = std::getenv("QRT_PROJECTION_SAFETY_DUAL_LANE_REPLAY");
    const bool dual_lane_replay = dual_lane_option && std::strcmp(dual_lane_option,"1")==0;
    const char* interleaved_option = std::getenv("QRT_PROJECTION_SAFETY_INTERLEAVED_REPLAY");
    const bool interleaved_replay = interleaved_option && std::strcmp(interleaved_option,"1")==0;
    const char* staged_half_option=std::getenv("QRT_PROJECTION_SAFETY_STAGED_HALF_REPLAY");
    const bool staged_half_replay=staged_half_option && std::strcmp(staged_half_option,"1")==0;
    const char* scaled_half_option=std::getenv("QRT_PROJECTION_SAFETY_SCALED_HALF_REPLAY");
    const bool scaled_half_replay=scaled_half_option && std::strcmp(scaled_half_option,"1")==0;
    const char* decoded_option = std::getenv("QRT_PROJECTION_SAFETY_DECODED_REPLAY");
    const bool decoded_replay = decoded_option && std::strcmp(decoded_option,"1")==0;
    const char* packed_tiles_option = std::getenv("QRT_PROJECTION_SAFETY_PACKED_TILES_REPLAY");
    const bool packed_tiles_replay = packed_tiles_option && std::strcmp(packed_tiles_option,"1")==0;
    const char* partition_option = std::getenv("QRT_PROJECTION_SAFETY_PARTITION_REPLAY");
    const bool partition_replay = partition_option && std::strcmp(partition_option,"1")==0;
    const char* strong_option = std::getenv("QRT_PROJECTION_SAFETY_STRONG_REPLAY");
    const bool strong_replay = strong_option && std::strcmp(strong_option,"1")==0;
    const char* spatial_option = std::getenv("QRT_PROJECTION_SAFETY_SPATIAL_REPLAY");
    const bool spatial_replay = spatial_option && std::strcmp(spatial_option,"1")==0;
    const char* interval_option = std::getenv("QRT_PROJECTION_SAFETY_INTERVAL_REPLAY");
    const bool interval_replay = interval_option && std::strcmp(interval_option,"1")==0;
    std::vector<unsigned> selected_indices;
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
        if ((staged_half_replay || scaled_half_replay || scalar_replay || tiled_replay || bounded_replay || scaled_replay || row_max_replay || f32_carry_replay || range_replay || partition_replay || interval_replay || strong_replay || spatial_replay || packed_tiles_replay || dual_lane_replay || decoded_replay || interleaved_replay) && selected) selected_indices.push_back(static_cast<unsigned>(i));
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
    if (staged_half_replay) {
        run_staged_half_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (scaled_half_replay) {
        run_scaled_half_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (interleaved_replay) {
        run_interleaved_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (decoded_replay) {
        run_decoded_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (spatial_replay) {
        run_spatial_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (dual_lane_replay) {
        run_dual_lane_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (packed_tiles_replay) {
        run_packed_tiles_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (strong_replay) {
        run_strong_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (interval_replay) {
        run_interval_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (partition_replay) {
        run_partition_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (range_replay) {
        run_range_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (f32_carry_replay) {
        run_f32_carry_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (row_max_replay) {
        run_row_max_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (scaled_replay) {
        run_scaled_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (bounded_replay) {
        run_bounded_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (tiled_replay) {
        run_tiled_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (scalar_replay) {
        run_scalar_projection_replays(dw, di, dout, weights, inputs, reference, output,
            selected_indices, rows, tokens, k);
        return;
    }
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
              << ",\"tokens\":" << tokens << ",\"source_tokens\":" << source_tokens
              << ",\"repeated_input_rows\":" << (tokens - source_tokens)
              << ",\"bf16_mismatches\":" << differences << ",\"first_mismatch\":" << first
              << ",\"correction_wall_ms\":" << ms << ",\"redzones_pass\":true,\"inputs_immutable\":true,\"inference_acceptance\":false}" << std::endl;
    require(differences == 0u, "corrected real QKV differs from reference");
}
}  // namespace projection_safety_test
