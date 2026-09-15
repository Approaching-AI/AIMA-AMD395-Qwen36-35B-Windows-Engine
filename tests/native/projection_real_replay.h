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
                  unsigned int ppb, bool extend_q8192 = false, bool output_projection = false) {
    constexpr unsigned int source_tokens = 7169u;
    const unsigned rows = output_projection ? 2048u : 8192u, k = output_projection ? 4096u : 2048u;
    const char* coarse_option = std::getenv("QRT_PROJECTION_SAFETY_COARSE_INTERVAL");
    const bool coarse_interval = coarse_option && !std::strcmp(coarse_option,"1");
    const char* owner_option = std::getenv("QRT_PROJECTION_SAFETY_COARSE_OWNER");
    const bool coarse_owner = owner_option && !std::strcmp(owner_option,"1");
    const char* dominant_option = std::getenv("QRT_PROJECTION_SAFETY_DOMINANT_HALF_REPLAY");
    const bool dominant_replay = dominant_option && !std::strcmp(dominant_option,"1");
    const char* partitioned_half_option = std::getenv("QRT_PROJECTION_SAFETY_PARTITIONED_HALF_REPLAY");
    const bool partitioned_half_replay = partitioned_half_option && !std::strcmp(partitioned_half_option,"1");
    const char* transfer_option = std::getenv("QRT_PROJECTION_SAFETY_CARRY_TRANSFER_REPLAY");
    const bool transfer_replay = transfer_option && !std::strcmp(transfer_option,"1");
    const char* weight_bucket_option = std::getenv("QRT_PROJECTION_SAFETY_WEIGHT_BUCKET_REPLAY");
    const bool weight_bucket_replay = weight_bucket_option && !std::strcmp(weight_bucket_option,"1");
    const char* matrix_option = std::getenv("QRT_PROJECTION_SAFETY_INTEGER_MATRIX_REPLAY");
    const bool matrix_replay = matrix_option && !std::strcmp(matrix_option,"1");
    const char* embedded_option = std::getenv("QRT_PROJECTION_SAFETY_EMBEDDED_HALF_REPLAY");
    const bool embedded_replay = embedded_option && !std::strcmp(embedded_option,"1");
    const char* cooperative = std::getenv("QRT_PROJECTION_SAFETY_COOPERATIVE_HALF_REPLAY");
    const char* staged_f32_option = std::getenv("QRT_PROJECTION_SAFETY_STAGED_HALF_F32_REPLAY");
    const bool staged_f32_replay = staged_f32_option && !std::strcmp(staged_f32_option,"1");
    require(!output_projection || (extend_q8192 && ppb == 10000u &&
        ((cooperative && !std::strcmp(cooperative,"1")) || staged_f32_replay || embedded_replay || matrix_replay || dominant_replay || weight_bucket_replay || transfer_replay || partitioned_half_replay || coarse_interval || coarse_owner)),
        "real OUT requires a full-shape replay comparison and original FA bound");
    const unsigned tokens = extend_q8192 ? 8192u : source_tokens;
    const size_t elements = static_cast<size_t>(rows) * tokens;
    auto inputs = read_replay_tensor<uint16_t>(input_path, static_cast<size_t>(source_tokens) * k, kBf16Guard);
    auto weights = read_replay_tensor<uint16_t>(weight_path, static_cast<size_t>(rows) * k, kBf16Guard);
    auto reference = read_replay_tensor<uint16_t>(reference_path,
        size_t(rows) * (output_projection ? tokens : source_tokens), kBf16Guard);
    if (extend_q8192) {
        // Projection rows are independent. Keep all original 7169 tokens and
        // repeat the first 1023 input rows to exercise the exact product shape.
        // The corresponding GB10 outputs are used only after GPU computation.
        auto extend = [&](std::vector<uint16_t>& data, unsigned width) {
            data.resize(size_t(tokens) * width + 2u * kGuard, kBf16Guard);
            std::copy_n(data.data() + kGuard, size_t(tokens - source_tokens) * width,
                data.data() + kGuard + size_t(source_tokens) * width);
        };
        extend(inputs, k);
        if (!output_projection) extend(reference, rows);
    }
    std::vector<float> output(elements + 2u * kGuard, kF32Guard);
    std::vector<float> input_bounds(tokens + 2u * kGuard, kF32Guard), weight_bounds(rows + 2u * kGuard, kF32Guard);
    DeviceBuffer<uint16_t> di(inputs), dw(weights);
    DeviceBuffer<float> dout(output), dix(input_bounds), dwx(weight_bounds);
    if (output_projection) {
        std::string stage, failure;
        const bool produced = resident_bf16_matrix_matmul_f32_output_with_heuristic_index(
            dw.data(),di.data(),dout.data(),rows,k,tokens,0u,nullptr,
            "real_out_producer",&stage,&failure);
        require(produced, (stage+": "+failure).c_str());
    } else hip_ok(launch_selected_bf16_projection_wmma_checked(dw.data(), di.data(), dout.data(), rows, tokens, 0u, 0u, nullptr), "real_wmma");
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
    const char* cooperative_half_option=std::getenv("QRT_PROJECTION_SAFETY_COOPERATIVE_HALF_REPLAY");
    const bool cooperative_half_replay=cooperative_half_option && std::strcmp(cooperative_half_option,"1")==0;
    const char* blocked_half_option=std::getenv("QRT_PROJECTION_SAFETY_BLOCKED_HALF_REPLAY");
    const bool blocked_half_replay=blocked_half_option && std::strcmp(blocked_half_option,"1")==0;
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
        const float error = upper * (static_cast<float>(ppb) * 1e-9f);
        const bool selected = distance <= 512u || tiny || (partitioned_half_replay
            ? qrt_bf16_midpoint::within_error(value,error) : margin <= error);
        candidates += selected;
        if (selected) selected_indices.push_back(static_cast<unsigned>(i));
        if (bf16(value) != reference[kGuard + i]) {
            ++initial_mismatches;
            if (distance > 512u) {
                ++midpoint_misses;
                if (!tiny && upper > 0.0f) required_ppb = (std::max)(required_ppb, std::ceil(static_cast<double>(margin) / (static_cast<double>(upper) * 1e-9)));
            }
            bound_misses += !selected;
        }
    }
    require(selected_indices.size() == candidates, "collected projection candidate count differs");
    const unsigned int blocks = selected_hawkeye_correction_maximum_blocks_per_launch();
    std::cout << "{\"type\":\"" << (output_projection ? "real_out_selector" : "real_qkv_selector") << "\",\"elements\":" << elements
              << ",\"initial_bf16_mismatches\":" << initial_mismatches
              << ",\"midpoint_misses\":" << midpoint_misses << ",\"bound_misses\":" << bound_misses
              << ",\"required_ppb_observed\":" << required_ppb << ",\"configured_ppb\":" << ppb
              << ",\"prospective_candidates\":" << candidates << ",\"maximum_blocks\":" << blocks
              << ",\"inference_acceptance\":false}" << std::endl;
    if (coarse_owner) {
        require(output_projection,"coarse owner requires full-attention OUT");
        run_coarse_owner_projection(dw,di,weights,inputs,reference,rows,tokens,k);
        return;
    }
    if (coarse_interval) {
        run_coarse_interval_projection(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k,ppb);
        return;
    }
    if (partitioned_half_replay) {
        run_partitioned_half_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (transfer_replay) {
        run_carry_transfer_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (dominant_replay) {
        run_dominant_half_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (weight_bucket_replay) {
        run_weight_bucket_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (matrix_replay) {
        run_matrix_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (embedded_replay) {
        run_embedded_half_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (staged_f32_replay) {
        run_staged_half_f32_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (cooperative_half_replay) {
        run_cooperative_half_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
    if (blocked_half_replay) {
        run_blocked_half_projection_replays(dw,di,dout,weights,inputs,reference,output,
            selected_indices,rows,tokens,k);
        return;
    }
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
