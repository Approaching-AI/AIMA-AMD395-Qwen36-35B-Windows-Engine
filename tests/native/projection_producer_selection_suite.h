#pragma once

namespace projection_safety_test {
// Complete product dimensions with independently exact dyadic inputs. These
// cases compare the existing provider implementations, not model inference.
unsigned run_matrix_producer_selection_suite() {
#if defined(QRT_ENABLE_HIPBLASLT_RESIDENT_MATRIX_PROVIDER)
    constexpr unsigned tokens = 8192u;
    unsigned completed = 0u;
    for (const auto shape : {std::pair{8192u, 2048u}, std::pair{4096u, 2048u},
                             std::pair{9216u, 2048u}, std::pair{2048u, 4096u}}) {
        const unsigned rows = shape.first, columns = shape.second;
        const size_t cells = size_t(rows) * tokens;
        std::vector<uint16_t> weights(size_t(rows) * columns + 2u * kGuard, kBf16Guard);
        std::vector<uint16_t> inputs(size_t(tokens) * columns + 2u * kGuard, kBf16Guard);
        std::vector<float> output(cells + 2u * kGuard, kF32Guard);
        for (unsigned r = 0u; r < rows; ++r) for (unsigned k = 0u; k < columns; ++k)
            weights[kGuard + size_t(r) * columns + k] = bf16(weight_value(r, k));
        for (unsigned t = 0u; t < tokens; ++t) for (unsigned k = 0u; k < columns; ++k)
            inputs[kGuard + size_t(t) * columns + k] = bf16(input_value(t, k));
        float reference[7][11]{};
        for (unsigned t = 0u; t < 7u; ++t) for (unsigned r = 0u; r < 11u; ++r) {
            int64_t numerator = 0;
            for (unsigned k = 0u; k < columns; ++k)
                numerator += int64_t(int((r * 7u + k * 3u) % 11u) - 5) *
                    (int((t * 3u + k) % 7u) - 3);
            // Independent exact dyadic reference. The native producer is
            // approximate: record unrounded FP32 differences diagnostically,
            // and compare BF16 endpoints as in the existing producer tests.
            reference[t][r] = float(numerator) / 128.0f;
        }
        DeviceBuffer<uint16_t> dw(weights), di(inputs); DeviceBuffer<float> out(output);
        hipStream_t stream = nullptr; hipEvent_t event = nullptr;
        hip_ok(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "producer stream");
        hip_ok(hipEventCreateWithFlags(&event, hipEventDisableTiming), "producer completion event");
        auto complete = [&]() {
            hip_ok(hipEventRecord(event, stream), "producer record");
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
            for (;;) {
                const auto status = hipEventQuery(event);
                if (status == hipSuccess) break;
                require(status == hipErrorNotReady && std::chrono::steady_clock::now() < deadline,
                    "producer completion deadline");
                std::this_thread::yield();
            }
        };
        ResidentBf16MatrixProvider* provider = nullptr;
        std::string failed_stage, failure;
        require(ensure_resident_bf16_matrix_provider(&provider, &failed_stage, &failure), "producer initialization");
        for (unsigned choice = 0u; choice < 32u; ++choice) {
            ResidentBf16MatrixPlan* plan = nullptr;
            failed_stage.clear(); failure.clear();
            const auto plan_start = std::chrono::steady_clock::now();
            const bool available = create_resident_bf16_matrix_plan(provider, rows, columns, tokens,
                true, choice, &plan, &failed_stage, &failure);
            const double plan_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - plan_start).count();
            if (!available) {
                require(choice != 0u && (failed_stage == "hipblaslt_resident_matrix_operation" ||
                    failed_stage == "hipblaslt_resident_matrix_heuristic" ||
                    failed_stage == "hipblaslt_resident_matrix_heuristic_state"), "unexpected producer plan failure");
                std::cout << "{\"type\":\"matrix_producer_choice\",\"rows\":" << rows
                    << ",\"tokens\":" << tokens << ",\"k\":" << columns << ",\"choice\":" << choice
                    << ",\"available\":false,\"failure_stage\":\"" << failed_stage
                    << "\",\"plan_ms\":" << plan_ms << ",\"kernel_submitted\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}" << std::endl;
                continue;
            }
            require(plan && plan->heuristic_index == choice && plan->output_f32, "producer plan identity");
            std::fill(output.begin(), output.end(), kF32Guard);
            hip_ok(hipMemcpy(out.base, output.data(), output.size() * sizeof(float), hipMemcpyHostToDevice), "producer reset");
            std::array<double, 3> times{};
            for (unsigned repetition = 0u; repetition < 4u; ++repetition) {
                const auto start = std::chrono::steady_clock::now();
                require(resident_bf16_matrix_matmul_f32_output_with_heuristic_index(dw.data(), di.data(), out.data(),
                    rows, columns, tokens, choice, stream, "matrix_producer_choice", &failed_stage, &failure), "producer launch");
                complete();
                if (repetition) times[repetition - 1u] = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
            }
            out.read(output);
            for (size_t i = 0u; i < kGuard; ++i)
                require(output[i] == kF32Guard && output[kGuard + cells + i] == kF32Guard, "producer output guard");
            auto actual_weights = weights, actual_inputs = inputs;
            dw.read(actual_weights); di.read(actual_inputs);
            require(actual_weights == weights && actual_inputs == inputs, "producer immutable operands");
            size_t mismatches = 0u, bf16_mismatches = 0u, first = cells;
            double maximum_error = 0.0;
            for (unsigned t = 0u; t < tokens; ++t) for (unsigned r = 0u; r < rows; ++r) {
                const size_t cell = size_t(t) * rows + r;
                const float actual = output[kGuard + cell], expected = reference[t % 7u][r % 11u];
                require(std::isfinite(actual), "nonfinite generated producer output");
                bf16_mismatches += bf16(actual) != bf16(expected);
                maximum_error = (std::max)(maximum_error, std::abs(double(actual) - expected));
                if (output[kGuard + cell] != reference[t % 7u][r % 11u]) {
                    if (!mismatches) first = cell;
                    ++mismatches;
                }
            }
            if (mismatches) {
                const unsigned t = unsigned(first / rows), r = unsigned(first % rows);
                double direct = 0.0;
                for (unsigned k = 0u; k < columns; ++k) {
                    const uint32_t wa = uint32_t(actual_weights[kGuard + size_t(r) * columns + k]) << 16u;
                    const uint32_t ia = uint32_t(actual_inputs[kGuard + size_t(t) * columns + k]) << 16u;
                    float w, a; std::memcpy(&w, &wa, 4u); std::memcpy(&a, &ia, 4u);
                    direct += double(w) * double(a);
                }
                const float expected = reference[t % 7u][r % 11u], direct_f32 = float(direct);
                uint32_t actual_bits, expected_bits, direct_bits;
                std::memcpy(&actual_bits, &output[kGuard + first], 4u);
                std::memcpy(&expected_bits, &expected, 4u); std::memcpy(&direct_bits, &direct_f32, 4u);
                require(direct_bits == expected_bits, "producer independent reference classes");
                std::cout << "{\"type\":\"matrix_producer_raw_difference\",\"rows\":" << rows
                    << ",\"tokens\":" << tokens << ",\"k\":" << columns << ",\"choice\":" << choice
                    << ",\"f32_value_mismatches\":" << mismatches << ",\"first_cell\":" << first
                    << ",\"first_token\":" << t << ",\"first_row\":" << r << ",\"actual_bits\":" << actual_bits
                    << ",\"expected_bits\":" << expected_bits << ",\"direct_cpu_bits\":" << direct_bits
                    << ",\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}" << std::endl;
            }
            auto ordered = times; std::sort(ordered.begin(), ordered.end());
            ++completed;
            std::cout << "{\"type\":\"matrix_producer_choice\",\"rows\":" << rows
                << ",\"tokens\":" << tokens << ",\"k\":" << columns << ",\"choice\":" << choice
                << ",\"available\":true,\"fast_bf16_compute\":" << (choice >= 16u ? "true" : "false")
                << ",\"workspace_bytes\":" << plan->workspace_bytes << ",\"plan_ms\":" << plan_ms
                << ",\"warmups\":1,\"completed_host_ms\":[" << times[0] << ',' << times[1] << ',' << times[2]
                << "],\"median_host_ms\":" << ordered[1] << ",\"elements\":" << cells
                << ",\"independent_cpu_dot_classes\":77,\"raw_f32_mismatches\":" << mismatches
                << ",\"raw_f32_max_abs_diff\":" << maximum_error
                << ",\"bf16_reference_mismatches\":" << bf16_mismatches
                << ",\"bf16_reference_pass\":" << (bf16_mismatches ? "false" : "true")
                << ",\"redzones_pass\":true,\"immutable_inputs\":true,\"timing_excludes_setup_upload_verification\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}" << std::endl;
        }
        hip_ok(hipEventDestroy(event), "producer event destroy");
        hip_ok(hipStreamDestroy(stream), "producer stream destroy");
    }
    return completed;
#else
    throw std::runtime_error("matrix producer comparison requires hipBLASLt build support");
#endif
}
} // namespace projection_safety_test
