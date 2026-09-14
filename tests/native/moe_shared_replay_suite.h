#pragma once
#include "../../native/providers/moe_accumulator/sm121_pv_error_bound.h"
namespace moe_batch_test {
void compare_shared_prevalidated(uint32_t tokens, bool down, unsigned mode) {
    const unsigned rows = down ? kHidden : kIntermediate, columns = down ? kIntermediate : kHidden;
    const auto input_slot = down ? MoeL2::SharedActivated : MoeL2::SharedInput;
    const auto weight_slot = down ? MoeL2::SharedDown : (mode == 1u ? MoeL2::SharedUp : MoeL2::SharedGate);
    const size_t elements = size_t(tokens) * rows;
    std::vector<uint16_t> input(size_t(tokens) * columns + 2u * kGuard, kSentinel);
    std::vector<uint16_t> weights(size_t(rows) * columns + 2u * kGuard, kSentinel);
    for (unsigned row = 0u; row < tokens; ++row) for (unsigned k = 0u; k < columns; ++k) {
        uint16_t value = bf16(float(int((row % 7u * 13u + k * 3u) % 61u) - 30) / 32.0f);
        if (mode == 2u && row % 7u == 1u && k % 3u == 0u) value = uint16_t((value & 0x807fu) | 0x0080u);
        input[kGuard + size_t(row) * columns + k] = value;
    }
    for (unsigned row = 0u; row < rows; ++row) for (unsigned k = 0u; k < columns; ++k) {
        uint16_t value = bf16(float(int((row % 11u * 7u + k * 5u) % 47u) - 23) / 64.0f);
        if (mode == 2u && row % 11u == 2u) value = uint16_t((value & 0x807fu) | (4u << 7u));
        weights[kGuard + size_t(row) * columns + k] = value;
    }
    uint16_t reference[7][11]{};
    require(tokens >= 7u, "shared reference classes");
    for (unsigned token = 0u; token < 7u; ++token) for (unsigned row = 0u; row < 11u; ++row)
        reference[token][row] = bf16(qrt_q1_moe_hawkeye::dot_bf16_hopper(
            input.data() + kGuard + size_t(token) * columns, weights.data() + kGuard + size_t(row) * columns, columns));
    std::vector<float> native(elements + 2u * kGuard, 12345.25f);
    std::vector<uint16_t> output(elements + 2u * kGuard, kSentinel);
    for (size_t cell = 0u; cell < elements; ++cell) {
        uint32_t bits = uint32_t(reference[(cell / rows) % 7u][(cell % rows) % 11u]) << 16u;
        if (mode == 1u || cell % 17u == 0u) bits = 0x3f808000u;
        std::memcpy(&native[kGuard + cell], &bits, sizeof(bits));
    }
    std::vector<float> input_norm(tokens + 2u * kGuard, 12345.25f), weight_norm(rows + 2u * kGuard, 12345.25f);
    std::vector<unsigned> input_flags(tokens + 2u * kGuard, 0xa5a5a5a5u), weight_flags(rows + 2u * kGuard, 0xa5a5a5a5u);
    Device<uint16_t> di(input), dw(weights), out(output); Device<float> dn(native), din(input_norm), dwn(weight_norm);
    Device<unsigned> dif(input_flags), dwf(weight_flags);
    g_state.sm121_moe_absolute_error_ppb = 1000u; g_state.scaled_l2 = false;
    g_state.prevalidated_float_active = false; g_state.prepared_replay_active = false;
    g_state.moe_l2[size_t(input_slot)] = din.data(); g_state.moe_l2[size_t(weight_slot)] = dwn.data();
    g_state.shared_replay_rows[size_t(input_slot)] = dif.data(); g_state.shared_replay_rows[size_t(weight_slot)] = dwf.data();
    std::vector<float> original_input_norm, original_weight_norm;
    std::vector<uint16_t> original_output;
    double times[2]{}; size_t selected = 0u, eligible = 0u;
    for (unsigned variant = 0u; variant < 2u; ++variant) {
        out.write(output); g_state.shared_prevalidated_float_active = variant != 0u;
        const auto start = std::chrono::steady_clock::now();
        require(launch_moe_l2(di.data(), input_slot, tokens, columns, nullptr) &&
            launch_moe_l2(dw.data(), weight_slot, rows, columns, nullptr), "shared norm dispatch");
        const unsigned blocks = unsigned((elements + kNativeThreads - 1u) / kNativeThreads);
        if (down) {
            auto kernel = variant ? shared_down_hawkeye_midpoint_correction_kernel<false, 4u>
                : shared_down_hawkeye_midpoint_correction_kernel<false>;
            hip_ok(launch_moe_correction(kernel, blocks, nullptr, input_slot, weight_slot,
                dn.data(), di.data(), dw.data(), out.data(), tokens, mode == 1u ? 32768u : 0u), "shared down replay");
        } else {
            auto kernel = variant ? shared_projection_hawkeye_midpoint_correction_kernel<false, 4u>
                : shared_projection_hawkeye_midpoint_correction_kernel<false>;
            hip_ok(launch_moe_correction(kernel, blocks, nullptr, input_slot, weight_slot,
                dn.data(), di.data(), dw.data(), out.data(), tokens, mode == 1u ? 32768u : 0u), "shared gate/up replay");
        }
        hipEvent_t event; hip_ok(hipEventCreate(&event), "shared completion event"); hip_ok(hipEventRecord(event), "shared record");
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
        for (;;) {
            const auto status = hipEventQuery(event); if (status == hipSuccess) break;
            require(status == hipErrorNotReady && std::chrono::steady_clock::now() < deadline, "shared completion deadline");
            std::this_thread::yield();
        }
        hip_ok(hipEventDestroy(event), "shared destroy");
        times[variant] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto actual = out.read(output.size());
        const auto ni = din.read(input_norm.size()), nw = dwn.read(weight_norm.size());
        if (!variant) { original_output = actual; original_input_norm = ni; original_weight_norm = nw; }
        require(actual == original_output &&
            !std::memcmp(ni.data(), original_input_norm.data(), ni.size() * sizeof(float)) &&
            !std::memcmp(nw.data(), original_weight_norm.data(), nw.size() * sizeof(float)), "shared original output or norm bits changed");
        for (size_t cell = 0u; cell < elements; ++cell)
            require(actual[kGuard + cell] == reference[(cell / rows) % 7u][(cell % rows) % 11u], "shared independent BF16 endpoint");
        for (size_t i = 0u; i < kGuard; ++i)
            require(actual[i] == kSentinel && actual[kGuard + elements + i] == kSentinel &&
                ni[i] == 12345.25f && ni[kGuard + tokens + i] == 12345.25f &&
                nw[i] == 12345.25f && nw[kGuard + rows + i] == 12345.25f, "shared norm/output guard");
        if (variant) {
            input_flags = dif.read(input_flags.size()); weight_flags = dwf.read(weight_flags.size());
            auto check_flags = [&](const std::vector<uint16_t>& values, const std::vector<unsigned>& flags, unsigned count) {
                for (unsigned row = 0u; row < count; ++row) {
                    bool valid = true;
                    for (unsigned k = 0u; k < columns; ++k) valid &= qrt_sm121_float_alignment::eligible(values[kGuard + size_t(row) * columns + k]);
                    require(flags[kGuard + row] == unsigned(valid), "shared eligibility mismatch");
                }
                for (size_t i = 0u; i < kGuard; ++i)
                    require(flags[i] == 0xa5a5a5a5u && flags[kGuard + count + i] == 0xa5a5a5a5u, "shared flag guard");
            };
            check_flags(input, input_flags, tokens); check_flags(weights, weight_flags, rows);
            for (size_t cell = 0u; cell < elements; ++cell) {
                const auto bits = qrt_sm121_pv_bound::bits(native[kGuard + cell]);
                const unsigned low = bits & 65535u, distance = low >= 32768u ? low - 32768u : 32768u - low;
                const bool candidate = distance <= (mode == 1u ? 32768u : 0u) ||
                    qrt_bf16_midpoint::within_error(native[kGuard + cell],
                        ni[kGuard + cell / rows] * nw[kGuard + cell % rows] * (1000.0f * 1e-9f));
                selected += candidate; eligible += candidate && input_flags[kGuard + cell / rows] && weight_flags[kGuard + cell % rows];
            }
        }
    }
    require(di.read(input.size()) == input && dw.read(weights.size()) == weights && dn.read(native.size()) == native, "shared immutable operands");
    g_state.moe_l2.fill(nullptr); g_state.shared_replay_rows.fill(nullptr); g_state.shared_prevalidated_float_active = false;
    std::printf("{\"kind\":\"shared_prevalidated_replay_comparison\",\"tokens\":%u,\"rows\":%u,\"columns\":%u,\"mode\":%u,\"weight_surface\":%u,\"elements\":%zu,\"independent_cpu_dot_classes\":77,\"selected_cells_cpu\":%zu,\"eligible_selected_cells_cpu\":%zu,\"original_host_ms\":%.6f,\"prevalidated_host_ms\":%.6f,\"norm_preparation_included\":true,\"norm_bits_equal\":true,\"bf16_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        tokens, rows, columns, mode, unsigned(weight_slot), elements, selected, eligible, times[0], times[1]);
}
} // namespace moe_batch_test
