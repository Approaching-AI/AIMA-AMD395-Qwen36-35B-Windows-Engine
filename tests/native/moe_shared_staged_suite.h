#pragma once
#include "../../native/providers/moe_accumulator/sm121_pv_error_bound.h"
namespace moe_batch_test {
void compare_shared_staged(uint32_t tokens, bool down, unsigned mode) {
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
    using Row=qrt_sm121_staged_half_projection::Row;
    Row marker;std::memset(&marker,0xa5,sizeof(marker));
    const size_t igroups=size_t(tokens)*(columns/16u),wgroups=size_t(rows)*(columns/16u);
    std::vector<Row> ip(igroups+2u*kGuard,marker),wp(wgroups+2u*kGuard,marker);
    Device<Row> dip(ip),dwp(wp);
    g_state.staged_half_replay_active=false;
    g_state.shared_staged_operands[size_t(input_slot)]=reinterpret_cast<uint16_t*>(dip.data());
    g_state.shared_staged_operands[size_t(weight_slot)]=reinterpret_cast<uint16_t*>(dwp.data());
    std::vector<float> original_input_norm, original_weight_norm;
    std::vector<uint16_t> original_output;
    double times[2][3]{};size_t selected=0u,eligible=0u,unsupported[2]{};
    for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<2u;++position){
        const unsigned variant=(attempt+position)%2u;
        out.write(output);dip.write(ip);dwp.write(wp);
        g_state.shared_prevalidated_float_active=true;g_state.shared_staged_half_active=variant!=0u;
        g_state.scaled_significand_fallback=false;
        const auto start = std::chrono::steady_clock::now();
        require(launch_moe_l2(di.data(), input_slot, tokens, columns, nullptr) &&
            launch_moe_l2(dw.data(), weight_slot, rows, columns, nullptr), "shared norm dispatch");
        const unsigned blocks = unsigned((elements + kNativeThreads - 1u) / kNativeThreads);
        if (down) {
            auto kernel = shared_down_hawkeye_midpoint_correction_kernel<false, 4u>;
            hip_ok(launch_moe_correction(kernel, blocks, nullptr, input_slot, weight_slot,
                dn.data(), di.data(), dw.data(), out.data(), tokens, mode == 1u ? 32768u : 0u), "shared down replay");
        } else {
            auto kernel = shared_projection_hawkeye_midpoint_correction_kernel<false, 4u>;
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
        if(attempt)times[variant][attempt-1u] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto actual = out.read(output.size());
        const auto ni = din.read(input_norm.size()), nw = dwn.read(weight_norm.size());
        if(!attempt&&!variant){original_output=actual;original_input_norm=ni;original_weight_norm=nw;}
        require(actual == original_output &&
            !std::memcmp(ni.data(), original_input_norm.data(), ni.size() * sizeof(float)) &&
            !std::memcmp(nw.data(), original_weight_norm.data(), nw.size() * sizeof(float)), "shared original output or norm bits changed");
        for (size_t cell = 0u; cell < elements; ++cell)
            require(actual[kGuard + cell] == reference[(cell / rows) % 7u][(cell % rows) % 11u], "shared independent BF16 endpoint");
        for (size_t i = 0u; i < kGuard; ++i)
            require(actual[i] == kSentinel && actual[kGuard + elements + i] == kSentinel &&
                ni[i] == 12345.25f && ni[kGuard + tokens + i] == 12345.25f &&
                nw[i] == 12345.25f && nw[kGuard + rows + i] == 12345.25f, "shared norm/output guard");
        {
            selected = eligible = 0u;
            input_flags = dif.read(input_flags.size()); weight_flags = dwf.read(weight_flags.size());
            auto check_flags = [&](const std::vector<uint16_t>& values, const std::vector<unsigned>& flags, unsigned count) {
                for (unsigned row = 0u; row < count; ++row) {
                    bool valid = true;
                    for (unsigned k = 0u; k < columns; ++k) {
                        const uint16_t x=values[kGuard + size_t(row) * columns + k];
                        const unsigned e=(x>>7u)&255u;
                        valid &= !(x&0x7fffu)||(e>=64u&&e<=190u);
                    }
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
                const unsigned common=input_flags[kGuard + cell / rows]&weight_flags[kGuard + cell % rows];
                selected += candidate; eligible += candidate && (common&1u);
            }
        }
        const auto actual_ip=dip.read(ip.size()),actual_wp=dwp.read(wp.size());
        for(unsigned side=0u;side<2u;++side){
            const auto& raw=side?weights:input;const auto& packed=side?actual_wp:actual_ip;const size_t count=side?wgroups:igroups;
            if(variant)unsupported[side]=0u;
            for(size_t group=0u;group<count;++group){
                const auto expected=variant?qrt_sm121_scaled_half_products::prepare(raw.data()+kGuard+group*16u):marker;
                require(!std::memcmp(&packed[kGuard+group],&expected,sizeof(Row)),"shared complete prepared encoding differs");
                if(variant)unsupported[side]+=qrt_sm121_scaled_half_products::unit(expected)==-32768;
            }
            for(size_t i=0u;i<kGuard;++i)require(!std::memcmp(&packed[i],&marker,sizeof(Row))&&!std::memcmp(&packed[kGuard+count+i],&marker,sizeof(Row)),"shared prepared guard");
        }
        require(di.read(input.size())==input&&dw.read(weights.size())==weights&&dn.read(native.size())==native,"shared per-attempt immutable operands");
    }
    require(di.read(input.size())==input&&dw.read(weights.size())==weights&&dn.read(native.size())==native,"shared immutable operands");
    g_state.moe_l2.fill(nullptr);g_state.shared_replay_rows.fill(nullptr);g_state.shared_staged_operands.fill(nullptr);
    g_state.shared_prevalidated_float_active=false;g_state.shared_staged_half_active=false;g_state.scaled_significand_fallback=false;
    for(unsigned variant=0u;variant<2u;++variant){std::array<double,3> ordered{times[variant][0],times[variant][1],times[variant][2]};std::sort(ordered.begin(),ordered.end());
      std::printf("{\"kind\":\"shared_staged_half_comparison\",\"variant\":%u,\"tokens\":%u,\"rows\":%u,\"columns\":%u,\"mode\":%u,\"weight_surface\":%u,\"elements\":%zu,\"independent_cpu_dot_classes\":77,\"selected_cells_cpu\":%zu,\"eligible_selected_cells_cpu\":%zu,\"unsupported_input_k16_groups\":%zu,\"unsupported_weight_k16_groups\":%zu,\"operand_bytes\":%zu,\"complete_host_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"norm_preparation_included\":true,\"operand_preparation_included\":true,\"norm_bits_equal\":true,\"all_attempts_verified\":true,\"rotated_variant_order\":true,\"warmups\":1,\"measured_attempts\":3,\"all_encoded_words_checked\":true,\"bf16_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        variant,tokens,rows,columns,mode,unsigned(weight_slot),elements,selected,eligible,unsupported[0],unsupported[1],(igroups+wgroups)*sizeof(Row),ordered[1],times[variant][0],times[variant][1],times[variant][2]);
    }
}
} // namespace moe_batch_test
