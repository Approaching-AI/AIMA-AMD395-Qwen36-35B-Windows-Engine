#pragma once
#include "../../native/providers/moe_accumulator/sm121_projection_interval_filter.h"

namespace projection_safety_test {
void run_interval_projection_replays(DeviceBuffer<uint16_t>& dw, DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout, const std::vector<uint16_t>& weights, const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference, const std::vector<float>& original_output,
    const std::vector<unsigned>& selected, unsigned rows, unsigned tokens, unsigned width) {
    namespace filter = qrt_sm121_projection_interval_filter;
    const size_t elements = size_t(rows) * tokens;
    require(!selected.empty() && std::is_sorted(selected.begin(), selected.end()) &&
        std::adjacent_find(selected.begin(), selected.end()) == selected.end(), "interval requires original unique candidates");
    constexpr unsigned sentinel = 0xa5a5a5a5u;
    std::vector<unsigned> index(selected.size() + 2u * kGuard, sentinel), remaining_initial(index.size(), sentinel);
    std::copy(selected.begin(), selected.end(), index.begin() + kGuard);
    std::vector<unsigned> wf(rows + 2u * kGuard, sentinel), inf(tokens + 2u * kGuard, sentinel), counter(2u + 2u * kGuard, sentinel);
    std::vector<uint8_t> mask_initial(elements + 2u * kGuard, 0xa5u), presence(elements, 0u);
    for (unsigned cell : selected) { require(cell < elements, "candidate range"); presence[cell] = 1u; }
    std::vector<float> raw_initial(selected.size() + 2u * kGuard, kF32Guard);
    std::fill(raw_initial.begin() + kGuard, raw_initial.end() - kGuard, std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<unsigned> ds(index), dr(remaining_initial), dc(counter), dfw(wf), dfi(inf);
    DeviceBuffer<uint8_t> dm(mask_initial); DeviceBuffer<float> raw(raw_initial);
    std::vector<float> control, raw_control;
    for (unsigned variant = 0u; variant < 2u; ++variant) {
        unsigned replay_count = unsigned(selected.size()); double milliseconds = 0.0;
        for (unsigned attempt = 0u; attempt < 2u; ++attempt) {
            hip_ok(hipMemcpy(dout.base, original_output.data(), original_output.size() * sizeof(float), hipMemcpyHostToDevice), "interval reset output");
            hip_ok(hipMemcpy(raw.base, raw_initial.data(), raw_initial.size() * sizeof(float), hipMemcpyHostToDevice), "interval reset raw");
            hip_ok(hipMemcpy(dr.base, remaining_initial.data(), remaining_initial.size() * sizeof(unsigned), hipMemcpyHostToDevice), "interval reset compact");
            const auto start = std::chrono::steady_clock::now();
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(rows), dim3(256u), 0u, nullptr,
                dw.data(), dfw.data(), rows, width);
            hip_ok(hipGetLastError(), "interval weight flags");
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(tokens), dim3(256u), 0u, nullptr,
                di.data(), dfi.data(), tokens, width);
            hip_ok(hipGetLastError(), "interval input flags");
            const unsigned* replay_indices = ds.data();
            if (variant) {
                hip_ok(filter::launch(dw.data(), di.data(), ds.data(), unsigned(selected.size()), dm.data(), elements,
                    dout.data(), rows, tokens, width, dr.data(), selected.size(), dc.data(), nullptr), "interval filter");
                unsigned status[2];
                hip_ok(hipMemcpy(status, dc.data(), sizeof(status), hipMemcpyDeviceToHost), "interval compact count");
                require(status[1] == 0u && status[0] <= selected.size(), "interval candidate status");
                replay_count = status[0]; replay_indices = dr.data();
            }
            if (replay_count) {
                hipLaunchKernelGGL(partition_projection_replay_kernel<0u>, dim3((replay_count * 4u + 255u) / 256u), dim3(256u), 0u, nullptr,
                    dw.data(), di.data(), dfw.data(), dfi.data(), replay_indices, dout.data(), raw.data(), rows, width, replay_count);
                hip_ok(hipGetLastError(), "interval remaining exact replay");
            }
            complete_partition_projection();
            if (attempt) milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        auto output = original_output, unrounded = raw_initial;
        dout.read(output); raw.read(unrounded);
        if (!variant) { control = output; raw_control = unrounded; }
        auto remaining = remaining_initial, count = counter; auto mask = mask_initial;
        dr.read(remaining); dc.read(count); dm.read(mask); dfw.read(wf); dfi.read(inf);
        size_t raw_bad = 0u, bf16_bad = 0u, unrounded_bad = 0u, admitted = 0u;
        std::vector<unsigned> expected_remaining;
        for (size_t cell = 0u; cell < elements; ++cell) {
            require(std::isfinite(output[kGuard + cell]), "interval output not finite");
            raw_bad += std::memcmp(&output[kGuard + cell], &control[kGuard + cell], sizeof(float)) != 0;
            bf16_bad += bf16(output[kGuard + cell]) != reference[kGuard + cell];
            if (variant) {
                const unsigned state = mask[kGuard + cell];
                require(presence[cell] ? (state == 1u || state == 2u) : state == 0u, "interval changed candidate identity");
                admitted += state == 2u;
                if (state == 1u) expected_remaining.push_back(unsigned(cell));
            }
        }
        for (unsigned slot = 0u; slot < replay_count; ++slot) {
            const unsigned cell = variant ? remaining[kGuard + slot] : selected[slot];
            const auto it = std::lower_bound(selected.begin(), selected.end(), cell);
            require(it != selected.end() && *it == cell, "interval introduced candidate");
            require(std::isfinite(unrounded[kGuard + slot]), "interval raw replay missing");
            unrounded_bad += std::memcmp(&unrounded[kGuard + slot], &raw_control[kGuard + size_t(it - selected.begin())], sizeof(float)) != 0;
        }
        for (size_t slot = replay_count; slot < selected.size(); ++slot)
            require(std::isnan(unrounded[kGuard + slot]), "unused raw tail changed");
        if (variant) {
            std::vector<unsigned> actual(remaining.begin() + kGuard, remaining.begin() + kGuard + replay_count);
            std::sort(actual.begin(), actual.end()); require(actual == expected_remaining, "interval compact permutation");
            require(count[kGuard] == replay_count && count[kGuard + 1u] == 0u && admitted + replay_count == selected.size(), "interval complete partition");
            for (size_t slot = replay_count; slot < selected.size(); ++slot)
                require(remaining[kGuard + slot] == sentinel, "unused compact tail changed");
        }
        auto after_w = weights, after_i = inputs; auto after_indices = index;
        dw.read(after_w); di.read(after_i); ds.read(after_indices);
        require(after_w == weights && after_i == inputs && after_indices == index, "interval immutable source changed");
        for (size_t i = 0u; i < kGuard; ++i) {
            require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard &&
                unrounded[i] == kF32Guard && unrounded[kGuard + selected.size() + i] == kF32Guard, "interval float redzone");
            require(mask[i] == 0xa5u && mask[kGuard + elements + i] == 0xa5u, "interval mask redzone");
            require(remaining[i] == sentinel && remaining[kGuard + selected.size() + i] == sentinel &&
                count[i] == sentinel && count[kGuard + 2u + i] == sentinel, "interval compact redzone");
            require(wf[i] == sentinel && wf[kGuard + rows + i] == sentinel && inf[i] == sentinel && inf[kGuard + tokens + i] == sentinel, "interval flags redzone");
        }
        std::cout << "{\"type\":\"interval_projection_real_replay\",\"variant\":" << variant
            << ",\"rows\":" << rows << ",\"tokens\":" << tokens << ",\"k\":" << width << ",\"elements\":" << elements
            << ",\"initial_candidates\":" << selected.size() << ",\"interval_admitted\":" << admitted << ",\"remaining_candidates\":" << replay_count
            << ",\"raw_bit_mismatches\":" << raw_bad << ",\"bf16_mismatches\":" << bf16_bad << ",\"unrounded_replayed_bit_mismatches\":" << unrounded_bad
            << ",\"unrounded_replayed_cells\":" << replay_count << ",\"preparation_filter_and_replay_host_ms\":" << milliseconds
            << ",\"filter_compaction_and_count_read_included\":" << (variant ? "true" : "false")
            << ",\"allocation_upload_and_verification_included\":false,\"warmup_sequences\":1,\"timed_sequences\":1,\"same_initial_candidate_identities\":true"
            << ",\"candidate_partition_pass\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false,\"performance_acceptance\":false}" << std::endl;
        require(!raw_bad && !bf16_bad && !unrounded_bad, "interval projection differs from original or GB10");
    }
}
} // namespace projection_safety_test
