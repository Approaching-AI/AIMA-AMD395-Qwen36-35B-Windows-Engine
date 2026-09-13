#pragma once
#include "../../native/providers/moe_accumulator/sm121_replay_partition.h"
#include "../../native/providers/moe_accumulator/sm121_range_projection.h"

namespace projection_safety_test {
template<unsigned Variant>
__global__ void partition_projection_replay_kernel(const uint16_t* weights, const uint16_t* inputs,
    const unsigned* weight_flags, const unsigned* input_flags, const unsigned* indices,
    float* output, float* unrounded, unsigned rows, unsigned width, unsigned count) {
    constexpr unsigned lanes = 4u;
    constexpr unsigned staging = 1u;
    const unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (slot >= count) return;
    const unsigned cell = indices[slot], row = cell % rows, token = cell / rows;
    const unsigned common_flags = weight_flags[row] & input_flags[token];
    const bool eligible = (common_flags & 1u) != 0u;
    float value;
    if constexpr (Variant < 2u)
        value = qrt_sm121_scalar_projection::validated_dot<lanes, staging>(inputs + size_t(token) * width,
            weights + size_t(row) * width, width, eligible);
    else
        value = qrt_sm121_range_projection::dot<lanes, staging>(inputs + size_t(token) * width,
            weights + size_t(row) * width, width, common_flags);
    if (!(threadIdx.x & (lanes - 1u))) {
        unrounded[slot] = value;
        output[cell] = device_bf16_round_to_float(value);
    }
}

void complete_partition_projection() {
    hipEvent_t event; hip_ok(hipEventCreate(&event), "scalar_event_create");
    hip_ok(hipEventRecord(event), "scalar_event_record");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) hip_ok(status, "scalar_event_query");
        require(std::chrono::steady_clock::now() < deadline, "scalar replay deadline");
        std::this_thread::yield();
    }
    hip_ok(hipEventDestroy(event), "scalar_event_destroy");
}

void run_partition_projection_replays(DeviceBuffer<uint16_t>& dw, DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout, const std::vector<uint16_t>& weights, const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference, const std::vector<float>& original_output,
    const std::vector<unsigned>& selected, unsigned rows, unsigned tokens, unsigned width) {
    const size_t elements = size_t(rows) * tokens;
    require(!selected.empty() && std::is_sorted(selected.begin(), selected.end()) &&
        std::adjacent_find(selected.begin(), selected.end()) == selected.end(),
        "partition comparison requires ascending unique candidate identities");
    constexpr unsigned flag_guard = 0xa5a5a5a5u;
    std::vector<unsigned> wf(rows + 2u * kGuard, flag_guard), inf(tokens + 2u * kGuard, flag_guard);
    std::vector<unsigned> index(selected.size() + 2u * kGuard, flag_guard);
    std::copy(selected.begin(), selected.end(), index.begin() + kGuard);
    DeviceBuffer<unsigned> dfw(wf), dfi(inf), indices(index);
    std::vector<unsigned> partition_initial(index.size(), flag_guard);
    std::vector<unsigned> count_initial(2u + 2u * kGuard, flag_guard);
    count_initial[kGuard] = count_initial[kGuard + 1u] = 0u;
    DeviceBuffer<unsigned> ordered(partition_initial), counts(count_initial);
    std::vector<float> raw_initial(selected.size() + 2u * kGuard, kF32Guard);
    std::fill(raw_initial.begin() + kGuard, raw_initial.end() - kGuard,
        std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<float> raw(raw_initial);
    std::vector<float> control, raw_control;
    for (unsigned variant = 0u; variant < 4u; ++variant) {
        const bool partitioned = variant == 1u || variant == 2u;
        double ms = 0.0;
        for (unsigned attempt = 0u; attempt < 2u; ++attempt) {
            hip_ok(hipMemcpy(dout.base, original_output.data(), original_output.size() * sizeof(float), hipMemcpyHostToDevice), "scalar_reset_output");
            hip_ok(hipMemcpy(raw.base, raw_initial.data(), raw_initial.size() * sizeof(float), hipMemcpyHostToDevice), "range_reset_raw");
            const auto start = std::chrono::steady_clock::now();
            hipLaunchKernelGGL(qrt_sm121_range_projection::classify_rows_kernel, dim3(rows), dim3(256u), 0u, nullptr,
                dw.data(), dfw.data(), rows, width);
            hip_ok(hipGetLastError(), "range_weight_eligibility");
            hipLaunchKernelGGL(qrt_sm121_range_projection::classify_rows_kernel, dim3(tokens), dim3(256u), 0u, nullptr,
                di.data(), dfi.data(), tokens, width);
            hip_ok(hipGetLastError(), "scalar_input_preparation");
            if (partitioned) {
                hip_ok(hipMemsetAsync(counts.data(), 0, 2u * sizeof(unsigned), nullptr), "partition_reset_counts");
                hipLaunchKernelGGL(qrt_sm121_replay_partition::indices_kernel,
                    dim3((unsigned(selected.size()) + 255u) / 256u), dim3(256u), 0u, nullptr,
                    indices.data(), ordered.data(), counts.data(), dfw.data(), dfi.data(), rows, unsigned(selected.size()));
                hip_ok(hipGetLastError(), "partition_selected_cells");
            }
            const unsigned* replay_indices = partitioned ? ordered.data() : indices.data();
            const unsigned lanes = 4u;
            const dim3 grid((unsigned(selected.size()) * lanes + 255u) / 256u);
#define QRT_SCALAR_REPLAY_CASE(v) if (variant == v) hipLaunchKernelGGL(partition_projection_replay_kernel<v>, grid, dim3(256u), 0u, nullptr, dw.data(), di.data(), dfw.data(), dfi.data(), replay_indices, dout.data(), raw.data(), rows, width, unsigned(selected.size()))
            QRT_SCALAR_REPLAY_CASE(0u); QRT_SCALAR_REPLAY_CASE(1u); QRT_SCALAR_REPLAY_CASE(2u); QRT_SCALAR_REPLAY_CASE(3u);
#undef QRT_SCALAR_REPLAY_CASE
            hip_ok(hipGetLastError(), "scalar_replay_launch"); complete_partition_projection();
            if (attempt) ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        auto output = original_output; dout.read(output);
        auto unrounded = raw_initial; raw.read(unrounded);
        if (!variant) { control = output; raw_control = unrounded; }
        size_t raw_bad = 0u, bf16_bad = 0u, unrounded_bad = 0u;
        auto partitioned_indices = partition_initial;
        auto partition_counts = count_initial;
        if (partitioned) { ordered.read(partitioned_indices); counts.read(partition_counts); }
        std::vector<std::pair<unsigned,uint32_t>> by_cell;
        if (partitioned) by_cell.reserve(selected.size());
        for (size_t i = 0u; i < selected.size(); ++i) {
            require(std::isfinite(unrounded[kGuard + i]), "candidate raw accumulator not written");
            uint32_t bits; std::memcpy(&bits, &unrounded[kGuard + i], 4u);
            if (partitioned) by_cell.emplace_back(partitioned_indices[kGuard + i], bits);
            else unrounded_bad += std::memcmp(&unrounded[kGuard + i], &raw_control[kGuard + i], 4u) != 0;
        }
        if (partitioned) {
            std::sort(by_cell.begin(), by_cell.end());
            for (size_t i = 0u; i < selected.size(); ++i) {
                require(by_cell[i].first == selected[i], "partition changed candidate identities");
                uint32_t expected; std::memcpy(&expected, &raw_control[kGuard + i], 4u);
                unrounded_bad += by_cell[i].second != expected;
            }
        }
        for (size_t i = 0u; i < elements; ++i) {
            require(std::isfinite(output[kGuard + i]), "scalar real output nonfinite");
            raw_bad += std::memcmp(&output[kGuard + i], &control[kGuard + i], 4u) != 0;
            bf16_bad += bf16(output[kGuard + i]) != reference[kGuard + i];
        }
        dfw.read(wf); dfi.read(inf); unsigned eligible_weights = 0u, eligible_inputs = 0u, certified_weights = 0u, certified_inputs = 0u;
        const auto check_flags = [&](const std::vector<uint16_t>& source, const std::vector<unsigned>& flags, unsigned count, unsigned& certified) {
            unsigned eligible = 0u;
            for (unsigned row = 0u; row < count; ++row) {
                bool valid = true; bool bounded = width > 0u && width <= 16384u && width % 16u == 0u;
                for (unsigned k = 0u; k < width; ++k) {
                    const uint16_t x = source[kGuard + size_t(row) * width + k]; const unsigned e = (x >> 7u) & 255u;
                    valid &= !(x & 0x7fffu) || (e >= 64u && e <= 190u);
                    bounded &= !(x & 0x7fffu) || (e >= 77u && e <= 179u);
                }
                require(flags[kGuard + row] == (unsigned(valid) | (unsigned(bounded) << 1u)), "real row eligibility mismatch"); eligible += valid; certified += bounded;
            }
            return eligible;
        };
        eligible_weights = check_flags(weights, wf, rows, certified_weights); eligible_inputs = check_flags(inputs, inf, tokens, certified_inputs);
        size_t certified_candidates = 0u;
        for (unsigned cell : selected) certified_candidates += (wf[kGuard + cell % rows] & inf[kGuard + cell / rows] & 2u) != 0u;
        size_t floating_candidates = 0u;
        for (unsigned cell : selected) floating_candidates += (wf[kGuard + cell % rows] & inf[kGuard + cell / rows] & 1u) != 0u;
        if (partitioned) {
            require(partition_counts[kGuard] == floating_candidates && partition_counts[kGuard + 1u] == selected.size() - floating_candidates, "partition counts differ");
            for (size_t i = 0; i < selected.size(); ++i) {
                const unsigned cell = partitioned_indices[kGuard + i];
                const bool is_float = (wf[kGuard + cell % rows] & inf[kGuard + cell / rows] & 1u) != 0u;
                require(is_float == (i < floating_candidates), "partition class order differs");
            }
            for (size_t i = 0; i < kGuard; ++i) {
                require(partitioned_indices[i] == flag_guard && partitioned_indices[kGuard + selected.size() + i] == flag_guard &&
                    partition_counts[i] == flag_guard && partition_counts[kGuard + 2u + i] == flag_guard, "partition scratch redzone");
            }
        }
        auto after_w = weights, after_i = inputs; dw.read(after_w); di.read(after_i);
        auto after_indices = index; indices.read(after_indices);
        require(after_w == weights && after_i == inputs && after_indices == index, "scalar immutable inputs changed");
        for (size_t i = 0u; i < kGuard; ++i) {
            require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard, "scalar output redzone");
            require(wf[i] == flag_guard && wf[kGuard + rows + i] == flag_guard && inf[i] == flag_guard && inf[kGuard + tokens + i] == flag_guard, "scalar flag redzone");
            require(unrounded[i] == kF32Guard && unrounded[kGuard + selected.size() + i] == kF32Guard, "raw accumulator redzone");
        }
        std::cout << "{\"type\":\"partition_projection_real_replay\",\"variant\":" << variant
            << ",\"rows\":" << rows << ",\"tokens\":" << tokens << ",\"k\":" << width
            << ",\"elements\":" << elements << ",\"candidates\":" << selected.size()
            << ",\"eligible_weights\":" << eligible_weights << ",\"eligible_inputs\":" << eligible_inputs
            << ",\"certified_weights\":" << certified_weights << ",\"certified_inputs\":" << certified_inputs
            << ",\"certified_candidates\":" << certified_candidates
            << ",\"raw_bit_mismatches\":" << raw_bad << ",\"bf16_mismatches\":" << bf16_bad
            << ",\"unrounded_candidate_bit_mismatches\":" << unrounded_bad
            << ",\"gpu_partitioned\":" << (partitioned ? "true" : "false")
            << ",\"range_normalization\":" << (variant >= 2u ? "true" : "false")
            << ",\"floating_candidates\":" << floating_candidates
            << ",\"integer_candidates\":" << selected.size() - floating_candidates
            << ",\"partition_in_timed_sequence\":" << (partitioned ? "true" : "false")
            << ",\"raw_values_compared_by_cell\":true,\"raw_capture_writes_included\":true"
            << ",\"preparation_and_replay_host_ms\":" << ms
            << ",\"warmup_sequences\":1,\"timed_sequences\":1,\"same_candidate_identities\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}" << std::endl;
        require(!raw_bad && !bf16_bad && !unrounded_bad, "captured projection differs from control or GB10");
    }
}
} // namespace projection_safety_test
