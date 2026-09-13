#pragma once
#include "../../native/providers/moe_accumulator/sm121_f32_carry_projection.h"

namespace projection_safety_test {
template<unsigned Variant>
__global__ void f32_carry_projection_replay_kernel(const uint16_t* weights, const uint16_t* inputs,
    const unsigned* weight_flags, const unsigned* input_flags, const unsigned* indices,
    float* output, float* unrounded, unsigned rows, unsigned width, unsigned count) {
    constexpr unsigned lanes = Variant < 3u ? 4u : 16u;
    constexpr unsigned staging = Variant < 3u ? 1u : 4u;
    const unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (slot >= count) return;
    const unsigned cell = indices[slot], row = cell % rows, token = cell / rows;
    const bool eligible = weight_flags[row] && input_flags[token];
    float value;
    if constexpr (Variant % 3u == 0u)
        value = qrt_sm121_scalar_projection::validated_dot<lanes, staging>(inputs + size_t(token) * width,
            weights + size_t(row) * width, width, eligible);
    else
        value = qrt_sm121_f32_carry_projection::dot<Variant % 3u - 1u, lanes, staging>(inputs + size_t(token) * width,
            weights + size_t(row) * width, width, eligible);
    if (!(threadIdx.x & (lanes - 1u))) {
        unrounded[slot] = value;
        output[cell] = device_bf16_round_to_float(value);
    }
}

void complete_f32_carry_projection() {
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

void run_f32_carry_projection_replays(DeviceBuffer<uint16_t>& dw, DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout, const std::vector<uint16_t>& weights, const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference, const std::vector<float>& original_output,
    const std::vector<unsigned>& selected, unsigned rows, unsigned tokens, unsigned width) {
    const size_t elements = size_t(rows) * tokens;
    constexpr unsigned flag_guard = 0xa5a5a5a5u;
    std::vector<unsigned> wf(rows + 2u * kGuard, flag_guard), inf(tokens + 2u * kGuard, flag_guard);
    std::vector<unsigned> index(selected.size() + 2u * kGuard, flag_guard);
    std::copy(selected.begin(), selected.end(), index.begin() + kGuard);
    DeviceBuffer<unsigned> dfw(wf), dfi(inf), indices(index);
    std::vector<float> raw_initial(selected.size() + 2u * kGuard, kF32Guard);
    std::fill(raw_initial.begin() + kGuard, raw_initial.end() - kGuard,
        std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<float> raw(raw_initial);
    std::vector<float> control, raw_control;
    for (unsigned variant = 0u; variant < 6u; ++variant) {
        double ms = 0.0;
        for (unsigned attempt = 0u; attempt < 2u; ++attempt) {
            hip_ok(hipMemcpy(dout.base, original_output.data(), original_output.size() * sizeof(float), hipMemcpyHostToDevice), "scalar_reset_output");
            hip_ok(hipMemcpy(raw.base, raw_initial.data(), raw_initial.size() * sizeof(float), hipMemcpyHostToDevice), "f32_carry_reset_raw");
            const auto start = std::chrono::steady_clock::now();
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(rows), dim3(256u), 0u, nullptr,
                dw.data(), dfw.data(), rows, width);
            hip_ok(hipGetLastError(), "f32_carry_weight_eligibility");
            hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(tokens), dim3(256u), 0u, nullptr,
                di.data(), dfi.data(), tokens, width);
            hip_ok(hipGetLastError(), "scalar_input_preparation");
            const unsigned lanes = variant < 3u ? 4u : 16u;
            const dim3 grid((unsigned(selected.size()) * lanes + 255u) / 256u);
#define QRT_SCALAR_REPLAY_CASE(v) if (variant == v) hipLaunchKernelGGL(f32_carry_projection_replay_kernel<v>, grid, dim3(256u), 0u, nullptr, dw.data(), di.data(), dfw.data(), dfi.data(), indices.data(), dout.data(), raw.data(), rows, width, unsigned(selected.size()))
            QRT_SCALAR_REPLAY_CASE(0u); QRT_SCALAR_REPLAY_CASE(1u); QRT_SCALAR_REPLAY_CASE(2u); QRT_SCALAR_REPLAY_CASE(3u); QRT_SCALAR_REPLAY_CASE(4u); QRT_SCALAR_REPLAY_CASE(5u);
#undef QRT_SCALAR_REPLAY_CASE
            hip_ok(hipGetLastError(), "scalar_replay_launch"); complete_f32_carry_projection();
            if (attempt) ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        auto output = original_output; dout.read(output);
        auto unrounded = raw_initial; raw.read(unrounded);
        if (!variant) { control = output; raw_control = unrounded; }
        size_t raw_bad = 0u, bf16_bad = 0u, unrounded_bad = 0u;
        for (size_t i = 0u; i < selected.size(); ++i) {
            require(std::isfinite(unrounded[kGuard + i]),
                "candidate raw accumulator not written");
            unrounded_bad += std::memcmp(&unrounded[kGuard + i], &raw_control[kGuard + i], 4u) != 0;
        }
        for (size_t i = 0u; i < elements; ++i) {
            require(std::isfinite(output[kGuard + i]), "scalar real output nonfinite");
            raw_bad += std::memcmp(&output[kGuard + i], &control[kGuard + i], 4u) != 0;
            bf16_bad += bf16(output[kGuard + i]) != reference[kGuard + i];
        }
        dfw.read(wf); dfi.read(inf); unsigned eligible_weights = 0u, eligible_inputs = 0u;
        const auto check_flags = [&](const std::vector<uint16_t>& source, const std::vector<unsigned>& flags, unsigned count) {
            unsigned eligible = 0u;
            for (unsigned row = 0u; row < count; ++row) {
                bool valid = true;
                for (unsigned k = 0u; k < width; ++k) {
                    const uint16_t x = source[kGuard + size_t(row) * width + k]; const unsigned e = (x >> 7u) & 255u;
                    valid &= !(x & 0x7fffu) || (e >= 64u && e <= 190u);
                }
                require(flags[kGuard + row] == unsigned(valid), "real row eligibility mismatch"); eligible += valid;
            }
            return eligible;
        };
        eligible_weights = check_flags(weights, wf, rows); eligible_inputs = check_flags(inputs, inf, tokens);
        auto after_w = weights, after_i = inputs; dw.read(after_w); di.read(after_i);
        auto after_indices = index; indices.read(after_indices);
        require(after_w == weights && after_i == inputs && after_indices == index, "scalar immutable inputs changed");
        for (size_t i = 0u; i < kGuard; ++i) {
            require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard, "scalar output redzone");
            require(wf[i] == flag_guard && wf[kGuard + rows + i] == flag_guard && inf[i] == flag_guard && inf[kGuard + tokens + i] == flag_guard, "scalar flag redzone");
            require(unrounded[i] == kF32Guard && unrounded[kGuard + selected.size() + i] == kF32Guard, "raw accumulator redzone");
        }
        std::cout << "{\"type\":\"f32_carry_projection_real_replay\",\"variant\":" << variant
            << ",\"rows\":" << rows << ",\"tokens\":" << tokens << ",\"k\":" << width
            << ",\"elements\":" << elements << ",\"candidates\":" << selected.size()
            << ",\"eligible_weights\":" << eligible_weights << ",\"eligible_inputs\":" << eligible_inputs
            << ",\"raw_bit_mismatches\":" << raw_bad << ",\"bf16_mismatches\":" << bf16_bad
            << ",\"unrounded_candidate_bit_mismatches\":" << unrounded_bad
            << ",\"raw_capture_writes_included\":true"
            << ",\"preparation_and_replay_host_ms\":" << ms
            << ",\"warmup_sequences\":1,\"timed_sequences\":1,\"sorted_common_candidate_order\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}" << std::endl;
        require(!raw_bad && !bf16_bad && !unrounded_bad, "captured projection differs from control or GB10");
    }
}
} // namespace projection_safety_test
