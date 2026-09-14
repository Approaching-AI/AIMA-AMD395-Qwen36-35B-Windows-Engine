#pragma once
#include "projection_strong_replay_suite.h"
#include "../../native/providers/moe_accumulator/sm121_decoded_projection.h"

namespace projection_safety_test {
template<unsigned Variant>
__global__ void decoded_projection_replay_kernel(const uint16_t* weights, const uint16_t* inputs,
    const unsigned* weight_flags, const unsigned* input_flags,
    const uint32_t* packed_weights, const uint32_t* packed_inputs, const unsigned* indices,
    float* output, float* unrounded, unsigned rows, unsigned width, unsigned count) {
    const unsigned slot = (blockIdx.x * blockDim.x + threadIdx.x) / 4u;
    if (slot >= count) return;
    const unsigned cell = indices[slot], row = cell % rows, token = cell / rows;
    const bool eligible = weight_flags[row] && input_flags[token];
    float value;
    if constexpr (Variant == 0u)
        value = qrt_sm121_scalar_projection::validated_dot<4u>(inputs + size_t(token) * width,
            weights + size_t(row) * width, width, eligible);
    else
        value = qrt_sm121_decoded_projection::dot<(Variant == 2u)>(
            inputs + size_t(token) * width, weights + size_t(row) * width,
            packed_inputs + size_t(token) * width,
            Variant == 2u ? packed_weights + size_t(row) * width : nullptr, width, eligible);
    if (!(threadIdx.x & 3u)) {
        unrounded[slot] = value;
        output[cell] = device_bf16_round_to_float(value);
    }
}

void run_decoded_projection_replays(DeviceBuffer<uint16_t>& dw, DeviceBuffer<uint16_t>& di,
    DeviceBuffer<float>& dout, const std::vector<uint16_t>& weights, const std::vector<uint16_t>& inputs,
    const std::vector<uint16_t>& reference, const std::vector<float>& original_output,
    const std::vector<unsigned>& selected, unsigned rows, unsigned tokens, unsigned width) {
    const size_t elements = size_t(rows) * tokens;
    constexpr unsigned marker = 0xa5a5a5a5u;
    std::vector<unsigned> wf(rows + 2u * kGuard, marker), inf(tokens + 2u * kGuard, marker);
    std::vector<unsigned> index(selected.size() + 2u * kGuard, marker);
    std::copy(selected.begin(), selected.end(), index.begin() + kGuard);
    DeviceBuffer<unsigned> dfw(wf), dfi(inf), indices(index);
    std::vector<uint32_t> pw(size_t(rows) * width + 2u * kGuard, marker);
    std::vector<uint32_t> pi(size_t(tokens) * width + 2u * kGuard, marker);
    DeviceBuffer<uint32_t> dpw(pw), dpi(pi);
    std::vector<float> raw_initial(selected.size() + 2u * kGuard, kF32Guard);
    std::fill(raw_initial.begin() + kGuard, raw_initial.end() - kGuard,
        std::numeric_limits<float>::quiet_NaN());
    DeviceBuffer<float> raw(raw_initial);
    std::vector<float> control, raw_control;
    for (unsigned variant = 0u; variant < 3u; ++variant) {
        std::array<double,3> times{};
        for (unsigned attempt = 0u; attempt < 4u; ++attempt) {
            hip_ok(hipMemcpy(dout.base, original_output.data(), original_output.size() * sizeof(float), hipMemcpyHostToDevice), "decoded_reset_output");
            hip_ok(hipMemcpy(raw.base, raw_initial.data(), raw_initial.size() * sizeof(float), hipMemcpyHostToDevice), "decoded_reset_raw");
            hip_ok(hipMemset(dpw.base, 0xa5, pw.size() * sizeof(uint32_t)), "decoded_reset_weights");
            hip_ok(hipMemset(dpi.base, 0xa5, pi.size() * sizeof(uint32_t)), "decoded_reset_inputs");
            const auto start = std::chrono::steady_clock::now();
            if (variant == 2u) {
                hipLaunchKernelGGL(qrt_sm121_decoded_projection::prepare_rows, dim3(rows), dim3(256u), 0u, nullptr,
                    dw.data(), dpw.data(), dfw.data(), rows, width);
            } else {
                hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(rows), dim3(256u), 0u, nullptr,
                    dw.data(), dfw.data(), rows, width);
            }
            hip_ok(hipGetLastError(), "decoded_prepare_weights");
            if (variant) {
                hipLaunchKernelGGL(qrt_sm121_decoded_projection::prepare_rows, dim3(tokens), dim3(256u), 0u, nullptr,
                    di.data(), dpi.data(), dfi.data(), tokens, width);
            } else {
                hipLaunchKernelGGL(qrt_sm121_scalar_projection::eligible_rows_kernel, dim3(tokens), dim3(256u), 0u, nullptr,
                    di.data(), dfi.data(), tokens, width);
            }
            hip_ok(hipGetLastError(), "decoded_prepare_inputs");
            const dim3 grid((unsigned(selected.size()) * 4u + 255u) / 256u);
#define QRT_DECODED_REPLAY_CASE(v) if (variant == v) hipLaunchKernelGGL(decoded_projection_replay_kernel<v>, grid, dim3(256u), 0u, nullptr, dw.data(), di.data(), dfw.data(), dfi.data(), dpw.data(), dpi.data(), indices.data(), dout.data(), raw.data(), rows, width, unsigned(selected.size()))
            QRT_DECODED_REPLAY_CASE(0u); QRT_DECODED_REPLAY_CASE(1u); QRT_DECODED_REPLAY_CASE(2u);
#undef QRT_DECODED_REPLAY_CASE
            hip_ok(hipGetLastError(), "decoded_replay_launch"); complete_strong_projection();
            if (attempt) times[attempt - 1u] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
        auto ordered = times; std::sort(ordered.begin(), ordered.end());
        auto output = original_output; dout.read(output);
        auto unrounded = raw_initial; raw.read(unrounded);
        if (!variant) { control = output; raw_control = unrounded; }
        size_t raw_bad = 0u, bf16_bad = 0u, unrounded_bad = 0u;
        for (size_t i = 0u; i < selected.size(); ++i) {
            require(std::isfinite(unrounded[kGuard + i]), "decoded raw accumulator not written");
            unrounded_bad += std::memcmp(&unrounded[kGuard + i], &raw_control[kGuard + i], 4u) != 0;
        }
        for (size_t i = 0u; i < elements; ++i) {
            require(std::isfinite(output[kGuard + i]), "decoded real output nonfinite");
            raw_bad += std::memcmp(&output[kGuard + i], &control[kGuard + i], 4u) != 0;
            bf16_bad += bf16(output[kGuard + i]) != reference[kGuard + i];
        }
        dfw.read(wf); dfi.read(inf); dpw.read(pw); dpi.read(pi);
        const auto verify = [&](const std::vector<uint16_t>& source, const std::vector<unsigned>& flags,
            const std::vector<uint32_t>& packed, unsigned count, bool encoded) {
            unsigned eligible = 0u;
            for (unsigned row = 0u; row < count; ++row) {
                bool valid = true;
                for (unsigned k = 0u; k < width; ++k) {
                    const size_t cell = kGuard + size_t(row) * width + k;
                    const uint16_t x = source[cell]; const unsigned e = (x >> 7u) & 255u;
                    valid &= !(x & 0x7fffu) || (e >= 64u && e <= 190u);
                    const uint32_t expected = (uint32_t(x) << 16u) | uint16_t((x & 0x7fffu) ? int(e) - 127 : -512);
                    require(packed[cell] == (encoded ? expected : marker), "decoded packed word mismatch");
                }
                require(flags[kGuard + row] == unsigned(valid), "decoded row eligibility mismatch");
                eligible += valid;
            }
            for (size_t i = 0u; i < kGuard; ++i) {
                require(flags[i] == marker && flags[kGuard + count + i] == marker, "decoded flag redzone");
                require(packed[i] == marker && packed[kGuard + size_t(count) * width + i] == marker, "decoded operand redzone");
            }
            return eligible;
        };
        const unsigned eligible_weights = verify(weights, wf, pw, rows, variant == 2u);
        const unsigned eligible_inputs = verify(inputs, inf, pi, tokens, variant != 0u);
        auto after_w = weights, after_i = inputs; dw.read(after_w); di.read(after_i);
        auto after_indices = index; indices.read(after_indices);
        require(after_w == weights && after_i == inputs && after_indices == index, "decoded immutable input changed");
        for (size_t i = 0u; i < kGuard; ++i) {
            require(output[i] == kF32Guard && output[kGuard + elements + i] == kF32Guard, "decoded output redzone");
            require(unrounded[i] == kF32Guard && unrounded[kGuard + selected.size() + i] == kF32Guard, "decoded raw redzone");
        }
        const size_t packed_bytes = (variant ? size_t(tokens) * width * 4u : 0u) +
            (variant == 2u ? size_t(rows) * width * 4u : 0u);
        std::cout << "{\"type\":\"decoded_projection_real_replay\",\"variant\":" << variant
            << ",\"lanes\":4,\"rows\":" << rows << ",\"tokens\":" << tokens << ",\"k\":" << width
            << ",\"elements\":" << elements << ",\"candidates\":" << selected.size()
            << ",\"eligible_weights\":" << eligible_weights << ",\"eligible_inputs\":" << eligible_inputs
            << ",\"packed_operand_bytes\":" << packed_bytes
            << ",\"raw_bit_mismatches\":" << raw_bad << ",\"bf16_mismatches\":" << bf16_bad
            << ",\"unrounded_candidate_bit_mismatches\":" << unrounded_bad
            << ",\"preparation_and_replay_host_ms\":" << ordered[1]
            << ",\"completed_host_samples_ms\":[" << times[0] << "," << times[1] << "," << times[2] << "]"
            << ",\"warmup_sequences\":1,\"timed_sequences\":3,\"raw_capture_writes_included\":true"
            << ",\"all_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}" << std::endl;
        require(!raw_bad && !bf16_bad && !unrounded_bad, "decoded projection differs from control or GB10");
    }
}
} // namespace projection_safety_test
