#pragma once
#include <array>

namespace weight_control_throughput {
struct Pair { unsigned input, weight; };
template<unsigned Variant>
__global__ void replay(const scaled::Row* inputs, const scaled::Row* packed_weights,
    const uint16_t* weights, const unsigned* metadata, const Pair* pairs,
    float* output, unsigned width, unsigned count) {
    const unsigned index = (blockIdx.x * blockDim.x + threadIdx.x) / 4u;
    if (index >= count) return;
    const Pair pair = pairs[index];
    const auto* input = inputs + size_t(pair.input) * (width / 16u);
    float result;
    if constexpr (Variant == 0u)
        result = qrt_sm121_staged_half_projection::dot<2u>(input,
            packed_weights + size_t(pair.weight) * (width / 16u), width);
    else result = projection::dot<2u>(input, weights + size_t(pair.weight) * width,
        metadata + size_t(pair.weight) * (width / 16u), width);
    if (!(threadIdx.x & 3u)) output[index] = result;
}
unsigned random_word(unsigned x) {
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u;
    x *= 0x846ca68bu; return x ^ (x >> 16u);
}
uint16_t operand(size_t index, unsigned seed) {
    const unsigned x = random_word(unsigned(index) ^ seed);
    if (x % 257u == 0u) return uint16_t((x & 1u) << 15u);
    return uint16_t(((x & 1u) << 15u) | ((117u + (x >> 1u) % 15u) << 7u) | ((x >> 8u) & 127u));
}
double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now() - start).count();
}
void run(unsigned weight_rows, unsigned input_rows, unsigned width, unsigned selected) {
    const size_t ww = size_t(weight_rows) * width, iw = size_t(input_rows) * width;
    const size_t wg = ww / 16u, ig = iw / 16u;
    std::vector<uint16_t> weights(ww), inputs(iw);
    for (size_t i = 0u; i < ww; ++i) weights[i] = operand(i, 0x3f592ce1u);
    for (size_t i = 0u; i < iw; ++i) inputs[i] = operand(i, 0x15a41f29u);
    std::vector<Pair> pairs(selected);
    const unsigned per_expert = weight_rows / 256u;
    // A stable expert-ordered selected workload. Every pair is in range and
    // spans all experts, weight rows and live activation rows. This synthetic
    // selection is not a model selector or a GB10 inference boundary.
    for (unsigned i = 0u; i < selected; ++i) {
        const unsigned expert = unsigned(size_t(i) * 256u / selected);
        pairs[i] = {random_word(i + 17u) % input_rows,
            expert * per_expert + random_word(i + 311u) % per_expert};
    }
    std::array<unsigned,256> cpu_indices{};
    std::array<float,256> cpu_values{};
    for (unsigned i = 0u; i < 256u; ++i) {
        const unsigned index = unsigned(size_t(i) * selected / 256u);
        cpu_indices[i] = index;
        const auto pair = pairs[index];
        cpu_values[i] = qrt_q1_moe_hawkeye::dot_bf16_hopper(
            inputs.data() + size_t(pair.input) * width,
            weights.data() + size_t(pair.weight) * width, width);
    }
    Device w((ww + 2u * guard) * 2u), in((iw + 2u * guard) * 2u);
    Device pw((wg + 2u * guard) * sizeof(scaled::Row));
    Device pi((ig + 2u * guard) * sizeof(scaled::Row));
    Device meta((wg + 2u * guard) * 4u), ids((selected + 2u * guard) * sizeof(Pair));
    Device out((selected + 2u * guard) * 4u);
    check(hipMemcpy(w.data<uint16_t>(), weights.data(), ww * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(in.data<uint16_t>(), inputs.data(), iw * 2u, hipMemcpyHostToDevice));
    check(hipMemcpy(ids.data<Pair>(), pairs.data(), selected * sizeof(Pair), hipMemcpyHostToDevice));
    finish();
    const auto cache_start = std::chrono::steady_clock::now();
    check(projection::prepare(w.data<uint16_t>(), ww, meta.data<unsigned>(), wg, weight_rows, width, nullptr));
    finish(); const double cache_ms = milliseconds(cache_start);
    const auto initial_metadata = read<unsigned>(meta, wg); guards(initial_metadata);
    // Validate every cached word and every reconstructed halfword against the
    // original prepared representation, independently of the replay kernels.
    for (size_t g = 0u; g < wg; ++g) {
        const auto expected = scaled::prepare(weights.data() + g * 16u);
        if (initial_metadata[guard + g] != expected.control)
            throw std::runtime_error("large-shape cached control differs");
        for (unsigned pair = 0u; pair < 8u; ++pair) {
            const unsigned value = unsigned(weights[g * 16u + pair * 2u]) |
                (unsigned(weights[g * 16u + pair * 2u + 1u]) << 16u);
            if (projection::encode_pair(value, initial_metadata[guard + g], pair) != expected.pairs[pair])
                throw std::runtime_error("large-shape reconstructed operands differ");
        }
    }
    double samples[3][3]{};
    std::vector<float> control_output;
    for (unsigned attempt = 0u; attempt < 4u; ++attempt) for (unsigned position = 0u; position < 3u; ++position) {
        const unsigned variant = (attempt + position) % 3u;
        check(hipMemset(out.pointer, 0xa5, (selected + 2u * guard) * 4u));
        finish(); const auto start = std::chrono::steady_clock::now();
        hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,
            dim3((ig + 255u) / 256u), dim3(256u), 0u, nullptr,
            in.data<uint16_t>(), pi.data<scaled::Row>(), input_rows, width);
        check(hipGetLastError());
        if (!variant) {
            hipLaunchKernelGGL(qrt_sm121_scaled_half_projection::prepare_rows,
                dim3((wg + 255u) / 256u), dim3(256u), 0u, nullptr,
                w.data<uint16_t>(), pw.data<scaled::Row>(), weight_rows, width);
            check(hipGetLastError());
        } else if (variant == 2u)
            check(projection::prepare(w.data<uint16_t>(), ww, meta.data<unsigned>(), wg, weight_rows, width, nullptr));
        for (unsigned offset = 0u; offset < selected; offset += 262144u) {
            const unsigned count = std::min(262144u, selected - offset);
#define WEIGHT_CONTROL_REPLAY(v) if (variant == v) hipLaunchKernelGGL(replay<v>, dim3((count + 63u) / 64u), dim3(256u), 0u, nullptr, pi.data<scaled::Row>(), pw.data<scaled::Row>(), w.data<uint16_t>(), meta.data<unsigned>(), ids.data<Pair>() + offset, out.data<float>() + offset, width, count)
            WEIGHT_CONTROL_REPLAY(0u); WEIGHT_CONTROL_REPLAY(1u); WEIGHT_CONTROL_REPLAY(2u);
#undef WEIGHT_CONTROL_REPLAY
            check(hipGetLastError());
        }
        finish(); const double elapsed = milliseconds(start);
        if (attempt) samples[variant][attempt - 1u] = elapsed;
        const auto actual = read<float>(out, selected); guards(actual);
        if (!attempt && !variant) control_output = actual;
        if (actual.size() != control_output.size() ||
            std::memcmp(actual.data(), control_output.data(), actual.size() * 4u))
            throw std::runtime_error("complete selected output differs");
        for (unsigned i = 0u; i < 256u; ++i)
            if (std::memcmp(&actual[guard + cpu_indices[i]], &cpu_values[i], 4u))
                throw std::runtime_error("independent sampled CPU dot differs");
    }
    const auto after_weights = read<uint16_t>(w, ww), after_inputs = read<uint16_t>(in, iw);
    const auto after_pairs = read<Pair>(ids, selected);
    guards(after_weights); guards(after_inputs); guards(after_pairs);
    if (std::memcmp(after_weights.data() + guard, weights.data(), ww * 2u) ||
        std::memcmp(after_inputs.data() + guard, inputs.data(), iw * 2u) ||
        std::memcmp(after_pairs.data() + guard, pairs.data(), selected * sizeof(Pair)) ||
        read<unsigned>(meta, wg) != initial_metadata)
        throw std::runtime_error("large-shape immutable source or controls changed");
    const auto packed_inputs = read<scaled::Row>(pi, ig); guards(packed_inputs);
    const auto packed_weights = read<scaled::Row>(pw, wg); guards(packed_weights);
    for (unsigned side = 0u; side < 2u; ++side) {
        const auto& raw = side ? weights : inputs;
        const auto& packed = side ? packed_weights : packed_inputs;
        for (size_t g = 0u; g < raw.size() / 16u; ++g) {
            const auto expected = scaled::prepare(raw.data() + g * 16u);
            if (std::memcmp(&packed[guard + g], &expected, sizeof(expected)))
                throw std::runtime_error("retained prepared operand differs");
        }
    }
    for (unsigned variant = 0u; variant < 3u; ++variant) {
        std::array<double,3> ordered{samples[variant][0],samples[variant][1],samples[variant][2]};
        std::sort(ordered.begin(),ordered.end());
        std::printf("{\"kind\":\"weight_control_projection_throughput\",\"variant\":%u,\"weight_rows\":%u,\"input_rows\":%u,\"k\":%u,\"selected\":%u,\"experts\":256,\"weight_bytes\":%zu,\"prepared_weight_bytes\":%zu,\"control_cache_bytes\":%zu,\"initial_cache_ms\":%.6f,\"completed_owner_median_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"initial_cache_in_owner\":%s,\"activation_preparation_in_owner\":true,\"original_weight_preparation_in_control\":true,\"warmups\":1,\"measured_attempts\":3,\"cpu_sampled_dots\":256,\"all_attempts_verified\":true,\"raw_bit_mismatches\":0,\"all_control_and_encoded_words_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"model_loaded\":false,\"gb10_qualified\":false,\"performance_acceptance\":false}\n",
            variant,weight_rows,input_rows,width,selected,ww*2u,wg*sizeof(scaled::Row),wg*4u,cache_ms,ordered[1],
            samples[variant][0],samples[variant][1],samples[variant][2],variant==2u?"true":"false");
        std::fflush(stdout);
    }
}
} // namespace weight_control_throughput
int throughput_main() {
    hipDeviceProp_t p{}; check(hipGetDeviceProperties(&p,0));
    if (std::strncmp(p.gcnArchName,"gfx1151",7u)) throw std::runtime_error("requires gfx1151");
    weight_control_throughput::run(262144u,8192u,2048u,4194304u);
    weight_control_throughput::run(524288u,65536u,512u,2097152u);
    return 0;
}
