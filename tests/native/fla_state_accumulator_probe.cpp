// CPU-only first-chunk state attribution. Never linked into the runtime.
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t state_elements = 32u * 128u * 128u;
float value(float x) { return x; }
float value(uint16_t x) { uint32_t bits = uint32_t(x) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
uint16_t bf16(float f) { uint32_t bits; std::memcpy(&bits, &f, 4); return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16); }
template<class T> std::vector<T> slice(const std::string& path, size_t total, size_t offset, size_t count) {
    if (offset > total || count > total - offset) throw std::runtime_error("invalid slice");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(total * sizeof(T))) throw std::runtime_error("capture size mismatch");
    std::vector<T> result(count); file.seekg(offset * sizeof(T));
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("capture read failed");
    for (T x : result) if (!std::isfinite(value(x))) throw std::runtime_error("nonfinite capture");
    return result;
}
template<int Block> float blackwell_dot(const uint16_t* a, const uint16_t* b) {
    float result = 0;
    for (int i = 0; i < 64; i += Block) result += qrt_q1_moe_hawkeye::dot_bf16_impl<26, 16, -133>(a + i, b + i, Block);
    return result;
}
float ieee_dot(const uint16_t* a, const uint16_t* b) {
    float result = 0; for (int i = 0; i < 64; ++i) result = std::fma(value(a[i]), value(b[i]), result); return result;
}
struct Stats {
    uint64_t elements = 0, mismatches = 0; double error2 = 0, norm2 = 0, maximum = 0;
    int64_t first = -1; float first_actual = 0, first_expected = 0;
    void add(float a, float b, int64_t index = -1) {
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite result");
        const double delta = double(a) - b; error2 += delta * delta; norm2 += double(b) * b;
        if (a != b) {
            ++mismatches; maximum = std::max(maximum, std::abs(delta));
            if (first < 0) { first = index < 0 ? static_cast<int64_t>(elements) : index; first_actual = a; first_expected = b; }
        }
        ++elements;
    }
    void print() const {
        std::cout << "{\"elements\":" << elements << ",\"mismatch_count\":" << mismatches
                  << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1.0e-300))
                  << ",\"maximum_absolute_error\":" << maximum << ",\"first_index\":" << first
                  << ",\"first_actual\":" << first_actual << ",\"first_expected\":" << first_expected << '}';
    }
};

// Each selected (head,value) row has an independent 128-element recurrent
// state. Carry it through every source token without injecting reference state.
void trajectory(const std::string& directory, size_t tokens) {
    const size_t chunks = (tokens + 63u) / 64u;
    auto path = [&](const char* name) { return directory + "/full-" + name + ".bin"; };
    auto k = slice<uint16_t>(path("k-normalized-bf16"), tokens * 2048u, 0, tokens * 2048u);
    auto w = slice<uint16_t>(path("w-bf16"), tokens * 4096u, 0, tokens * 4096u);
    auto u = slice<uint16_t>(path("u-bf16"), tokens * 4096u, 0, tokens * 4096u);
    auto g = slice<float>(path("g-cumsum-f32"), tokens * 32u, 0, tokens * 32u);
    auto reference_h = slice<uint16_t>(path("chunk-state-bf16"), chunks * state_elements, 0, chunks * state_elements);
    auto reference_v = slice<uint16_t>(path("v-new-bf16"), tokens * 4096u, 0, tokens * 4096u);
    auto reference_final = slice<float>(path("native-final-state-f32"), state_elements, 0, state_elements);
    const std::array<std::array<size_t, 2>, 16> selected{{
        {0,0}, {2,88}, {8,64}, {1,17}, {4,34}, {6,51}, {10,68}, {12,85},
        {14,102}, {16,119}, {18,8}, {20,25}, {22,42}, {24,59}, {28,76}, {31,127}
    }};
    struct Variant { const char* name; bool split_projection, fused_update, ieee; };
    const std::array<Variant, 4> variants{{
        {"blackwell_k128_k64_fma", false, true, false},
        {"blackwell_two_k64_k64_fma", true, true, false},
        {"blackwell_k128_k64_unfused", false, false, false},
        {"ieee_k128_k64_fma", false, true, true}
    }};
    constexpr float log2e = 1.4426950408889634074f;
    std::cout << ",\"trajectory\":{\"sampled_state_rows\":16,\"reference_state_injected\":false,\"coordinates_head_value\":[";
    for (size_t i = 0; i < selected.size(); ++i) { if (i) std::cout << ','; std::cout << '[' << selected[i][0] << ',' << selected[i][1] << ']'; }
    std::cout << "],\"variants\":[";
    for (size_t mode = 0; mode < variants.size(); ++mode) {
        const auto& variant = variants[mode]; Stats states, values, terminal;
        for (const auto& coordinate : selected) {
            const size_t head = coordinate[0], v = coordinate[1];
            std::array<float, 128> state{};
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                const size_t first = chunk * 64u, valid = std::min(size_t(64), tokens - first);
                std::array<uint16_t, 128> rounded{};
                for (size_t d = 0; d < 128; ++d) {
                    rounded[d] = bf16(state[d]); const size_t index = chunk * state_elements + (head * 128u + v) * 128u + d;
                    states.add(value(rounded[d]), value(reference_h[index]), static_cast<int64_t>(index));
                }
                const float gate_last = g[(first + valid - 1u) * 32u + head], decay = std::exp2(gate_last * log2e);
                std::array<uint16_t, 64> residual{};
                for (size_t t = 0; t < valid; ++t) {
                    const size_t row = (first + t) * 32u + head; const auto* left = &w[row * 128u];
                    float projection = 0;
                    if (variant.ieee) {
                        for (size_t d = 0; d < 128; ++d) projection = std::fma(value(left[d]), value(rounded[d]), projection);
                    } else if (variant.split_projection) {
                        projection = blackwell_dot<64>(left, rounded.data()) + blackwell_dot<64>(left + 64, rounded.data() + 64);
                    } else projection = qrt_q1_moe_hawkeye::dot_bf16_impl<26,16,-133>(left, rounded.data(), 128);
                    const float current = value(u[row * 128u + v]) - projection;
                    values.add(value(bf16(current)), value(reference_v[row * 128u + v]), static_cast<int64_t>(row * 128u + v));
                    residual[t] = bf16(current * std::exp2((gate_last - g[row]) * log2e));
                }
                for (size_t d = 0; d < 128; ++d) {
                    std::array<uint16_t, 64> left{};
                    for (size_t t = 0; t < valid; ++t) left[t] = k[((first + t) * 16u + head / 2u) * 128u + d];
                    const float update = variant.ieee ? ieee_dot(left.data(), residual.data()) : blackwell_dot<64>(left.data(), residual.data());
                    state[d] = variant.fused_update ? std::fma(state[d], decay, update) : state[d] * decay + update;
                }
            }
            for (size_t d = 0; d < 128; ++d) {
                const size_t index = (head * 128u + v) * 128u + d;
                terminal.add(state[d], reference_final[index], static_cast<int64_t>(index));
            }
        }
        if (mode) std::cout << ',';
        std::cout << "{\"name\":\"" << variant.name << "\",\"chunk_state_bf16\":"; states.print();
        std::cout << ",\"v_new_bf16\":"; values.print(); std::cout << ",\"final_state_f32\":"; terminal.print(); std::cout << '}';
    }
    std::cout << "]}";
}
}

int main(int argc, char** argv) try {
    if ((argc != 4 && argc != 5) || (argc == 5 && std::string(argv[4]) != "--trajectory-sample")) {
        std::cerr << "usage: fla-state-accumulator-probe <capture-dir> <source-tokens> <native-first-chunk-state-f32|-> [--trajectory-sample]\n"; return 2;
    }
    char* end = nullptr; const unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (!*argv[2] || !end || *end || parsed < 65 || parsed > 8192) return 2;
    const size_t tokens = parsed, chunks = (tokens + 63u) / 64u;
    auto path = [&](const char* name) { return std::string(argv[1]) + "/full-" + name + ".bin"; };
    auto k = slice<uint16_t>(path("k-normalized-bf16"), tokens * 2048u, 0, 64u * 2048u);
    auto u = slice<uint16_t>(path("u-bf16"), tokens * 4096u, 0, 64u * 4096u);
    auto g = slice<float>(path("g-cumsum-f32"), tokens * 32u, 0, 64u * 32u);
    auto seed = slice<float>(path("initial_state-f32"), state_elements, 0, state_elements);
    if (std::any_of(seed.begin(), seed.end(), [](float x) { return x != 0.0f; })) throw std::runtime_error("requires captured zero initial state");
    auto reference = slice<uint16_t>(path("chunk-state-bf16"), chunks * state_elements, state_elements, state_elements);
    std::vector<float> native;
    if (std::string(argv[3]) != "-") native = slice<float>(argv[3], state_elements, 0, state_elements);
    // Zero seed makes the first-chunk residual exactly U. The gated product is
    // rounded before the state-update dot, exactly like the reference formula.
    constexpr float log2e = 1.4426950408889634074f;
    for (size_t t = 0; t < 64; ++t) for (size_t h = 0; h < 32; ++h) {
        const float gate = std::exp2((g[63u * 32u + h] - g[t * 32u + h]) * log2e);
        for (size_t d = 0; d < 128; ++d) u[(t * 32u + h) * 128u + d] = bf16(value(u[(t * 32u + h) * 128u + d]) * gate);
    }
    struct Variant { const char* name; float (*dot)(const uint16_t*, const uint16_t*); };
    const std::array<Variant, 3> variants{{{"blackwell_k64", blackwell_dot<64>},
        {"blackwell_two_k32", blackwell_dot<32>}, {"ieee_fma_sequential", ieee_dot}}};
    Stats native_reference;
    if (!native.empty()) for (size_t i = 0; i < state_elements; ++i) native_reference.add(value(bf16(native[i])), value(reference[i]));
    std::cout << std::setprecision(17) << "{\"kind\":\"cpu_first_chunk_state_attribution\",\"source_tokens\":" << tokens
              << ",\"replay_tokens\":64,\"zero_initial_state\":true,\"exponent\":\"host_exp2f_not_sm121_sfu\",\"inference_acceptance\":false,\"native_reference_bf16\":";
    native_reference.print(); std::cout << ",\"variants\":[";
    for (size_t mode = 0; mode < variants.size(); ++mode) {
        Stats expected, native_f32;
        for (size_t h = 0; h < 32; ++h) for (size_t v = 0; v < 128; ++v) {
            std::array<uint16_t, 64> right{};
            for (size_t t = 0; t < 64; ++t) right[t] = u[(t * 32u + h) * 128u + v];
            for (size_t d = 0; d < 128; ++d) {
                std::array<uint16_t, 64> left{};
                for (size_t t = 0; t < 64; ++t) left[t] = k[(t * 16u + h / 2u) * 128u + d];
                const size_t index = (h * 128u + v) * 128u + d;
                const float result = variants[mode].dot(left.data(), right.data());
                expected.add(value(bf16(result)), value(reference[index]));
                if (!native.empty()) native_f32.add(result, native[index]);
            }
        }
        if (mode) std::cout << ',';
        std::cout << "{\"name\":\"" << variants[mode].name << "\",\"reference_bf16\":"; expected.print();
        std::cout << ",\"native_f32\":"; native_f32.print(); std::cout << '}';
    }
    std::cout << ']';
    if (argc == 5) trajectory(argv[1], tokens);
    std::cout << "}\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 3; }
