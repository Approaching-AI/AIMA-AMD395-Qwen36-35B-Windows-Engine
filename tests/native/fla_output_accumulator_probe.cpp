// CPU-only output-stage attribution using captured GB10 intermediate inputs.
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

template<class T> bool load(const char* path, std::vector<T>& data) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(data.size() * sizeof(T))) return false;
    file.seekg(0);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(T));
    return static_cast<bool>(file);
}
float value(uint16_t bits) {
    const uint32_t raw = uint32_t(bits) << 16;
    float result;
    std::memcpy(&result, &raw, 4);
    return result;
}
uint16_t bf16(float input) {
    uint32_t bits;
    std::memcpy(&bits, &input, 4);
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float exact_dot(const uint16_t* left, const uint16_t* right, int count, int block) {
    float sum = 0;
    for (int i = 0; i < count; i += block) {
        sum += qrt_q1_moe_hawkeye::dot_bf16_impl<26, 16, -133>(left + i, right + i, block);
    }
    return sum;
}
float ieee_dot(const uint16_t* left, const uint16_t* right, int count) {
    float sum = 0;
    for (int i = 0; i < count; ++i) sum = std::fma(value(left[i]), value(right[i]), sum);
    return sum;
}
bool selected(int index, int tokens) {
    return tokens <= 64 || ((uint32_t(index) * UINT32_C(0x9e3779b9)) >> 21) == 0;
}
int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: fla-output-accumulator-probe <q-bf16> <k-bf16> <v-new-bf16> <chunk-state-bf16> <g-cumsum-f32> <gb10-output-bf16> <tokens>\n";
        return 2;
    }
    char* end = nullptr;
    const long parsed = std::strtol(argv[7], &end, 10);
    if (!end || *end || parsed < 1 || parsed > 8192) return 2;
    const int tokens = static_cast<int>(parsed), chunks = (tokens + 63) / 64;
    std::vector<uint16_t> q(tokens * 16 * 128), k(q.size()), v(tokens * 32 * 128);
    std::vector<uint16_t> h(chunks * 32 * 128 * 128), reference(v.size());
    std::vector<float> g(tokens * 32);
    if (!load(argv[1], q) || !load(argv[2], k) || !load(argv[3], v) ||
        !load(argv[4], h) || !load(argv[5], g) || !load(argv[6], reference)) return 3;
    for (const auto* surface : {&q, &k, &v, &h, &reference}) {
        for (uint16_t x : *surface) if (!std::isfinite(value(x))) return 4;
    }
    for (float x : g) if (!std::isfinite(x)) return 4;
    struct Variant { const char* name; int projection_block; bool output_ieee; };
    const std::array<Variant, 4> variants{{
        {"blackwell_k128_blackwell_k64", 128, false},
        {"blackwell_two_k64_blackwell_k64", 64, false},
        {"blackwell_k128_ieee_k64", 128, true},
        {"ieee_k128_ieee_k64", 0, true}
    }};
    constexpr float scale = 0.08838834764831845f, log2e = 1.4426950408889634074f;
    std::cout << std::setprecision(17)
              << "{\"kind\":\"cpu_output_stage_attribution\",\"tokens\":" << tokens
              << ",\"sampled\":" << (tokens > 64 ? "true" : "false")
              << ",\"exponent\":\"host_exp2f_not_sm121_sfu\",\"inference_acceptance\":false,\"variants\":[";
    for (size_t mode = 0; mode < variants.size(); ++mode) {
        const auto& variant = variants[mode];
        uint64_t count = 0, mismatch = 0;
        double error2 = 0, norm2 = 0, maximum = 0;
        int first = -1;
        float first_actual = 0, first_expected = 0;
        for (int t = 0; t < tokens; ++t) for (int head = 0; head < 32; ++head) {
            const int row = t * 32 + head;
            bool row_selected = false;
            for (int d = 0; d < 128; ++d) row_selected |= selected(row * 128 + d, tokens);
            if (!row_selected) continue;
            const auto* query = &q[(t * 16 + head / 2) * 128];
            auto project = [&](const uint16_t* right) {
                return variant.projection_block ? exact_dot(query, right, 128, variant.projection_block)
                                                : ieee_dot(query, right, 128);
            };
            std::array<uint16_t, 64> scores{};
            for (int s = 0; s <= t % 64; ++s) {
                const int source = t / 64 * 64 + s;
                const float dot = project(&k[(source * 16 + head / 2) * 128]);
                const float exponent = (g[row] - g[source * 32 + head]) * log2e;
                scores[s] = bf16(dot * std::exp2(exponent));
            }
            for (int d = 0; d < 128; ++d) {
                const int index = row * 128 + d;
                if (!selected(index, tokens)) continue;
                const auto* state = &h[((t / 64 * 32 + head) * 128 + d) * 128];
                const float old = project(state) * std::exp2(g[row] * log2e);
                std::array<uint16_t, 64> values{};
                for (int s = 0; s < 64 && t / 64 * 64 + s < tokens; ++s) {
                    values[s] = v[((t / 64 * 64 + s) * 32 + head) * 128 + d];
                }
                const float local = variant.output_ieee ? ieee_dot(scores.data(), values.data(), 64)
                                                        : exact_dot(scores.data(), values.data(), 64, 64);
                const uint16_t actual = bf16(std::fma(old, scale, local * scale));
                const float expected = value(reference[index]);
                const double delta = double(value(actual)) - expected;
                ++count; error2 += delta * delta; norm2 += double(expected) * expected;
                if (actual != reference[index]) {
                    ++mismatch; maximum = std::max(maximum, std::abs(delta));
                    if (first < 0) { first = index; first_actual = value(actual); first_expected = expected; }
                }
            }
        }
        if (mode) std::cout << ',';
        std::cout << "{\"name\":\"" << variant.name << "\",\"elements\":" << count
                  << ",\"bit_mismatches\":" << mismatch << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1.0e-300))
                  << ",\"maximum_absolute_error\":" << maximum << ",\"first_index\":" << first
                  << ",\"first_actual\":" << first_actual << ",\"first_expected\":" << first_expected << '}';
    }
    std::cout << "]}\n";
}
