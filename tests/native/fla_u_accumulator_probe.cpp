// CPU-only attribution using captured V, beta, inverse, and GB10 U surfaces.
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

bool load(const char* path, std::vector<uint16_t>& data) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(data.size() * 2)) return false;
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(data.data()), data.size() * 2);
    return static_cast<bool>(stream);
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
template<int Width, int Group, int Block>
float exact_dot(const uint16_t* left, const uint16_t* right) {
    float result = 0.0f;
    for (int start = 0; start < 64; start += Block) {
        result += qrt_q1_moe_hawkeye::dot_bf16_impl<Width, Group, (Width == 26 ? -133 : -132)>(
            left + start, right + start, Block);
    }
    return result;
}
float ieee_dot(const uint16_t* left, const uint16_t* right) {
    float result = 0.0f;
    for (int i = 0; i < 64; ++i) result = std::fma(value(left[i]), value(right[i]), result);
    return result;
}
int main(int argc, char** argv) {
    if (argc != 5 && argc != 6) {
        std::cerr << "usage: fla-u-accumulator-probe <v-bf16> <beta-bf16> <inverse-bf16> <gb10-u-bf16> [tokens]\n";
        return 2;
    }
    int tokens = 64;
    if (argc == 6) {
        char* end = nullptr;
        const long parsed = std::strtol(argv[5], &end, 10);
        if (!end || *end || parsed < 1 || parsed > 8192) return 2;
        tokens = static_cast<int>(parsed);
    }
    std::vector<uint16_t> v(tokens * 32 * 128), beta(tokens * 32);
    std::vector<uint16_t> inverse(tokens * 32 * 64), reference(v.size());
    if (!load(argv[1], v) || !load(argv[2], beta) || !load(argv[3], inverse) || !load(argv[4], reference)) return 3;
    for (size_t i = 0; i < v.size(); ++i) {
        const float product = value(v[i]) * value(beta[i / 128]);
        if (!std::isfinite(product)) return 4;
        v[i] = bf16(product);
    }
    struct Variant { const char* name; float (*dot)(const uint16_t*, const uint16_t*); };
    const std::array<Variant, 4> variants{{
        {"blackwell_k64", exact_dot<26, 16, 64>},
        {"blackwell_two_k32", exact_dot<26, 16, 32>},
        {"group8_width25_k64", exact_dot<25, 8, 64>},
        {"ieee_fma_sequential", ieee_dot}
    }};
    std::cout << "{\"kind\":\"cpu_u_accumulator_attribution\",\"tokens\":" << tokens
              << ",\"sampled\":" << (tokens > 64 ? "true" : "false")
              << ",\"inference_acceptance\":false,\"variants\":[";
    for (size_t mode = 0; mode < variants.size(); ++mode) {
        uint64_t compared = 0, mismatch = 0;
        int first = -1;
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < 32; ++h) for (int d = 0; d < 128; ++d) {
            const int index = (t * 32 + h) * 128 + d;
            if (tokens > 64 && ((uint32_t(index) * UINT32_C(0x9e3779b9)) >> 24) != 0) continue;
            std::array<uint16_t, 64> right{};
            for (int s = 0; s < 64 && t / 64 * 64 + s < tokens; ++s) {
                right[s] = v[((t / 64 * 64 + s) * 32 + h) * 128 + d];
            }
            if (!std::isfinite(value(reference[index]))) return 5;
            const auto actual = bf16(variants[mode].dot(&inverse[(t * 32 + h) * 64], right.data()));
            ++compared;
            if (actual != reference[index]) { ++mismatch; if (first < 0) first = index; }
        }
        if (mode) std::cout << ',';
        std::cout << "{\"name\":\"" << variants[mode].name << "\",\"elements\":" << compared
                  << ",\"bit_mismatches\":" << mismatch << ",\"first_index\":" << first << '}';
    }
    std::cout << "]}\n";
}
