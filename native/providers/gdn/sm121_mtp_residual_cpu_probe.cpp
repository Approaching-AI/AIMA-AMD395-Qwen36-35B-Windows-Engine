#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "sm121_mtp_residual_math.h"

template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() % sizeof(T))
        throw std::runtime_error("invalid input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (bytes > 128u * 1024u * 1024u) throw std::runtime_error("file exceeds bound");
    std::vector<T> values(bytes / sizeof(T)); file.seekg(0);
    file.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!file) throw std::runtime_error("short input file");
    return values;
}

float row_inverse(const uint16_t* input, const uint16_t* residual, const unsigned char* table) {
    std::array<float, 256> lanes{}, next{};
    for (unsigned int lane = 0; lane < 256u; ++lane)
        lanes[lane] = qrt_sm121_mtp::residual_lane_sumsq(input, residual, lane);
    for (unsigned int mask = 16u; mask; mask >>= 1) {
        for (unsigned int lane = 0; lane < 256u; ++lane)
            next[lane] = qrt_sm121_q1::add(lanes[lane], lanes[lane ^ mask]);
        lanes = next;
    }
    std::array<float, 8> warps{};
    for (unsigned int i = 0; i < 8u; ++i) warps[i] = lanes[i * 32u];
    return qrt_sm121_mtp::inverse(qrt_sm121_mtp::residual_sum_warps(warps.data()), table);
}

int main(int argc, char** argv) try {
    if (argc != 7) throw std::runtime_error("input residual weights expected rsqrt_table residual_output");
    const auto input = read_file<uint16_t>(argv[1]), residual = read_file<uint16_t>(argv[2]);
    const auto weights = read_file<uint16_t>(argv[3]), expected = read_file<uint16_t>(argv[4]);
    const auto table = read_file<unsigned char>(argv[5]);
    const size_t rows = input.size() / 2048u;
    if (!rows || rows > 8192u || input.size() != rows * 2048u || weights.size() != 2048u ||
        residual.size() != input.size() || expected.size() != input.size() ||
        !qrt_sm121_rsqrt::valid_layout(table.data(), table.size())) throw std::runtime_error("shape or table");
    std::vector<uint16_t> combined(input.size());
    size_t mismatches = 0, first = expected.size(); uint16_t first_actual = 0;
    double maximum_error = 0.0;
    for (size_t row = 0; row < rows; ++row) {
        const size_t offset = row * 2048u;
        const float rstd = row_inverse(input.data() + offset, residual.data() + offset, table.data());
        for (size_t i = 0; i < 2048u; ++i) {
            const size_t index = offset + i;
            combined[index] = qrt_sm121_mtp::residual_endpoint(input[index], residual[index]);
            const uint16_t actual = qrt_sm121_mtp::normalized(qrt_sm121_q1::widen(combined[index]), rstd, weights[i]);
            const float value = qrt_sm121_q1::widen(actual), reference = qrt_sm121_q1::widen(expected[index]);
            if (!std::isfinite(value) || !std::isfinite(reference)) throw std::runtime_error("nonfinite boundary");
            maximum_error = std::max(maximum_error, std::abs(double(value) - reference));
            if (actual != expected[index]) {
                ++mismatches; if (first == expected.size()) { first = index; first_actual = actual; }
            }
        }
    }
    std::ofstream output(argv[6], std::ios::binary);
    output.write(reinterpret_cast<const char*>(combined.data()), combined.size() * sizeof(uint16_t));
    output.close(); if (!output) throw std::runtime_error("residual output write");
    std::cout << "{\"rows\":" << rows << ",\"elements\":" << expected.size()
              << ",\"bf16_mismatches\":" << mismatches << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"first_difference\":";
    if (mismatches) std::cout << "{\"index\":" << first << ",\"actual_bits\":" << first_actual
                              << ",\"expected_bits\":" << expected[first] << '}';
    else std::cout << "null";
    std::cout << ",\"native_inference_acceptance\":false}\n";
    return mismatches ? 1 : 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
