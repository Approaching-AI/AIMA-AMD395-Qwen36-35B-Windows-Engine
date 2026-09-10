// CPU-only comparison of a captured live GDN call with independent BF16/FP32
// reference tensors. References are never inference inputs. The bounded token
// count keeps memory below 1 GiB; the output is a compact diagnostic JSON record.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> std::vector<T> read(const char *path, size_t elements) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(elements * sizeof(T)))
        throw std::runtime_error(std::string("tensor span mismatch: ") + path);
    std::vector<T> result(elements);
    file.seekg(0);
    file.read(reinterpret_cast<char *>(result.data()), elements * sizeof(T));
    if (!file) throw std::runtime_error("short tensor read");
    return result;
}

uint32_t bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}
uint16_t bf16(float value) {
    const uint32_t raw = bits(value);
    return static_cast<uint16_t>((raw + 0x7fffu + ((raw >> 16u) & 1u)) >> 16u);
}
float value(float input) { return input; }
float value(uint16_t input) {
    const uint32_t raw = uint32_t(input) << 16u;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

template<class T> void compare(const char *name, const std::vector<float> &actual,
                              const std::vector<T> &reference, unsigned width,
                              unsigned stride, unsigned offset) {
    uint64_t numeric = 0, bf16_count = 0, bit_count = 0, nonfinite = 0;
    double error = 0, norm = 0, maximum = 0;
    std::vector<size_t> first;
    for (size_t i = 0; i < reference.size(); ++i) {
        const float a = actual[(i / width) * stride + offset + i % width];
        const float b = value(reference[i]);
        const bool wrong_bits = bits(a) != bits(b);
        numeric += a != b;
        bit_count += wrong_bits;
        bf16_count += bf16(a) != bf16(b);
        if (wrong_bits && first.size() < 64u) first.push_back(i);
        if (!std::isfinite(a) || !std::isfinite(b)) { ++nonfinite; continue; }
        const double delta = double(a) - b;
        error += delta * delta;
        norm += double(b) * b;
        maximum = (std::max)(maximum, std::fabs(delta));
    }
    std::cout << '"' << name << "\":{\"elements\":" << reference.size()
              << ",\"numeric_mismatches\":" << numeric
              << ",\"f32_bit_mismatches\":" << bit_count
              << ",\"bf16_mismatches\":" << bf16_count
              << ",\"nonfinite\":" << nonfinite
              << ",\"relative_l2\":" << std::sqrt(error / (std::max)(norm, 1e-300))
              << ",\"maximum_absolute_error\":" << maximum << ",\"first_differences\":[";
    for (size_t n = 0; n < first.size(); ++n) {
        const size_t i = first[n];
        if (n) std::cout << ',';
        std::cout << "{\"index\":" << i << ",\"actual_f32_bits\":"
                  << bits(actual[(i / width) * stride + offset + i % width])
                  << ",\"expected_f32_bits\":" << bits(value(reference[i])) << '}';
    }
    std::cout << "]}";
}

int main(int argc, char **argv) {
    try {
        if (argc != 13) throw std::runtime_error(
            "usage: compare_gdn_capture RAW GATES OUTPUT STATE Q K V G BETA REF_OUTPUT REF_STATE TOKENS");
        size_t consumed = 0;
        const unsigned long count = std::stoul(argv[12], &consumed);
        if (!count || count > 8192u || consumed != std::strlen(argv[12]))
            throw std::runtime_error("token count must be between 1 and 8192");
        const size_t t = count;
        const auto raw = read<float>(argv[1], t * 8192u);
        const auto gates = read<float>(argv[2], t * 64u);
        const auto output = read<float>(argv[3], t * 4096u);
        const auto state = read<float>(argv[4], 524288u);
        const auto q = read<uint16_t>(argv[5], t * 2048u);
        const auto k = read<uint16_t>(argv[6], t * 2048u);
        const auto v = read<uint16_t>(argv[7], t * 4096u);
        const auto g = read<float>(argv[8], t * 32u);
        const auto beta = read<uint16_t>(argv[9], t * 32u);
        const auto ref_output = read<uint16_t>(argv[10], t * 4096u);
        const auto ref_state = read<float>(argv[11], 524288u);
        std::cout << std::setprecision(17)
                  << "{\"kind\":\"live_native_gdn_capture_comparison\",\"inference_acceptance\":false,\"tokens\":"
                  << t << ",\"surfaces\":{";
        compare("q", raw, q, 2048u, 8192u, 0u); std::cout << ',';
        compare("k", raw, k, 2048u, 8192u, 2048u); std::cout << ',';
        compare("v", raw, v, 4096u, 8192u, 4096u); std::cout << ',';
        compare("g", gates, g, 32u, 64u, 0u); std::cout << ',';
        compare("beta", gates, beta, 32u, 64u, 32u); std::cout << ',';
        compare("output", output, ref_output, 4096u, 4096u, 0u); std::cout << ',';
        compare("state", state, ref_state, 128u, 128u, 0u);
        std::cout << "}}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
