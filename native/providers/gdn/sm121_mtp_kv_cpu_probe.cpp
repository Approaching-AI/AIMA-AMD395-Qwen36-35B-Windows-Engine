#include <array>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "sm121_mtp_kv_math.h"

template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(T))
        throw std::runtime_error("invalid input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (!bytes || bytes > 128u * 1024u * 1024u)
        throw std::runtime_error("empty file or size exceeds bound");
    std::vector<T> result(bytes / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char*>(result.data()), bytes);
    if (!file) throw std::runtime_error("short input file");
    return result;
}

int main(int argc, char** argv) try {
    if (argc != 8)
        throw std::runtime_error("kv key_weights expected_norm expected_kv positions rsqrt_table rope_table");
    const auto input = read_file<uint16_t>(argv[1]);
    const auto weights = read_file<uint16_t>(argv[2]);
    const auto expected_norm = read_file<uint16_t>(argv[3]);
    const auto expected_kv = read_file<uint16_t>(argv[4]);
    const auto positions = read_file<uint32_t>(argv[5]);
    const auto table = read_file<unsigned char>(argv[6]);
    const auto rope = read_file<uint16_t>(argv[7]);
    const size_t rows = input.size() / 1024u;
    if (!rows || rows > 8192u || input.size() != rows * 1024u || weights.size() != 256u ||
        expected_norm.size() != rows * 512u || expected_kv.size() != input.size() || positions.size() != rows ||
        rope.size() % 64u || !qrt_sm121_rsqrt::valid_layout(table.data(), table.size()))
        throw std::runtime_error("inconsistent KV shapes or table");
    for (uint32_t position : positions) if (position >= 262144u || position >= rope.size() / 64u)
        throw std::runtime_error("position outside original MTP cache");
    size_t norm_bad = 0, key_bad = 0, value_bad = 0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t head = 0; head < 2u; ++head) {
            std::array<float, 256> values{};
            std::array<float, 128> lanes{}, next{};
            std::array<float, 4> warps{};
            std::array<uint16_t, 256> normalized{};
            for (size_t i = 0; i < 256u; ++i)
                values[i] = qrt_sm121_q1::widen(input[row * 1024u + head * 256u + i]);
            for (unsigned int lane = 0; lane < 128u; ++lane)
                lanes[lane] = qrt_sm121_q1::head_norm_lane_sumsq(values.data(), lane, true);
            for (unsigned int offset = 16u; offset; offset >>= 1u) {
                for (unsigned int lane = 0; lane < 128u; ++lane)
                    next[lane] = qrt_sm121_q1::add(lanes[lane], lanes[lane ^ offset]);
                lanes = next;
            }
            for (size_t warp = 0; warp < 4u; ++warp) warps[warp] = lanes[warp * 32u];
            const float inverse = qrt_sm121_mtp::key_inverse(warps.data(), table.data());
            for (unsigned int i = 0; i < 256u; ++i) {
                normalized[i] = qrt_sm121_mtp::normalized(values[i], inverse, weights[i]);
                norm_bad += normalized[i] != expected_norm[row * 512u + head * 256u + i];
            }
            for (unsigned int i = 0; i < 256u; ++i) {
                const size_t index = row * 1024u + head * 256u + i;
                const uint16_t rotated = qrt_sm121_mtp::key_rotated(normalized.data(), i,
                    rope.data() + size_t(positions[row]) * 64u);
                key_bad += rotated != expected_kv[index];
                value_bad += input[index + 512u] != expected_kv[index + 512u];
            }
        }
    }
    std::cout << "{\"rows\":" << rows << ",\"norm_elements\":" << rows * 512u
              << ",\"key_elements\":" << rows * 512u << ",\"value_elements\":" << rows * 512u
              << ",\"norm_bf16_mismatches\":" << norm_bad << ",\"key_bf16_mismatches\":" << key_bad
              << ",\"value_bf16_mismatches\":" << value_bad << ",\"inference_acceptance\":false}\n";
    return norm_bad || key_bad || value_bad ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
