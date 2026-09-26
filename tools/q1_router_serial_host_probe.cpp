#include "native/providers/moe_accumulator/sm121_packed_dense.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<uint16_t> read_bf16(const char* name, size_t count) {
    std::ifstream stream(name, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(count * sizeof(uint16_t))) {
        throw std::runtime_error(std::string("BF16 file shape: ") + name);
    }
    std::vector<uint16_t> result(count);
    stream.seekg(0);
    stream.read(reinterpret_cast<char*>(result.data()), count * sizeof(uint16_t));
    if (!stream) throw std::runtime_error(std::string("BF16 file read: ") + name);
    return result;
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) return 2;
        const auto input = read_bf16(argv[1], 2048u);
        const auto weight = read_bf16(argv[2], 256u * 2048u);
        const auto reference = read_bf16(argv[3], 256u);
        const unsigned target = static_cast<unsigned>(std::stoul(argv[4]));
        if (target >= 256u) return 2;
        unsigned mismatches = 0u, first = 256u;
        uint16_t target_bf16 = 0u;
        uint32_t target_f32_bits = 0u;
        for (unsigned expert = 0; expert < 256u; ++expert) {
            const float value = qrt_sm121_packed_dense::router_serial_dot(
                input.data(), weight.data() + size_t(expert) * 2048u);
            const uint16_t rounded = qrt_sm121_q1::bf16(value);
            if (rounded != reference[expert]) {
                if (first == 256u) first = expert;
                ++mismatches;
            }
            if (expert == target) {
                target_bf16 = rounded;
                std::memcpy(&target_f32_bits, &value, sizeof(value));
            }
        }
        std::cout << "{\"mismatches\":" << mismatches
                  << ",\"first_mismatch_expert\":";
        if (first == 256u) std::cout << "null";
        else std::cout << first;
        std::cout << ",\"target_expert\":" << target
                  << ",\"target_bf16\":\"" << std::hex << std::setw(4)
                  << std::setfill('0') << target_bf16
                  << "\",\"target_f32_bits\":\"" << std::setw(8)
                  << target_f32_bits << "\"}" << std::dec << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
