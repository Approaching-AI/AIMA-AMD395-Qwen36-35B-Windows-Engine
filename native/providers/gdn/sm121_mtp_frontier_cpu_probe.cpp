#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "sm121_mtp_math.h"

template<class T> std::vector<T> read_file(const char* path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(T))
        throw std::runtime_error("invalid input file");
    const size_t bytes = static_cast<size_t>(file.tellg());
    if (!bytes || bytes > 128u * 1024u * 1024u)
        throw std::runtime_error("file exceeds bound or is empty");
    std::vector<T> output(bytes / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(output.data()), bytes);
    if (!file) throw std::runtime_error("short input file");
    return output;
}

float row_inverse(const uint16_t* row, const unsigned char* table, bool split1024) {
    std::array<float, 512> lanes{}, next{};
    for (unsigned int lane = 0; lane < lanes.size(); ++lane)
        lanes[lane] = split1024 ? qrt_sm121_mtp::lane_sumsq_split1024(row, lane)
                                : qrt_sm121_mtp::lane_sumsq(row, lane);
    for (unsigned int offset = 16; offset; offset >>= 1) {
        for (unsigned int lane = 0; lane < lanes.size(); ++lane)
            next[lane] = qrt_sm121_q1::add(lanes[lane], lanes[lane ^ offset]);
        lanes = next;
    }
    std::array<float, 16> warps{};
    for (unsigned int warp = 0; warp < warps.size(); ++warp)
        warps[warp] = lanes[warp * 32u];
    return qrt_sm121_mtp::inverse(qrt_sm121_mtp::sum_warps(warps.data()), table);
}

int main(int argc, char** argv) try {
    const bool split1024 = argc > 1 && std::string(argv[argc - 1]) == "--split1024";
    const int arguments = argc - (split1024 ? 1 : 0);
    if (arguments != 5 && arguments != 7)
        throw std::runtime_error("input weights expected rsqrt_table [second_input second_weights] [--split1024]");
    const auto input = read_file<uint16_t>(argv[1]);
    const auto weights = read_file<uint16_t>(argv[2]);
    const auto expected = read_file<uint16_t>(argv[3]);
    const auto table = read_file<unsigned char>(argv[4]);
    const bool fusion = arguments == 7;
    const auto second_input = fusion ? read_file<uint16_t>(argv[5]) : std::vector<uint16_t>{};
    const auto second_weights = fusion ? read_file<uint16_t>(argv[6]) : std::vector<uint16_t>{};
    const size_t rows = input.size() / 2048u;
    const size_t parts = fusion ? 2u : 1u;
    if (!rows || rows > 8192u || rows * 2048u != input.size() || weights.size() != 2048u ||
        expected.size() != rows * 2048u * parts ||
        (fusion && (second_input.size() != input.size() || second_weights.size() != weights.size())))
        throw std::runtime_error("inconsistent normalization shapes");
    if (!qrt_sm121_rsqrt::valid_layout(table.data(), table.size()))
        throw std::runtime_error("invalid rsqrt table layout");
    size_t mismatches = 0, first = expected.size();
    uint16_t first_actual = 0;
    double maximum_error = 0;
    for (size_t row = 0; row < rows; ++row) {
        for (size_t part = 0; part < parts; ++part) {
            const uint16_t* values = (part ? second_input : input).data() + row * 2048u;
            const uint16_t* norm_weights = (part ? second_weights : weights).data();
            const float rstd = row_inverse(values, table.data(), split1024);
            for (size_t column = 0; column < 2048u; ++column) {
                const uint16_t actual = qrt_sm121_mtp::normalized(
                    qrt_sm121_q1::widen(values[column]), rstd, norm_weights[column]);
                const size_t index = row * 2048u * parts + part * 2048u + column;
                const float actual_value = qrt_sm121_q1::widen(actual);
                const float expected_value = qrt_sm121_q1::widen(expected[index]);
                if (!std::isfinite(actual_value) || !std::isfinite(expected_value))
                    throw std::runtime_error("nonfinite normalization boundary");
                maximum_error = std::max(maximum_error, std::abs(double(actual_value) - expected_value));
                if (actual != expected[index]) {
                    ++mismatches;
                    if (first == expected.size()) { first = index; first_actual = actual; }
                }
            }
        }
    }
    std::cout << "{\"mode\":\"" << (fusion ? "fusion_inputs" : "normalization")
              << "\",\"split1024\":" << (split1024 ? "true" : "false")
              << ",\"rows\":" << rows << ",\"elements\":" << expected.size()
              << ",\"bf16_mismatches\":" << mismatches
              << ",\"maximum_absolute_error\":" << maximum_error << ",\"first_difference\":";
    if (mismatches) std::cout << "{\"index\":" << first << ",\"actual_bits\":" << first_actual
                              << ",\"expected_bits\":" << expected[first] << '}';
    else std::cout << "null";
    std::cout << ",\"native_inference_acceptance\":false}\n";
    return mismatches ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
