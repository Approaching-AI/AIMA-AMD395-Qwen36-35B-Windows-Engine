// Compare the actual packed gate helper with independently captured BF16 rows.
// Plan fields: layer position input expected-a expected-b weight-a weight-b.
#include "native/providers/moe_accumulator/sm121_packed_dense.h"
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<uint16_t> read(const std::string& path, size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != std::streamoff(count * 2)) throw std::runtime_error(path);
    std::vector<uint16_t> result(count);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()), count * 2);
    if (!file) throw std::runtime_error("read");
    return result;
}

template<class Input> uint16_t gate(const Input* input, const uint16_t* weight) {
    float partial[16];
    for (unsigned lane = 0; lane < 16; ++lane)
        partial[lane] = qrt_sm121_packed_dense::gate_lane_dot(input, weight, lane);
    // Host equivalent of the existing physical 16-lane shuffle. The expected
    // values are original GPU outputs; no reference output enters arithmetic.
    for (unsigned offset = 8; offset; offset /= 2)
        for (unsigned lane = 0; lane < offset; ++lane)
            partial[lane] = qrt_sm121_q1::add(partial[lane], partial[lane + offset]);
    return qrt_sm121_q1::bf16(partial[0]);
}

int main(int argc, char** argv) try {
    if (argc != 2) throw std::runtime_error("original operand plan");
    std::ifstream plan(argv[1]);
    if (!plan) throw std::runtime_error("plan");
    unsigned layer, position;
    std::string x_path, a_path, b_path, wa_path, wb_path;
    std::map<std::string, std::vector<uint16_t>> weights;
    size_t samples = 0, elements = 0, mismatches = 0;
    while (plan >> layer >> position >> x_path >> a_path >> b_path >> wa_path >> wb_path) {
        if (++samples > 512 || layer >= 40) throw std::runtime_error("case extent");
        for (const auto& path : {wa_path, wb_path})
            if (!weights.count(path)) weights[path] = read(path, 32 * 2048);
        const auto x = read(x_path, 2048), a = read(a_path, 32), b = read(b_path, 32);
        std::vector<float> widened(2048);
        for (unsigned i = 0; i < 2048; ++i) widened[i] = qrt_sm121_q1::widen(x[i]);
        for (unsigned projection = 0; projection < 2; ++projection)
            for (unsigned row = 0; row < 32; ++row) {
                const auto* weight = weights.at(projection ? wb_path : wa_path).data() + row * 2048;
                const uint16_t expected = (projection ? b : a)[row];
                const uint16_t values[] = {gate(x.data(), weight), gate(widened.data(), weight)};
                for (unsigned carrier = 0; carrier < 2; ++carrier) {
                    ++elements;
                    if (values[carrier] != expected) {
                        ++mismatches;
                        std::cout << "{\"layer\":" << layer << ",\"position\":" << position
                            << ",\"projection\":" << projection << ",\"row\":" << row
                            << ",\"carrier\":" << carrier << ",\"actual_bf16\":" << values[carrier]
                            << ",\"expected_bf16\":" << expected << "}\n";
                    }
                }
            }
    }
    if (!samples || !plan.eof()) throw std::runtime_error("empty or malformed plan");
    std::cout << "{\"samples\":" << samples << ",\"compared_elements\":" << elements
        << ",\"bf16_mismatches\":" << mismatches
        << ",\"native_gpu_executed\":false,\"inference_acceptance\":false}\n";
    return mismatches ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
}
