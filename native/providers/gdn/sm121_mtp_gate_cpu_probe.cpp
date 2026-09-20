#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "sm121_mtp_gate_math.h"

std::vector<uint16_t> read(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() <= 0 || f.tellg() % 2 || f.tellg() > 128u * 1024u * 1024u)
        throw std::runtime_error("invalid original gate input file");
    std::vector<uint16_t> data(static_cast<size_t>(f.tellg()) / 2u);
    f.seekg(0); f.read(reinterpret_cast<char*>(data.data()), data.size() * 2u);
    if (!f) throw std::runtime_error("short original gate input");
    return data;
}
int main(int argc, char** argv) try {
    if (argc != 5) throw std::runtime_error("q_projection context expected_gated sigmoid_bf16_table");
    const auto q = read(argv[1]), context = read(argv[2]), expected = read(argv[3]), table = read(argv[4]);
    const size_t rows = context.size() / 4096u;
    if (!rows || rows > 8192u || context.size() != rows * 4096u ||
        expected.size() != context.size() || q.size() != rows * 8192u || table.size() != 65536u)
        throw std::runtime_error("inconsistent original gate shapes");
    size_t mismatches = 0, first = context.size();
    for (size_t i = 0; i < context.size(); ++i) {
        const size_t feature = i % 4096u;
        const uint16_t gate = q[(i / 4096u) * 8192u + (feature / 256u) * 512u + 256u + feature % 256u];
        const auto actual = qrt_sm121_mtp::gated_context(context[i], gate, table.data());
        if (actual != expected[i]) { ++mismatches; if (first == context.size()) first = i; }
    }
    std::cout << "{\"rows\":" << rows << ",\"elements\":" << context.size()
        << ",\"bf16_mismatches\":" << mismatches << ",\"first_difference\":";
    if (mismatches) std::cout << first; else std::cout << "null";
    std::cout << ",\"inference_acceptance\":false}\n";
    return mismatches ? 1 : 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 2;
}
