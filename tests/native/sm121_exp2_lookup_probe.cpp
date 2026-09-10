#include "../../native/providers/gdn/sm121_exp2_table.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <stdexcept>

std::vector<unsigned char> read(const char* path, size_t limit) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0 || uint64_t(f.tellg()) > limit) throw std::runtime_error("invalid file size");
    std::vector<unsigned char> data(static_cast<size_t>(f.tellg())); f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), data.size());
    if (!f) throw std::runtime_error("read failed");
    return data;
}
int main(int argc, char** argv) try {
    using namespace qrt_sm121_exp2;
    if (argc == 2 && std::string(argv[1]) == "--domain-only") {
        if (bits(evaluate(nullptr, -0.0f)) != 0x3f800000u || bits(evaluate(nullptr, 0.0f)) != 0x3f800000u ||
            bits(evaluate(nullptr, value(0x80000001u))) != 0x3f800000u ||
            bits(evaluate(nullptr, -152.0f)) != 0 || bits(evaluate(nullptr, value(0xff800000u))) != 0 ||
            bits(evaluate(nullptr, 1.0f)) != 0x7fc00000u || bits(evaluate(nullptr, value(0xffc00000u))) != 0x7fc00000u ||
            valid_layout(nullptr, table_bytes)) return 3;
        std::cout << "domain guards pass\n"; return 0;
    }
    if (argc != 4) throw std::runtime_error("usage: lookup <table> <arguments> <expected> | --domain-only");
    const auto data = read(argv[1], table_bytes), arguments = read(argv[2], 1u << 20), expected = read(argv[3], 1u << 20);
    if (!valid_layout(data.data(), data.size()) || arguments.size() != expected.size() || arguments.size() % 4)
        throw std::runtime_error("invalid layout");
    size_t mismatches = 0;
    for (size_t i = 0; i < arguments.size(); i += 4) {
        uint32_t a, e; std::memcpy(&a, arguments.data() + i, 4); std::memcpy(&e, expected.data() + i, 4);
        mismatches += bits(evaluate(data.data(), value(a))) != e;
    }
    std::cout << "{\"elements\":" << arguments.size() / 4 << ",\"mismatches\":" << mismatches << "}\n";
    return mismatches ? 3 : 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
