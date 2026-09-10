#include "../../native/providers/gdn/sm121_rsqrt_table.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
std::vector<unsigned char> read(const char* path, size_t limit) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() < 0 || uint64_t(f.tellg()) > limit) throw std::runtime_error("invalid file size");
    std::vector<unsigned char> data(static_cast<size_t>(f.tellg())); f.seekg(0);
    f.read(reinterpret_cast<char*>(data.data()), data.size());
    if (!f) throw std::runtime_error("read failed");
    return data;
}
int main(int argc, char** argv) try {
    using namespace qrt_sm121_rsqrt;
    using qrt_sm121_exp2::bits; using qrt_sm121_exp2::value;
    if (argc == 2 && std::string(argv[1]) == "--domain-only") {
        for (uint32_t u : {0u,1u,0x007fffffu}) if (bits(evaluate(nullptr, value(u))) != 0x7f800000u) return 3;
        for (uint32_t u : {0x80000000u,0xbf800000u,0x7fc00000u}) if (bits(evaluate(nullptr, value(u))) != 0x7fc00000u) return 3;
        if (bits(evaluate(nullptr, value(0x7f800000u))) != 0 || valid_layout(nullptr, table_bytes)) return 3;
        // Synthetic pages test layout and scaling only; they are never accepted
        // by the production SHA-256 loader as a numerical compatibility table.
        std::vector<uint32_t> data(table_bytes / 4, 0u);
        auto* raw = reinterpret_cast<unsigned char*>(data.data());
        std::memcpy(raw, "QRSQTBL1", 8);
        const uint32_t header[] = {1,8,begin,end,0x7f800000u,0,pages,1};
        std::memcpy(raw + 8, header, sizeof(header)); std::memcpy(raw + 40, &payload_bytes, 8);
        for (uint32_t page = 0; page < pages; ++page) data[12u + page * 2u] = 0x3f800000u;
        data[12u + 32768u * 2u] = 0x3f400000u;
        if (!valid_layout(raw, table_bytes) || evaluate(raw, 4.0f) != 0.5f || evaluate(raw, 0.25f) != 2.0f ||
            evaluate(raw, 2.0f) != 0.75f || evaluate(raw, 0.5f) != 1.5f) return 3;
        data[13] = (2u << 30u) | uint32_t(payload_bytes - 256u);
        if (valid_layout(raw, table_bytes)) return 3;
        std::cout << "domain, exponent scaling and layout guards pass\n"; return 0;
    }
    if (argc != 4) throw std::runtime_error("lookup <table> <arguments> <expected> | --domain-only");
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
