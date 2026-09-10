// Exercise the production CPU/device lookup against the held-out real control.
// The large generated table is supplied explicitly and is not a test fixture.
#include "gdn/sm121_silu_table.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class T> std::vector<T> read(const char *path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || file.tellg() % sizeof(T)) throw std::runtime_error("invalid input span");
    const auto bytes = file.tellg();
    std::vector<T> values(static_cast<size_t>(bytes) / sizeof(T));
    file.seekg(0); file.read(reinterpret_cast<char *>(values.data()), bytes);
    if (!file) throw std::runtime_error("short input");
    return values;
}
int main(int argc, char **argv) {
    try {
        if (argc != 4) throw std::runtime_error("TABLE ARGUMENTS_F32 EXPECTED_BF16");
        auto table = read<unsigned char>(argv[1]);
        const auto arguments = read<float>(argv[2]);
        const auto expected = read<uint16_t>(argv[3]);
        if (!qrt_sm121_silu::valid_layout(table.data(), table.size()) ||
            arguments.empty() || arguments.size() != expected.size()) throw std::runtime_error("invalid table or control");
        size_t wrong = 0;
        for (size_t i = 0; i < arguments.size(); ++i)
            wrong += qrt_sm121_silu::evaluate(table.data(), arguments[i]) != expected[i];
        const auto *keys = reinterpret_cast<const uint32_t *>(table.data() + qrt_sm121_silu::key_start);
        const auto *values = reinterpret_cast<const uint16_t *>(table.data() + qrt_sm121_silu::value_start);
        for (uint32_t i = 0; i < qrt_sm121_silu::transition_count; ++i) {
            if (qrt_sm121_silu::evaluate(table.data(), qrt_sm121_exp2::value(keys[i])) != values[i])
                throw std::runtime_error("transition lookup failed");
            if (i && ((keys[i] - 1u) & 0x7fffffffu) < 0x7f800000u &&
                qrt_sm121_silu::evaluate(table.data(), qrt_sm121_exp2::value(keys[i] - 1u)) != values[i - 1u])
                throw std::runtime_error("preceding transition lookup failed");
        }
        table[8] ^= 1u;
        if (qrt_sm121_silu::valid_layout(table.data(), table.size())) throw std::runtime_error("invalid schema accepted");
        table[8] ^= 1u; table[64] ^= 1u;
        if (qrt_sm121_silu::valid_layout(table.data(), table.size())) throw std::runtime_error("invalid directory accepted");
        std::cout << "{\"control_elements\":" << arguments.size() << ",\"bf16_differences\":" << wrong
                  << ",\"transition_boundaries_pass\":true,\"malformed_layout_rejected\":true,\"inference_acceptance\":false}\n";
        return wrong ? 1 : 0;
    } catch (const std::exception &error) { std::cerr << error.what() << '\n'; return 2; }
}
