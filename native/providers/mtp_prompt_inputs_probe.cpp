// Offline input replay. Captured sampled IDs are diagnostics, never inference inputs.
#include "mtp_prompt_inputs.h"
#include <fstream>
#include <iostream>
#include <vector>

int main(int argc, char** argv) try {
    if (argc != 2) return 2;
    std::ifstream input(argv[1], std::ios::binary | std::ios::ate);
    if (!input) return 2;
    const auto bytes = input.tellg();
    if (bytes <= 0 || bytes % 4 || bytes > 262144 * 4) return 2;
    std::vector<uint32_t> prompt(static_cast<size_t>(bytes) / 4u);
    input.seekg(0);
    input.read(reinterpret_cast<char*>(prompt.data()), bytes);
    if (!input) return 2;
    size_t first = 0, rows = 0;
    unsigned int discarded = 0, sampled = 0, calls = 0;
    while (std::cin >> first >> rows >> discarded >> sampled) {
        if (++calls > 64u || discarded > 1u || rows > 8192u) return 2;
        std::vector<uint32_t> output(rows);
        if (!qrt_mtp_prompt_inputs::shift(prompt.data(), prompt.size(), first,
                rows, discarded != 0u, sampled, output.data())) return 2;
        // Full shifted inputs are emitted for independent offline comparison.
        std::cout << '[';
        for (size_t i = 0; i < rows; ++i) std::cout << (i ? "," : "") << output[i];
        std::cout << "]\n";
    }
    return std::cin.eof() && calls ? 0 : 2;
} catch (...) { return 2; }
