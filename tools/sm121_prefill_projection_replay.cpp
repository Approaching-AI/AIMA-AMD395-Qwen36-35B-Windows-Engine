#include "../native/providers/moe_accumulator/sm121_prefill_projection.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T>
std::vector<T> read(const char *path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != std::streamoff(count * sizeof(T)))
        throw std::runtime_error(std::string("invalid tensor extent: ") + path);
    std::vector<T> result(count);
    f.seekg(0);
    f.read(reinterpret_cast<char *>(result.data()), count * sizeof(T));
    if (!f) throw std::runtime_error("short tensor read");
    return result;
}
uint16_t bf16(float x) {
    uint32_t bits;
    std::memcpy(&bits, &x, sizeof(bits));
    return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
float widen(uint16_t x) {
    uint32_t bits = uint32_t(x) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

int main(int argc, char **argv) try {
    if (argc != 8) throw std::runtime_error(
        "usage: replay router|shared-gate-up|shared-down|attention-output|linear-ba tokens features K input weights expected");
    using namespace qrt_sm121_prefill_projection;
    const std::string kind = argv[1];
    const Stage stage = kind == "router" ? Stage::Router :
        kind == "shared-gate-up" ? Stage::SharedGateUp :
        kind == "attention-output" ? Stage::AttentionOutput :
        kind == "linear-ba" ? Stage::LinearBA : Stage::SharedDown;
    if (kind != "router" && kind != "shared-gate-up" && kind != "shared-down" &&
        kind != "attention-output" && kind != "linear-ba")
        throw std::runtime_error("unknown projection");
    const unsigned tokens = std::stoul(argv[2]), features = std::stoul(argv[3]);
    const unsigned count = std::stoul(argv[4]);
    const bool shape = stage == Stage::Router ? (features == 256 && count == 2048) :
        stage == Stage::SharedGateUp ? ((features == 512 || features == 1024) && count == 2048) :
        stage == Stage::AttentionOutput ? (features == 2048 && count == 4096) :
        stage == Stage::LinearBA ? ((features == 32 || features == 64) && count == 2048) :
        (features == 2048 && count == 512);
    if (!shape || !tokens || tokens > 4096 || uint64_t(tokens) * features * count > (1ull << 30))
        throw std::runtime_error("replay exceeds bounded model projection shape");
    const auto input = read<uint16_t>(argv[5], size_t(tokens) * count);
    const auto weights = read<uint16_t>(argv[6], size_t(features) * count);
    const auto expected = read<uint16_t>(argv[7], size_t(tokens) * features);
    for (const auto *tensor : {&input, &weights, &expected})
        for (uint16_t x : *tensor)
            if ((x & 0x7f80u) == 0x7f80u) throw std::runtime_error("nonfinite tensor");
    const Plan selected = plan(stage, tokens);
    size_t mismatches = 0;
    double maximum_error = 0;
    std::vector<std::array<unsigned, 4>> first;
    for (unsigned t = 0; t < tokens; ++t) {
        for (unsigned row = 0; row < features; ++row) {
            const auto actual = bf16(dot<1>(input.data() + size_t(t) * count,
                weights.data() + size_t(row) * count, count, selected, t, row));
            const auto reference = expected[size_t(t) * features + row];
            if (actual == reference) continue;
            ++mismatches;
            maximum_error = std::max(maximum_error, double(std::abs(widen(actual) - widen(reference))));
            if (first.size() < 16) first.push_back({t, row, actual, reference});
        }
    }
    std::cout << "{\"diagnostic_only\":true,\"tokens\":" << tokens
              << ",\"elements\":" << expected.size() << ",\"mismatches\":" << mismatches
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"split_count\":" << selected.splits
              << ",\"accumulators\":" << selected.accumulators
              << ",\"bf16_partials\":" << (selected.bf16_partials ? "true" : "false")
              << ",\"serial_bf16_tile\":" << selected.serial_bf16_tile
              << ",\"first_mismatches\":[";
    for (size_t i = 0; i < first.size(); ++i) {
        if (i) std::cout << ',';
        const auto &x = first[i];
        std::cout << "{\"token\":" << x[0] << ",\"feature\":" << x[1]
                  << ",\"actual_bits\":" << x[2] << ",\"reference_bits\":" << x[3] << '}';
    }
    std::cout << "]}\n";
    return mismatches ? 1 : 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 2;
}
