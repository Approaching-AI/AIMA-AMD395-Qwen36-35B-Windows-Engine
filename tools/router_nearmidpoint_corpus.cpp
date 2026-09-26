#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

#include "native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include "native/providers/moe_accumulator/router_nearmidpoint_exact.h"

static uint32_t read_u32(const uint8_t *p) {
    uint32_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

static uint16_t read_u16(const uint8_t *p) {
    uint16_t value = 0;
    std::memcpy(&value, p, sizeof(value));
    return value;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    std::ifstream stream(argv[1], std::ios::binary | std::ios::ate);
    if (!stream) return 3;
    const size_t size = static_cast<size_t>(stream.tellg());
    std::vector<uint8_t> data(size);
    stream.seekg(0);
    stream.read(reinterpret_cast<char *>(data.data()), size);
    if (!stream || size < 16 || std::memcmp(data.data(), "Q1RBLS01", 8) != 0) return 4;
    const uint32_t weight_count = read_u32(data.data() + 8);
    const uint32_t case_count = read_u32(data.data() + 12);
    const size_t expected = 16 + static_cast<size_t>(weight_count) * (4 + 1048576) +
        static_cast<size_t>(case_count) * (12 + 4096 + 512);
    if (size != expected || case_count != 147) return 5;
    const uint8_t *p = data.data() + 16;
    std::map<uint32_t, const uint8_t *> weights;
    for (uint32_t i = 0; i < weight_count; ++i) {
        const uint32_t layer = read_u32(p);
        p += 4;
        weights[layer] = p;
        p += 1048576;
    }
    uint64_t baseline_wrong[2] = {}, selected_wrong[2] = {};
    uint64_t recomputed[2] = {}, changed[2] = {};
    for (uint32_t ordinal = 0; ordinal < case_count; ++ordinal) {
        const uint32_t group = read_u32(p), row = read_u32(p + 4), layer = read_u32(p + 8);
        p += 12;
        const uint8_t *input_bytes = p;
        p += 4096;
        const uint8_t *reference_bytes = p;
        p += 512;
        if (group > 1 || !weights.count(layer)) return 6;
        uint16_t input[2048], weight[2048];
        for (uint32_t k = 0; k < 2048; ++k) input[k] = read_u16(input_bytes + k * 2);
        for (uint32_t expert = 0; expert < 256; ++expert) {
            for (uint32_t k = 0; k < 2048; ++k) {
                weight[k] = read_u16(weights[layer] + (expert * 2048 + k) * 2);
            }
            const float hawkeye = qrt_q1_moe_hawkeye::dot_bf16_hopper(input, weight, 2048);
            const auto decision = qrt_router_nearmidpoint::select(input, weight, 2048, hawkeye);
            const uint16_t reference = read_u16(reference_bytes + expert * 2);
            baseline_wrong[group] += decision.baseline != reference;
            selected_wrong[group] += decision.selected != reference;
            recomputed[group] += decision.recomputed;
            changed[group] += decision.selected != decision.baseline;
            if (decision.selected != decision.baseline) {
                std::cout << "change " << group << ' ' << row << ' ' << layer << ' '
                          << expert << " reference " << std::hex << reference
                          << " baseline " << decision.baseline
                          << " selected " << decision.selected << std::dec
                          << " margin " << decision.midpoint_distance << '\n';
            }
        }
    }
    std::cout << "qualified baseline_wrong " << baseline_wrong[0]
              << " selected_wrong " << selected_wrong[0]
              << " recomputed " << recomputed[0] << " changed " << changed[0] << '\n';
    std::cout << "draft baseline_wrong " << baseline_wrong[1]
              << " selected_wrong " << selected_wrong[1]
              << " recomputed " << recomputed[1] << " changed " << changed[1] << '\n';
    return baseline_wrong[0] == 1 && selected_wrong[0] == 0 &&
                   recomputed[0] == 24 && changed[0] == 1 &&
                   baseline_wrong[1] == 3 && selected_wrong[1] == 3 &&
                   recomputed[1] == 39 && changed[1] == 0
        ? 0 : 7;
}
