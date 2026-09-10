// CPU attribution only. Captured exponents/checkpoints never enter the engine.
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr size_t frame = 32u * 128u * 128u;
uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
float fp32(uint16_t x) { const uint32_t u = uint32_t(x) << 16; float f; std::memcpy(&f, &u, 4); return f; }
uint16_t bf16(float x) { const uint32_t u = bits(x); return uint16_t((u + 0x7fffu + ((u >> 16) & 1)) >> 16); }
template<class T> std::vector<T> read(const std::string& path, size_t count) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f || f.tellg() != static_cast<std::streamoff>(count * sizeof(T))) throw std::runtime_error("wrong input size: " + path);
    std::vector<T> v(count); f.seekg(0); f.read(reinterpret_cast<char*>(v.data()), count * sizeof(T));
    if (!f) throw std::runtime_error("cannot read: " + path);
    return v;
}
struct Stats {
    size_t elements = 0, raw = 0, rounded = 0;
    long long first = -1; uint32_t actual = 0, expected = 0;
    void add(float a, float b, size_t index) {
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite state");
        ++elements; rounded += bf16(a) != bf16(b);
        if (bits(a) != bits(b)) {
            ++raw;
            if (first < 0) { first = static_cast<long long>(index); actual = bits(a); expected = bits(b); }
        }
    }
    void print() const {
        std::cout << "{\"elements\":" << elements << ",\"raw_bit_mismatches\":" << raw
                  << ",\"bf16_bit_mismatches\":" << rounded << ",\"first_index\":" << first
                  << ",\"actual_bits\":" << actual << ",\"expected_bits\":" << expected << '}';
    }
};
struct Mode { const char* name; unsigned exp_mask; bool split, fused; };
struct Inputs {
    size_t tokens;
    std::vector<uint16_t> k, w, u, v;
    std::vector<float> g, exponent, seed, checkpoints, terminal;
    float exp(size_t chunk, size_t token, size_t head, unsigned mask) const {
        const size_t last = chunk * 64u + 63u;
        const float x = token == tokens ? g[last * 32u + head] : g[last * 32u + head] - g[token * 32u + head];
        const bool captured = mask & (token == tokens ? 2u : 1u);
        return captured ? exponent[(token == tokens ? tokens + chunk : token) * 32u + head]
                        : std::exp2(x * 1.4426950408889634074f);
    }
    std::array<float, 128> step(const std::array<float, 128>& state, size_t chunk, size_t head, size_t value,
                               const Mode& mode, Stats& values) const {
        std::array<uint16_t, 128> h{};
        for (size_t key = 0; key < 128; ++key) h[key] = bf16(state[key]);
        std::array<uint16_t, 64> residual{};
        for (size_t t = 0; t < 64; ++t) {
            const size_t token = chunk * 64u + t, offset = (token * 32u + head) * 128u;
            const auto* left = w.data() + offset;
            float projection = 0;
            if (mode.split) {
                projection = qrt_q1_moe_hawkeye::dot_bf16_impl<26,16,-133>(left, h.data(), 64);
                projection += qrt_q1_moe_hawkeye::dot_bf16_impl<26,16,-133>(left + 64, h.data() + 64, 64);
            } else projection = qrt_q1_moe_hawkeye::dot_bf16_impl<26,16,-133>(left, h.data(), 128);
            const float current = fp32(u[offset + value]) - projection;
            values.add(fp32(bf16(current)), fp32(v[offset + value]), offset + value);
            residual[t] = bf16(current * exp(chunk, token, head, mode.exp_mask));
        }
        std::array<float, 128> next{};
        const float decay = exp(chunk, tokens, head, mode.exp_mask);
        for (size_t key = 0; key < 128; ++key) {
            std::array<uint16_t, 64> left{};
            for (size_t t = 0; t < 64; ++t) left[t] = k[((chunk * 64u + t) * 16u + head / 2u) * 128u + key];
            const float update = qrt_q1_moe_hawkeye::dot_bf16_impl<26,16,-133>(left.data(), residual.data(), 64);
            next[key] = mode.fused ? std::fma(state[key], decay, update) : state[key] * decay + update;
        }
        return next;
    }
};
}

int main(int argc, char** argv) try {
    if (argc != 4 && (argc != 5 || std::string(argv[4]) != "--all-rows"))
        throw std::runtime_error("usage: fla-raw-prefix-probe <prefix-dir> <capture-dir> <tokens> [--all-rows]");
    size_t consumed = 0;
    const size_t tokens = std::stoul(argv[3], &consumed), chunks = tokens / 64u;
    if (consumed != std::strlen(argv[3]) || tokens < 64 || tokens > 1024 || tokens % 64) throw std::runtime_error("invalid tokens");
    const std::string prefix = std::string(argv[1]) + '/', capture = std::string(argv[2]) + '/';
    Inputs in{tokens,
        read<uint16_t>(prefix + "k-normalized-bf16.bin", tokens * 2048u),
        read<uint16_t>(prefix + "w-bf16.bin", tokens * 4096u),
        read<uint16_t>(prefix + "u-bf16.bin", tokens * 4096u),
        read<uint16_t>(prefix + "v-new-bf16.bin", tokens * 4096u),
        read<float>(prefix + "g-cumsum-f32.bin", tokens * 32u),
        read<float>(capture + "state-exponent-values-f32.bin", (tokens + chunks) * 32u),
        read<float>(prefix + "initial_state-f32.bin", frame),
        read<float>(capture + "checkpoint-f32.bin", chunks * frame),
        read<float>(capture + "terminal-state-f32.bin", frame)};
    std::vector<std::array<size_t,2>> rows{
        {0,0},{2,88},{8,64},{1,17},{4,34},{6,51},{10,68},{12,85},
        {14,102},{16,119},{18,8},{20,25},{22,42},{24,59},{28,76},{31,127}};
    if (argc == 5) {
        rows.clear();
        for (size_t h = 0; h < 32; ++h) for (size_t v = 0; v < 128; ++v) rows.push_back({h,v});
    }
    const Mode modes[] = {{"host_exp_fma",0,false,true}, {"captured_gate_fma",1,false,true},
        {"captured_decay_fma",2,false,true}, {"captured_exp_fma",3,false,true},
        {"captured_exp_unfused",3,false,false}, {"captured_exp_split_projection",3,true,true}};
    std::cout << std::setprecision(12) << "{\"tokens\":" << tokens
              << ",\"sampled_rows\":" << rows.size()
              << ",\"reference_state_feeds_carried_trajectory\":false,\"variants\":[";
    for (size_t m = 0; m < sizeof(modes) / sizeof(*modes); ++m) {
        Stats state_stats, value_stats, terminal_stats, isolated_state, isolated_value;
        std::vector<Stats> by_chunk(chunks);
        for (const auto& row : rows) {
            const size_t offset = (row[0] * 128u + row[1]) * 128u;
            std::array<float,128> state{};
            std::memcpy(state.data(), in.seed.data() + offset, sizeof(state));
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                const auto* expected = in.checkpoints.data() + chunk * frame + offset;
                for (size_t key = 0; key < 128; ++key) {
                    const size_t index = chunk * frame + offset + key;
                    state_stats.add(state[key], expected[key], index);
                    by_chunk[chunk].add(state[key], expected[key], index);
                }
                state = in.step(state, chunk, row[0], row[1], modes[m], value_stats);
                // Separate one-step control; never overwrite the carried state.
                std::array<float,128> reference_input{};
                std::memcpy(reference_input.data(), expected, sizeof(reference_input));
                const auto local = in.step(reference_input, chunk, row[0], row[1], modes[m], isolated_value);
                const auto* next = chunk + 1 == chunks ? in.terminal.data() + offset
                                  : in.checkpoints.data() + (chunk + 1) * frame + offset;
                for (size_t key = 0; key < 128; ++key) isolated_state.add(local[key], next[key], chunk * frame + offset + key);
            }
            for (size_t key = 0; key < 128; ++key) terminal_stats.add(state[key], in.terminal[offset + key], offset + key);
        }
        if (m) std::cout << ',';
        std::cout << "{\"name\":\"" << modes[m].name << "\",\"checkpoints\":"; state_stats.print();
        std::cout << ",\"v_new\":"; value_stats.print(); std::cout << ",\"terminal\":"; terminal_stats.print();
        std::cout << ",\"isolated_step_state\":"; isolated_state.print();
        std::cout << ",\"isolated_step_v_new\":"; isolated_value.print();
        std::cout << ",\"by_chunk\":[";
        for (size_t c = 0; c < chunks; ++c) { if (c) std::cout << ','; by_chunk[c].print(); }
        std::cout << "]}";
    }
    std::cout << "]}\n";
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
