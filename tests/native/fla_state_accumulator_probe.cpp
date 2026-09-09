// CPU-only first-chunk state attribution. Never linked into the runtime.
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr size_t state_elements = 32u * 128u * 128u;
float value(float x) { return x; }
float value(uint16_t x) { uint32_t bits = uint32_t(x) << 16; float f; std::memcpy(&f, &bits, 4); return f; }
uint16_t bf16(float f) { uint32_t bits; std::memcpy(&bits, &f, 4); return uint16_t((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16); }
template<class T> std::vector<T> slice(const std::string& path, size_t total, size_t offset, size_t count) {
    if (offset > total || count > total - offset) throw std::runtime_error("invalid slice");
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() != static_cast<std::streamoff>(total * sizeof(T))) throw std::runtime_error("capture size mismatch");
    std::vector<T> result(count); file.seekg(offset * sizeof(T));
    file.read(reinterpret_cast<char*>(result.data()), count * sizeof(T));
    if (!file) throw std::runtime_error("capture read failed");
    for (T x : result) if (!std::isfinite(value(x))) throw std::runtime_error("nonfinite capture");
    return result;
}
template<int Block> float blackwell_dot(const uint16_t* a, const uint16_t* b) {
    float result = 0;
    for (int i = 0; i < 64; i += Block) result += qrt_q1_moe_hawkeye::dot_bf16_impl<26, 16, -133>(a + i, b + i, Block);
    return result;
}
float ieee_dot(const uint16_t* a, const uint16_t* b) {
    float result = 0; for (int i = 0; i < 64; ++i) result = std::fma(value(a[i]), value(b[i]), result); return result;
}
struct Stats {
    uint64_t elements = 0, mismatches = 0; double error2 = 0, norm2 = 0, maximum = 0;
    int64_t first = -1; float first_actual = 0, first_expected = 0;
    void add(float a, float b, int64_t index = -1) {
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("nonfinite result");
        const double delta = double(a) - b; error2 += delta * delta; norm2 += double(b) * b;
        if (a != b) {
            ++mismatches; maximum = std::max(maximum, std::abs(delta));
            if (first < 0) { first = index < 0 ? static_cast<int64_t>(elements) : index; first_actual = a; first_expected = b; }
        }
        ++elements;
    }
    void print() const {
        std::cout << "{\"elements\":" << elements << ",\"mismatch_count\":" << mismatches
                  << ",\"relative_l2\":" << std::sqrt(error2 / std::max(norm2, 1.0e-300))
                  << ",\"maximum_absolute_error\":" << maximum << ",\"first_index\":" << first
                  << ",\"first_actual\":" << first_actual << ",\"first_expected\":" << first_expected << '}';
    }
};

uint32_t float_bits(float value) { uint32_t bits; std::memcpy(&bits, &value, 4); return bits; }
float from_bits(uint32_t bits) { float value; std::memcpy(&value, &bits, 4); return value; }
uint32_t exponent_key(float input) { return input == 0 ? 0u : float_bits(input); }

struct RawStats {
    Stats numeric;
    uint64_t bit_mismatches = 0;
    std::array<uint16_t, 32u * 128u> mismatches_by_row{};
    int64_t first_bit_index = -1;
    uint32_t first_actual_bits = 0, first_expected_bits = 0;
    void add(float actual, float expected, size_t index) {
        numeric.add(actual, expected, static_cast<int64_t>(index));
        if (float_bits(actual) != float_bits(expected)) {
            ++bit_mismatches;
            ++mismatches_by_row.at(index / 128u);
            if (first_bit_index < 0) {
                first_bit_index = static_cast<int64_t>(index);
                first_actual_bits = float_bits(actual); first_expected_bits = float_bits(expected);
            }
        }
    }
    void print() const {
        const size_t differing_rows = static_cast<size_t>(std::count_if(mismatches_by_row.begin(), mismatches_by_row.end(),
            [](uint16_t count) { return count != 0; }));
        std::cout << "{\"numeric\":"; numeric.print();
        std::cout << ",\"bit_mismatch_count\":" << bit_mismatches << ",\"first_bit_index\":" << first_bit_index
                  << ",\"first_actual_bits\":" << first_actual_bits
                  << ",\"first_expected_bits\":" << first_expected_bits << ",\"differing_row_count\":" << differing_rows
                  << ",\"differing_rows_truncated\":" << (differing_rows > 16 ? "true" : "false") << ",\"differing_rows\":[";
        size_t printed = 0;
        for (size_t row = 0; row < mismatches_by_row.size(); ++row) if (mismatches_by_row[row]) {
            if (printed == 16) break;
            if (printed++) std::cout << ',';
            std::cout << "{\"head\":" << row / 128u << ",\"value\":" << row % 128u
                      << ",\"bit_mismatch_count\":" << mismatches_by_row[row] << '}';
        }
        std::cout << "]}";
    }
};

// Diagnostic reference-derived control ONLY. Infer an exponent's F32 value
// from captured round_f32(pre_decay_dot * exponent), intersecting constraints
// for identical exponent inputs. Ambiguous/conflicting values are never chosen.
// This is not a general SFU implementation and must not enter the runtime.
struct CapturedExponentControl {
    struct Bounds { uint32_t low = 0, high = 0; bool seen = false, inconsistent = false; };
    std::unordered_map<uint32_t, Bounds> required;
    uint64_t constraints = 0, unusable_products = 0, calls = 0, hits = 0, changed_calls = 0;
    uint64_t resolved = 0, ambiguous = 0, unobserved = 0, inconsistent = 0, host_differences = 0;

    CapturedExponentControl(const std::string& directory, size_t tokens, const std::vector<float>& g,
                            const std::string& prefix = "full-") {
        constexpr float log2e = 1.4426950408889634074f;
        required.reserve(tokens * 32u);
        for (size_t first = 0; first < tokens; first += 64) {
            const size_t last = std::min(first + 64u, tokens) - 1;
            for (size_t h = 0; h < 32; ++h) {
                required.emplace(exponent_key(g[last * 32u + h] * log2e), Bounds{});
                for (size_t t = first; t <= last; ++t)
                    required.emplace(exponent_key((g[last * 32u + h] - g[t * 32u + h]) * log2e), Bounds{});
            }
        }
        const size_t elements = tokens * 32u * 64u;
        auto before = slice<float>(directory + "/" + prefix + "a-dot-f32.bin", elements, 0, elements);
        auto after = slice<float>(directory + "/" + prefix + "a-f32.bin", elements, 0, elements);
        for (size_t t = 0; t < tokens; ++t) for (size_t h = 0; h < 32; ++h) for (size_t s = 0; s < t % 64u; ++s) {
            const size_t index = (t * 32u + h) * 64u + s;
            const float x = (g[t * 32u + h] - g[(t / 64u * 64u + s) * 32u + h]) * log2e;
            auto found = required.find(exponent_key(x));
            if (found == required.end()) continue;
            // Underflowed products can admit an arbitrarily wide exponent
            // interval; never infer uniqueness from a small local search there.
            if (!std::isnormal(before[index]) || !std::isnormal(after[index])) { ++unusable_products; continue; }
            const float ratio = static_cast<float>(double(after[index]) / double(before[index]));
            if (!(ratio > 0) || !std::isnormal(ratio)) { ++unusable_products; continue; }
            const uint32_t center = float_bits(ratio), begin = center > 2 ? center - 2 : 0;
            uint32_t low = UINT32_MAX, high = 0;
            for (uint32_t candidate = begin; candidate <= center + 2 && candidate < 0x7f800000u; ++candidate) {
                const float product = before[index] * from_bits(candidate);
                if (float_bits(product) == float_bits(after[index])) { low = std::min(low, candidate); high = std::max(high, candidate); }
            }
            auto& bounds = found->second;
            if (low == UINT32_MAX) { bounds.inconsistent = true; ++unusable_products; continue; }
            if (low == begin || high == center + 2) { ++unusable_products; continue; }
            ++constraints;
            if (!bounds.seen) { bounds.low = low; bounds.high = high; bounds.seen = true; }
            else { bounds.low = std::max(bounds.low, low); bounds.high = std::min(bounds.high, high); }
            if (bounds.low > bounds.high) bounds.inconsistent = true;
        }
        for (const auto& entry : required) {
            const auto& b = entry.second;
            if (b.inconsistent) ++inconsistent;
            else if (!b.seen) ++unobserved;
            else if (b.low != b.high) ++ambiguous;
            else { ++resolved; host_differences += float_bits(std::exp2(from_bits(entry.first))) != b.low; }
        }
    }
    float evaluate(float input) {
        ++calls; const float fallback = std::exp2(input);
        const auto found = required.find(exponent_key(input));
        if (found == required.end()) return fallback;
        const auto& b = found->second;
        if (!b.seen || b.inconsistent || b.low != b.high) return fallback;
        ++hits; changed_calls += float_bits(fallback) != b.low;
        return from_bits(b.low);
    }
    void reset_counts() { calls = hits = changed_calls = 0; }
    void print() const {
        std::cout << "{\"reference_derived\":true,\"production_implementation\":false,\"required_inputs\":" << required.size()
                  << ",\"resolved_inputs\":" << resolved << ",\"ambiguous_inputs\":" << ambiguous << ",\"unobserved_inputs\":" << unobserved
                  << ",\"inconsistent_inputs\":" << inconsistent << ",\"resolved_host_differences\":" << host_differences
                  << ",\"product_constraints\":" << constraints << ",\"unusable_products\":" << unusable_products << '}';
    }
};

// Each selected (head,value) row has an independent 128-element recurrent
// state. Carry it through every source token without injecting reference state.
void trajectory(const std::string& directory, size_t tokens, bool captured_exp_control = false) {
    const size_t chunks = (tokens + 63u) / 64u;
    auto path = [&](const char* name) { return directory + "/full-" + name + ".bin"; };
    auto k = slice<uint16_t>(path("k-normalized-bf16"), tokens * 2048u, 0, tokens * 2048u);
    auto w = slice<uint16_t>(path("w-bf16"), tokens * 4096u, 0, tokens * 4096u);
    auto u = slice<uint16_t>(path("u-bf16"), tokens * 4096u, 0, tokens * 4096u);
    auto g = slice<float>(path("g-cumsum-f32"), tokens * 32u, 0, tokens * 32u);
    auto reference_h = slice<uint16_t>(path("chunk-state-bf16"), chunks * state_elements, 0, chunks * state_elements);
    auto reference_v = slice<uint16_t>(path("v-new-bf16"), tokens * 4096u, 0, tokens * 4096u);
    auto reference_final = slice<float>(path("native-final-state-f32"), state_elements, 0, state_elements);
    const std::array<std::array<size_t, 2>, 16> selected{{
        {0,0}, {2,88}, {8,64}, {1,17}, {4,34}, {6,51}, {10,68}, {12,85},
        {14,102}, {16,119}, {18,8}, {20,25}, {22,42}, {24,59}, {28,76}, {31,127}
    }};
    struct Variant { const char* name; bool split_projection, fused_update, ieee; bool seeded_update = false; unsigned captured_exp_mask = 0; };
    std::vector<Variant> variants{
        {"blackwell_k128_k64_fma", false, true, false},
        {"blackwell_two_k64_k64_fma", true, true, false},
        {"blackwell_k128_k64_unfused", false, false, false},
        {"ieee_k128_k64_fma", false, true, true},
        {"blackwell_k128_seeded_k64", false, false, false, true}
    };
    std::unique_ptr<CapturedExponentControl> exponent_control;
    if (captured_exp_control) {
        exponent_control = std::make_unique<CapturedExponentControl>(directory, tokens, g);
        variants.push_back({"blackwell_k128_k64_captured_gate", false, true, false, false, 1});
        variants.push_back({"blackwell_k128_k64_captured_decay", false, true, false, false, 2});
        variants.push_back({"blackwell_k128_k64_captured_gate_decay", false, true, false, false, 3});
    }
    constexpr float log2e = 1.4426950408889634074f;
    std::cout << ",\"trajectory\":{\"tokens\":" << tokens << ",\"sampled_state_rows\":16,\"reference_state_injected\":false,\"coordinates_head_value\":[";
    for (size_t i = 0; i < selected.size(); ++i) { if (i) std::cout << ','; std::cout << '[' << selected[i][0] << ',' << selected[i][1] << ']'; }
    std::cout << ']';
    if (exponent_control) { std::cout << ",\"captured_exp_control\":"; exponent_control->print(); }
    std::cout << ",\"variants\":[";
    for (size_t mode = 0; mode < variants.size(); ++mode) {
        const auto& variant = variants[mode]; Stats states, values, terminal, isolated_values;
        struct Boundary { size_t head, v, chunk, key; float actual, expected, previous, decay, update; };
        std::vector<Boundary> first_boundaries;
        if (exponent_control) exponent_control->reset_counts();
        auto exponent = [&](float input, unsigned mask) {
            return exponent_control && (variant.captured_exp_mask & mask) ? exponent_control->evaluate(input) : std::exp2(input);
        };
        auto project = [&](const uint16_t* left, const uint16_t* right) {
            float sum = 0;
            if (variant.ieee) {
                for (size_t d = 0; d < 128; ++d) sum = std::fma(value(left[d]), value(right[d]), sum);
            } else if (variant.split_projection) {
                sum = blackwell_dot<64>(left, right) + blackwell_dot<64>(left + 64, right + 64);
            } else sum = qrt_q1_moe_hawkeye::dot_bf16_impl<26,16,-133>(left, right, 128);
            return sum;
        };
        for (const auto& coordinate : selected) {
            const size_t head = coordinate[0], v = coordinate[1];
            std::array<float, 128> state{}, previous_state{}, previous_update{};
            float previous_decay = 0;
            bool found_first_boundary = false;
            for (size_t chunk = 0; chunk < chunks; ++chunk) {
                const size_t first = chunk * 64u, valid = std::min(size_t(64), tokens - first);
                std::array<uint16_t, 128> rounded{};
                for (size_t d = 0; d < 128; ++d) {
                    rounded[d] = bf16(state[d]); const size_t index = chunk * state_elements + (head * 128u + v) * 128u + d;
                    states.add(value(rounded[d]), value(reference_h[index]), static_cast<int64_t>(index));
                    if (!found_first_boundary && value(rounded[d]) != value(reference_h[index])) {
                        first_boundaries.push_back({head, v, chunk, d, state[d], value(reference_h[index]),
                                                    previous_state[d], previous_decay, previous_update[d]});
                        found_first_boundary = true;
                    }
                }
                const float gate_last = g[(first + valid - 1u) * 32u + head], decay = exponent(gate_last * log2e, 2);
                std::array<uint16_t, 64> residual{};
                for (size_t t = 0; t < valid; ++t) {
                    const size_t row = (first + t) * 32u + head; const auto* left = &w[row * 128u];
                    const float current = value(u[row * 128u + v]) - project(left, rounded.data());
                    values.add(value(bf16(current)), value(reference_v[row * 128u + v]), static_cast<int64_t>(row * 128u + v));
                    residual[t] = bf16(current * exponent((gate_last - g[row]) * log2e, 1));
                    // Parallel input-isolation check only: never feed this value
                    // or reference state into the carried trajectory above.
                    const auto* reference_row = &reference_h[chunk * state_elements + (head * 128u + v) * 128u];
                    const float isolated = value(u[row * 128u + v]) - project(left, reference_row);
                    isolated_values.add(value(bf16(isolated)), value(reference_v[row * 128u + v]), static_cast<int64_t>(row * 128u + v));
                }
                for (size_t d = 0; d < 128; ++d) {
                    std::array<uint16_t, 64> left{};
                    for (size_t t = 0; t < valid; ++t) left[t] = k[((first + t) * 16u + head / 2u) * 128u + d];
                    if (variant.seeded_update) {
                        state[d] = qrt_q1_moe_hawkeye::accumulate_bf16_impl<26,16,-133>(
                            state[d] * decay, left.data(), residual.data(), 64);
                    } else {
                        const float update = variant.ieee ? ieee_dot(left.data(), residual.data()) : blackwell_dot<64>(left.data(), residual.data());
                        previous_state[d] = state[d]; previous_update[d] = update;
                        state[d] = variant.fused_update ? std::fma(state[d], decay, update) : state[d] * decay + update;
                    }
                }
                previous_decay = decay;
            }
            for (size_t d = 0; d < 128; ++d) {
                const size_t index = (head * 128u + v) * 128u + d;
                terminal.add(state[d], reference_final[index], static_cast<int64_t>(index));
            }
        }
        if (mode) std::cout << ',';
        std::cout << "{\"name\":\"" << variant.name << "\",\"chunk_state_bf16\":"; states.print();
        std::cout << ",\"v_new_bf16\":"; values.print(); std::cout << ",\"final_state_f32\":"; terminal.print();
        std::cout << ",\"same_input_projection\":{\"uses_reference_chunk_state\":true,\"feeds_trajectory\":false,\"v_new_bf16\":";
        isolated_values.print(); std::cout << "},\"reference_derived_exp_mask\":" << variant.captured_exp_mask;
        if (exponent_control) std::cout << ",\"captured_exp_calls\":" << exponent_control->calls << ",\"captured_exp_hits\":" << exponent_control->hits
                                       << ",\"captured_exp_changed_calls\":" << exponent_control->changed_calls;
        // Raw state comes from this CPU trajectory, not a hidden reference
        // checkpoint. The seeded-MMA negative control has no separate dot/FMA.
        if (!variant.seeded_update) {
            std::cout << ",\"first_state_boundaries\":[";
            for (size_t i = 0; i < first_boundaries.size(); ++i) {
                if (i) std::cout << ',';
                const auto& b = first_boundaries[i];
                std::cout << "{\"head\":" << b.head << ",\"value\":" << b.v << ",\"chunk\":" << b.chunk << ",\"key\":" << b.key
                          << ",\"actual_f32\":" << b.actual << ",\"actual_f32_bits\":" << float_bits(b.actual)
                          << ",\"expected_bf16\":" << b.expected << ",\"previous_f32\":" << b.previous
                          << ",\"previous_decay_f32\":" << b.decay << ",\"previous_update_f32\":" << b.update
                          << ",\"reference_raw_state_available\":false}";
            }
            std::cout << ']';
        }
        std::cout << '}';
    }
    std::cout << "]}";
}
}

int main(int argc, char** argv) try {
    const bool single_chunk_raw = argc >= 4 && std::string(argv[3]) == "--single-chunk-raw";
    if ((argc < 4 || argc > 6) ||
        (single_chunk_raw ? argc > 5 || (argc == 5 && std::string(argv[4]) != "--captured-exp-control")
                          : (argc >= 5 && std::string(argv[4]) != "--trajectory-sample") ||
                            (argc == 6 && std::string(argv[5]) != "--captured-exp-control"))) {
        std::cerr << "usage: fla-state-accumulator-probe <capture-dir> <source-tokens> <native-first-chunk-state-f32|-> [--trajectory-sample [--captured-exp-control]]\n"
                     "       fla-state-accumulator-probe <single-chunk-capture-dir> 64 --single-chunk-raw [--captured-exp-control]\n"; return 2;
    }
    char* end = nullptr; const unsigned long parsed = std::strtoul(argv[2], &end, 10);
    if (!*argv[2] || !end || *end || (single_chunk_raw ? parsed != 64 : parsed < 65 || parsed > 8192)) return 2;
    const size_t tokens = parsed, chunks = (tokens + 63u) / 64u;
    auto path = [&](const char* name) { return std::string(argv[1]) + (single_chunk_raw ? "/" : "/full-") + name + ".bin"; };
    auto k = slice<uint16_t>(path("k-normalized-bf16"), tokens * 2048u, 0, 64u * 2048u);
    auto u = slice<uint16_t>(path("u-bf16"), tokens * 4096u, 0, 64u * 4096u);
    auto g = slice<float>(path("g-cumsum-f32"), tokens * 32u, 0, 64u * 32u);
    std::vector<uint16_t> reference;
    std::vector<float> reference_raw, native;
    if (single_chunk_raw) {
        // A separately captured, zero-seeded single chunk exposes its raw F32
        // terminal state. It is NOT the first 64 tokens of a longer capture.
        auto initial = slice<uint16_t>(path("chunk-state-bf16"), state_elements, 0, state_elements);
        if (std::any_of(initial.begin(), initial.end(), [](uint16_t x) { return value(x) != 0; }))
            throw std::runtime_error("requires captured zero initial state");
        const auto residual = slice<uint16_t>(path("v-new-bf16"), 64u * 4096u, 0, 64u * 4096u);
        if (residual != u) throw std::runtime_error("zero-seed V-new must equal captured U bit-for-bit");
        reference_raw = slice<float>(path("state-f32"), state_elements, 0, state_elements);
        reference.reserve(state_elements);
        for (float x : reference_raw) reference.push_back(bf16(x));
    } else {
        auto seed = slice<float>(path("initial_state-f32"), state_elements, 0, state_elements);
        if (std::any_of(seed.begin(), seed.end(), [](float x) { return x != 0.0f; })) throw std::runtime_error("requires captured zero initial state");
        reference = slice<uint16_t>(path("chunk-state-bf16"), chunks * state_elements, state_elements, state_elements);
        if (std::string(argv[3]) != "-") native = slice<float>(argv[3], state_elements, 0, state_elements);
    }
    // Zero seed makes the first-chunk residual exactly U. The gated product is
    // rounded before the state-update dot, exactly like the reference formula.
    constexpr float log2e = 1.4426950408889634074f;
    std::unique_ptr<CapturedExponentControl> single_chunk_exp;
    if (single_chunk_raw && argc == 5)
        single_chunk_exp = std::make_unique<CapturedExponentControl>(argv[1], tokens, g, "");
    std::vector<std::array<uint32_t, 6>> changed_gate_products;
    for (size_t t = 0; t < 64; ++t) for (size_t h = 0; h < 32; ++h) {
        const float argument = (g[63u * 32u + h] - g[t * 32u + h]) * log2e;
        const float host_gate = std::exp2(argument), gate = single_chunk_exp ? single_chunk_exp->evaluate(argument) : host_gate;
        for (size_t d = 0; d < 128; ++d) {
            const size_t index = (t * 32u + h) * 128u + d;
            const uint16_t host_product = bf16(value(u[index]) * host_gate), product = bf16(value(u[index]) * gate);
            if (host_product != product) changed_gate_products.push_back({static_cast<uint32_t>(t), static_cast<uint32_t>(h),
                static_cast<uint32_t>(d), u[index], host_product, product});
            u[index] = product;
        }
    }
    struct Variant { const char* name; float (*dot)(const uint16_t*, const uint16_t*); };
    const std::array<Variant, 3> variants{{{"blackwell_k64", blackwell_dot<64>},
        {"blackwell_two_k32", blackwell_dot<32>}, {"ieee_fma_sequential", ieee_dot}}};
    Stats native_reference;
    if (!native.empty()) for (size_t i = 0; i < state_elements; ++i) native_reference.add(value(bf16(native[i])), value(reference[i]));
    std::cout << std::setprecision(17) << "{\"kind\":\""
              << (single_chunk_raw ? "cpu_single_chunk_raw_state_attribution" : "cpu_first_chunk_state_attribution")
              << "\",\"source_tokens\":" << tokens
              << ",\"replay_tokens\":64,\"zero_initial_state\":true,\"exponent\":\"host_exp2f_not_sm121_sfu\",\"inference_acceptance\":false,\"native_reference_bf16\":";
    native_reference.print();
    if (single_chunk_raw) {
        std::cout << ",\"raw_initial_state_available\":false,\"zero_seed_basis\":\"capture_recipe_assumption_with_zero_bf16_H_and_U_equals_V_new_checks\"";
    }
    if (single_chunk_exp) {
        std::cout << ",\"captured_exp_control\":"; single_chunk_exp->print();
        std::cout << ",\"captured_exp_calls\":" << single_chunk_exp->calls << ",\"captured_exp_hits\":" << single_chunk_exp->hits
                  << ",\"captured_exp_changed_calls\":" << single_chunk_exp->changed_calls
                  << ",\"changed_gated_bf16_product_count\":" << changed_gate_products.size() << ",\"first_changed_products\":[";
        for (size_t i = 0; i < std::min(changed_gate_products.size(), size_t(16)); ++i) {
            if (i) std::cout << ',';
            const auto& p = changed_gate_products[i];
            std::cout << "{\"token\":" << p[0] << ",\"head\":" << p[1] << ",\"value\":" << p[2]
                      << ",\"input_bf16_bits\":" << p[3] << ",\"host_product_bf16_bits\":" << p[4]
                      << ",\"captured_product_bf16_bits\":" << p[5] << '}';
        }
        std::cout << ']';
    }
    std::cout << ",\"variants\":[";
    for (size_t mode = 0; mode < variants.size(); ++mode) {
        Stats expected, native_f32; RawStats raw;
        for (size_t h = 0; h < 32; ++h) for (size_t v = 0; v < 128; ++v) {
            std::array<uint16_t, 64> right{};
            for (size_t t = 0; t < 64; ++t) right[t] = u[(t * 32u + h) * 128u + v];
            for (size_t d = 0; d < 128; ++d) {
                std::array<uint16_t, 64> left{};
                for (size_t t = 0; t < 64; ++t) left[t] = k[(t * 16u + h / 2u) * 128u + d];
                const size_t index = (h * 128u + v) * 128u + d;
                const float result = variants[mode].dot(left.data(), right.data());
                expected.add(value(bf16(result)), value(reference[index]));
                if (!native.empty()) native_f32.add(result, native[index]);
                if (!reference_raw.empty()) raw.add(result, reference_raw[index], index);
            }
        }
        if (mode) std::cout << ',';
        std::cout << "{\"name\":\"" << variants[mode].name << "\",\"reference_bf16\":"; expected.print();
        std::cout << ",\"native_f32\":"; native_f32.print();
        if (!reference_raw.empty()) { std::cout << ",\"reference_raw_f32\":"; raw.print(); }
        std::cout << '}';
    }
    std::cout << ']';
    if (!single_chunk_raw && argc >= 5) trajectory(argv[1], tokens, argc == 6);
    std::cout << "}\n"; return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 3; }
