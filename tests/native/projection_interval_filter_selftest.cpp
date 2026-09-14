#include "../../native/providers/moe_accumulator/sm121_projection_interval_filter.h"
#include "../../native/providers/moe_accumulator/sm121_group16_modulo.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
namespace filter = qrt_sm121_projection_interval_filter;
namespace bound = qrt_sm121_pv_bound;
namespace canonical = qrt_q1_moe_hawkeye;
constexpr unsigned guard = 65u;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t value) { if (value != hipSuccess) throw std::runtime_error(hipGetErrorString(value)); }
void complete() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now() < deadline, "filter deadline");
        std::this_thread::yield();
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* base = nullptr; std::vector<T> initial;
    explicit Buffer(const std::vector<T>& values) : initial(values) {
        check(hipMalloc(reinterpret_cast<void**>(&base), values.size() * sizeof(T)));
        check(hipMemcpy(base, values.data(), values.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Buffer() { if (base) (void)hipFree(base); }
    T* data() { return base + guard; }
    std::vector<T> read() {
        auto values = initial; check(hipMemcpy(values.data(), base, values.size() * sizeof(T), hipMemcpyDeviceToHost));
        for (unsigned i = 0u; i < guard; ++i)
            require(values[i] == initial[i] && values[values.size() - 1u - i] == initial[initial.size() - 1u - i], "filter redzone");
        return values;
    }
};
unsigned random_state = 0x395bf16u;
unsigned random_word() { random_state ^= random_state << 13u; random_state ^= random_state >> 17u; random_state ^= random_state << 5u; return random_state; }
void run(unsigned rows, unsigned tokens, unsigned width, unsigned mode, unsigned selection, bool corrupt) {
    const unsigned cells = rows * tokens;
    std::vector<uint16_t> weights(size_t(rows) * width + 2u * guard, 0x5a5au), inputs(size_t(tokens) * width + 2u * guard, 0x5a5au);
    auto fill = [&](std::vector<uint16_t>& values, unsigned count, bool weight) {
        for (unsigned row = 0u; row < count; ++row) for (unsigned k = 0u; k < width; ++k) {
            unsigned exponent = mode == 1u ? 80u + random_word() % 95u : 120u + random_word() % 9u;
            uint16_t value = uint16_t((exponent << 7u) | (random_word() & 0x807fu));
            if (mode == 2u && weight && row % 9u == 0u) value = uint16_t((4u << 7u) | (value & 0x807fu));
            if (mode == 2u && !weight && row % 7u == 0u) value &= 0x8000u;
            values[guard + size_t(row) * width + k] = value;
        }
    };
    fill(weights, rows, true); fill(inputs, tokens, false);
    std::vector<unsigned> selected;
    if (selection) for (unsigned cell = cells; cell--;) if (selection == 1u || cell % 7u == 0u) selected.push_back(cell);
    if (corrupt) { require(!selected.empty(), "empty corrupt test"); selected.back() = cells + 17u; }
    std::vector<unsigned> indices(selected.size() + 2u * guard, 0xa5a5a5a5u), remaining(indices.size(), 0xa5a5a5a5u);
    std::copy(selected.begin(), selected.end(), indices.begin() + guard);
    std::vector<unsigned> counts(2u + 2u * guard, 0xa5a5a5a5u);
    std::vector<uint8_t> mask(cells + 2u * guard, 0xa5u);
    std::vector<float> output(cells + 2u * guard, 12345.25f);
    Buffer<uint16_t> dw(weights), di(inputs); Buffer<unsigned> ds(indices), dr(remaining), dc(counts);
    Buffer<uint8_t> dm(mask); Buffer<float> out(output);
    require(filter::launch(dw.data(), di.data(), ds.data(), unsigned(selected.size()), dm.data(), cells - 1u,
        out.data(), rows, tokens, width, dr.data(), selected.size(), dc.data(), nullptr) == hipErrorInvalidValue, "short mask accepted");
    if (!selected.empty()) require(filter::launch(dw.data(), di.data(), ds.data(), unsigned(selected.size()), dm.data(), cells,
        out.data(), rows, tokens, width, dr.data(), selected.size() - 1u, dc.data(), nullptr) == hipErrorInvalidValue, "short compact output accepted");
    check(filter::launch(dw.data(), di.data(), ds.data(), unsigned(selected.size()), dm.data(), cells,
        out.data(), rows, tokens, width, dr.data(), selected.size(), dc.data(), nullptr));
    complete(); output = out.read(); mask = dm.read(); remaining = dr.read(); counts = dc.read();
    require(dw.read() == weights && di.read() == inputs && ds.read() == indices, "filter immutable source");
    require(counts[guard] <= selected.size(), "filter compact overflow");
    std::vector<unsigned> expected_remaining; unsigned admitted = 0u, independent_dots = 0u;
    std::vector<bool> present(cells, false);
    for (unsigned cell : selected) if (cell < cells) present[cell] = true;
    for (unsigned cell = 0u; cell < cells; ++cell) {
        const unsigned state = mask[guard + cell];
        require(state == (present[cell] ? 1u : 0u) || (present[cell] && state == 2u), "filter candidate identity");
        if (!present[cell] || state == 1u) {
            require(output[guard + cell] == 12345.25f, "unadmitted output changed");
            if (present[cell]) expected_remaining.push_back(cell);
            continue;
        }
        ++admitted; ++independent_dots;
        auto carry = canonical::value_from_float(0.0f, -133);
        for (unsigned base = 0u; base < width; base += 16u) {
            canonical::Value terms[17]; terms[0] = carry;
            for (unsigned k = 0u; k < 16u; ++k) terms[k + 1u] = canonical::multiply_bf16(
                inputs[guard + size_t(cell / rows) * width + base + k], weights[guard + size_t(cell % rows) * width + base + k], -133);
            carry = canonical::group_sum<26, -133>(terms, 17u);
        }
        const float exact = canonical::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
        require(bound::bf16(output[guard + cell]) == bound::bf16(exact), "false interval admission");
    }
    std::vector<unsigned> actual(remaining.begin() + guard, remaining.begin() + guard + counts[guard]);
    std::sort(actual.begin(), actual.end()); require(actual == expected_remaining, "remaining candidate permutation");
    for (size_t i = counts[guard]; i < selected.size(); ++i) require(remaining[guard + i] == 0xa5a5a5a5u, "unused compact tail modified");
    require(corrupt ? counts[guard + 1u] != 0u : counts[guard + 1u] == 0u, "device identity guard");
    std::printf("{\"kind\":\"projection_interval_filter_safety\",\"rows\":%u,\"tokens\":%u,\"width\":%u,\"mode\":%u,\"selection\":%u,\"corrupt_index_case\":%s,\"selected\":%zu,\"admitted\":%u,\"remaining\":%u,\"independent_cpu_dots\":%u,\"false_admissions\":0,\"candidate_identity_pass\":true,\"host_capacity_guards_pass\":true,\"device_index_guards_pass\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        rows, tokens, width, mode, selection, corrupt ? "true" : "false", selected.size(), admitted, counts[guard], independent_dots);
}
int main() try {
    hipDeviceProp_t device{}; check(hipGetDeviceProperties(&device, 0)); require(std::string(device.gcnArchName).find("gfx1151") == 0u, "requires gfx1151");
    for (unsigned shape = 0u; shape < 4u; ++shape) for (unsigned mode = 0u; mode < 3u; ++mode) for (unsigned selection = 0u; selection < 3u; ++selection) {
        const unsigned rows[] = {1u, 17u, 32u, 129u}, tokens[] = {1u, 19u, 65u, 65u}, widths[] = {16u, 64u, 256u, 2048u};
        run(rows[shape], tokens[shape], widths[shape], mode, selection, false);
    }
    run(17u, 19u, 64u, 2u, 2u, true);
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "interval_filter_error=%s\n", error.what()); return 1; }
