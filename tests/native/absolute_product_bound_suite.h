#pragma once
#include <hip/hip_runtime.h>
#include "../../native/providers/moe_accumulator/bf16_positive_sum_bound.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

namespace absolute_product_bound_test {
constexpr unsigned guard = 65u;
constexpr uint16_t operand_guard = 0x5a5au;
constexpr unsigned flag_guard = 0xa5a5a5a5u;
constexpr float output_guard = -12345.25f;
uint32_t seed = 0x3958192u;
uint32_t random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
void require(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
void check(hipError_t status) { if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status)); }
double bf16(uint16_t raw) { return double(qrt_bf16_positive_sum_bound::value(uint32_t(raw) << 16u)); }
bool eligible(uint16_t raw) { const unsigned e = (raw >> 7u) & 255u; return !(raw & 0x7fffu) || (e >= 64u && e <= 191u); }
void complete() {
    hipEvent_t event; check(hipEventCreate(&event)); check(hipEventRecord(event));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    for (;;) {
        const auto status = hipEventQuery(event);
        if (status == hipSuccess) break;
        if (status != hipErrorNotReady) check(status);
        require(std::chrono::steady_clock::now() < deadline, "absolute product matrix timeout");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    check(hipEventDestroy(event));
}
template<class T> struct Buffer {
    T* base = nullptr;
    explicit Buffer(const std::vector<T>& data) {
        check(hipMalloc(reinterpret_cast<void**>(&base), data.size() * sizeof(T)));
        check(hipMemcpy(base, data.data(), data.size() * sizeof(T), hipMemcpyHostToDevice));
    }
    ~Buffer() { if (base) (void)hipFree(base); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    T* data() { return base + guard; }
    void read(std::vector<T>& host) { check(hipMemcpy(host.data(), base, host.size() * sizeof(T), hipMemcpyDeviceToHost)); }
    void unchanged(const std::vector<T>& expected) {
        auto actual = expected; read(actual); require(actual == expected, "matrix modified an input or its redzone");
    }
};
struct Shape { unsigned rows, tokens, k, mode; };
template<class Launch> void run(Shape s, Launch launch) {
    const size_t cells = size_t(s.rows) * s.tokens;
    std::vector<uint16_t> weights(size_t(s.rows) * s.k + 2u * guard, operand_guard);
    std::vector<uint16_t> inputs(size_t(s.tokens) * s.k + 2u * guard, operand_guard);
    std::vector<unsigned> wf(s.rows + 2u * guard, flag_guard), xf(s.tokens + 2u * guard, flag_guard);
    std::vector<double> wn(s.rows), xn(s.tokens), reference(cells);
    for (unsigned which = 0u; which < 2u; ++which) {
        auto& values = which ? inputs : weights;
        auto& flags = which ? xf : wf;
        auto& norms = which ? xn : wn;
        const unsigned rows = which ? s.tokens : s.rows;
        for (unsigned row = 0u; row < rows; ++row) {
            bool valid = true; double squares = 0.0;
            for (unsigned k = 0u; k < s.k; ++k) {
                unsigned exponent = 119u + random_word() % 10u;
                if (s.mode == 1u) exponent = 64u + random_word() % 128u;
                if (s.mode == 3u) exponent = 64u;
                if (s.mode == 4u) exponent = 191u;
                uint16_t value = uint16_t((exponent << 7u) | (random_word() & 0x807fu));
                if (row % 19u == 0u) value = uint16_t((k & 1u) << 15u);
                if (s.mode == 2u && k == (which ? 3u : 1u) && row % 7u == 1u)
                    value = which ? 0x8001u : 0x6000u;
                values[guard + size_t(row) * s.k + k] = value;
                valid &= eligible(value); const double real = bf16(value); squares += real * real;
            }
            flags[guard + row] = valid; norms[row] = std::sqrt(squares);
        }
    }
    for (unsigned token = 0u; token < s.tokens; ++token) for (unsigned row = 0u; row < s.rows; ++row) {
        double sum = 0.0;
        for (unsigned k = 0u; k < s.k; ++k)
            sum += std::fabs(bf16(weights[guard + size_t(row) * s.k + k]) * bf16(inputs[guard + size_t(token) * s.k + k]));
        reference[size_t(token) * s.rows + row] = sum;
    }
    Buffer<uint16_t> dw(weights), dx(inputs);
    Buffer<unsigned> dwf(wf), dxf(xf);
    for (unsigned partial = 0u; partial < 2u; ++partial) {
        const size_t first = partial ? s.rows + 1u : 0u;
        const unsigned count = unsigned(cells - first - (partial ? 1u : 0u));
        require(count > 0u, "invalid synthetic window");
        std::vector<float> output(count + 2u * guard, output_guard);
        Buffer<float> bounds(output);
        launch(dw.data(), dx.data(), dwf.data(), dxf.data(), bounds.data(),
            s.rows, s.tokens, s.k, first, count);
        check(hipGetLastError()); complete(); bounds.read(output);
        unsigned underestimates = 0u, finite_bounds = 0u, fallback = 0u, tighter = 0u;
        double maximum_ratio = 0.0;
        for (unsigned index = 0u; index < count; ++index) {
            const size_t absolute = first + index;
            const unsigned row = unsigned(absolute % s.rows), token = unsigned(absolute / s.rows);
            const float upper = output[guard + index];
            require(!std::isnan(upper) && upper >= 0.0f, "invalid absolute product bound");
            if (!wf[guard + row] || !xf[guard + token]) {
                ++fallback; require(std::isinf(upper), "ineligible row did not retain exact replay");
            }
            underestimates += double(upper) < reference[absolute];
            finite_bounds += std::isfinite(upper);
            tighter += double(upper) < wn[row] * xn[token];
            if (std::isfinite(upper) && reference[absolute] > 0.0)
                maximum_ratio = std::max(maximum_ratio, double(upper) / reference[absolute]);
        }
        for (unsigned i = 0u; i < guard; ++i)
            require(output[i] == output_guard && output[guard + count + i] == output_guard, "bound output redzone changed");
        dw.unchanged(weights); dx.unchanged(inputs); dwf.unchanged(wf); dxf.unchanged(xf);
        std::printf("{\"kind\":\"absolute_product_matrix_bound\",\"rows\":%u,\"tokens\":%u,\"k\":%u,\"mode\":%u,\"first_element\":%zu,\"cells\":%u,\"finite_bounds\":%u,\"fallback_cells\":%u,\"underestimates\":%u,\"tighter_than_cauchy_cells\":%u,\"maximum_finite_ratio\":%.9g,\"redzones_pass\":true,\"inputs_immutable\":true,\"inference_acceptance\":false}\n",
            s.rows,s.tokens,s.k,s.mode,first,count,finite_bounds,fallback,underestimates,tighter,maximum_ratio);
        require(!underestimates, "matrix sum underestimated independent double product sum");
    }
}
template<class Launch> void suite(Launch launch) {
    hipDeviceProp_t properties{}; check(hipGetDeviceProperties(&properties, 0));
    require(!std::strncmp(properties.gcnArchName, "gfx1151", 7u), "requires gfx1151");
    for (const Shape s : {Shape{3,5,16,0}, {129,65,512,0}, {257,131,2048,1},
                         {129,67,4096,2}, {37,71,2048,3}, {131,17,2048,4}}) run(s,launch);
}
}
