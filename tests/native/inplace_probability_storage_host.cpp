#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
#include "native/providers/ck_fmha/inplace_probability_storage.h"

namespace p = qrt_inplace_probability_storage;
uint32_t bits(float x) { uint32_t u; __builtin_memcpy(&u, &x, 4); return u; }
float original(size_t i) { return float(int((i * 137u + i / 17u) % 65521u) - 32760) * 0.0625f; }
uint16_t payload(size_t i) { return uint16_t(i * 197u + i / 31u); }

int main() {
    for (unsigned x = 0; x < 65536u; ++x) {
        const auto value = p::encode(uint16_t(x));
        assert(bits(value) == (0x3f800000u | x));
        assert(std::isnormal(value) && value >= 1.0f && value < 2.0f);
        assert(p::decode(value) == x);
    }
    uint64_t layouts = 0, cells = 0;
    for (unsigned n = 1; n <= qrt_long_attention_layout::maximum_tokens; ++n) {
        for (unsigned q : {1u, 17u, 32u, 64u, 127u, 128u}) {
            const auto a = p::layout(q, n);
            if (q > n) { assert(!a.elements); continue; }
            const size_t rows = size_t(q) * 16u;
            const size_t score_bytes = rows * n * 4u;
            const size_t scale_bytes = rows * ((n + 31u) / 32u + 1u) * 4u;
            const size_t error_bytes = rows * 256u * 4u;
            assert(a.probability == 0u && a.scales * 4u == score_bytes);
            assert(a.errors * 4u == score_bytes + scale_bytes);
            assert(a.indices * 4u == score_bytes + scale_bytes + error_bytes);
            assert(a.count * 4u == score_bytes + scale_bytes + error_bytes * 2u);
            assert(a.elements * 4u == a.count * 4u + 4u);
            const auto old = qrt_long_attention_layout::layout(q, n);
            assert((old.elements - a.elements) * 4u == rows * n * 2u);
            assert((rows * n - 1u) * 4u + 4u == a.scales * 4u);
            ++layouts;
        }
    }
    for (auto q : {0u, 129u, std::numeric_limits<unsigned>::max()})
        assert(!p::layout(q, 264736u).elements);
    assert(!p::layout(1u, 0u).elements && !p::layout(1u, 264737u).elements);
    struct Shape { unsigned start, queries; };
    const Shape shapes[] = {{0,1},{1,32},{1,128},{8191,2},{8192,33},
        {16384,17},{32768,3},{65520,17},{131072,1},{264735,1}};
    std::mt19937 random(71937u);
    for (const auto shape : shapes) {
        const unsigned n = shape.start + shape.queries, rows = shape.queries * 16u;
        const size_t count = size_t(rows) * n;
        constexpr size_t guard = 32u;
        std::vector<float> arena(count + 2u * guard, -37.25f);
        std::vector<uint16_t> separate(count + 2u * guard, 0xa5a5u);
        for (size_t i = 0; i < count; ++i) arena[i + guard] = original(i);
        auto* memory = reinterpret_cast<uint16_t*>(arena.data() + guard);
        auto* packed = separate.data() + guard;
        std::vector<unsigned> order(rows); std::iota(order.begin(), order.end(), 0u);
        std::shuffle(order.begin(), order.end(), random);
        // Independent row order models concurrently scheduled CTAs. Within a
        // row, each original K32 score is read before its own probability is
        // stored. Every other score cell must remain available until consumed.
        for (const unsigned row : order) {
            const unsigned tokens = shape.start + row / 16u + 1u;
            const unsigned end = std::min(n, ((tokens + 31u) / 32u) * 32u);
            for (unsigned base = 0; base < end; base += 32u) {
                const unsigned last = std::min(end, base + 32u);
                for (unsigned key = base; key < last; ++key) {
                    const size_t index = size_t(row) * n + key;
                    assert(bits(arena[guard + index]) == bits(original(index)));
                }
                for (unsigned key = base; key < last; ++key) {
                    const size_t index = size_t(row) * n + key;
                    const auto value = key < tokens ? payload(index) : uint16_t(0u);
                    p::store<true>(memory, index, value);
                    p::store<false>(packed, index, value);
                    assert(p::load<true>(memory, index) == p::load<false>(packed, index));
                    assert(bits(arena[guard + index]) == (0x3f800000u | value));
                    ++cells;
                }
            }
        }
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned tokens = shape.start + row / 16u + 1u;
            const unsigned end = std::min(n, ((tokens + 31u) / 32u) * 32u);
            for (unsigned key = 0; key < n; ++key) {
                const size_t index = size_t(row) * n + key;
                if (key < end) assert(p::load<true>(memory, index) == p::load<false>(packed, index));
                else {
                    assert(bits(arena[guard + index]) == bits(original(index)));
                    assert(separate[guard + index] == 0xa5a5u);
                }
            }
        }
        for (size_t i = 0; i < guard; ++i) {
            assert(arena[i] == -37.25f && arena[guard + count + i] == -37.25f);
            assert(separate[i] == 0xa5a5u && separate[guard + count + i] == 0xa5a5u);
        }
    }
    const auto old = qrt_long_attention_layout::layout(128u, 264736u);
    const auto candidate = p::layout(128u, 264736u);
    std::printf("{\"payload_encodings\":65536,\"layouts\":%llu,\"consumed_cells\":%llu,"
        "\"maximum_old_scratch_bytes\":%zu,\"maximum_candidate_scratch_bytes\":%zu,"
        "\"maximum_saved_bytes\":%zu,\"strict_aliasing\":true,\"guards_pass\":true,"
        "\"native_acceptance\":false}\n", (unsigned long long)layouts, (unsigned long long)cells,
        old.elements * 4u, candidate.elements * 4u, (old.elements - candidate.elements) * 4u);
}
