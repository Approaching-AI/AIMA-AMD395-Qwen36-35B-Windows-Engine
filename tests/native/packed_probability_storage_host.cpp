#include <algorithm>
#include <cassert>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>
#include "native/providers/ck_fmha/inplace_probability_storage.h"
#include "native/providers/ck_fmha/packed_probability_storage.h"

namespace packed = qrt_packed_probability_storage;
uint32_t representation(const float* value) {
    uint32_t result; __builtin_memcpy(&result, value, sizeof(result)); return result;
}
float original(size_t i) { return float(int((i * 137u + i / 17u) % 65521u) - 32760) * 0.0625f; }
uint16_t payload(size_t i) { return uint16_t(i * 197u + i / 31u); }

int main() {
    uint64_t encodings = 0, cells = 0, pairs = 0;
    for (unsigned first = 0; first < 65536u; ++first) {
        for (unsigned second : {0u, first, first ^ 65535u, (first * 197u) & 65535u}) {
            float memory[3] = {-37.25f, -37.25f, -37.25f};
            packed::store_pair(memory + 1u, 0u, 0u, 1u, uint16_t(first), uint16_t(second));
            assert(packed::load(memory + 1u, 0u, 0u, 1u) == first);
            // Even an odd-length row has a complete score word for the pair.
            assert(packed::load(memory + 1u, 0u, 1u, 1u) == second);
            uint16_t expected[2] = {uint16_t(first), uint16_t(second)};
            uint32_t bits; __builtin_memcpy(&bits, expected, sizeof(bits));
            assert(representation(memory + 1u) == bits);
            assert(memory[0] == -37.25f && memory[2] == -37.25f);
            ++encodings;
        }
    }
    struct Shape { unsigned start, queries; };
    const Shape shapes[] = {{0,1},{1,32},{1,128},{31,2},{32,33},{63,2},
        {8191,2},{8192,33},{16384,17},{32768,3},{65520,17},{131072,1},{264735,1}};
    std::mt19937 random(801729u);
    for (const auto shape : shapes) {
        const unsigned stride = shape.start + shape.queries, rows = shape.queries * 16u;
        const size_t count = size_t(rows) * stride;
        constexpr size_t guard = 32u;
        std::vector<float> storage(count + 2u * guard, -37.25f);
        auto* scores = storage.data() + guard;
        for (size_t i = 0; i < count; ++i) scores[i] = original(i);
        std::vector<unsigned> pending(rows), position(rows);
        std::iota(pending.begin(), pending.end(), 0u);
        // Interleave individual K32 tiles from different rows. All 32 source
        // scores are consumed before any pair in that row is published.
        while (!pending.empty()) {
            const size_t selected = random() % pending.size();
            const unsigned row = pending[selected], base = position[row];
            const unsigned tokens = shape.start + row / 16u + 1u;
            const unsigned end = std::min(stride, ((tokens + 31u) / 32u) * 32u);
            const unsigned last = std::min(end, base + 32u);
            uint16_t probability[32]{};
            for (unsigned key = base; key < last; ++key) {
                const size_t index = size_t(row) * stride + key;
                const float expected = original(index);
                assert(representation(scores + index) == representation(&expected));
                probability[key - base] = key < tokens ? payload(index) : uint16_t(0u);
            }
            for (unsigned key = base; key < last; key += 2u) {
                assert(key / 2u < last && key / 2u < stride);
                packed::store_pair(scores, row, key, stride, probability[key - base], probability[key - base + 1u]);
                ++pairs;
            }
            for (unsigned key = base; key < last; ++key) {
                assert(packed::load(scores, row, key, stride) == probability[key - base]);
                ++cells;
            }
            position[row] = last;
            if (last == end) { pending[selected] = pending.back(); pending.pop_back(); }
        }
        for (unsigned row = 0; row < rows; ++row) {
            const unsigned tokens = shape.start + row / 16u + 1u;
            const unsigned end = std::min(stride, ((tokens + 31u) / 32u) * 32u);
            for (unsigned key = 0; key < end; ++key) {
                const auto expected = key < tokens ? payload(size_t(row) * stride + key) : uint16_t(0u);
                assert(packed::load(scores, row, key, stride) == expected);
            }
            if (end & 1u) assert(packed::load(scores, row, end, stride) == 0u);
            for (unsigned word = (end + 1u) / 2u; word < stride; ++word) {
                const size_t index = size_t(row) * stride + word;
                const float expected = original(index);
                assert(representation(scores + index) == representation(&expected));
            }
        }
        for (size_t i = 0; i < guard; ++i)
            assert(storage[i] == -37.25f && storage[guard + count + i] == -37.25f);
    }
    const auto original_layout = qrt_long_attention_layout::layout(128u, 264736u);
    const auto candidate = qrt_inplace_probability_storage::layout(128u, 264736u);
    std::printf("{\"payload_pairs\":%llu,\"consumed_cells\":%llu,\"pair_writes\":%llu,"
        "\"interleaved_rows\":true,\"odd_stride_and_causal_tail\":true,\"strict_aliasing\":true,"
        "\"single_writer_per_score_word\":true,\"maximum_scratch_bytes\":%zu,"
        "\"maximum_saved_bytes\":%zu,\"probability_read_bytes\":2,\"gpu_execution\":false}\n",
        (unsigned long long)encodings, (unsigned long long)cells, (unsigned long long)pairs,
        candidate.elements * 4u, (original_layout.elements - candidate.elements) * 4u);
}
