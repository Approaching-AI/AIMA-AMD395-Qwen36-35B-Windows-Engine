#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include "sm121_compact_integer_dot4.h"

namespace compact = qrt_sm121_compact_integer_dot4;
static_assert(std::is_trivial<compact::Row>::value, "HIP shared rows must need no initialization");
using qrt_q1_moe_hawkeye::Value;
static uint32_t seed = 0x3958192u;
static uint32_t random_word() { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; return seed; }
static int coefficient(uint16_t word) { return word & 0x8000u ? int(word) - 65536 : int(word); }
static void same(Value actual, Value expected, unsigned group, unsigned path) {
    if (actual.significand != expected.significand || actual.exponent != expected.exponent || actual.negative != expected.negative) {
        std::fprintf(stderr, "group=%u path=%u actual=%u,%d,%u expected=%u,%d,%u\n", group, path,
            actual.significand, actual.exponent, actual.negative, expected.significand, expected.exponent, expected.negative);
        std::abort();
    }
}
int main() {
    uint64_t roundtrips = 0, supported_roundtrips = 0, products = 0;
    // Every bit pattern appears alone, in a uniform row, and with two different
    // scale owners. The fallback must retain negative zero and exceptional bits.
    for (unsigned bits = 0; bits < 65536u; ++bits) {
        for (unsigned context = 0; context < 4u; ++context) {
            uint16_t words[16]{};
            if (context == 1u) for (auto& v : words) v = uint16_t(bits);
            if (context == 2u) words[15] = 0x3f80u;
            if (context == 3u) words[15] = 0x7f7fu;
            words[bits % 15u] = uint16_t(bits);
            const auto row = compact::prepare(words);
            for (unsigned i = 0; i < 16u; ++i) { assert(compact::original(row, i) == words[i]); ++roundtrips; }
            supported_roundtrips += compact::unit(row) != 0;
        }
    }
    unsigned paths[3]{};
    Value previous{0u, -133, false};
    for (unsigned group = 0; group < 500000u; ++group) {
        uint16_t a[16]{}, b[16]{}; Value values[17];
        const unsigned ae = 1u + random_word() % 220u, be = 1u + random_word() % 220u;
        const unsigned spread = 1u + group % 34u;
        for (unsigned i = 0; i < 16u; ++i) {
            a[i] = uint16_t((random_word() & 0x807fu) | ((ae + random_word() % spread) << 7u));
            b[i] = uint16_t((random_word() & 0x807fu) | ((be + random_word() % spread) << 7u));
            if (group % 11u == 0u && i % 3u == 0u) a[i] = 0;
            if (group % 13u == 0u && i % 3u == 1u) b[i] = 0x8000;
            if (group % 17u == 0u) { a[i] = uint16_t((random_word() & 0x807fu) | ((ae + i % 9u) << 7u)); b[i] = uint16_t((random_word() & 0x807fu) | ((be + 8u - i % 9u) << 7u)); }
            if (group % 19u == 0u && i == 2u) a[i] = 1u;
            if (group % 23u == 0u && i == 3u) b[i] = 0x7fc1u;
            if (group % 29u == 0u) { a[i] = uint16_t(0x3fffu | ((group & 1u) << 15u)); b[i] = 0x3fffu; }
            if (group % 31u == 0u) { a[i] = uint16_t((0x3f81u + (i / 2u % 4u) * 128u) | ((i & 1u) << 15u)); b[i] = 0x3f85u; }
            if (group % 37u == 0u) { a[i] = i % 2u ? 0x8000u : 0x7f7fu; b[i] = i % 2u ? 0x7f7fu : 0u; }
            if (group % 41u == 0u) { a[i] = uint16_t(0x0081u + (i & 1u)); b[i] = 0x0081u; }
            if (group % 43u == 0u) { a[i] = i == 2u ? 0x7f7fu : uint16_t(0x0081u + (i & 1u)); b[i] = i == 2u ? 0u : 0x0081u; }
            values[i + 1u] = qrt_q1_moe_hawkeye::multiply_bf16(a[i], b[i], -133);
        }
        const auto pa = compact::prepare(a), pb = compact::prepare(b), saved_a = pa, saved_b = pb;
        for (unsigned i = 0; i < 16u; ++i) { assert(compact::original(pa, i) == a[i]); assert(compact::original(pb, i) == b[i]); }
        values[0] = {(random_word() & 0x7fffffu) | 0x800000u, int16_t(int(ae + be) - 254 + int(group % 67u) - 26), bool(group & 1u)};
        if (group % 3u == 0u || group % 31u == 0u || group % 37u == 0u || group % 43u == 0u) values[0] = {0u, -133, bool(group & 1u)};
        if (group % 29u == 0u) values[0] = {0xffffffu, 0, bool(group & 1u)};
        if (group % 41u == 0u) values[0] = {0u, -133, false};
        if (group % 4u == 1u) values[0] = previous;
        const auto expected = qrt_q1_moe_hawkeye::group_sum<26, -133>(values, 17u);
        unsigned path = 99u;
        const auto actual = compact::accumulate(values[0], pa, pb, &path);
        same(actual, expected, group, path);
        assert(path < 3u); ++paths[path]; previous = expected;
        if (compact::unit(pa) && compact::unit(pb)) {
            int64_t independent = 0;
            int maximum = values[0].exponent > -133 ? values[0].exponent : -133;
            for (unsigned i = 0; i < 16u; ++i) {
                independent += int64_t(coefficient(compact::word(pa, i))) * coefficient(compact::word(pb, i));
                if (values[i + 1u].exponent > maximum) maximum = values[i + 1u].exponent;
            }
            assert(compact::product(pa, pb) == independent);
            assert(compact::maximum(values[0], pa, pb) == maximum); ++products;
        }
        assert(!std::memcmp(&pa, &saved_a, sizeof(pa)) && !std::memcmp(&pb, &saved_b, sizeof(pb)));
    }
    assert(paths[0] && paths[1] && paths[2]);
    std::printf("{\"kind\":\"compact_integer_dot4_host\",\"row_bytes\":%zu,\"all_bf16_encodings\":65536,\"roundtrip_words\":%llu,\"supported_contexts\":%llu,\"independent_products\":%llu,\"ordered_carries\":500000,\"paths_fallback_exact_remainder\":[%u,%u,%u],\"raw_mismatches\":0,\"immutable_inputs\":true}\n",
        sizeof(compact::Row), (unsigned long long)roundtrips, (unsigned long long)supported_roundtrips, (unsigned long long)products, paths[0], paths[1], paths[2]);
}
