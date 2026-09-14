#include "sm121_projection_interval.h"
#include "sm121_group16_modulo.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
namespace interval = qrt_sm121_projection_interval;
namespace bound = qrt_sm121_pv_bound;
namespace canonical = qrt_q1_moe_hawkeye;
unsigned seed = 0x3958192u;
unsigned random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
int main(int argc, char** argv) {
    if (argc == 2 && !std::strcmp(argv[1], "--directed")) {
        uint32_t a, b;
        while (std::cin >> a >> b)
            std::cout << bound::bits(interval::directed_add(bound::value(a), bound::value(b), false))
                << ' ' << bound::bits(interval::directed_add(bound::value(a), bound::value(b), true)) << '\n';
        return 0;
    }
    unsigned admitted = 0u;
    for (unsigned dot = 0u; dot < 2048u; ++dot) {
        const float initial = dot % 7u ? 0.0f : bound::value((random_word() & 0x807fffffu) | (145u << 23u));
        auto carry = canonical::value_from_float(initial, -133);
        interval::Interval bounds{initial, initial};
        for (unsigned group = 0u; group < 128u; ++group) {
            canonical::Value values[17]; values[0] = carry;
            interval::Row left, right;
            double product = 0.0, absolute = 0.0;
            for (unsigned k = 0u; k < 16u; ++k) {
                const unsigned base = dot % 5u ? 116u : 80u;
                const unsigned spread = dot % 5u ? 1u + dot % 25u : 95u;
                uint16_t a = uint16_t(((base + random_word() % spread) << 7u) | (random_word() & 0x807fu));
                uint16_t b = uint16_t(((base + random_word() % spread) << 7u) | (random_word() & 0x807fu));
                if (dot % 11u == 0u) a = 0u;
                if (dot % 13u == 0u && k % 3u) b &= 0x8000u;
                if (dot % 17u == 0u) { a = uint16_t(0x3fc1u | ((k & 1u) << 15u)); b = 0x3f81u; }
                interval::include(left, a); interval::include(right, b);
                values[k + 1u] = canonical::multiply_bf16(a, b, -133);
                const double term = double(bound::value(uint32_t(a) << 16u)) * double(bound::value(uint32_t(b) << 16u));
                product += term; absolute += std::abs(term);
            }
            carry = canonical::group_sum<26, -133>(values, 17u);
            const float expected = canonical::value_to_float(qrt_sm121_group16::finish_accumulator(carry));
            // This host case supplies a rounded mathematical product. Native
            // WMMA errors are checked separately by the original-tensor probe.
            bounds = interval::group(bounds, float(product), float(absolute), left, right);
            if (!interval::valid(bounds) || expected < bounds.lower || expected > bounds.upper) {
                std::fprintf(stderr, "interval miss dot=%u group=%u bounds=%.9g,%.9g expected=%.9g\n",
                    dot, group, bounds.lower, bounds.upper, expected);
                return 1;
            }
            if (group == 127u && interval::same_bf16(bounds)) {
                ++admitted; assert(bound::bf16(bounds.lower) == bound::bf16(expected));
            }
        }
    }
    assert(admitted > 100u);
    for (uint16_t value : {uint16_t(0x7f80u), uint16_t(0x7fc1u), uint16_t(79u << 7u), uint16_t(175u << 7u), uint16_t(1u)}) {
        interval::Row bad, good; interval::include(bad, value); interval::include(good, 0x3f80u);
        assert(!bad.valid && !interval::same_bf16(interval::group({0, 0}, 0, 1, bad, good)));
    }
    interval::Row zero, good; interval::include(good, 0x3f80u);
    assert(!interval::same_bf16(interval::group(interval::invalid(), 0, 0, zero, good)));
    assert(!interval::same_bf16(interval::group({0, 0}, bound::infinity(), 0, good, good)));
    assert(!interval::same_bf16(interval::group({0, 0}, 0, -1, good, good)));
    const auto retained = interval::group({-3, -2}, 0, 0, zero, good);
    assert(retained.lower == -3 && retained.upper == -2);
    std::printf("{\"ordered_groups\":262144,\"dots\":2048,\"admitted\":%u,\"undercoverage\":0,\"false_admissions\":0,\"exceptional_fallback_pass\":true}\n", admitted);
}
