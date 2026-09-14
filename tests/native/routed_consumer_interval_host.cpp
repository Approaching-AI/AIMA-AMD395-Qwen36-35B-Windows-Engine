#include "triton_moe/routed_consumer_interval.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>

int main() {
    namespace c = qrt_routed_consumer;
    std::array<uint16_t, 65536> table{};
    // Synthetic table with independent, nonmonotonic plateaus. The real CUDA
    // SiLU table is exercised by the native full-model audit, not inferred here.
    for (unsigned u = 0; u < table.size(); ++u)
        table[u] = c::rounded(float(int((c::ordered(uint16_t(u)) / 4u) % 17u) - 8) / 8.0f);
    uint64_t intervals = 0, admitted_gate = 0, admitted_up = 0, endpoints = 0;
    for (unsigned u = 0; u < 65536; ++u) {
        assert(c::unordered(c::ordered(uint16_t(u))) == u);
        for (unsigned tail : {0u, 0x7dffu, 0x8000u, 0xffffu}) {
            const float raw = c::value((u << 16u) | tail);
            if (std::isfinite(raw)) {
                assert(c::bits(c::outward(raw, true)) == c::bits(std::nextafter(raw, INFINITY)));
                assert(c::bits(c::outward(raw, false)) == c::bits(std::nextafter(raw, -INFINITY)));
            }
            for (float ratio : {0.0f, 0x1p-20f, 0x1p-9f, 0.125f}) {
                const float error = std::fabs(raw) * ratio;
                const auto r = c::range(raw, error);
                ++intervals;
                if (!r.valid) continue;
                assert(c::contains(r, raw));
                // Check against a separately computed double interval and the
                // adjacent encoded values, without using the range arithmetic.
                const double lo = double(raw) - double(error), hi = double(raw) + double(error);
                const auto a = uint16_t(c::bits(raw) >> 16u), b = uint16_t(a + 1u);
                assert(c::ordered(a) >= r.low && c::ordered(a) <= r.high);
                assert(c::ordered(b) >= r.low && c::ordered(b) <= r.high);
                assert(c::ordered(c::rounded(float(lo))) >= r.low);
                assert(c::ordered(c::rounded(float(hi))) <= r.high);
                const uint16_t gate = uint16_t((u * 313u) & 65535u);
                bool gate_same = true, up_same = true;
                const uint16_t first = c::unordered(r.low);
                for (unsigned key = r.low; key <= r.high; ++key) {
                    const uint16_t endpoint = c::unordered(uint16_t(key));
                    gate_same &= table[endpoint] == table[first];
                    // BF16 product has at most sixteen significand bits. Use
                    // independent double arithmetic and BF16 ties-to-even.
                    auto reference = [&](uint16_t up) {
                        const double product = double(c::widen(table[gate])) * double(c::widen(up));
                        const float narrowed = float(product);
                        uint32_t word; __builtin_memcpy(&word, &narrowed, sizeof(word));
                        const uint32_t high = word >> 16u, tail_bits = word & 65535u;
                        return uint16_t(high + (tail_bits > 32768u || (tail_bits == 32768u && (high & 1u))));
                    };
                    up_same &= reference(endpoint) == reference(first);
                    ++endpoints;
                }
                assert(c::gate_constant(r, table.data()) == gate_same);
                const bool certificate = c::up_constant(r, gate, table.data());
                assert(!certificate || up_same);
                admitted_gate += gate_same; admitted_up += certificate;
            }
        }
    }
    for (float raw : {INFINITY, -INFINITY, NAN, 0.0f, -0.0f, c::value(1u)})
        assert(!c::range(raw, 1.0f).valid);
    for (float error : {INFINITY, NAN, -1.0f}) assert(!c::range(1.0f, error).valid);
    assert(!c::gate_constant({0x8000u, 0x8001u, true}, nullptr));
    assert(!c::up_constant({0x8000u, 0x8001u, true}, 0u, nullptr));
    // A plateau at the endpoints is insufficient when an interior table value
    // differs: enumeration must catch the nonmonotonic interior.
    table[0x3f80u] = table[0x3f82u] = 0x3f80u; table[0x3f81u] = 0x3f00u;
    assert(!c::gate_constant({c::ordered(0x3f80u), c::ordered(0x3f82u), true}, table.data()));
    assert(admitted_gate && admitted_up);
    std::printf("consumer_interval_host intervals=%llu enumerated_endpoints=%llu gate_certificates=%llu up_certificates=%llu pass=1\n",
        (unsigned long long)intervals, (unsigned long long)endpoints,
        (unsigned long long)admitted_gate, (unsigned long long)admitted_up);
}
