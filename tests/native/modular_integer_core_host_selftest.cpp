#include "../../native/providers/moe_accumulator/sm121_modular_integer_core.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
namespace core = qrt_sm121_modular_core;
uint32_t state = 0x3958192u;
uint32_t random_word() { state ^= state << 13u; state ^= state >> 17u; state ^= state << 5u; return state; }
double half(uint16_t bits) {
    const unsigned exponent = (bits >> 10u) & 31u;
    return exponent ? ((bits & 32768u) ? -1.0 : 1.0) * std::ldexp(double(1024u + (bits & 1023u)), int(exponent) - 25) : 0.0;
}
int independent_core(uint16_t bits, int unit) {
    if (unit < 0 || !(bits & 0x7fffu)) return 0;
    const int magnitude = int(std::ldexp(double(128u + (bits & 127u)), int((bits >> 7u) & 255u) - unit));
    return bits & 32768u ? -magnitude : magnitude;
}
int main() {
    size_t encodings = 0u, recovery_checks = 0u, accepted = 0u, fallback = 0u, remainder_checks = 0u;
    for (unsigned bits = 0u; bits < 65536u; ++bits) {
        const int exponent = int((bits >> 7u) & 255u);
        if (!exponent || exponent == 255) continue;
        for (int shift = -16; shift <= 7; ++shift) {
            const int unit = exponent - shift;
            if (unit < 1 || unit > 247) continue;
            const int expected = independent_core(uint16_t(bits), unit);
            const int original = qrt_sm121_integer_core::signed_core(uint16_t(bits), unit);
            assert(original == expected && std::abs(expected) <= 32640);
            assert(half(core::half_bits(unsigned(std::abs(expected)), expected < 0)) == expected);
            ++encodings;
        }
    }
    // Every possible low16 residue, positive/negative quotient and fractional
    // cancellation around the exact recovery boundary. The condition is on
    // the actual FP32 value, not the pre-rounding requested perturbation.
    for (unsigned low = 0u; low < 65536u; ++low) for (int64_t high : {int64_t(-65536), int64_t(0), int64_t(65536), int64_t(-16777216), int64_t(16777216), core::maximum_dot - 131072, -core::maximum_dot}) {
        const int64_t expected = high + low;
        if (expected < -core::maximum_dot || expected > core::maximum_dot) continue;
        for (double delta : {-32768.0, -32767.5, -8192.0, -0.5, 0.0, 0.5, 8192.0, 32767.5, 32768.0}) {
            const float approximate = float(double(expected) + delta);
            if (std::abs(double(approximate) - double(expected)) >= 32768.0) continue;
            int64_t result = 0x3958192;
            assert(core::recover(approximate, uint32_t(expected), &result) && result == expected);
            ++recovery_checks;
        }
    }
    for (float invalid : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), 17180000256.0f, -17180000256.0f, 32768.0f, -32768.0f}) {
        int64_t result = 0x3958192;
        assert(!core::recover(invalid, 0u, &result) && result == 0x3958192);
    }
    core::Value previous{0u, -133, false};
    for (unsigned group = 0u; group < 262144u; ++group) {
        core::Row a{}, b{}; core::Value terms[17];
        const unsigned ae = 1u + random_word() % 220u, be = 1u + random_word() % 220u, spread = 1u + group % 34u;
        for (unsigned i = 0u; i < 16u; ++i) {
            a.original[i] = uint16_t((random_word() & 0x807fu) | ((ae + random_word() % spread) << 7u));
            b.original[i] = uint16_t((random_word() & 0x807fu) | ((be + random_word() % spread) << 7u));
            if (group % 11u == 0u && i % 3u == 0u) a.original[i] = 0u;
            if (group % 13u == 0u && i % 3u == 1u) b.original[i] = 0x8000u;
            if (group % 17u == 0u) {
                a.original[i] = uint16_t((random_word() & 0x807fu) | ((ae + i % 9u) << 7u));
                b.original[i] = uint16_t((random_word() & 0x807fu) | ((be + 8u - i % 9u) << 7u));
            }
            if (group % 19u == 0u && i == 2u) a.original[i] = 1u;
            if (group % 23u == 0u && i == 3u) b.original[i] = 0x7fc1u;
            if (group % 29u == 0u) { a.original[i] = uint16_t(0x3fffu | ((group & 1u) << 15u)); b.original[i] = 0x3fffu; }
            if (group % 31u == 0u) { a.original[i] = uint16_t((0x3f81u + (i / 2u % 4u) * 128u) | ((i & 1u) << 15u)); b.original[i] = 0x3f85u; }
            if (group % 37u == 0u) { a.original[i] = i % 2u ? 0x8000u : 0x7f7fu; b.original[i] = i % 2u ? 0x7f7fu : 0u; }
            if (group % 41u == 0u) { a.original[i] = uint16_t(0x0081u + (i & 1u)); b.original[i] = 0x0081u; }
            if (group % 43u == 0u) { a.original[i] = i == 2u ? 0x7f7fu : uint16_t(0x0081u + (i & 1u)); b.original[i] = i == 2u ? 0u : 0x0081u; }
            terms[i + 1u] = qrt_q1_moe_hawkeye::multiply_bf16(a.original[i], b.original[i], -133);
        }
        const auto raw_a = a, raw_b = b; core::prepare(a); core::prepare(b);
        assert(!std::memcmp(raw_a.original, a.original, 32u) && !std::memcmp(raw_b.original, b.original, 32u));
        const auto before_a = a, before_b = b;
        int64_t mathematical = 0;
        for (unsigned i = 0u; i < 16u; ++i) {
            const int x = independent_core(a.original[i], a.unit), y = independent_core(b.original[i], b.unit);
            assert(half(a.half[i]) == x && half(b.half[i]) == y);
            assert(((a.magnitudes[i / 2u] >> ((i & 1u) * 16u)) & 65535u) == unsigned(std::abs(x)));
            assert(((b.magnitudes[i / 2u] >> ((i & 1u) * 16u)) & 65535u) == unsigned(std::abs(y)));
            mathematical += int64_t(x) * y;
        }
        for (unsigned shift = 0u; shift <= 16u; ++shift) {
            const uint32_t mask = (1u << shift) - 1u; int64_t expected_discarded = 0;
            for (unsigned i = 0u; i < 16u; ++i) {
                const int64_t product = int64_t(independent_core(a.original[i], a.unit)) * independent_core(b.original[i], b.unit);
                expected_discarded += product % (int64_t(1) << shift);
            }
            const auto low = core::remainders(a, b, mask);
            assert((uint32_t(low.residue) & 65535u) == (uint32_t(mathematical) & 65535u));
            assert(low.discarded == expected_discarded); ++remainder_checks;
        }
        terms[0] = {(random_word() & 0x7fffffu) | 0x800000u, int16_t(a.maximum + b.maximum - 254 + int(group % 67u) - 26), bool(group & 1u)};
        if (group % 3u == 0u || group % 31u == 0u || group % 37u == 0u || group % 41u == 0u || group % 43u == 0u) terms[0] = {0u, -133, false};
        if (group % 4u == 1u) terms[0] = previous;
        previous = qrt_q1_moe_hawkeye::group_sum<26, -133>(terms, 17u);
        core::AlignedSum actual{{0xdeadbeefu, true}, 123};
        if (core::sum(terms[0], a, b, float(mathematical), &actual)) ++accepted;
        else {
            assert(actual.value.magnitude == 0xdeadbeefu && actual.value.negative && actual.max_exponent == 123);
            actual = core::fallback(terms[0], a, b); ++fallback;
        }
        const auto normalized = qrt_sm121_canonical::normalize(actual.value.magnitude, actual.value.negative, actual.max_exponent);
        if (normalized.significand != previous.significand || normalized.exponent != previous.exponent || normalized.negative != previous.negative) {
            std::fprintf(stderr, "group=%u actual=%u,%d,%u expected=%u,%d,%u\n", group, normalized.significand, normalized.exponent, normalized.negative, previous.significand, previous.exponent, previous.negative); return 1;
        }
        assert(!std::memcmp(&a, &before_a, sizeof(a)) && !std::memcmp(&b, &before_b, sizeof(b)));
    }
    assert(accepted > 50000u && fallback > 10000u);
    std::printf("{\"kind\":\"modular_integer_core_host\",\"exact_half_encodings\":%zu,\"conditional_recovery_checks\":%zu,\"shared_remainder_checks\":%zu,\"canonical_groups\":262144,\"accepted_groups\":%zu,\"fallback_groups\":%zu,\"raw_mismatches\":0,\"immutable_inputs\":true,\"native_wmma_checked\":false,\"hardware_error_bound_proven\":false}\n", encodings, recovery_checks, remainder_checks, accepted, fallback);
}
