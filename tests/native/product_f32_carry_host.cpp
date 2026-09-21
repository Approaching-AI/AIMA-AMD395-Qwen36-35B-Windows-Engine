#ifndef QRT_PRODUCT_CARRY_TEST_HEADER
#define QRT_PRODUCT_CARRY_TEST_HEADER "native/providers/moe_accumulator/sm121_product_f32_carry.h"
#endif
#include QRT_PRODUCT_CARRY_TEST_HEADER
#include <array>
#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace candidate = qrt_sm121_product_f32_carry;
namespace original = qrt_q1_moe_hawkeye;
using Words = std::array<uint16_t, 256>;
static unsigned long long accepted = 0u, declined = 0u, boundaries = 0u, ftz_groups = 0u;

static void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static uint32_t bits(float x) { return qrt_sm121_f32_carry::bits(x); }
static uint32_t random_word(uint32_t x) {
    x ^= x << 13u; x ^= x >> 17u; x ^= x << 5u; return x;
}
static uint16_t word(unsigned exponent, unsigned fraction, unsigned sign = 0u) {
    return uint16_t((exponent << 7u) | (fraction & 127u) | ((sign & 1u) << 15u));
}
static float reference(const Words& a, const Words& b, unsigned width) {
    // The original wide signed sum is independent of the candidate's scalar
    // products, modulo decoding, exponent guards and FP32 carry encoding.
    original::Value carry{0u, -133, false};
    for (unsigned base = 0u; base < width; base += 16u) {
        original::Value group[17]; group[0] = carry;
        for (unsigned i = 0u; i < 16u; ++i)
            group[i + 1u] = original::multiply_bf16(a[base + i], b[base + i], -133);
        carry = original::group_sum<26, -133>(group, 17u);
    }
    return original::value_to_float(carry);
}
template<unsigned Width>
static bool check(const Words& a, const Words& b, int expected_admission = -1) {
    float actual = -123.25f;
    const bool ok = candidate::dot<Width>(a.data(), b.data(), &actual);
    if (expected_admission >= 0) require(ok == (expected_admission != 0), "unexpected admission");
    if (!ok) { ++declined; require(actual == -123.25f, "rejected dot changed output"); return false; }
    ++accepted;
    require(bits(actual) == bits(reference(a, b, Width)), "accepted dot differs from wide integer sum");
    float carry = 0.0f, flushed = 0.0f;
    for (unsigned base = 0u; base < Width; base += 16u) {
        candidate::Group group;
        for (unsigned i = 0u; i < 16u; ++i) group.set(i, a[base + i], b[base + i]);
        float next = -123.25f;
        require(candidate::accumulate(carry, group, &next), "accepted prefix declined");
        carry = next;
        require(bits(carry) == bits(reference(a, b, base + 16u)), "ordered carry boundary differs");
        ++boundaries;
        // Simulate a backend that flushes FP32 subnormal multiplication
        // results. Subnormal BF16 inputs were omitted by the checked group.
        for (float& product : group.aligned.products) {
            const uint32_t raw = bits(product);
            if (!(raw & 0x7f800000u)) product = qrt_sm121_float_alignment::from_bits(raw & 0x80000000u);
        }
        require(candidate::accumulate(flushed, group, &next), "FTZ prefix declined");
        flushed = next;
        require(bits(flushed) == bits(carry), "FTZ changed an accepted carry");
        ++ftz_groups;
    }
    return true;
}

static void cancellation(Words& a, Words& b, unsigned exponent) {
    a.fill(0u); b.fill(0u);
    a[0] = word(127u, 0u); b[0] = word(exponent, 0u);
    a[1] = word(127u, 0u, 1u); b[1] = b[0];
}
static int probe(bool subnormal) {
    Words a{}, b{};
    if (subnormal) {
        cancellation(a, b, 127u);
        a[2] = 127u; b[2] = word(228u, 127u);
    } else { a[0] = word(64u, 0u); b[0] = word(64u, 0u); }
    float value = -123.25f;
    const bool ok = candidate::dot<16u>(a.data(), b.data(), &value);
    if (ok && bits(value) != bits(reference(a, b, 16u))) {
        std::printf("negative_control_detected=%s expected=%08x actual=%08x\n",
            subnormal ? "subnormal" : "floor", bits(reference(a, b, 16u)), bits(value));
        return 3;
    }
    require(!ok && value == -123.25f, "boundary probe must decline without publishing");
    return 0;
}

int main(int argc, char** argv) {
    require(std::fesetround(FE_TONEAREST) == 0, "rounding mode");
    if (argc == 2) return probe(std::strcmp(argv[1], "--probe-subnormal") == 0);
    require(argc == 1, "unexpected arguments");
    require(probe(true) == 0 && probe(false) == 0, "boundary probes");
    Words a{}, b{};
    // Every normal exponent pair, both product signs, and low/high mantissas.
    // No anchor masks an out-of-window product maximum in this sweep.
    for (unsigned ae = 1u; ae < 255u; ++ae) for (unsigned be = 1u; be < 255u; ++be)
        for (unsigned mode = 0u; mode < 4u; ++mode) {
            a.fill(0u); b.fill(0u);
            for (unsigned i = 0u; i < 16u; ++i) {
                a[i] = word(ae, mode & 1u ? 127u : 0u, mode >> 1u);
                b[i] = word(be, mode & 1u ? 127u : 0u);
            }
            const int exponent = int(ae + be) - 254;
            check<16u>(a, b, exponent >= -64 && exponent <= 64);
        }
    // Every BF16 encoding appears as an operand. Inverse exponents exercise
    // accepted pairs whose individual operands are outside the old domain.
    for (unsigned raw = 0u; raw < 65536u; ++raw) {
        a.fill(0u); b.fill(0u);
        const unsigned exponent = (raw >> 7u) & 255u;
        a[0] = uint16_t(raw);
        b[0] = word(exponent > 0u && exponent < 255u ? 255u - exponent : 100u,
                    random_word(raw + 17u), raw >> 15u);
        a[15] = b[15] = word(127u, 0u);
        check<16u>(a, b, exponent != 255u);
    }
    // A subnormal term close to the guard can become the entire endpoint
    // when two dominant normal products cancel. Test both positions/signs.
    for (unsigned maximum : {63u, 127u, 191u}) for (unsigned fraction = 1u; fraction < 128u; ++fraction)
        for (unsigned gap = 23u; gap <= 30u; ++gap) for (unsigned mode = 0u; mode < 4u; ++mode) {
            const int product_exponent = int(maximum) - 127;
            const int be = product_exponent - int(gap) + 253;
            if (be < 1 || be > 254) continue;
            cancellation(a, b, maximum);
            a[2] = word(0u, fraction, mode >> 1u); b[2] = word(unsigned(be), 127u);
            if (mode & 1u) { const auto t = a[2]; a[2] = b[2]; b[2] = t; }
            check<16u>(a, b, gap >= 27u);
        }
    // Ordered K64/K128/K256 dots with mixed product magnitudes and signs,
    // including maximal modulo sums, signed zeros and cancellation.
    for (unsigned row = 0u; row < 4096u; ++row) {
        for (unsigned i = 0u; i < 256u; ++i) {
            const uint32_t x = random_word(row * 7919u + i * 17u + 1u);
            const uint32_t y = random_word(x ^ 0x3958192u);
            const unsigned mode = row % 8u;
            unsigned ae = 1u + x % 254u;
            int be = 254 - int(ae) + int((row / 8u) % 129u) - 64;
            be = be < 1 ? 1 : be > 254 ? 254 : be;
            a[i] = word(ae, x, x >> 8u); b[i] = word(unsigned(be), y, y >> 8u);
            if (mode < 3u) {
                a[i] = word(159u, 127u, mode == 1u || (mode == 2u && (i & 1u)));
                b[i] = word(159u, 127u);
            }
            if (mode == 3u && i % 16u > 1u) a[i] = uint16_t(x & 0x8000u);
            if (mode == 4u) { a[i] = word(1u, x); b[i] = word(i % 16u ? 1u : 189u, y); }
            if (mode == 5u && (i / 16u) % 2u) { a[i] = uint16_t(x & 0x8000u); b[i] = uint16_t(y & 0x8000u); }
        }
        check<64u>(a, b, 1); check<128u>(a, b, 1); check<256u>(a, b, 1);
    }
    // Late rejection must leave the caller's output intact even after
    // fifteen accepted groups. Cover nonfinite, subnormal-only, and range.
    const uint16_t rejected[][2] = {
        {0x7f80u, 0u}, {0xff80u, 0x3f80u}, {0x7fc1u, 0u}, {0x7fffu, 0x3f80u},
        {1u, 1u}, {0x7f7fu, 0x7f7fu}, {word(94u,0u),word(95u,0u)},
        {word(160u,0u),word(159u,0u)}};
    for (const auto& pair : rejected) {
        a.fill(word(127u, 0u)); b.fill(word(127u, 0u));
        for (unsigned i = 240u; i < 256u; ++i) { a[i] = pair[0]; b[i] = pair[1]; }
        check<256u>(a, b, 0);
    }
    // Public accumulation rejects a malformed incoming carry before touching
    // output, independently of the valid group metadata.
    candidate::Group group;
    for (unsigned i = 0u; i < 16u; ++i) group.set(i, word(127u,0u), word(127u,0u));
    for (uint32_t raw : {1u,0x7f800000u,0xff800000u,0x7fc00001u,0x12000000u,0x65000000u}) {
        float output = -123.25f;
        require(!candidate::accumulate(qrt_sm121_float_alignment::from_bits(raw), group, &output), "invalid carry admitted");
        require(output == -123.25f, "rejected carry changed output");
    }
    require(accepted > 100000u && declined > 100000u && boundaries == ftz_groups, "coverage counters");
    std::printf("{\"component\":\"checked_k16_product_f32_carry\",\"accepted_dots\":%llu,\"declined_dots\":%llu,\"ordered_integer_boundaries\":%llu,\"simulated_ftz_groups\":%llu,\"raw_mismatches\":0,\"native_executed\":false}\n",
        accepted, declined, boundaries, ftz_groups);
    return 0;
}
