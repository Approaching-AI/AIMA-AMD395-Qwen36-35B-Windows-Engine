#include "../../native/providers/moe_accumulator/sm121_certified_f32_group.h"
#include "float_alignment_cases.h"
#include <cstdio>
#include <initializer_list>

namespace candidate = qrt_sm121_certified_f32_group;
namespace original = qrt_q1_moe_hawkeye;
namespace cases = qrt_float_alignment_cases;

int main() {
    unsigned admitted = 0u, rejected = 0u, normalized = 0u, extended = 0u;
    unsigned dominant = 0u, higher = 0u, different_grid = 0u;
    uint32_t random = 0x3958192u;
    for (unsigned row = 0u; row < 16384u; ++row) {
        original::Value reference{0u, -133, false};
        for (unsigned group = 0u; group < 128u; ++group) {
            uint16_t left[16], right[16];
            float products[16]; original::Value terms[17]; terms[0] = reference;
            int actual_maximum = reference.exponent > -133 ? reference.exponent : -133;
            for (unsigned i = 0u; i < 16u; ++i) {
                auto p = cases::input(row, group % 16u, i);
                if (row % 4u) {
                    random = cases::random(random); const uint32_t a = random;
                    random = cases::random(random); const uint32_t b = random;
                    const unsigned spread = row % 4u == 1u ? 4u : row % 4u == 2u ? 10u : 32u;
                    p.left = uint16_t((a & 0x807fu) | ((118u + a % spread) << 7u));
                    p.right = uint16_t((b & 0x807fu) | ((118u + b % spread) << 7u));
                    if (row % 16u == 7u) { p.left &= 0xff80u; p.right &= 0xff80u; }
                    if (row % 16u == 11u && i % 3u == 0u) { p.left &= 0x8000u; p.right &= 0x8000u; }
                }
                left[i] = p.left; right[i] = p.right;
                terms[i + 1u] = original::multiply_bf16(p.left, p.right, -133);
                if (terms[i + 1u].exponent > actual_maximum) actual_maximum = terms[i + 1u].exponent;
            }
            const float carry = original::value_to_float(reference);
            const bool exact_carry = !reference.significand ||
                (reference.exponent >= -126 && reference.exponent <= 127 &&
                 reference.significand >= 0x800000u && reference.significand <= 0xffffffu);
            const auto expected = original::group_sum<26, -133>(terms, 17u);
            int maximum = 999;
            const bool accepted = candidate::exponent(carry, candidate::bounds::prepare(left),
                candidate::bounds::prepare(right), &maximum);
            if (accepted && exact_carry) {
                ++admitted;
                dominant += maximum == reference.exponent;
                higher += maximum > reference.exponent;
                different_grid += maximum != actual_maximum;
                for (unsigned i = 0u; i < 16u; ++i)
                    products[i] = qrt_sm121_float_alignment::from_bits(uint32_t(left[i]) << 16u) *
                        qrt_sm121_float_alignment::from_bits(uint32_t(right[i]) << 16u);
                float actual = 123.0f;
                if (candidate::accumulate(carry, products, ((left[0] ^ right[0]) & 0x8000u) != 0u,
                        maximum, &actual)) {
                    ++normalized;
                    if (candidate::carry_math::bits(actual) != cases::output_bits(expected)) {
                        std::printf("DIFF row=%u group=%u maximum=%d actual_maximum=%d expected=%08x actual=%08x\n",
                            row, group, maximum, actual_maximum, cases::output_bits(expected), candidate::carry_math::bits(actual));
                        return 1;
                    }
                } else {
                    ++extended;
                    if (actual != 123.0f) return 2;
                }
            } else {
                ++rejected;
                if (!accepted && maximum != 999) return 3;
            }
            reference = expected;
        }
    }
    for (uint32_t raw : {1u,0x80000001u,0x7f800000u,0xff800000u,0x7fc00001u}) {
        int maximum = 999;
        if (candidate::exponent(qrt_sm121_float_alignment::from_bits(raw), 0x17f7fu, 0x17f7fu, &maximum) || maximum != 999) return 4;
    }
    std::printf("{\"kind\":\"certified_f32_group_host\",\"ordered_groups\":2097152,\"admitted\":%u,\"rejected\":%u,\"normalized\":%u,\"extended\":%u,\"carry_dominates\":%u,\"higher_than_carry\":%u,\"higher_than_original_grid\":%u,\"raw_mismatches\":0,\"inference_acceptance\":false}\n",
        admitted,rejected,normalized,extended,dominant,higher,different_grid);
    return !admitted || !rejected || !dominant || !higher || !different_grid;
}
