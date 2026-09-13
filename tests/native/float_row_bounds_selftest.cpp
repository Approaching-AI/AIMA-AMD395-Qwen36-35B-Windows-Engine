#include "../../native/providers/moe_accumulator/sm121_float_row_bounds.h"
#include "../../native/providers/moe_accumulator/sm121_canonical_normalize.h"
#include "float_alignment_cases.h"
#include <cstdio>

namespace bounds = qrt_sm121_float_row_bounds;
namespace original = qrt_q1_moe_hawkeye;
uint32_t random_state = 0x3957169u;
uint32_t next() { random_state = qrt_float_alignment_cases::random(random_state); return random_state; }

int main() {
    unsigned accepted = 0u, dominant = 0u, higher = 0u, fallback = 0u;
    // Independently enumerate every BF16 word, including signed zero and all
    // exceptional exponents. Metadata must reject every ineligible nonzero.
    for (unsigned word = 0u; word < 65536u; ++word) {
        uint16_t row[16]{}; row[word % 16u] = uint16_t(word);
        const auto actual = bounds::prepare(row);
        const unsigned e = (word >> 7u) & 255u;
        const bool zero = !(word & 0x7fffu), valid = zero || (e >= 64u && e <= 190u);
        unsigned trailing = 0u, sig = 128u | (word & 127u);
        while (!(sig & 1u)) { ++trailing; sig >>= 1u; }
        if (bool(actual & bounds::valid_bit) != valid || (actual & 255u) != (zero ? 0u : e) ||
            (valid && ((actual >> 8u) & 255u) != (zero ? 255u : e + trailing))) return 2;
    }
    for (unsigned row = 0u; row < 16384u; ++row) {
        original::Value carry{0u, -133, false}, expected = carry;
        if (row % 16u == 15u) {
            carry = {0x800000u | (next() & 0x7fffffu), int16_t(int(row % 279u) - 127), bool(row & 32u)};
            expected = carry;
        }
        for (unsigned group = 0u; group < 128u; ++group) {
            uint16_t left[16], right[16]; original::Value values[17]; values[0] = expected;
            for (unsigned i = 0u; i < 16u; ++i) {
                if (row % 4u == 0u) {
                    const auto pair = qrt_float_alignment_cases::input(row / 4u, group, i);
                    left[i] = pair.left; right[i] = pair.right;
                } else {
                    const uint32_t a = next(), b = next();
                    const unsigned spread = row % 4u == 1u ? 4u : row % 4u == 2u ? 10u : 32u;
                    left[i] = uint16_t((a & 0x807fu) | ((118u + a % spread) << 7u));
                    right[i] = uint16_t((b & 0x807fu) | ((118u + b % spread) << 7u));
                    if (row % 16u == 7u) { left[i] &= 0xff80u; right[i] &= 0xff80u; }
                    if (row % 16u == 11u && i % 3u == 0u) { left[i] &= 0x8000u; right[i] &= 0x8000u; }
                }
                values[i + 1u] = original::multiply_bf16(left[i], right[i], -133);
            }
            const auto before = carry;
            qrt_sm121_group16::AlignedSum sum;
            if (bounds::sum(carry, left, right, bounds::prepare(left), bounds::prepare(right), &sum)) {
                ++accepted; dominant += sum.max_exponent == before.exponent;
                higher += sum.max_exponent > before.exponent;
                carry = qrt_sm121_canonical::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
            } else {
                ++fallback; uint32_t products[16];
                for (unsigned i = 0u; i < 16u; ++i) products[i] = qrt_sm121_group16::pack_product(values[i + 1u]);
                sum = qrt_sm121_group16::sum_packed(carry, products);
                carry = qrt_sm121_canonical::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent);
            }
            expected = original::group_sum<26, -133>(values, 17u);
            if (carry.significand != expected.significand || carry.exponent != expected.exponent || carry.negative != expected.negative) {
                std::printf("DIFF row=%u group=%u expected=%u,%d,%u actual=%u,%d,%u\n",row,group,
                    expected.significand,int(expected.exponent),unsigned(expected.negative),carry.significand,int(carry.exponent),unsigned(carry.negative));
                return 3;
            }
        }
    }
    std::printf("{\"kind\":\"float_row_bounds_cpu\",\"bf16_encodings\":65536,\"ordered_groups\":2097152,\"canonical_value_mismatches\":0,\"certified_groups\":%u,\"carry_dominates\":%u,\"exact_higher_grid\":%u,\"fallback_groups\":%u,\"inference_acceptance\":false}\n",accepted,dominant,higher,fallback);
    return !dominant || !higher || !fallback;
}
