"""Check 18-bit sparse reconstruction against independent original K16 arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WideCoreTests(unittest.TestCase):
    def test_wide_products_sparse_remainders_and_ordered_carries(self):
        code = r'''
#include "sm121_wide_integer_core.h"
#include "sm121_canonical_normalize.h"
#include <cstdio>
#include <cstring>
namespace wide = qrt_sm121_wide_core;
namespace original = qrt_q1_moe_hawkeye;
uint32_t seed = 0x3958192u;
uint32_t next() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
int integer_half(uint16_t bits) { _Float16 value; std::memcpy(&value, &bits, 2u); return int(float(value)); }
uint32_t bits(original::Value value) {
    float result = original::value_to_float(qrt_sm121_group16::finish_accumulator(value));
    uint32_t out; std::memcpy(&out, &result, 4u); return out;
}
int main() {
    unsigned old_replayed = 0u, new_replayed = 0u, old_fallbacks = 0u, new_fallbacks = 0u;
    original::Value carry{0u, -133, false};
    for (unsigned test = 0u; test < 262144u; ++test) {
        wide::Row a{}, b{}; qrt_sm121_integer_core::Row old_a{}, old_b{};
        const unsigned spreads[] = {1u, 4u, 8u, 10u, 13u, 25u, 81u, 200u};
        const unsigned spread = spreads[test % 8u];
        for (unsigned i = 0u; i < 16u; ++i) {
            const unsigned ae = 1u + next() % spread, be = 1u + next() % spread;
            a.original[i] = uint16_t((next() & 0x807fu) | ((spread < 81u ? ae + 100u : ae) << 7u));
            b.original[i] = uint16_t((next() & 0x807fu) | ((spread < 81u ? be + 100u : be) << 7u));
            if (test % 17u == 0u) a.original[i] = uint16_t(test * 16u + i);
            if (test % 19u == 0u) b.original[i] = uint16_t(test * 997u + i);
            if (test % 23u == 0u && i % 3u == 0u) a.original[i] = b.original[i] = uint16_t(i & 1u ? 0x8000u : 0u);
            if (test % 29u == 0u) a.original[i] = b.original[i] = uint16_t(127u << 7u | (i & 1u ? 0x8000u : 127u));
            old_a.original[i] = a.original[i]; old_b.original[i] = b.original[i];
        }
        wide::prepare(a); wide::prepare(b); qrt_sm121_integer_core::prepare(old_a); qrt_sm121_integer_core::prepare(old_b);
        int high = 0, low = 0, combined = 0; int64_t mathematical = 0, old_mathematical = 0;
        uint32_t sum_a = 0u, sum_b = 0u;
        original::Value reference_terms[17]; reference_terms[0] = carry;
        for (unsigned i = 0u; i < 16u; ++i) {
            const int ah = integer_half(a.high[i]), al = integer_half(a.low[i]);
            const int bh = integer_half(b.high[i]), bl = integer_half(b.low[i]);
            if (ah < 0 || ah > 511 || al < 0 || al > 511 || bh < 0 || bh > 511 || bl < 0 || bl > 511) return 1;
            const int ac = ah * 512 + al - int(wide::center), bc = bh * 512 + bl - int(wide::center);
            sum_a += unsigned(ah * 512 + al); sum_b += unsigned(bh * 512 + bl);
            mathematical += int64_t(ac) * bc;
            high += ah * bh; low += al * bl; combined += (ah + al) * (bh + bl);
            old_mathematical += int64_t(qrt_sm121_integer_core::signed_core(old_a.original[i], old_a.unit)) *
                qrt_sm121_integer_core::signed_core(old_b.original[i], old_b.unit);
            reference_terms[i + 1u] = original::multiply_bf16(a.original[i], b.original[i], -133);
        }
        if (sum_a != wide::unsigned_row_sum(a) || sum_b != wide::unsigned_row_sum(b) ||
            combined > 16711744 || mathematical != wide::reconstruct(high, low, combined, sum_a, sum_b)) return 2;
        const auto reference = original::group_sum<26, -133>(reference_terms, 17u);
        qrt_sm121_group16::AlignedSum sum{}, old_sum{}; unsigned replayed = 0u, old_replay = 0u;
        const bool accepted = wide::sum_integer_product(carry, a, b, mathematical, &sum, &replayed);
        const bool old_accepted = qrt_sm121_integer_core::sum_integer_product(carry, old_a, old_b, old_mathematical, &old_sum, &old_replay);
        new_fallbacks += !accepted; old_fallbacks += !old_accepted;
        new_replayed += accepted ? replayed : 16u; old_replayed += old_accepted ? old_replay : 16u;
        auto actual = accepted ? qrt_sm121_canonical::normalize(sum.value.magnitude, sum.value.negative, sum.max_exponent) : reference;
        auto old_actual = old_accepted ? qrt_sm121_canonical::normalize(old_sum.value.magnitude, old_sum.value.negative, old_sum.max_exponent) : reference;
        if (bits(actual) != bits(reference) || bits(old_actual) != bits(reference)) {
            std::printf("DIFF case=%u carry=%08x expected=%08x old=%08x wide=%08x maximum=%d\n", test, bits(carry), bits(reference), bits(old_actual), bits(actual), sum.max_exponent);
            return 3;
        }
        carry = test % 16u == 15u ? original::Value{0u, -133, false} : actual;
    }
    std::printf("{\"cases\":262144,\"row_bytes\":132,\"raw_mismatches\":0,\"old_replayed_pairs\":%u,\"wide_replayed_pairs\":%u,\"old_row_fallbacks\":%u,\"wide_row_fallbacks\":%u,\"inference_acceptance\":false}\n", old_replayed, new_replayed, old_fallbacks, new_fallbacks);
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-wide-core-") as directory:
            exe = str(Path(directory) / "wide")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            "-I", str(ROOT / "native/providers/moe_accumulator"),
                            "-x", "c++", "-", "-o", exe], input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
