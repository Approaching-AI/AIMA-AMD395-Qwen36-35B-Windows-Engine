"""Bound positive BF16 products under deliberately downward FP32 accumulation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PositiveSumBoundTests(unittest.TestCase):
    def test_downward_accumulation_and_ftz_are_bounded(self):
        source = r'''
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include "bf16_positive_sum_bound.h"
namespace bound = qrt_bf16_positive_sum_bound;
uint32_t seed = 0x3958192u;
uint32_t random_word() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; }
double bf16(uint16_t raw) { return double(bound::value(uint32_t(raw) << 16u)); }
float add_down_ftz(float accumulated, double product) {
    const double exact = double(accumulated) + product;
    float rounded = float(exact);
    if (double(rounded) > exact) rounded = bound::value(bound::bits(rounded) - 1u);
    if (rounded < 0x1p-126f) return 0.0f;
    return rounded;
}
int main() {
    unsigned cases = 0u;
    for (unsigned k : {16u, 512u, 2048u, 4096u}) {
        for (unsigned mode = 0u; mode < 8u; ++mode) for (unsigned trial = 0u; trial < 64u; ++trial) {
            float approximate = 0.0f;
            double reference = 0.0;
            for (unsigned column = 0u; column < k; ++column) {
                unsigned a_exp = 64u + random_word() % 128u, b_exp = 64u + random_word() % 128u;
                if (mode == 1u) a_exp = b_exp = 64u;
                if (mode == 2u) a_exp = b_exp = 191u;
                if (mode == 3u) { a_exp = 119u + random_word() % 10u; b_exp = 119u + random_word() % 10u; }
                if (mode == 4u) { a_exp = 64u + column % 128u; b_exp = 254u - a_exp; }
                if (mode == 5u) { a_exp = column ? 64u : 127u; b_exp = 127u; }
                if (mode == 6u) a_exp = b_exp = 1u; // Exercise the additive FTZ allowance separately.
                const uint16_t a = mode == 7u ? 0u : uint16_t((a_exp << 7u) | (random_word() & 127u));
                const uint16_t b = uint16_t((b_exp << 7u) | (random_word() & 127u));
                const double product = bf16(a) * bf16(b);
                reference += product;
                approximate = add_down_ftz(approximate, product);
            }
            const float upper = bound::finish(approximate, k);
            assert(!std::isnan(upper) && double(upper) >= reference);
            ++cases;
        }
    }
    for (uint32_t invalid : {0x7f800000u, 0xff800000u, 0x7fc00001u, 0x80000000u, 0xbf800000u})
        assert(bound::bits(bound::finish(bound::value(invalid), 4096u)) == 0x7f800000u);
    std::printf("positive_product_bound_cases=%u downward_rounding_and_ftz_underestimates=0\n", cases);
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-positive-sum-bound-") as tmp:
            exe = str(Path(tmp) / "bound")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=undefined", "-I", str(ROOT / "native/providers/moe_accumulator"),
                 "-x", "c++", "-", "-o", exe],
                input=source, text=True, check=True, timeout=30,
            )
            subprocess.run([exe], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
