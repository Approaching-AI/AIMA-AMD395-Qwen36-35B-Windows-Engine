"""Compare collapsed normalization with independent wide signed group arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CanonicalNormalizeTests(unittest.TestCase):
    def test_integer_boundaries_and_underflow_match_original_wide_reference(self):
        code = r'''
#include "sm121_canonical_normalize.h"
#include <cstdint>
#include <cstdio>
int main() {
    uint32_t seed = 0x3958192u;
    for (unsigned i = 0u; i < 4194304u; ++i) {
        seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u;
        uint32_t magnitude = seed;
        switch (i % 16u) {
            case 0: magnitude = 0u; break;
            case 1: magnitude = 1u << (i / 16u % 32u); break;
            case 2: magnitude = (1u << (i / 16u % 32u)) - 1u; break;
            case 3: magnitude = 0xffffffffu; break;
        }
        const int maximum = int(i / 512u % 646u) - 133;
        const bool negative = (i & 1u) != 0u;
        using namespace qrt_q1_moe_hawkeye;
        // These two signed values reconstruct the full uint32 magnitude at
        // the reference's26-bit alignment grid, including its low two bits.
        const Value terms[] = {{magnitude >> 2u, int16_t(maximum), negative},
            {(magnitude & 3u) << 23u, int16_t(maximum - 25), negative}};
        auto expected = group_sum<26, -133>(terms, 2u);
        // The low-level normalizer preserves the supplied zero sign; the
        // final float conversion is responsible for clearing signed zero.
        if (!magnitude) expected.negative = negative;
        const auto actual = qrt_sm121_canonical::normalize(magnitude, negative, maximum);
        if (actual.significand != expected.significand || actual.exponent != expected.exponent ||
            actual.negative != expected.negative) {
            std::fprintf(stderr, "case=%u magnitude=%u exponent=%d\n", i, magnitude, maximum);
            return 1;
        }
    }
    std::puts("cases=4194304 normalization_mismatches=0 independent_wide_signed_reference=1");
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-normalize-") as tmp:
            exe = str(Path(tmp) / "normalize")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                "-I", str(ROOT / "native/providers/moe_accumulator"), "-x", "c++", "-", "-o", exe],
                input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
