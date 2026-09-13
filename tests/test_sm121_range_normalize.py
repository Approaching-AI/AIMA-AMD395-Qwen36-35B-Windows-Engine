"""Compare the certified normalizer with an independent exact dyadic value."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RangeNormalizeTests(unittest.TestCase):
    def test_normalization_against_exact_double(self):
        code = r'''
#include "range_projection_cases.h"
#include "../../native/providers/moe_accumulator/sm121_range_normalize.h"
#include <cmath>
#include <cstdio>
namespace range = qrt_sm121_range;
namespace cases = qrt_float_alignment_cases;
int main() {
    unsigned checks = 0u;
    const uint32_t edges[] = {0u,1u,2u,3u,0x7fffffu,0x800000u,0xffffffu,
        0x1000000u,0x1000001u,0x7fffffffu,0x80000000u,
        qrt_sm121_group16::kMinNegativeModulo,qrt_sm121_group16::kMaxMagnitude};
    for (int maximum = -100; maximum <= 119; ++maximum) {
        for (unsigned i = 0; i < 65536u; ++i) {
            const uint32_t magnitude = i < sizeof(edges)/sizeof(edges[0]) ? edges[i] :
                cases::random(i * 7919u + unsigned(maximum + 100)) % (qrt_sm121_group16::kMaxMagnitude + 1u);
            const double exact = std::ldexp(double(magnitude), maximum - 25);
            float rounded = float(exact);
            if (double(rounded) > exact) rounded = std::nextafter(rounded, 0.0f);
            for (unsigned negative = 0; negative < 2; ++negative) {
                const auto actual = range::normalize(magnitude, negative != 0u, maximum);
                const auto original = qrt_sm121_canonical::normalize(magnitude, negative != 0u, maximum);
                const float wanted = negative && magnitude ? -rounded : rounded;
                uint32_t expected; __builtin_memcpy(&expected, &wanted, 4u);
                if (actual.significand != original.significand || actual.exponent != original.exponent ||
                    actual.negative != original.negative || cases::output_bits(actual) != expected) {
                    std::printf("RANGE_DIFF magnitude=%u maximum=%d negative=%u wanted=%08x actual=%08x\n",
                        magnitude, maximum, negative, expected, cases::output_bits(actual)); return 1;
                }
                ++checks;
            }
        }
    }
    std::printf("{\"normalizations\":%u,\"exact_dyadic_reference\":true,\"raw_mismatches\":0}\n", checks);
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-range-normalize-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            "-I", str(ROOT / "tests/native"), "-x", "c++", "-", "-o", executable],
                           input=code, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
