"""Verify the complete signed16 decomposition domain before native matrix tests."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PositiveKaratsubaTests(unittest.TestCase):
    def test_all_core_encodings_and_independent_wide_products(self):
        code = r'''
#include "sm121_positive_karatsuba.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace k3 = qrt_sm121_positive_karatsuba;
int signed_core(uint16_t value) { return value & 0x8000u ? int(value) - 65536 : int(value); }
int main() {
    for (unsigned value = 0u; value <= 510u; ++value) {
        const uint16_t bits = k3::positive_half_bits(value);
        _Float16 actual; std::memcpy(&actual, &bits, sizeof(bits));
        if (float(actual) != float(value)) return 2;
    }
    for (unsigned core = 0u; core < 65536u; ++core) {
        const auto d = k3::split(uint16_t(core));
        if (d.high * 256 + d.low != signed_core(uint16_t(core)) ||
            d.high < -128 || d.high > 127 || d.low < 0 || d.low > 255 ||
            k3::sum_digit(uint16_t(core)) > 510u || int(k3::sum_digit(uint16_t(core))) != d.high + d.low + 128) return 1;
    }
    uint32_t seed = 0x3958192u;
    auto next = [&]() { seed ^= seed << 13u; seed ^= seed >> 17u; seed ^= seed << 5u; return seed; };
    for (unsigned test = 0u; test < 262144u; ++test) {
        int parts[3]{}; unsigned left_sum = 0u, right_sum = 0u; int64_t independent = 0;
        for (unsigned i = 0u; i < 16u; ++i) {
            uint16_t a = uint16_t(next()), b = uint16_t(next());
            if (test % 8u == 0u) a = uint16_t(test / 8u * 2u + (i & 1u));
            if (test % 8u == 1u) a = b = uint16_t((i & 1u) ? 32767u : 32768u);
            const auto x = k3::split(a), y = k3::split(b);
            parts[0] += x.high * y.high; parts[1] += x.low * y.low;
            parts[2] += (x.high + x.low + 128) * (y.high + y.low + 128);
            left_sum += k3::sum_digit(a); right_sum += k3::sum_digit(b);
            independent += int64_t(signed_core(a)) * signed_core(b);
        }
        if (std::abs(parts[0]) > 262144 || std::abs(parts[1]) > 1040400 ||
            std::abs(parts[2]) > 4161600 || independent != k3::reconstruct(parts[0], parts[1], parts[2], left_sum, right_sum)) return 3;
    }
    std::puts("core_encodings=65536 wide_K16_dots=262144 reconstruction_mismatches=0");
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-karatsuba-") as directory:
            exe = str(Path(directory) / "core")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            "-include", "initializer_list", "-I", str(ROOT / "native/providers/moe_accumulator"),
                            "-x", "c++", "-", "-o", exe], input=code, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
