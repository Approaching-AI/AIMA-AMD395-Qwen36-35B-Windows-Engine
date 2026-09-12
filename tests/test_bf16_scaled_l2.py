"""Compare conservative FP32 metadata with independent full-range BF16 norms."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScaledL2Tests(unittest.TestCase):
    def test_full_range_and_mixed_rows(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include "bf16_scaled_l2.h"
#include "scaled_l2_reference.h"
namespace p = qrt_bf16_scaled_l2;
float upper(const uint16_t *row, unsigned columns) {
    unsigned maximum = 0;
    for (unsigned k = 0; k < columns; ++k) maximum = std::max(maximum, p::exponent(row[k]));
    if (maximum == 0 || maximum == 255) return p::finish(0, maximum);
    float partial[256]{};
    for (unsigned lane = 0; lane < 256; ++lane)
        for (unsigned k = lane; k < columns; k += 256)
            partial[lane] = p::add_up(partial[lane], p::square(row[k], maximum));
    for (unsigned step = 128; step; step >>= 1)
        for (unsigned lane = 0; lane < step; ++lane)
            partial[lane] = p::add_up(partial[lane], partial[lane + step]);
    return p::finish(partial[0], maximum);
}
void check(float actual, long double norm) {
    if (norm == 0) { assert(actual == 0); return; }
    if (!std::isfinite(norm)) { assert(std::isinf(actual)); return; }
    assert(static_cast<long double>(actual) >= norm * static_cast<long double>(1.00002f));
    if (std::isfinite(actual)) assert(static_cast<long double>(actual) <= norm * 1.0001L);
}
int main() {
    unsigned rows = 0;
    std::vector<uint16_t> row(2048);
    for (unsigned raw = 0; raw < 65536; ++raw) {
        std::fill(row.begin(), row.end(), uint16_t(raw));
        const long double v = scaled_l2_test::bf16(uint16_t(raw));
        for (unsigned columns : {512u, 2048u}) {
            check(upper(row.data(), columns), v * std::sqrt(static_cast<long double>(columns)));
            ++rows;
        }
    }
    for (unsigned columns : {1u, 17u, 255u, 256u, 511u, 512u, 2048u}) {
        const auto mixed = scaled_l2_test::fixture(1024, columns);
        for (unsigned r = 0; r < 1024; ++r) {
            const auto *input = mixed.data() + size_t(r) * columns;
            check(upper(input, columns), scaled_l2_test::norm(input, columns));
            ++rows;
        }
    }
    for (unsigned e : {1u, 64u, 127u, 254u}) {
        for (unsigned m = 0; m < 0x800000u; m += 8191u) {
            const float input = p::value((e << 23u) | m);
            for (int shift = -300; shift <= 300; ++shift) {
                const long double expected = std::ldexp(static_cast<long double>(input), shift);
                const float actual = p::scale_up(input, shift);
                assert(static_cast<long double>(actual) >= expected);
                const float rounded = static_cast<float>(expected);
                assert(actual <= std::nextafter(rounded, std::numeric_limits<float>::infinity()));
            }
        }
    }
    std::printf("scaled_l2_rows=%u full_bf16_encodings=65536 bound_failures=0\n", rows);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-scaled-l2-') as tmp:
            exe = str(Path(tmp) / 'scaled-l2')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined',
                            '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-I', str(ROOT / 'tests/native'), '-x', 'c++', '-', '-o', exe],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
