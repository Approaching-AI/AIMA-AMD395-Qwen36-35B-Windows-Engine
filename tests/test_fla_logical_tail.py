"""Keep padded scan values outside the active triangular arithmetic."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function


ROOT = Path(__file__).resolve().parents[1]


class FlaLogicalTailTests(unittest.TestCase):
    def test_inactive_scan_roundoff_cannot_poison_the_inverse(self):
        source = (ROOT / "native/providers/gdn/blackwell_kkt.h").read_text()
        kernel = function(source, "__global__ void gate_kernel(")
        harness = r'''
#include <cmath>
#include <vector>
#include <limits>
#include "native/providers/gdn/sm121_exp2_table.h"
#define __global__
constexpr unsigned kChunk = 64;
struct Index { unsigned x; } blockIdx{}, threadIdx{}, blockDim{256};
''' + kernel + r'''
int main() {
    constexpr unsigned elements = 64u * 32u * 64u;
    const float padding = std::nextafter(-1.0f, 0.0f);
    const float exponent = (padding + 1.0f) * 1.4426950408889634074f;
    // This is the original failure mechanism: even a zero padded dot becomes
    // NaN when evaluated outside the negative-only exponential table domain.
    if (!std::isnan(0.0f * qrt_sm121_exp2::evaluate(nullptr, exponent))) return 1;
    for (unsigned valid : {1u, 2u, 16u, 17u, 31u, 32u, 33u, 63u, 64u}) {
        std::vector<float> g(64u * 32u, padding), a(elements + 32u, 123.0f);
        for (unsigned token = 0; token < valid; ++token)
            for (unsigned head = 0; head < 32u; ++head) g[token * 32u + head] = -1.0f;
        for (unsigned index = 0; index < elements; ++index) {
            const unsigned token = index / 2048u, column = index % 64u;
            a[index] = token >= valid ? std::numeric_limits<float>::quiet_NaN()
                                     : column < token ? 0.25f : 0.0f;
        }
        // Exercise the actual production kernel, including excess launch lanes.
        for (unsigned index = 0; index < elements + 32u; ++index) {
            blockIdx.x = index / 256u; threadIdx.x = index % 256u;
            gate_kernel(a.data(), g.data(), 64u, valid, nullptr);
        }
        for (unsigned index = 0; index < elements; ++index) {
            const unsigned token = index / 2048u, column = index % 64u;
            const float expected = token < valid && column < token ? 0.25f : 0.0f;
            if (a[index] != expected) return 2;
        }
        for (unsigned index = elements; index < a.size(); ++index)
            if (a[index] != 123.0f) return 3;
    }
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-fla-logical-tail-") as tmp:
            executable = str(Path(tmp) / "tail")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-ffp-contract=off",
                 "-Wall", "-Wextra", "-Werror", "-I", str(ROOT), "-x", "c++", "-",
                 "-o", executable],
                input=harness, text=True, check=True, timeout=30,
            )
            subprocess.run([executable], check=True, timeout=5)


if __name__ == "__main__":
    unittest.main()
