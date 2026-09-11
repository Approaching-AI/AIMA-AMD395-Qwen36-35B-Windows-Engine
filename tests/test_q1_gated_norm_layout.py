"""Regress the captured short-decode variance that differs from prefill."""

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q1GatedNormLayoutTests(unittest.TestCase):
    def test_original_short_row_uses_four_adjacent_values_per_lane(self):
        fixture = json.loads((ROOT / 'tests/fixtures/q1_gated_norm_short_row.json').read_text())
        whole = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        functions = '__device__ float gated_rmsnorm_core_load(' + whole.split(
            '__device__ float gated_rmsnorm_core_load(', 1)[1].split(
                '__device__ float device_sm121_rsqrt_from_gfx1151(', 1)[0]
        source = r'''
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstddef>
#define __device__
float device_bf16_to_float(uint16_t input) {
    uint32_t bits = uint32_t(input) << 16; float result;
    std::memcpy(&result, &bits, 4); return result;
}
float device_mul_separate(float a, float b) { volatile float out = a * b; return out; }
''' + functions + r'''
uint32_t sum(const float* fp32, const uint16_t* bf16, unsigned segment) {
    std::array<float, 32> sums{};
    for (unsigned i = 0; i < 128u / segment; ++i)
        sums[i] = gated_rmsnorm_triton_segment_sum(fp32, bf16, i * segment, segment);
    for (unsigned step = 64u / segment; step; step /= 2u)
        for (unsigned i = 0; i < step; ++i) sums[i] += sums[i + step];
    uint32_t bits; std::memcpy(&bits, sums.data(), 4); return bits;
}
int main() {
    const uint16_t input[128] = {''' + ','.join(map(str, fixture['core_bf16'])) + r'''};
    float fp32[128];
    for (unsigned i = 0; i < 128; ++i) fp32[i] = device_bf16_to_float(input[i]);
    if (sum(fp32, nullptr, 4) != ''' + str(fixture['short_row_sumsq_f32_bits']) + r'''u) return 1;
    if (sum(nullptr, input, 4) != ''' + str(fixture['short_row_sumsq_f32_bits']) + r'''u) return 2;
    if (sum(fp32, nullptr, 8) != ''' + str(fixture['multirow_sumsq_f32_bits']) + r'''u) return 3;
    if (gated_rmsnorm_triton_segment_sum(fp32, nullptr, 0) !=
        gated_rmsnorm_triton_segment_sum(fp32, nullptr, 0, 8)) return 4;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q1-gated-') as directory:
            executable = str(Path(directory) / 'gated-layout')
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2',
                            '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                            '-x', 'c++', '-', '-o', executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
