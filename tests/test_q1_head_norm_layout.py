"""Regress the real K-cache value that diverged under the Q reduction layout."""

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q1HeadNormLayoutTests(unittest.TestCase):
    def test_original_strided_key_uses_four_warps(self):
        fixture = json.loads((ROOT / 'tests/fixtures/q1_key_norm_head256.json').read_text())
        key = fixture['ptx_replayed_key_layout']
        query = fixture['prior_query_layout']
        source = r'''
#include "native/providers/gdn/sm121_q1_math.h"
#include <array>
using namespace qrt_sm121_q1;
float sum(const float *values, bool is_key) {
    std::array<float, 128> partial{};
    std::array<float, 4> warp{};
    const unsigned warps = head_norm_warps(is_key);
    for (unsigned lane = 0; lane < warps * 32; ++lane)
        partial[lane] = head_norm_lane_sumsq(values, lane, is_key);
    for (unsigned w = 0; w < warps; ++w) {
        for (unsigned offset = 16; offset; offset >>= 1)
            for (unsigned lane = 0; lane < offset; ++lane)
                partial[w * 32 + lane] = add(partial[w * 32 + lane],
                                             partial[w * 32 + lane + offset]);
        warp[w] = partial[w * 32];
    }
    return head_norm_warp_sum(warp.data(), is_key);
}
int main() {
    const uint16_t input[256] = {''' + ','.join(map(str, fixture['key_bf16'])) + r'''};
    float values[256];
    for (unsigned i = 0; i < 256; ++i) values[i] = widen(input[i]);
    if (head_norm_warps(true) != 4 || head_norm_warps(false) != 2) return 1;
    if (qrt_sm121_exp2::bits(sum(values, true)) != ''' + str(key['sumsq_bits']) + r'''u) return 2;
    if (qrt_sm121_exp2::bits(sum(values, false)) != ''' + str(query['sumsq_bits']) + r'''u) return 3;
    const float variance = add(multiply(sum(values, true), 1.f / 256.f), 1.e-6f);
    if (qrt_sm121_exp2::bits(variance) != ''' + str(key['variance_bits']) + r'''u) return 4;
    const float scale = add(1.f, widen(''' + str(fixture['feature_weight_bf16']) + r'''));
    const float input_value = values[''' + str(fixture['feature']) + r'''];
    // The inverse values come from the captured SM121 instruction table.
    const float key_inverse = qrt_sm121_exp2::value(''' + str(key['inverse_bits']) + r'''u);
    const float query_inverse = qrt_sm121_exp2::value(''' + str(query['inverse_bits']) + r'''u);
    const uint16_t expected = ''' + str(fixture['reference_output_bf16']) + r''';
    if (bf16(multiply(multiply(input_value, key_inverse), scale)) != expected) return 5;
    if (bf16(multiply(multiply(input_value, query_inverse), scale)) == expected) return 6;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-q1-head-norm-') as directory:
            executable = str(Path(directory) / 'head-norm')
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2',
                            '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
