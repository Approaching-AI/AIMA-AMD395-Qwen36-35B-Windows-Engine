"""Reproduce the original embedding norm at two real long-decode origins."""

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q1EmbeddingNormTests(unittest.TestCase):
    def test_original_variance_and_all_bf16_outputs(self):
        fixture = json.loads((ROOT / 'tests/fixtures/q1_embedding_norm.json').read_text())
        source = r'''
#include "native/providers/gdn/sm121_q1_math.h"
using namespace qrt_sm121_q1;
int main() {
'''
        source += 'const uint16_t weights[2048] = {'
        source += ','.join(map(str, fixture['weight_bf16'])) + '};\n'
        for index, case in enumerate(fixture['cases']):
            source += '{\nconst uint16_t inputs[2048] = {'
            source += ','.join(map(str, case['embedding_bf16'])) + '};\n'
            source += 'const uint16_t expected[2048] = {'
            source += ','.join(map(str, case['expected_norm_bf16'])) + '};\n'
            source += r'''
float warps[8];
for (unsigned w = 0; w < 8; ++w) {
    float partial[32];
    for (unsigned lane = 0; lane < 32; ++lane) {
        float values[8];
        for (unsigned i = 0; i < 8; ++i)
            values[i] = widen(inputs[(w * 32 + lane) * 8 + i]);
        partial[lane] = embedding_lane_sumsq(values);
    }
    for (unsigned step = 16; step; step >>= 1)
        for (unsigned i = 0; i < step; ++i)
            partial[i] = add(partial[i], partial[i + step]);
    warps[w] = partial[0];
}
for (unsigned step = 4; step; step >>= 1)
    for (unsigned i = 0; i < step; ++i)
        warps[i] = add(warps[i], warps[i + step]);
const float variance = add(multiply(warps[0], 1.f / 2048.f), 1.e-6f);
'''
            source += f'''
if (qrt_sm121_exp2::bits(warps[0]) != {case['expected_sumsq_bits']}u) return {index * 3 + 1};
if (qrt_sm121_exp2::bits(variance) != {case['expected_variance_bits']}u) return {index * 3 + 2};
// The inverse is captured from original CUDA at this measured variance;
// table/instruction emulation has its own independent domain tests.
const float inverse = qrt_sm121_exp2::value({case['original_inverse_bits']}u);
for (unsigned i = 0; i < 2048; ++i)
    if (bf16(embedding_norm_value(widen(inputs[i]), inverse, weights[i])) != expected[i])
        return {index * 3 + 3};
}}
'''
        source += 'return 0;\n}\n'
        with tempfile.TemporaryDirectory(prefix='qrt-q1-embedding-norm-') as directory:
            executable = str(Path(directory) / 'embedding-norm')
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2',
                            '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
