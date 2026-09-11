"""Keep the original K projection and Q output reductions distinct."""

from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Q1StateDotOrderTests(unittest.TestCase):
    def test_real_recurrent_output_and_projection(self):
        fixture = json.loads((ROOT / 'tests/fixtures/q1_state_dot_order.json').read_text())
        source = r'''
#include "native/providers/gdn/sm121_q1_math.h"
#include <array>
using namespace qrt_sm121_q1;
int main() {
'''
        for index, name in enumerate(['output', 'projection']):
            case = fixture[name]
            source += '{\nconst uint32_t state_bits[128] = {'
            source += ','.join(str(v) + 'u' for v in case['state_f32_bits']) + '};\n'
            source += 'const uint32_t right_bits[128] = {'
            source += ','.join(str(v) + 'u' for v in case['right_f32_bits']) + '};\n'
            source += r'''
std::array<float, 16384> state{};
float right[128];
for (unsigned stride : {1u, 128u}) {
    for (unsigned i = 0; i < 128; ++i) {
        state[i * stride] = qrt_sm121_exp2::value(state_bits[i]);
        right[i] = qrt_sm121_exp2::value(right_bits[i]);
    }
'''
            decay = case.get('decay_bits', 1065353216)
            apply_decay = 'true' if name == 'projection' else 'false'
            source += f'''const float value = state_dot(state.data(), stride,
    qrt_sm121_exp2::value({decay}u), right, {case['value_dim']}u, {apply_decay});
if (qrt_sm121_exp2::bits(value) != {case['expected_f32_bits']}u) return {index + 1};
'''
            if name == 'output':
                source += f"if (bf16(value) != {case['reference_bf16']}) return 3;\n"
            source += '}\n}\n'
        source += 'return 0;\n}\n'
        with tempfile.TemporaryDirectory(prefix='qrt-q1-state-dot-') as directory:
            executable = str(Path(directory) / 'state-dot')
            subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O2',
                            '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT), '-x', 'c++', '-', '-o', executable],
                           input=source, text=True, check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=5)


if __name__ == '__main__':
    unittest.main()
