"""Exercise actual OUT audit ownership, strict opt-in, and the captured miss."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OutL1ShadowTests(unittest.TestCase):
    def test_actual_host_owner_and_captured_selector_counterexample(self):
        audit = (ROOT / 'native/providers/q8192_out_l1_shadow_audit.h').read_text()
        owner = audit[audit.index('inline hipError_t run('):]
        provider = (ROOT / 'native/providers/whole_provider.cpp').read_text()
        start = provider.index('__device__ bool selected_bf16_projection_hawkeye_candidate(')
        selector = provider[start:provider.index('// Collect absolute output indices', start)]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'out_l1_actual_run.h').write_text(
                'namespace qrt_out_l1_shadow {\nconstexpr unsigned guard=128, common=4, counters=19;\n' + owner)
            (temp / 'out_l1_actual_selector.h').write_text(selector.replace('__device__', ''))
            executable = temp / 'out-l1-host'
            build = subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-I', str(temp), '-I', str(ROOT / 'native/providers'),
                            str(ROOT / 'tests/native/out_l1_shadow_host_mock.cpp'), '-o', str(executable)],
                           capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('out_l1_host_pass', result.stdout)


if __name__ == '__main__':
    unittest.main()
