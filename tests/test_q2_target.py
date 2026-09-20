"""Exercise the real forty-layer graph, private selection and borrowed lifetimes."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
from tests.q2_target_harness import write_target_harness

ROOT = Path(__file__).resolve().parents[1]
PROVIDERS = ROOT / 'native/providers/gdn'


class Q2TargetTests(unittest.TestCase):
    def test_complete_graph_failures_and_owner_lifetimes(self):
        with tempfile.TemporaryDirectory(prefix='qrt-q2-target-') as temporary:
            directory = Path(temporary)
            write_target_harness(directory)
            exe = directory / 'target'
            built = subprocess.run([os.getenv('CXX', 'c++'), '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                '-ffp-contract=off', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                '-I', str(directory), '-I', str(PROVIDERS),
                str(ROOT / 'tests/native/q2_target_host.cpp'), '-o', str(exe)],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(built.returncode, 0, built.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('all 89 submission failures drained', run.stdout)
            self.assertIn('all 80 publication failures drained', run.stdout)


if __name__ == '__main__':
    unittest.main()
