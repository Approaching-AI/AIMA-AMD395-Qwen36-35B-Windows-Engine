"""Run actual quad GDN ownership with host-emulated subgroup transport.

The GPU DPP instructions remain covered by the native fixture. The host test
uses the existing serial strong group for subgroup transport and an independent
wide integer accumulator for expected tensor values.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QuadFloatGdnTests(unittest.TestCase):
    def test_actual_kernels_original_values_tail_alias_and_state(self):
        header = (ROOT / 'native/providers/gdn/quad_float_matrices.h').read_text()
        with tempfile.TemporaryDirectory(prefix='qrt-quad-gdn-') as directory:
            work = Path(directory)
            (work / 'quad_under_test.h').write_text(
                header[header.index('namespace qrt_fla_quad_float {'):])
            executable = work / 'check'
            build = subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                '-ffp-contract=off', '-pthread', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(work),
                str(ROOT / 'tests/native/quad_float_gdn_host.cpp'),
                '-o', str(executable)], capture_output=True, text=True, timeout=40)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True,
                                 text=True, timeout=55)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('quad_gdn_host_transport=pass', run.stdout)


if __name__ == '__main__':
    unittest.main()
