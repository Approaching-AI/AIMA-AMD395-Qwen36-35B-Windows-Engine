"""Exercise the actual compacted state/output kernels with host CTA barriers.

The 128 host threads cover every initialization and use the kernel's original
thread-strided loops. Native tests retain the production 256-thread launch.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AbsoluteBoundGdnTests(unittest.TestCase):
    def test_actual_kernels_replay_ownership_tails_and_state(self):
        header = (ROOT / 'native/providers/gdn/absolute_bound_matrices.h').read_text()
        with tempfile.TemporaryDirectory(prefix='qrt-bound-gdn-') as directory:
            work = Path(directory)
            (work / 'absolute_bound_under_test.h').write_text(
                header[header.index('namespace qrt_fla_absolute_bound {'):])
            executable = work / 'check'
            build = subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                '-ffp-contract=off', '-pthread', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(work),
                str(ROOT / 'tests/native/absolute_bound_gdn_host.cpp'),
                '-o', str(executable)], capture_output=True, text=True, timeout=40)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True,
                                 text=True, timeout=90)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('absolute_bound_gdn_host=pass', run.stdout)
            self.assertEqual(run.stdout.count('original_states_outputs=pass'), 8)
            print(run.stdout.strip(), flush=True)


if __name__ == '__main__':
    unittest.main()
