"""Actual candidate ownership and replay, with independent integer endpoints."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SeparateStateGdnTests(unittest.TestCase):
    def test_actual_fast_abort_and_replay_preserve_tensor_owners(self):
        header = (ROOT / 'native/providers/gdn/separate_state_replay.h').read_text()
        retained = (ROOT / 'native/providers/gdn/blackwell_lifetime_matrices.h').read_text()
        start = retained.index('template<unsigned Columns>\n__global__ void state_kernel(')
        body = retained[start:retained.index('\n// Query/checkpoint', start)]
        body = body.replace('void state_kernel(', 'void replay_kernel(', 1)
        body = body.replace('const unsigned char* table) {', 'const unsigned char* table,const unsigned* completed) {', 1)
        body = body.replace('    static_assert(Columns==4u || Columns==8u);',
            '    static_assert(Columns==4u || Columns==8u);\n    if(completed[blockIdx.y*(128u/Columns)+blockIdx.x])return;', 1)
        self.assertIn(body, header, 'Replay arithmetic must remain the retained state body')
        with tempfile.TemporaryDirectory(prefix='qrt-separate-state-') as directory:
            work = Path(directory)
            (work / 'separate_state_under_test.h').write_text(header[header.index('namespace qrt_fla_separate_state {'):])
            executable = work / 'check'
            build = subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O1', '-ffp-contract=off', '-pthread',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all', '-I', str(ROOT), '-I', str(work),
                str(ROOT / 'tests/native/separate_state_gdn_host.cpp'), '-o', str(executable)],
                capture_output=True, text=True, timeout=40)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=55)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn('separate_state_host=pass', run.stdout)


if __name__ == '__main__':
    unittest.main()
