"""Execute the actual read-only row-identity owner with a deferred stream."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProjectionRowReuseAuditTests(unittest.TestCase):
    def test_actual_owner_verifies_hash_candidates_and_drains_failures(self):
        source = (ROOT / 'native/providers/projection_row_reuse_audit.h').read_text()
        actual = source[source.index('inline hipError_t run('):]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'row_reuse_actual_owner.h').write_text(
                'namespace qrt_projection_row_reuse_audit {\n'
                'namespace identity=qrt_projection_row_identity;\n'
                'constexpr unsigned tokens=8192u,guard=128u;\n' + actual)
            executable = temp / 'row-reuse-owner'
            build = subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-I', str(temp), '-I', str(ROOT / 'native/providers'),
                str(ROOT / 'tests/native/projection_row_reuse_host_mock.cpp'),
                '-o', str(executable)
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('row_reuse_owner_pass', run.stdout)
            self.assertIn('verified_reusable_rows=6144 unique_or_unproven_rows=2048', run.stderr)
            self.assertIn('verified_reusable_rows=6143 unique_or_unproven_rows=2049', run.stderr)
            self.assertIn('hash_collision_rows=1', run.stderr)
            print(run.stdout.strip())


if __name__ == '__main__':
    unittest.main()
