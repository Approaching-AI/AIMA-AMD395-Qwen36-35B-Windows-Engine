"""Exercise the actual asynchronous variance-replay owner and its failure paths."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OutVarianceReplayTests(unittest.TestCase):
    def test_actual_owner_cleanup_reports_and_context(self):
        source = (ROOT / 'native/providers/q8192_out_variance_replay.h').read_text()
        owner = source[source.index('struct Stats {'):]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'out_variance_replay_actual_owner.h').write_text(
                'namespace qrt_out_variance_replay {\n'
                'using Row=qrt_sm121_staged_half_projection::Row;\n'
                'constexpr unsigned rows=2048,tokens=8192,width=4096,fields=8;\n'
                'constexpr size_t cells=size_t(rows)*tokens;\n'
                'constexpr size_t prepared_groups=size_t(rows+tokens)*(width/16);\n'
                'constexpr size_t workspace_bytes=prepared_groups*sizeof(Row)+size_t(rows+tokens+tokens*fields)*4;\n' + owner)
            executable = temp / 'out-variance-owner'
            build = subprocess.run([
                'c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                '-I', str(temp), '-I', str(ROOT / 'native/providers'),
                str(ROOT / 'tests/native/out_variance_replay_host_mock.cpp'), '-o', str(executable)
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=45)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('out_variance_replay_host_pass', run.stdout)
            print(run.stdout.strip(), flush=True)
