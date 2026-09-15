"""Execute the actual empirical-bound owner, finishing kernel and cleanup."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OutL1ReplayTests(unittest.TestCase):
    def test_bounds_and_every_transport_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'out-l1-replay')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all', '-I', str(ROOT / 'native/providers'),
                            str(ROOT / 'tests/native/out_l1_replay_host_mock.cpp'), '-o', exe],
                           check=True, timeout=30)
            result = subprocess.run([exe], capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('out_l1_replay_host_pass', result.stdout)
            print(result.stdout.strip(), flush=True)
