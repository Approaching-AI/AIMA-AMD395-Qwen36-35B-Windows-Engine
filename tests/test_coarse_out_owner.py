"""Check actual owner offsets, bounded replay and drain-before-release failures."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CoarseOutOwnerTests(unittest.TestCase):
    def test_owner_failure_cleanup_and_full_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'coarse-out-owner')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT / 'native/providers'),
                            str(ROOT / 'tests/native/coarse_out_owner_host_mock.cpp'),
                            '-o', exe], check=True, timeout=30)
            result = subprocess.run([exe], capture_output=True, text=True, timeout=45)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('coarse_out_owner_host_pass', result.stdout)
            print(result.stdout.strip(), flush=True)
