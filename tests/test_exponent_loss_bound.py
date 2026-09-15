import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExponentLossBoundTests(unittest.TestCase):
    def test_metadata_encloses_actual_canonical_alignment_losses(self):
        with tempfile.TemporaryDirectory(prefix='qrt-exponent-loss-') as directory:
            executable = Path(directory) / 'exponent-loss'
            subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall',
                '-Wextra', '-Werror', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all',
                str(ROOT / 'tests/native/exponent_loss_bound_host_selftest.cpp'),
                '-o', str(executable),
            ], check=True, timeout=30)
            result = subprocess.run([str(executable)], check=False,
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['summary_bytes'], 8)
        self.assertEqual(report['payloads'], 65536)
        self.assertEqual(report['original_prefix_checkpoints'], 16384 * 8 * 3)
        self.assertEqual(report['undercoverage'], 0)
        self.assertEqual(report['false_certificates'], 0)
        self.assertLess(report['revised_product_charges'], report['legacy_product_charges'])
        self.assertGreaterEqual(report['revised_certificates'], report['legacy_certificates'])
        print(result.stdout.strip(), flush=True)


if __name__ == '__main__':
    unittest.main()
