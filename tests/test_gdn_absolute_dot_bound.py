import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GdnAbsoluteDotBoundTests(unittest.TestCase):
    def test_original_accumulator_bounds_and_both_consumers(self):
        with tempfile.TemporaryDirectory(prefix='qrt-gdn-bound-') as directory:
            binary = Path(directory) / 'check'
            build = subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                '-Wall', '-Wextra', '-Werror', '-ffp-contract=off',
                '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                str(ROOT / 'tests/native/gdn_absolute_dot_bound_host.cpp'),
                '-o', str(binary)], capture_output=True, text=True, timeout=40)
            self.assertEqual(build.returncode, 0, build.stderr)
            result = subprocess.run([str(binary)],
                                    capture_output=True, text=True, timeout=40)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['dots'], 32768)
        self.assertEqual(report['zero_dots'], 4096)
        self.assertGreater(report['state_admitted'], 4096)
        self.assertGreater(report['output_admitted'], 4096)
        self.assertEqual(report['invalid_rejections'], 10)
        self.assertEqual(report['false_admissions'], 0)
        self.assertEqual(report['bound_violations'], 0)
        print(result.stdout.strip(), flush=True)


if __name__ == '__main__':
    unittest.main()
