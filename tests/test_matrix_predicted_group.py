import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatrixPredictedGroupTests(unittest.TestCase):
    def test_predictions_never_replace_actual_alignment_checks(self):
        with tempfile.TemporaryDirectory(prefix='qrt-matrix-predicted-group-') as directory:
            executable = Path(directory) / 'predicted-group'
            subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall',
                '-Wextra', '-Werror', '-fsanitize=address,undefined',
                '-fno-sanitize-recover=all', '-I',
                str(ROOT / 'native/providers/moe_accumulator'),
                str(ROOT / 'tests/native/matrix_predicted_group_host_selftest.cpp'),
                '-o', str(executable),
            ], check=True, timeout=30)
            result = subprocess.run([str(executable)], check=True,
                                    capture_output=True, text=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertEqual(report['canonical_groups'], 8192 * 16)
        self.assertEqual(report['raw_mismatches'], 0)
        self.assertTrue(report['rejected_outputs_unchanged'])
        self.assertEqual(report['plan_bytes'], 8)
        self.assertGreater(report['parallel_prefix_hits'], 10000)
        self.assertGreater(report['parallel_prefix_fallbacks'], 0)
        self.assertGreater(report['identical_float_coefficients'], 10000)
        self.assertGreater(report['matrix_plans'], 100000)
        self.assertEqual(report['integer_core_matrix_products'], 4)
        self.assertFalse(report['hardware_error_bound_required'])
        print(result.stdout.strip(), flush=True)


if __name__ == '__main__':
    unittest.main()
