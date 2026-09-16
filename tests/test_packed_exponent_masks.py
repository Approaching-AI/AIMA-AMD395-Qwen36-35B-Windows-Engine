from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedExponentMaskTests(unittest.TestCase):
    def test_exact_certificates_carries_and_declined_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'exponent-mask')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            str(ROOT / 'tests/native/packed_exponent_masks_host_selftest.cpp'),
                            '-o', exe], check=True, timeout=30)
            result = subprocess.run([exe], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['bf16_encodings'], 65536)
            self.assertEqual(report['certificate_checks'], 327680)
            self.assertEqual(report['canonical_groups'], 262144)
            self.assertGreater(report['certificates'], 0)
            self.assertGreater(report['misses'], 0)
            self.assertGreater(report['declined_accumulations'], 0)
            self.assertEqual(report['raw_mismatches'], 0)
            self.assertTrue(report['immutable_inputs'])
            self.assertTrue(report['declined_output_unchanged'])
            print(result.stdout.strip(), flush=True)
