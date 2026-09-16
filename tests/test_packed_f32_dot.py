from pathlib import Path
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedF32DotTests(unittest.TestCase):
    def test_original_endpoints_and_declined_output_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'packed-f32-dot')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            str(ROOT / 'tests/native/packed_f32_dot_host.cpp'),
                            '-o', exe], check=True, timeout=30)
            result = subprocess.run([exe], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['calls'], 12288)
            self.assertEqual(report['raw_mismatches'], 0)
            self.assertTrue(report['immutable_inputs'])
            self.assertTrue(report['declined_output_unchanged'])
            print(result.stdout.strip(), flush=True)
