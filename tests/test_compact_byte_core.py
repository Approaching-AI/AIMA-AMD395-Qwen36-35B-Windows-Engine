"""Check compact recovery and scalar fallback against independent wide sums."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompactByteCoreTests(unittest.TestCase):
    def test_representation_alignment_and_original_fallback(self):
        with tempfile.TemporaryDirectory(prefix="qrt-compact-byte-") as directory:
            executable = Path(directory) / "compact-byte"
            subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall",
                "-Wextra", "-Werror", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", "-I", str(ROOT / "native/providers/moe_accumulator"),
                str(ROOT / "tests/native/compact_byte_core_host.cpp"), "-o", str(executable),
            ], check=True, timeout=30)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=45)
        report = json.loads(result.stdout)
        self.assertEqual(report["representation_checks"], 65536 * 5 * 16)
        self.assertEqual(report["cases"], 4096 * 16 * 14)
        self.assertEqual(report["fallback_checks"], report["cases"])
        self.assertEqual(report["row_bytes"], 76)
        self.assertEqual(report["raw_mismatches"], 0)
        self.assertEqual(report["maximum_modulo_overlap_cases"], 2)
        self.assertTrue(report["rejected_outputs_unchanged"])
        self.assertTrue(report["immutable_rows"])
        self.assertFalse(report["native_matrix_checked"])
        print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    unittest.main()
