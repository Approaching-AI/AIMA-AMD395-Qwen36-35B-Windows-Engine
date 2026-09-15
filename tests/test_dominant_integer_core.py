"""Verify the 32-bit dominant-carry reduction against original wide arithmetic."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DominantIntegerCoreTests(unittest.TestCase):
    def test_exact_alignment_compensation_and_rejection(self):
        with tempfile.TemporaryDirectory(prefix="qrt-dominant-integer-") as directory:
            executable = Path(directory) / "dominant-integer"
            subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall",
                "-Wextra", "-Werror", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", "-I", str(ROOT / "native/providers/moe_accumulator"),
                str(ROOT / "tests/native/dominant_integer_core_host.cpp"), "-o", str(executable),
            ], check=True, timeout=30)
            result = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertEqual(report["cases"], 8192 * 16 * 12)
        self.assertEqual(report["raw_mismatches"], 0)
        self.assertEqual(report["maximum_modulo_overlap_cases"], 2)
        self.assertTrue(report["rejected_outputs_unchanged"])
        self.assertTrue(report["immutable_rows"])
        self.assertFalse(report["prediction_required"])
        print(result.stdout.strip(), flush=True)


if __name__ == "__main__":
    unittest.main()
