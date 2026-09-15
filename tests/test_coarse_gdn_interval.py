"""Check C64 envelopes at the actual GDN rounding boundaries."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CoarseGdnIntervalTests(unittest.TestCase):
    def test_canonical_consumers_and_invalid_ranges(self):
        with tempfile.TemporaryDirectory(prefix="qrt-coarse-gdn-") as directory:
            executable = Path(directory) / "coarse-gdn"
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall",
                 "-Wextra", "-Werror", "-ffp-contract=off",
                 "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/coarse_gdn_interval_host.cpp"),
                 "-o", str(executable)], check=True, timeout=30,
            )
            result = subprocess.run([str(executable)], check=True,
                                    capture_output=True, text=True, timeout=45)
        report = json.loads(result.stdout)
        self.assertEqual(report["cases"], 24576)
        self.assertEqual(report["canonical_prefixes"], 73728)
        self.assertEqual(report["undercoverage"], 0)
        self.assertEqual(report["false_admissions"], 0)
        self.assertTrue(report["rejection_outputs_unchanged"])
        for name in ("wu_admitted", "residual_admitted", "output_admitted"):
            self.assertGreater(report[name], 1000)
        self.assertFalse(report["hardware_error_bound_proven"])
        print(result.stdout.strip(), flush=True)
