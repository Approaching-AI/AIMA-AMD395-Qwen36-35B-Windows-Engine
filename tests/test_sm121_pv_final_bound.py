"""Require the deferred PV envelope to dominate the retained recurrence."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Sm121PvFinalBoundTests(unittest.TestCase):
    def test_envelope_dominance_and_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix="qrt-pv-final-bound-") as tmp:
            exe = str(Path(tmp) / "pv-final-bound")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", str(ROOT / "tests/native/final_pv_bound_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
