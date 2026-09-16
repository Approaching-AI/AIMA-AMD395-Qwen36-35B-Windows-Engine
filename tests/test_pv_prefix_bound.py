"""Verify original PV envelopes, suffix products and BF16 midpoint decisions."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PvPrefixBoundTests(unittest.TestCase):
    def test_original_envelope_and_midpoints(self):
        with tempfile.TemporaryDirectory(prefix="qrt-pv-prefix-") as tmp:
            executable = str(Path(tmp) / "bound")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/pv_prefix_bound_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
