"""Check dyadic composition and every accepted original K16 carry state."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DyadicCarryScanTests(unittest.TestCase):
    def test_algebra_boundaries_and_predicted_trajectories(self):
        with tempfile.TemporaryDirectory(prefix="qrt-dyadic-scan-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined,float-cast-overflow",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/dyadic_carry_scan_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
