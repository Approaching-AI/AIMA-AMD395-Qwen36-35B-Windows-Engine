"""Exercise lossless exp2 decoding, domain guards, and corrupted layouts."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InterpolatedExp2Tests(unittest.TestCase):
    def test_all_residual_widths_endpoints_and_invalid_layouts(self):
        with tempfile.TemporaryDirectory(prefix="qrt-exp2-interpolated-") as directory:
            binary = str(Path(directory) / "selftest")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            str(ROOT / "tests/native/sm121_exp2_interpolated_selftest.cpp"),
                            "-o", binary], check=True, timeout=30)
            subprocess.run([binary], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
