"""Exact product-exponent summaries and conservative saturated-field fallback."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedExponentsTests(unittest.TestCase):
    def test_original_maxima_and_declines(self):
        with tempfile.TemporaryDirectory(prefix="qrt-packed-exponents-") as temporary:
            executable = str(Path(temporary) / "exponents")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                            "-fno-sanitize-recover=all", str(ROOT / "tests/native/packed_exponents_host_selftest.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
