"""Exact signed16 decomposition; native nearest recovery is tested separately."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CenteredKaratsubaTests(unittest.TestCase):
    def test_exact_integer_decomposition(self):
        with tempfile.TemporaryDirectory(prefix="qrt-centered-karatsuba-") as temporary:
            executable = str(Path(temporary) / "centered")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                            "-fno-sanitize-recover=all", str(ROOT / "tests/native/centered_karatsuba_host_selftest.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
