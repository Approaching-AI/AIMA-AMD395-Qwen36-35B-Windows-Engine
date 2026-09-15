"""Check original K16 loss coverage independently of the native matrix model."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CoarseProjectionBoundTests(unittest.TestCase):
    def test_original_prefixes_and_conditional_native_error(self):
        with tempfile.TemporaryDirectory(prefix="qrt-coarse-projection-") as tmp:
            exe = str(Path(tmp) / "bound")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/coarse_projection_bound_host_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
