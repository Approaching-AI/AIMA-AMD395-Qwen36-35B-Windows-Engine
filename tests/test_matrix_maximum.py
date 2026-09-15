"""Check the exponent encoding, conditional recovery band and original carries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MatrixMaximumTests(unittest.TestCase):
    def test_encoding_recovery_and_canonical_groups(self):
        with tempfile.TemporaryDirectory(prefix="qrt-matrix-maximum-") as tmp:
            exe = str(Path(tmp) / "matrix-maximum")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/matrix_maximum_host_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
