"""Check row certificates against the independent wide K16 accumulator."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Sm121MantissaRowCertificateTests(unittest.TestCase):
    def test_carry_alignment_and_coarse_ranges(self):
        with tempfile.TemporaryDirectory(prefix="qrt-mantissa-row-") as tmp:
            exe = str(Path(tmp) / "row-certificate")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", str(ROOT / "tests/native/mantissa_row_certificate_probe.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe, "--selftest"], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
