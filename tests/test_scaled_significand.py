"""Check exponent-aligned scalar products against original integer arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ScaledSignificandTests(unittest.TestCase):
    def test_all_significands_signs_shifts_and_ordered_carries(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/scaled_significand_selftest.cpp"), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
