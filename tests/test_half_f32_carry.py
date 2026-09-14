"""Check compact operands with an FP32 carry against original wide K16 sums."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class HalfF32CarryTests(unittest.TestCase):
    def test_original_groups_and_rejection(self):
        with tempfile.TemporaryDirectory(prefix="qrt-half-f32-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined,float-cast-overflow",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/half_f32_carry_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
