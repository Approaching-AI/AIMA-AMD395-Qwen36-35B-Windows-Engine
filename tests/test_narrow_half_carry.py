"""Check narrow half carries against independent original K16 arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NarrowHalfCarryTests(unittest.TestCase):
    def test_original_carries_through_k8192(self):
        with tempfile.TemporaryDirectory(prefix="qrt-narrow-half-carry-") as directory:
            executable = str(Path(directory) / "audit")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/narrow_half_carry_host.cpp"),
                            "-o", executable], check=True, timeout=60)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
