"""Validate complete-row metadata and conservative K16 exponent certificates."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RowMaximumTests(unittest.TestCase):
    def test_all_encodings_and_carry_certificates(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/row_maximum_selftest.cpp"), "-o", executable],
                           check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=15)


if __name__ == "__main__":
    unittest.main()
