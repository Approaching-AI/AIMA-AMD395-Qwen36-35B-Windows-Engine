"""Coprime residue recovery over the complete H7 integer operand domain."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CrtIntegerCoreTests(unittest.TestCase):
    def test_encodings_and_integer_recovery(self):
        with tempfile.TemporaryDirectory(prefix="qrt-crt-integer-") as temporary:
            executable = str(Path(temporary) / "crt")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                            "-fno-sanitize-recover=all", str(ROOT / "tests/native/crt_integer_core_host_selftest.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
