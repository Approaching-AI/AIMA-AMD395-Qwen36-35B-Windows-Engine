"""Check the bounded native-normalizer specification against canonical Values."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeRzCarryTests(unittest.TestCase):
    def test_normalization_domain(self):
        with tempfile.TemporaryDirectory(prefix="qrt-native-rz-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-fsanitize=address,undefined,float-cast-overflow",
                 "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/native_rz_carry_selftest.cpp"), "-o", executable],
                check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
