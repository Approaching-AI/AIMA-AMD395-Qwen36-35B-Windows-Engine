"""Check integer recovery and individual truncation independently of GPU DOT2."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PairResidueTests(unittest.TestCase):
    def test_integer_residue_recovery_and_original_truncation(self):
        with tempfile.TemporaryDirectory(prefix="qrt-pair-residue-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/pair_residue_host.cpp"), "-o", executable],
                check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
