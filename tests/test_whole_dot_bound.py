"""Check complete-dot and remaining-segment envelopes against original K16."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WholeDotBoundTests(unittest.TestCase):
    def test_original_carries_and_suffixes(self):
        with tempfile.TemporaryDirectory(prefix="qrt-whole-dot-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-fsanitize=address,undefined,float-cast-overflow",
                 "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/whole_dot_bound_host.cpp"),
                 "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
