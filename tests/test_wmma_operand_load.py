"""Verify exact bounded K16 copies for aligned and misaligned source rows."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WmmaOperandLoadTests(unittest.TestCase):
    def test_original_words_and_inactive_rows(self):
        with tempfile.TemporaryDirectory(prefix="qrt-wmma-load-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/wmma_operand_load_host.cpp"),
                 "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
