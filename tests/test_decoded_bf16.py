"""Exhaustive operand decoding and original K16 group/carry equivalence."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class DecodedBf16Tests(unittest.TestCase):
    def test_original_group_and_carry(self):
        with tempfile.TemporaryDirectory(prefix="qrt-decoded-bf16-") as tmp:
            exe=str(Path(tmp)/"decoded")
            subprocess.run([os.environ.get("CXX","c++"),"-std=c++17","-O2","-Wall","-Wextra","-Werror",
                "-fsanitize=undefined,float-cast-overflow","-fno-sanitize-recover=all",
                str(ROOT/"tests/native/decoded_bf16_selftest.cpp"),"-o",exe],check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=30)

if __name__=="__main__":unittest.main()
