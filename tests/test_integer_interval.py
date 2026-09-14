"""Validate integer-matrix rounding intervals against independent wide K16 sums."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class IntegerIntervalTests(unittest.TestCase):
    def test_original_group_and_carry(self):
        with tempfile.TemporaryDirectory(prefix="qrt-integer-interval-") as tmp:
            exe=str(Path(tmp)/"decoded")
            subprocess.run([os.environ.get("CXX","c++"),"-std=c++17","-O2","-Wall","-Wextra","-Werror",
                "-fsanitize=address,undefined,float-cast-overflow","-fno-sanitize-recover=all",
                "-I",str(ROOT/"native/providers/moe_accumulator"),str(ROOT/"tests/native/integer_interval_selftest.cpp"),"-o",exe],check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=30)

if __name__=="__main__":unittest.main()
