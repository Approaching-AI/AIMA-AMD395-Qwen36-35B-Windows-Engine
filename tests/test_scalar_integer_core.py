"""Validate exact scalar byte dot4 decomposition and original raw K16 carry states."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ScalarIntegerCoreTests(unittest.TestCase):
    def test_original_group_and_carry(self):
        with tempfile.TemporaryDirectory(prefix="qrt-scalar-core-") as tmp:
            exe=str(Path(tmp)/"decoded")
            subprocess.run([os.environ.get("CXX","c++"),"-std=c++17","-O2","-Wall","-Wextra","-Werror",
                "-fsanitize=address,undefined,float-cast-overflow","-fno-sanitize-recover=all",
                "-I",str(ROOT/"native/providers/moe_accumulator"),str(ROOT/"tests/native/scalar_integer_core_selftest.cpp"),"-o",exe],check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=30)

if __name__=="__main__":unittest.main()
