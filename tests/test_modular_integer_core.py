"""Validate the conditional residue lemma and complete canonical compensation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModularIntegerCoreTests(unittest.TestCase):
    def test_encoding_recovery_and_independent_canonical_groups(self):
        with tempfile.TemporaryDirectory(prefix="qrt-modular-core-") as tmp:
            exe = str(Path(tmp) / "modular-core")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=address,undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/modular_integer_core_host_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
