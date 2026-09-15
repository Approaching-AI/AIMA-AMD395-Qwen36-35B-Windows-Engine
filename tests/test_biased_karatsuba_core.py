"""Check exact digit preparation, bounded bias subtraction and wide reconstruction."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class BiasedKaratsubaCoreTests(unittest.TestCase):
    def test_all_core_encodings_and_independent_wide_dots(self):
        with tempfile.TemporaryDirectory(prefix="qrt-biased-core-") as tmp:
            exe = str(Path(tmp) / "biased-core")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", str(ROOT / "tests/native/biased_karatsuba_host_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
