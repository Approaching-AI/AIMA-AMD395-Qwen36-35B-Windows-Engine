"""Check lossless packed metadata and folded products against original integers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FoldedHalfProductsTests(unittest.TestCase):
    def test_original_representation_products_and_carries(self):
        with tempfile.TemporaryDirectory(prefix="qrt-folded-half-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/folded_half_products_host.cpp"), "-o", executable],
                check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
