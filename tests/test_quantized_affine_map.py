"""Validate exact integer map composition, including checked overflow."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QuantizedAffineMapTests(unittest.TestCase):
    def test_composition_and_signed_rounding(self):
        with tempfile.TemporaryDirectory(prefix="qrt-quantized-map-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/quantized_affine_map_host.cpp"), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
