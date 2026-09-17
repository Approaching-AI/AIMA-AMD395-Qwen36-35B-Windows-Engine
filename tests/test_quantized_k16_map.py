"""Check independent predictions, map domains and every accepted K16 carry."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QuantizedK16MapTests(unittest.TestCase):
    def test_generated_carries_and_bad_predictions(self):
        with tempfile.TemporaryDirectory(prefix="qrt-k16-map-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/quantized_k16_map_audit.cpp"), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
