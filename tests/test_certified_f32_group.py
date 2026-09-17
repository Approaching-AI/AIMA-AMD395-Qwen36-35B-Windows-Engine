"""Check certified K16 grids against the independent wide accumulator."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CertifiedGroupTests(unittest.TestCase):
    def test_ordered_carries_and_grid_rejection(self):
        with tempfile.TemporaryDirectory(prefix="qrt-certified-group-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/certified_f32_group_host.cpp"), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
