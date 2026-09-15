"""Check lossless slab address maps and their complete padded extents."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SlabHalfLayoutTests(unittest.TestCase):
    def test_independent_padded_layouts(self):
        with tempfile.TemporaryDirectory(prefix="qrt-slab-half-") as tmp:
            exe = str(Path(tmp) / "slab-half")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all", str(ROOT / "tests/native/slab_half_layout_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
