"""Check shared operand reads against independent ordered wide K16 arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedDotTileTests(unittest.TestCase):
    def test_all_carries_and_original_fallback(self):
        with tempfile.TemporaryDirectory(prefix="qrt-dot-tile-") as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/packed_dot_tile_host.cpp"), "-o", exe],
                           check=True, timeout=45)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
