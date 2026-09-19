"""Packed pair representations and consumed-score lifetimes under sanitizers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PackedProbabilityStorageTests(unittest.TestCase):
    def test_pair_transport_and_interleaved_rows(self):
        with tempfile.TemporaryDirectory(prefix="qrt-packed-probability-") as tmp:
            executable = Path(tmp) / "storage"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
                            "-fstrict-aliasing", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                            "-I", str(ROOT), str(ROOT / "tests/native/packed_probability_storage_host.cpp"),
                            "-o", str(executable)], check=True, timeout=45)
            subprocess.run([str(executable)], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
