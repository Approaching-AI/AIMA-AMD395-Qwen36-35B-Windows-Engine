"""Check compact matrix groups and register queue source selection."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CompactMatrixGroupTests(unittest.TestCase):
    def test_partial_matrix_original_ordered_carries(self):
        with tempfile.TemporaryDirectory(prefix="qrt-partial-matrix-group-") as directory:
            executable = str(Path(directory) / "audit")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/partial_matrix_group_host.cpp"),
                            "-o", executable], check=True, timeout=60)
            subprocess.run([executable], check=True, timeout=60)

    def test_matrix_remainder_original_ordered_carries(self):
        with tempfile.TemporaryDirectory(prefix="qrt-matrix-remainder-group-") as directory:
            executable = str(Path(directory) / "audit")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/matrix_remainder_group_host.cpp"),
                            "-o", executable], check=True, timeout=60)
            subprocess.run([executable], check=True, timeout=60)

    def test_original_carries_and_queue_sources(self):
        with tempfile.TemporaryDirectory(prefix="qrt-compact-matrix-group-") as directory:
            executable = str(Path(directory) / "audit")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/compact_matrix_group_host.cpp"),
                            "-o", executable], check=True, timeout=60)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
