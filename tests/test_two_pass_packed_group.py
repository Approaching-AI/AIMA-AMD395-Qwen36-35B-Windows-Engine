"""Compare the strided two-pass primitive with independent original K16 sums."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TwoPassPackedGroupTests(unittest.TestCase):
    def test_original_groups_with_strides_and_carry_edges(self):
        with tempfile.TemporaryDirectory(prefix="qrt-two-pass-group-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined,float-cast-overflow",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/two_pass_packed_group_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
