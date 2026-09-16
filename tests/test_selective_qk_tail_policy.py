"""Execute strict option parsing and cold-only complete slab ownership."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SelectiveQkTailPolicyTests(unittest.TestCase):
    def test_boundary_shapes_and_parser(self):
        with tempfile.TemporaryDirectory(prefix="qrt-selective-tail-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/selective_qk_tail_policy_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
