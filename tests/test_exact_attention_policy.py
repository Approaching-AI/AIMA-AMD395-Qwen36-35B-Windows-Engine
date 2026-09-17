"""Run strict combined-attention option and shape-boundary checks."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ExactAttentionPolicyTests(unittest.TestCase):
    def test_parser_and_owner_eligibility(self):
        with tempfile.TemporaryDirectory(prefix="qrt-exact-attention-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/exact_attention_policy_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
