"""Execute the actual bounded scheduler with ownership and failure injection."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AttentionSlabScheduleTests(unittest.TestCase):
    def test_query_ownership_and_failure_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="qrt-attention-slabs-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/attention_slab_schedule_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
