"""Exercise production host ownership with a dependency-graph HIP model."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PipelineOwnerTests(unittest.TestCase):
    def test_dependencies_reuse_and_partial_failures(self):
        with tempfile.TemporaryDirectory(prefix="qrt-pipeline-owner-") as directory:
            exe = str(Path(directory) / "owner")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/pipelined_segment_owner_host.cpp"),
                            "-o", exe], check=True, timeout=60)
            subprocess.run([exe], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
