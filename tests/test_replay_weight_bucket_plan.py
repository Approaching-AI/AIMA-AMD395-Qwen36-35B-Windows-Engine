"""Check all supported row counts, bucket boundaries and workspace arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReplayWeightBucketPlanTests(unittest.TestCase):
    def test_shapes_and_independent_mappings(self):
        with tempfile.TemporaryDirectory(prefix="qrt-weight-buckets-") as tmp:
            exe = str(Path(tmp) / "weight-buckets")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                            "-fno-sanitize-recover=all", str(ROOT / "tests/native/replay_weight_bucket_plan_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
