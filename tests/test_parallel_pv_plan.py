"""Check bounded partial storage and the separately rounded PV envelope."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ParallelPvPlanTests(unittest.TestCase):
    def test_layout_bounds_and_independent_wide_recurrence(self):
        with tempfile.TemporaryDirectory(prefix="qrt-parallel-pv-") as tmp:
            exe = str(Path(tmp) / "parallel-pv-plan")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", str(ROOT / "tests/native/parallel_pv_plan_selftest.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
