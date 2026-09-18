"""Check lookahead plans against actual original K16 carries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NarrowQkPlanTests(unittest.TestCase):
    def test_actual_carry_acceptance_and_rejection(self):
        with tempfile.TemporaryDirectory(prefix="qrt-narrow-qk-plan-") as directory:
            executable = str(Path(directory) / "audit")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/narrow_qk_plan_host.cpp"),
                            "-o", executable], check=True, timeout=60)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
