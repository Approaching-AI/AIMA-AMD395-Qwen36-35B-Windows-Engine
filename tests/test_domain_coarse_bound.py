"""Verify C64 domain specialization against the complete original recurrence."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DomainCoarseBoundTests(unittest.TestCase):
    def test_raw_recurrence_and_exponent_edges(self):
        with tempfile.TemporaryDirectory(prefix="qrt-domain-coarse-") as tmp:
            exe = str(Path(tmp) / "bound")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/domain_coarse_bound_host.cpp"),
                            "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
