"""Check canonical PV chunk coverage including both rescale rounding paths."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PvTransferBoundTests(unittest.TestCase):
    def test_original_recurrences_and_conditional_matrix_error(self):
        with tempfile.TemporaryDirectory(prefix="qrt-pv-transfer-") as tmp:
            executable = str(Path(tmp) / "bound")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
                            "-fsanitize=undefined", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/pv_transfer_bound_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
