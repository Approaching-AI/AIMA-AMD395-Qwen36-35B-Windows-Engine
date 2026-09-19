"""Original payloads, same-cell score lifetime and complete storage bounds."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InplaceProbabilityStorageTests(unittest.TestCase):
    def test_payload_lifetime_and_all_supported_extents(self):
        with tempfile.TemporaryDirectory(prefix="qrt-inplace-probability-") as tmp:
            exe = Path(tmp) / "storage"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
                            "-fstrict-aliasing", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                            "-I", str(ROOT),
                            str(ROOT / "tests/native/inplace_probability_storage_host.cpp"),
                            "-o", str(exe)], check=True, timeout=45)
            subprocess.run([str(exe)], check=True, timeout=45)


if __name__ == "__main__":
    unittest.main()
