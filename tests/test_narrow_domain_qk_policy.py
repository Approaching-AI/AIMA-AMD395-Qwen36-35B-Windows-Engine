"""Check the default-off QK owner and incompatible option boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NarrowDomainQkPolicyTests(unittest.TestCase):
    def test_parser_and_owner_boundaries(self):
        with tempfile.TemporaryDirectory(prefix="qrt-narrow-qk-policy-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/narrow_domain_qk_policy_host.cpp"),
                            "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
