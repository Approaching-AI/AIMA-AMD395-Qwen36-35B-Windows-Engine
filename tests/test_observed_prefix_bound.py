"""Check observed-prefix envelopes and exact suffixes against wide arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ObservedPrefixBoundTests(unittest.TestCase):
    def test_observed_magnitudes_and_remaining_loss(self):
        with tempfile.TemporaryDirectory(prefix="qrt-observed-prefix-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-fsanitize=address,undefined,float-cast-overflow",
                 "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/observed_prefix_bound_host.cpp"), "-o", executable],
                check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
