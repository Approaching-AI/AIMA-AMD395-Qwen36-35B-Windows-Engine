"""Check coalesced F32/F64 metadata against independent positive F64 sums."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CooperativeNormMetadataTests(unittest.TestCase):
    def test_norm_and_maximum_bounds(self):
        with tempfile.TemporaryDirectory(prefix="qrt-cooperative-norm-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-fsanitize=address,undefined,float-cast-overflow",
                 "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/cooperative_norm_metadata_host.cpp"),
                 "-o", executable], check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
