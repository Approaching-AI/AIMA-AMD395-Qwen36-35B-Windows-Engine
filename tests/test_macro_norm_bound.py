"""Check block norms, observed scalar-add losses and original canonical intervals."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MacroNormBoundTests(unittest.TestCase):
    def test_separable_metadata_and_original_loss(self):
        with tempfile.TemporaryDirectory(prefix="qrt-macro-norm-") as directory:
            executable = str(Path(directory) / "check")
            subprocess.run(
                [os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                 "-fsanitize=address,undefined,float-cast-overflow",
                 "-fno-sanitize-recover=all",
                 str(ROOT / "tests/native/macro_norm_bound_host.cpp"), "-o", executable],
                check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=60)


if __name__ == "__main__":
    unittest.main()
