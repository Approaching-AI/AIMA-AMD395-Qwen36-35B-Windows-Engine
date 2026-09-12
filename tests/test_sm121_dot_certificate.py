"""Check every certified intermediate against independent signed64 K16 arithmetic."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class DotCertificateTests(unittest.TestCase):
    def test_certified_prefixes_match_original_integer_reference(self):
        with tempfile.TemporaryDirectory(prefix="qrt-dot-certificate-") as tmp:
            exe = str(Path(tmp) / "certificate")
            subprocess.run([os.environ.get("CXX", "c++"), "-O2", "-std=c++17",
                "-I", str(ROOT / "native/providers/moe_accumulator"),
                str(ROOT / "tests/native/dot_certificate_reference.cpp"), "-o", exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)

if __name__ == "__main__":
    unittest.main()
