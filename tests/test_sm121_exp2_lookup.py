from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Exp2LookupTests(unittest.TestCase):
    def test_exterior_and_invalid_domain_do_not_access_table_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "lookup"
            subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tests/native/sm121_exp2_lookup_probe.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(binary), "--domain-only"], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
