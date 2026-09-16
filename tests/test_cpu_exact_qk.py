from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CpuExactQkTests(unittest.TestCase):
    def test_preparation_padding_and_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "cpu-qk"
            subprocess.run([
                "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                str(ROOT / "tests/native/cpu_exact_qk_host.cpp"), "-o", str(binary)
            ], check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(binary)], check=True, capture_output=True,
                                    text=True, timeout=120)
            self.assertIn('"padding_and_flags_pass":true', result.stdout)


if __name__ == "__main__":
    unittest.main()
