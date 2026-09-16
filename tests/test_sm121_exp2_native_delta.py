from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeExp2DeltaTests(unittest.TestCase):
    def test_packed_corrections_and_unsigned_edges(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "delta"
            subprocess.run([
                "c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                str(ROOT / "tests/native/sm121_exp2_native_delta_host.cpp"), "-o", str(binary)
            ], check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(binary)], check=True, capture_output=True,
                                    text=True, timeout=30)
            self.assertIn("signed_edges_and_escape=pass", result.stdout)


if __name__ == "__main__":
    unittest.main()
