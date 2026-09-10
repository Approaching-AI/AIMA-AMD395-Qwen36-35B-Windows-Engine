from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class Sm121Group16ModuloTests(unittest.TestCase):
    def test_native_modulo_decoder_matches_wide_signed_reference(self):
        compiler = shutil.which("c++") or shutil.which("clang++")
        if not compiler:
            self.skipTest("C++ compiler is unavailable")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "sm121-group16-modulo"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-fsanitize=undefined", "-fno-sanitize-recover=all",
                 "-I", str(root),
                 str(root / "tests/native/sm121_group16_modulo.cpp"),
                 "-o", str(executable)],
                check=True, capture_output=True, text=True, timeout=30,
            )
            result = subprocess.run(
                [str(executable)], check=True, capture_output=True,
                text=True, timeout=30,
            )
            self.assertIn("sm121_group16_modulo=pass", result.stdout)


if __name__ == "__main__":
    unittest.main()
