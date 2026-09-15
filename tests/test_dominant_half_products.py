"""Independent original-word bound and full K16 arithmetic comparisons."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DominantHalfProductsTests(unittest.TestCase):
    def test_original_bf16_bounds_and_wide_arithmetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'check')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-Wno-unknown-pragmas', '-fsanitize=address,undefined',
                            '-fno-sanitize-recover=all',
                            str(ROOT / 'tests/native/dominant_half_products_selftest.cpp'),
                            '-o', exe], check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=45)
