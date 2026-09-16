from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class QueryLdsLayoutTests(unittest.TestCase):
    def test_cooperative_writes_and_all_query_consumers(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = str(Path(tmp) / 'query-layout')
            subprocess.run(['c++', '-std=c++17', '-O2', '-Wall', '-Wextra',
                            '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            '-I', str(ROOT / 'native/providers'),
                            str(ROOT / 'tests/native/query_lds_layout_host.cpp'),
                            '-o', exe], check=True, timeout=30)
            result = subprocess.run([exe], capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('query_lds_layout_host_pass', result.stdout)
            print(result.stdout.strip(), flush=True)
