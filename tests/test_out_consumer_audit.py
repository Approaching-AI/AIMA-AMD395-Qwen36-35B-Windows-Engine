"""Execute the actual diagnostic owner and residual interval edge cases."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OutConsumerAuditTests(unittest.TestCase):
    def test_actual_owner_and_interval_transport(self):
        source = (ROOT / 'native/providers/q8192_out_consumer_audit.h').read_text()
        owner = source[source.index('inline hipError_t run('):]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / 'out_consumer_actual_owner.h').write_text(
                'namespace qrt_out_consumer_audit {\n'
                'constexpr unsigned rows=2048,tokens=8192,width=4096,guard=128,fields=15;\n' + owner)
            executable = temp / 'out-consumer-host'
            build = subprocess.run([
                os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-ffp-contract=off',
                '-fsanitize=address,undefined', '-fno-omit-frame-pointer',
                '-I', str(temp), '-I', str(ROOT / 'native/providers'),
                str(ROOT / 'tests/native/out_consumer_audit_host_mock.cpp'), '-o', str(executable)
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('out_consumer_host_pass', run.stdout)
            print(run.stdout.strip())


if __name__ == '__main__':
    unittest.main()
