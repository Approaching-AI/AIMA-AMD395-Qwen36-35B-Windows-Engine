from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RoutedConsumerIntervalTests(unittest.TestCase):
    def test_actual_owner_and_observed_miss(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'consumer-owner'
            build = subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                                    '-I', str(ROOT / 'native/providers'),
                                    str(ROOT / 'tests/native/routed_consumer_owner_host.cpp'), '-o', str(exe)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('pass=1', run.stdout)

    def test_all_encodings_and_independent_consumer_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = Path(directory) / 'consumer-interval'
            build = subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                                    '-fsanitize=undefined,float-cast-overflow', '-fno-sanitize-recover=all',
                                    '-I', str(ROOT / 'native/providers'),
                                    str(ROOT / 'tests/native/routed_consumer_interval_host.cpp'), '-o', str(exe)],
                                   capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn('pass=1', run.stdout)


if __name__ == '__main__':
    unittest.main()
