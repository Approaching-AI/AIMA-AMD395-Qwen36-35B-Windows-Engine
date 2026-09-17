"""Enumerate the input box and table segments used by convolution certificates."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ConvConsumerIntervalTests(unittest.TestCase):
    def test_joint_interval_and_continuation_halo(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "conv-consumer-host"
            build = subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                "-ffp-contract=off", "-fsanitize=address,undefined",
                "-fno-omit-frame-pointer", "-I", str(ROOT / "native/providers"),
                str(ROOT / "tests/native/conv_consumer_interval_host.cpp"),
                "-o", str(executable),
            ], capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = json.loads(run.stdout)
            self.assertEqual(result["segments"], 64304)
            self.assertEqual(result["boundary_rejections"], 64303)
            self.assertEqual(result["false_certificates"], 0)
            self.assertGreater(result["enumerated_combinations"], 32768)
            self.assertGreater(result["constant_combinations"], 0)
            print(run.stdout.strip())


if __name__ == "__main__":
    unittest.main()
