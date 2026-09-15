import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class GdnConsumerIntervalTests(unittest.TestCase):
    def test_all_representable_points_and_rejection_ownership(self):
        compiler = shutil.which("clang++") or shutil.which("g++")
        if compiler is None:
            self.skipTest("C++ compiler unavailable")
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "gdn-consumer-interval"
            subprocess.run(
                [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 str(ROOT / "tests/native/gdn_consumer_interval_host.cpp"),
                 "-o", str(executable)], check=True, capture_output=True, timeout=30,
            )
            result = subprocess.run(
                [str(executable)], check=True, capture_output=True, text=True, timeout=30,
            )
        report = json.loads(result.stdout)
        self.assertEqual(report["cases"], 32768)
        self.assertGreater(report["state_admitted"], 1000)
        self.assertGreater(report["output_admitted"], 1000)
        self.assertEqual(report["state_points"], 7 * report["state_admitted"])
        self.assertEqual(report["output_points"], 49 * report["output_admitted"])
        self.assertEqual(report["exceptional_rejections"], 20)
        self.assertEqual(report["false_admissions"], 0)
        self.assertTrue(report["rejection_outputs_unchanged"])
        print(result.stdout.strip(), flush=True)
