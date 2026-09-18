"""Distinguish strict diagnostic latency bounds from completed runtime work."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaCompletionGuardTests(unittest.TestCase):
    def test_clock_boundaries_and_monotonic_enclosing_interval(self):
        with tempfile.TemporaryDirectory(prefix="qrt-fla-completion-") as temp:
            executable = str(Path(temp)/"guard")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                "-fno-sanitize-recover=all", str(ROOT/"tests/native/fla_completion_guard_host.cpp"),
                "-o", executable], check=True, capture_output=True, timeout=30)
            result = subprocess.run([executable], check=True, capture_output=True, text=True, timeout=10)
            report = json.loads(result.stdout)
            self.assertEqual(report["comparisons"], 121)
            self.assertEqual(report["host_fallbacks"], 30)
            self.assertEqual(report["rejections"], 36)
            self.assertEqual(report["slow_completed"], 20)
            self.assertEqual(report["invalid_completed_clocks"], 16)
            self.assertEqual(report["guard_ms"], 100)
            self.assertTrue(report["pass"])
