"""Compile the actual server bridge against controlled native API responses."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ServerPrefixRoutingTests(unittest.TestCase):
    def test_only_complete_checkpoints_allow_prefix_and_fallback_never_repeats_output(self):
        with tempfile.TemporaryDirectory(prefix="qrt-server-prefix-routing-") as temporary:
            executable = str(Path(temporary) / "prefix-routing-test")
            subprocess.run(
                [os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror",
                 "-I", str(ROOT / "native/src"),
                 str(ROOT / "tests/native/server_prefix_routing.c"), "-o", executable],
                check=True, timeout=30,
            )
            result = subprocess.run([executable], capture_output=True, text=True,
                                    check=True, timeout=5)
        self.assertIn("11 bridge checkpoint routing and callback cases passed", result.stdout)
        self.assertEqual(result.stderr.count("prefix_cache_fallback"), 1)
        self.assertEqual(result.stderr.count("prefix_cache_hit"), 1)


if __name__ == "__main__":
    unittest.main()
