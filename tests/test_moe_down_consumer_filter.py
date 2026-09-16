"""Exercise the actual asynchronous MoE consumer filter owner and dependencies."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class MoeDownConsumerFilterTests(unittest.TestCase):
    def test_actual_async_owner_failures_dependencies_and_guards(self):
        source = (ROOT / "native/providers/triton_moe/down_consumer_filter.h").read_text()
        constants = source[source.index("constexpr unsigned tokens"):source.index("__device__ inline bool omitted")]
        owner = source[source.index("class Owner {"):]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "moe_down_filter_actual_owner.h").write_text(
                "namespace qrt_moe_down_consumer_filter {\n" + constants + owner)
            executable = temp / "owner-host"
            build = subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-ffp-contract=off",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                "-I", str(temp), "-I", str(ROOT / "native/providers"),
                str(ROOT / "tests/native/moe_down_consumer_filter_owner_host.cpp"), "-o", str(executable)
            ], text=True, capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], text=True, capture_output=True, timeout=30,
                                 env={**os.environ, "ASAN_OPTIONS": "quarantine_size_mb=16"})
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("shared_event_ordered=1 drain_before_free=1 pass=1", run.stdout)
            print(run.stdout.strip())


if __name__ == "__main__":
    unittest.main()
