"""Exercise interval certificates against the original combine and actual owner."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MoeDownConsumerAuditTests(unittest.TestCase):
    def test_actual_async_owner_failures_and_guards(self):
        source = (ROOT / "native/providers/triton_moe/down_consumer_audit.h").read_text()
        constants = source[source.index("constexpr unsigned tokens"):source.index("__device__ inline void tally")]
        owner = source[source.index("class Owner {"):]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "moe_down_actual_owner.h").write_text(
                "namespace qrt_moe_down_consumer_audit {\n"
                "namespace interval = qrt_moe_down_consumer;\n" + constants + owner)
            executable = temp / "owner-host"
            build = subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-ffp-contract=off",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                "-I", str(temp), "-I", str(ROOT / "native/providers"),
                str(ROOT / "tests/native/moe_down_consumer_owner_host.cpp"), "-o", str(executable)
            ], text=True, capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], text=True, capture_output=True, timeout=30,
                                 env={**os.environ, "ASAN_OPTIONS": "quarantine_size_mb=16"})
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("moe_down_owner_host", run.stdout)
            print(run.stdout.strip())

    def test_original_combine_interval_corners(self):
        provider = (ROOT / "native/providers/triton_moe/qrt_triton_moe_q8192_provider.cpp").read_text()
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            (temp / "moe_down_actual_combine.h").write_text(
                function(provider, "void full_v3_fused_combine_residual_kernel(") + "\n")
            executable = temp / "interval-host"
            build = subprocess.run([
                os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-ffp-contract=off",
                "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                "-I", str(temp), "-I", str(ROOT / "native/providers"),
                str(ROOT / "tests/native/moe_down_consumer_interval_host.cpp"), "-o", str(executable)
            ], text=True, capture_output=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([str(executable)], text=True, capture_output=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertIn("moe_down_interval_host", run.stdout)
            print(run.stdout.strip())


if __name__ == "__main__":
    unittest.main()
