from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SharedGateTests(unittest.TestCase):
    def test_real_gb10_midpoint_and_cancellation_cases(self):
        fixture = ROOT / "tests/fixtures/sm121_shared_gate.bin"
        metadata = json.loads(fixture.with_suffix(".json").read_text())
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(), metadata["sha256"])
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "shared-gate"
            subprocess.run(["c++", "-std=c++17", "-O2", "-ffp-contract=off",
                            "-Wall", "-Wextra", "-Werror",
                            str(ROOT / "tests/native/sm121_shared_gate_probe.cpp"), "-o", str(binary)],
                           check=True, capture_output=True, text=True, timeout=30)
            result = subprocess.run([str(binary), str(fixture)], capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
