import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import capture_sm121_rsqrt_table as capture


class RsqrtCaptureTests(unittest.TestCase):
    def test_exponent_scaling_including_negative_odd_exponents(self):
        # A power of two has an exact mantissa; use independently computed
        # reciprocal roots, covering every normal input exponent.
        def bits(value):
            return struct.unpack("<I", struct.pack("<f", value))[0]
        for exponent in range(-126, 128):
            base = bits(1.0 / math.sqrt(2.0 if exponent % 2 else 1.0))
            actual = capture.scaled_result((exponent + 127) << 23, base)
            self.assertEqual(actual, bits(1.0 / math.sqrt(math.ldexp(1.0, exponent))))

    def test_ftz_zero_infinity_and_rejected_domain(self):
        for value in (0, 1, 0x007FFFFF):
            self.assertEqual(capture.scaled_result(value, 0), 0x7F800000)
        self.assertEqual(capture.scaled_result(0x7F800000, 0), 0)
        for value in (-1, 0x7F800001, 0x7FC00000, 0x80000000):
            with self.assertRaises(ValueError):
                capture.scaled_result(value, 0)

    def test_preflight_never_imports_gpu_and_host_mismatch_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("torch", "triton", "numpy"):
                (root / (name + ".py")).write_text('raise RuntimeError("GPU imports forbidden")\n')
            command = [sys.executable, str(ROOT / "scripts/capture_sm121_rsqrt_table.py"),
                       "--source-commit", "0" * 40, "--output-dir", str(root / "preflight")]
            environment = dict(os.environ, PYTHONPATH=str(root))
            result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, result.stderr)
            record = json.loads(result.stdout)
            for key in ("completed", "kernel_executed", "model_or_prompt_inputs", "inference_acceptance"):
                self.assertFalse(record[key])
            again = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=10)
            self.assertIn("never overwrite", again.stderr)
            command[-1] = str(root / "wrong-host")
            result = subprocess.run(command + ["--execute", "--expected-host", "not-this-host"],
                                    env=environment, capture_output=True, text=True, timeout=10)
            self.assertIn("execution host mismatch", result.stderr)
            self.assertFalse((root / "wrong-host").exists())


if __name__ == "__main__":
    unittest.main()
