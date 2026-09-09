import importlib.util
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("state_exp_capture", ROOT / "scripts/fla_state_exponent_capture.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class StateExponentCaptureTests(unittest.TestCase):
    def gates(self, tokens=64):
        return struct.pack(f"<{tokens * 32}f", *[-(t % 64) * 0.01 * (h + 1) for t in range(tokens) for h in range(32)])

    def test_gate_and_decay_layout_and_separate_f32_rounding(self):
        gates = self.gates(128)
        output = probe.exponent_arguments(gates, 128)
        self.assertEqual(len(output), (128 + 2) * 32 * 4)
        actual = struct.unpack(f"<{len(output) // 4}f", output)
        values = struct.unpack("<4096f", gates)
        constant = struct.unpack("<f", struct.pack("<I", 0x3FB8AA3B))[0]
        for t in range(128):
            for head in range(32):
                last = (t // 64 + 1) * 64 - 1
                expected = probe.f32(probe.f32(values[last * 32 + head] - values[t * 32 + head]) * constant)
                self.assertEqual(actual[t * 32 + head], expected)
        for chunk in range(2):
            for head in range(32):
                self.assertEqual(actual[4096 + chunk * 32 + head], probe.f32(values[((chunk + 1) * 64 - 1) * 32 + head] * constant))
        self.assertTrue(all(value == 0 for value in actual[63 * 32:64 * 32]))

    def test_invalid_bounds_shapes_and_gates_fail_on_cpu(self):
        for tokens in (0, 63, 65, 1025, True):
            with self.assertRaisesRegex(ValueError, "complete 64-token"):
                probe.exponent_arguments(b"", tokens)
        with self.assertRaisesRegex(ValueError, "byte count"):
            probe.exponent_arguments(b"", 64)
        for value in (float("nan"), float("inf"), 0.1):
            with self.assertRaisesRegex(ValueError, "finite and nonpositive"):
                probe.exponent_arguments(struct.pack("<f", value) + self.gates()[4:], 64)
        data = bytearray(self.gates())
        data[32 * 4:32 * 4 + 4] = struct.pack("<f", 0.0)
        data[0:4] = struct.pack("<f", -0.5)
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            probe.exponent_arguments(bytes(data), 64)

    def test_output_requires_exact_arguments_valid_values_and_deterministic_duplicates(self):
        arguments = struct.pack("<4f", 0, -0.0, -1, -1)
        values = struct.pack("<4f", 1, 1, 0.5, 0.5)
        result = probe.validate_samples(arguments, arguments, values)
        self.assertEqual((result["elements"], result["unique_inputs"]), (4, 2))
        with self.assertRaisesRegex(ValueError, "argument or output"):
            probe.validate_samples(arguments, arguments[:-4], values)
        for invalid in (0.99, float("nan"), -0.5, 1.5):
            with self.assertRaisesRegex(ValueError, "invalid or unwritten"):
                probe.validate_samples(arguments, arguments, struct.pack("<f", invalid) + values[4:])
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            probe.validate_samples(arguments, arguments, values[:-4] + struct.pack("<f", 0.6))

    def test_prepared_argument_volume_is_bounded(self):
        self.assertEqual(len(probe.exponent_arguments(self.gates(1024), 1024)) * 2, 266240)
        with self.assertRaises((OverflowError, ValueError)):
            probe.exponent_arguments(struct.pack("<f", -3.4e38) * (64 * 32), 64)

    def test_offline_compiler_requires_gpu_invisibility_before_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "triton.py").write_text('raise RuntimeError("GPU import forbidden")\n')
            environment = dict(os.environ, PYTHONPATH=str(root))
            for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
                environment.pop(name, None)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/fla_state_exponent_capture.py"),
                                     "--output-dir", str(root / "output")], env=environment,
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CPU-only compiler requires", result.stderr)
            self.assertNotIn("GPU import forbidden", result.stderr)
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
