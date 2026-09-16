"""Independent exact-arithmetic and file-contract checks for the WMMA probe."""
from fractions import Fraction
import importlib.util
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("wmma_error_probe", ROOT / "tools/wmma_error_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def rational(word, fraction_bits, exponent_bits, bias):
    fraction = word & ((1 << fraction_bits) - 1)
    exponent = (word >> fraction_bits) & ((1 << exponent_bits) - 1)
    sign = -1 if word >> (fraction_bits + exponent_bits) else 1
    mantissa = Fraction(fraction, 1 << fraction_bits) + bool(exponent)
    power = max(exponent, 1) - bias
    scale = Fraction(1 << power) if power >= 0 else Fraction(1, 1 << -power)
    return sign * mantissa * scale


class WmmaErrorProbeTests(unittest.TestCase):
    def test_exact_units_and_nearest_rounding(self):
        rng = random.Random(3958192)
        for word in range(65536):
            if (word >> 7) & 255 != 255:
                self.assertEqual(Fraction(probe.bf16_integer(word), 1 << 133), rational(word, 7, 8, 127))
        for _ in range(4096):
            word = rng.randrange(1 << 32)
            if (word >> 23) & 255 == 255:
                continue
            exact = probe.fp32_integer(word)
            self.assertEqual(Fraction(exact, 1 << 266), rational(word, 23, 8, 127))
            self.assertEqual(probe.round_fp32(exact) & 0x7FFFFFFF, word & 0x7FFFFFFF)
        # Independent halfway cases on every normal binade, both signs;
        # also the subnormal/normal transition and overflow threshold.
        for exponent in range(1, 255):
            for fraction in (0, 1, 2, 0x3FFFFF, 0x7FFFFE):
                lower = (exponent << 23) | fraction
                a, b = probe.fp32_integer(lower), probe.fp32_integer(lower + 1)
                midpoint = (a + b) // 2
                expected = lower + (lower & 1)
                for sign in (1, -1):
                    mask = 0x80000000 if sign < 0 else 0
                    self.assertEqual(probe.round_fp32(sign * midpoint), expected | mask)
                    self.assertEqual(probe.round_fp32(sign * (midpoint - 1)), lower | mask)
                    self.assertEqual(probe.round_fp32(sign * (midpoint + 1)), (lower + 1) | mask)
        self.assertEqual(probe.round_fp32((1 << 116)), 0)
        self.assertEqual(probe.round_fp32((1 << 116) + 1), 1)
        self.assertEqual(probe.round_fp32(probe.fp32_integer(0x7FFFFF) + (1 << 116)), 0x800000)
        self.assertEqual(probe.round_fp32(probe.fp32_integer(0x7F7FFFFF) + (1 << 369)), 0x7F800000)

    def test_exact_bound_counterexample_and_cancellation(self):
        encode = lambda value: struct.unpack("<I", struct.pack("<f", value))[0] >> 16
        left = [encode(-221), encode(87)] + [0] * 14
        right = [encode(-29), encode(-68)] + [0] * 14
        exact, absolute, error = probe.compare_dot(left, right, 1140228088)
        self.assertEqual(exact, 493 << 266)
        self.assertEqual(absolute, (6409 + 5916) << 266)
        self.assertEqual(error, 1 << 254)
        self.assertLessEqual(error << 25, absolute)
        self.assertGreater(error << 26, absolute)
        exact, absolute, error = probe.compare_dot([encode(1), encode(1)], [encode(1), encode(-1)], 1)
        self.assertEqual(exact, 0)
        self.assertEqual(absolute, 2 << 266)
        self.assertEqual(error, 1 << 117)

    def test_input_parser_and_analyzer(self):
        with tempfile.TemporaryDirectory(prefix="qrt-wmma-error-") as directory:
            path = Path(directory)
            source, output = path / "input.bin", path / "output.bin"
            generated = probe.generate(source, cases=1)
            self.assertEqual(generated["cells"], 12288)
            exe = path / "probe"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all", str(ROOT / "tests/native/wmma_error_probe.cpp"), "-o", str(exe)],
                           check=True, timeout=30)
            result = subprocess.run([str(exe), str(source), "--check-input"], capture_output=True, text=True, check=True, timeout=20)
            self.assertFalse(json.loads(result.stdout)["native_wmma_checked"])
            raw = source.read_bytes()
            header = probe.HEADER.unpack_from(raw)
            words = struct.unpack("<" + "H" * (header[2] * 512), raw[probe.HEADER.size:])
            cells = header[2] * 256
            expected = []
            for tile in range(header[2]):
                for row in range(16):
                    for column in range(16):
                        a = tile * 256 + row * 16
                        b = cells + tile * 256 + column * 16
                        exact = sum(probe.bf16_integer(words[a + k]) * probe.bf16_integer(words[b + k]) for k in range(16))
                        expected.append(probe.round_fp32(exact))
            output.write_bytes(probe.HEADER.pack(probe.OUTPUT_MAGIC, *header[1:]) + struct.pack("<" + "I" * cells, *expected))
            report = probe.analyze(source, output)
            self.assertEqual(sum(row["single_round_value_mismatches"] for row in report["reports"]), 0)
            self.assertEqual(sum(row["bound_misses"]["24"] for row in report["reports"]), 0)
            self.assertFalse(report["universal_coefficient_proven"])
            source.write_bytes(raw[:-1])
            result = subprocess.run([str(exe), str(source), "--check-input"], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("input span", result.stderr)
            with self.assertRaisesRegex(ValueError, "input span"):
                probe.analyze(source, output)


if __name__ == "__main__":
    unittest.main()
