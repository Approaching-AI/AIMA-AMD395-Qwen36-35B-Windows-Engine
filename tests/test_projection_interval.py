"""Independent integer endpoints for directed addition and canonical K16 coverage."""
from pathlib import Path
import os
import random
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def integer_float(bits):
    exponent, fraction = (bits >> 23) & 255, bits & 0x7fffff
    value = fraction if not exponent else (fraction | 0x800000) << (exponent - 1)
    return -value if bits >> 31 else value


def directed_bits(value, positive):
    magnitude = abs(value)
    width = magnitude.bit_length()
    shift = max(0, width - 24)
    leading = magnitude >> shift
    encoded = leading if width <= 23 else ((width - 23) << 23) | (leading & 0x7fffff)
    if magnitude != leading << shift and ((value > 0) == positive):
        encoded += 1
    return encoded | (0x80000000 if value < 0 else 0)


class ProjectionIntervalTests(unittest.TestCase):
    def test_directed_add_and_original_groups(self):
        with tempfile.TemporaryDirectory(prefix="qrt-projection-interval-") as tmp:
            exe = str(Path(tmp) / "interval")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            "-I", str(ROOT / "native/providers/moe_accumulator"),
                            str(ROOT / "tests/native/projection_interval_selftest.cpp"), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)
            rng = random.Random(0x3958192)
            pairs = [(0, 0), (0x80000000, 0x80000000), (1, 0x80000001)]
            for _ in range(20000):
                # Include cancellation, halfway cases, signs, subnormals and
                # exponent gaps beyond FP64 precision; sums stay finite.
                a = rng.getrandbits(32) & 0x807fffff | rng.randrange(0, 246) << 23
                b = rng.getrandbits(32) & 0x807fffff | rng.randrange(0, 246) << 23
                pairs.append((a, b))
                if len(pairs) % 17 == 0:
                    pairs.append((a, a ^ 0x80000000))
            output = subprocess.run([exe, "--directed"], input="".join(f"{a} {b}\n" for a, b in pairs),
                                    capture_output=True, text=True, check=True, timeout=30).stdout.splitlines()
            self.assertEqual(len(output), len(pairs))
            for pair, line in zip(pairs, output):
                exact = sum(map(integer_float, pair))
                expected = [directed_bits(exact, False), directed_bits(exact, True)]
                actual = [int(v) for v in line.split()]
                actual = [0 if v == 0x80000000 else v for v in actual]
                self.assertEqual(actual, expected, pair)


if __name__ == "__main__":
    unittest.main()
