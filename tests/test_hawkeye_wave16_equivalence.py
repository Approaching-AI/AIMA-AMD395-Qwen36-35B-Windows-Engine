import random
import unittest


ZERO_EXPONENT = -133
INTERNAL_SIGNIFICAND_WIDTH = 26
INTERNAL_TO_FP32_SHIFT = 2


def multiply_bf16(left, right):
    negative = ((left ^ right) & 0x8000) != 0
    left_exponent = (left >> 7) & 0xFF
    right_exponent = (right >> 7) & 0xFF
    left_fraction = left & 0x7F
    right_fraction = right & 0x7F
    left_significand = (
        left_fraction | 0x80 if left_exponent != 0 else left_fraction
    )
    right_significand = (
        right_fraction | 0x80 if right_exponent != 0 else right_fraction
    )
    if left_exponent == 0:
        left_exponent = 1
    if right_exponent == 0:
        right_exponent = 1
    significand = (left_significand * right_significand) << 9
    if significand == 0:
        return (0, ZERO_EXPONENT, negative)
    return (
        significand,
        left_exponent + right_exponent - 254,
        negative,
    )


def normalize_group(signed_significand, max_exponent):
    negative = signed_significand < 0
    magnitude = abs(signed_significand)
    width = magnitude.bit_length()
    if width == 0:
        return (0, ZERO_EXPONENT, negative)
    exponent = (
        max_exponent + width - INTERNAL_SIGNIFICAND_WIDTH
    )
    if width > INTERNAL_SIGNIFICAND_WIDTH:
        normalized = magnitude >> (width - INTERNAL_SIGNIFICAND_WIDTH)
    else:
        normalized = magnitude << (INTERNAL_SIGNIFICAND_WIDTH - width)
    if exponent < -126:
        underflow_shift = -126 - exponent
        normalized = 0 if underflow_shift >= 64 else normalized >> underflow_shift
        exponent = -126
    normalized >>= INTERNAL_TO_FP32_SHIFT
    if normalized == 0:
        return (0, ZERO_EXPONENT, negative)
    return (normalized, exponent, negative)


def reference_group(accumulator, left, right):
    values = [accumulator]
    values.extend(multiply_bf16(l, r) for l, r in zip(left, right))
    max_exponent = max(value[1] for value in values)
    signed_significand = 0
    for significand, exponent, negative in values:
        shift = max_exponent - exponent
        if shift >= 32:
            continue
        aligned = (significand << INTERNAL_TO_FP32_SHIFT) >> shift
        signed_significand += -aligned if negative else aligned
    return normalize_group(signed_significand, max_exponent)


def wave16_group(accumulator, left, right):
    products = [multiply_bf16(l, r) for l, r in zip(left, right)]
    max_exponent = max(
        [accumulator[1]] + [product[1] for product in products]
    )
    lane_values = []
    for significand, exponent, negative in products:
        shift = max_exponent - exponent
        aligned = (
            0
            if shift >= 32
            else (significand << INTERNAL_TO_FP32_SHIFT) >> shift
        )
        lane_values.append(-aligned if negative else aligned)
    accumulator_shift = max_exponent - accumulator[1]
    accumulator_aligned = (
        0
        if accumulator_shift >= 32
        else (accumulator[0] << INTERNAL_TO_FP32_SHIFT)
        >> accumulator_shift
    )
    lane_values[0] += (
        -accumulator_aligned if accumulator[2] else accumulator_aligned
    )
    # Mirrors the exact integer __shfl_down tree used by the device helper.
    for offset in (8, 4, 2, 1):
        lane_values = [
            lane_values[index]
            + (lane_values[index + offset] if index + offset < 16 else 0)
            for index in range(16)
        ]
    return normalize_group(lane_values[0], max_exponent)


def random_finite_bf16(generator):
    while True:
        value = generator.randrange(0x10000)
        if value & 0x7F80 != 0x7F80:
            return value


class HawkeyeWave16EquivalenceTests(unittest.TestCase):
    def test_wave16_group_matches_serial_hawkeye_group_sum(self):
        generator = random.Random(0x3951096)
        edge_values = [
            0x0000,
            0x8000,
            0x0001,
            0x8001,
            0x007F,
            0x807F,
            0x0080,
            0x8080,
            0x3F80,
            0xBF80,
            0x7F7F,
            0xFF7F,
        ]
        accumulator = (0, ZERO_EXPONENT, False)
        checked = 0
        for group_index in range(50000):
            if group_index < len(edge_values):
                left = [edge_values[group_index]] * 16
                right = list(reversed(edge_values))
                right = (right * 2)[:16]
            else:
                left = [random_finite_bf16(generator) for _ in range(16)]
                right = [random_finite_bf16(generator) for _ in range(16)]
            expected = reference_group(accumulator, left, right)
            actual = wave16_group(accumulator, left, right)
            self.assertEqual(expected, actual)
            accumulator = expected
            checked += 1
            if group_index % 257 == 256:
                accumulator = (0, ZERO_EXPONENT, bool(group_index & 1))
        self.assertEqual(checked, 50000)


if __name__ == "__main__":
    unittest.main()
