"""Exact rational error recurrence, conditional on the observed DOT2 model."""
from fractions import Fraction
import unittest


def power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)


def exponent(value):
    candidate = value.numerator.bit_length() - value.denominator.bit_length()
    return candidate - int(value < power(candidate))


def envelope(headroom):
    return integer_envelope(255 << headroom)


def integer_envelope(limit):
    product = limit ** 2
    product_quantum = power(2 * (limit.bit_length() - 1) - 24)
    error = Fraction(0)
    steps = []
    for pair in range(1, 9):
        incoming = 2 * (pair - 1) * product + error
        carry_quantum = power(exponent(incoming) - 26) if incoming else Fraction(0)
        # At product-dominated alignment, carry truncation plus both product
        # truncations is bounded by3q. At carry-dominated alignment the FP32
        # carry lies on the alignment lattice and the bound is2q. The initial
        # positive zero has no carry truncation or negative correction.
        alignment = max((2 if pair == 1 else 3) * product_quantum, 2 * carry_quantum)
        pre_round = incoming + 2 * product + alignment
        rounding = power(exponent(pre_round) - 24)
        error += alignment + rounding
        steps.append(error)
    return steps


class ConditionalIntegerBoundTests(unittest.TestCase):
    def test_centered_signed16_combination_domain(self):
        self.assertEqual(integer_envelope(256)[-1], Fraction(85, 256))
        self.assertLess(integer_envelope(256)[-1], Fraction(1, 2))
        self.assertEqual(integer_envelope(255)[-1], Fraction(245, 1024))

    def test_h4_mod256_domain_and_wider_limits(self):
        self.assertEqual(envelope(4), [Fraction(x, 4) for x in (6, 17, 37, 61, 101, 149, 197, 245)])
        self.assertLess(envelope(4)[-1], 128)
        self.assertEqual(envelope(5)[-1], 245)
        self.assertEqual(envelope(6)[-1], 980)
        # Failure to certify the wider domains is not a counterexample.
        self.assertGreaterEqual(envelope(5)[-1], 128)
        self.assertGreaterEqual(envelope(6)[-1], 128)


if __name__ == "__main__":
    unittest.main()
