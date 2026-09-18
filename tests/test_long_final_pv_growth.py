"""Check the long PV growth denominator and outward FP32 inflation exactly."""
from fractions import Fraction
import json
import struct


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def upper(x):
    word = struct.unpack("I", struct.pack("f", x))[0]
    return struct.unpack("f", struct.pack("I", word + 2))[0]


epsilon = Fraction(1, 1 << 17)
for groups in range(514, 16547, 2):
    calls = groups * 3 // 2
    denominator = 1 - calls * epsilon
    assert 0 < calls * epsilon < Fraction(19, 100)
    computed_denominator = f32(1 - f32(calls * 2.0**-17))
    assert Fraction(computed_denominator) == denominator
    inflation = upper(f32(1 / computed_denominator))
    assert Fraction(inflation) >= 1 / denominator
    assert 64 * groups > 16 * calls / denominator

# The binomial proof applies to every integer length. Check exact rational
# endpoint identities at the attention boundaries as an implementation audit.
for groups in (514, 1024, 2048, 4096, 8192, 16384, 16546):
    calls = groups * 3 // 2
    assert (1 + epsilon) ** calls * (1 - calls * epsilon) <= 1

print(json.dumps(dict(long_even_lengths=8017, maximum_groups=16546,
                     maximum_calls=24819, exact_denominators=True,
                     outward_inflation=True, additive_floor_covered=True)))
