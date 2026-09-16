#!/usr/bin/env python3
"""Generate native zero-C WMMA cases and measure error with exact integers.

Every finite BF16 product is an integer multiple of 2**-266. Python integers
therefore compare native FP32 errors with L1 bounds without a floating oracle.
This is a finite hardware diagnostic, never a proof of a universal coefficient.
"""
import argparse
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys

INPUT_MAGIC = 0x574D4931
OUTPUT_MAGIC = 0x574D4F31
HEADER = struct.Struct("<8I")
FAMILIES = ("positive_same_exponent", "signed_same_exponent", "moderate_span",
            "eligible_full_span", "two_term_cancellation", "rounding_boundary",
            "exact_pair_cancellation", "near_pair_cancellation", "small_tail",
            "alternating_exponents", "eligible_endpoints", "negative_products")
POWERS = tuple(range(19, 28))


def mix(value):
    value &= 0xFFFFFFFF
    value = ((value ^ (value >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    value = ((value ^ (value >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    return value ^ (value >> 16)


def permute(order, k):
    return (k, 15 - k, (k % 8) * 2 + k // 8, (k * 5 + 3) % 16)[order]


def operand(family, sample, row, k, right):
    index = k // 2 if family in (6, 7) else k
    random = mix(0x3958192 ^ sample * 7919 ^ row * 104729 ^ index * 65537 ^ right * 99991)
    mantissa, sign = random & 127, (random >> 7) & 1
    exponent = 127
    if family == 0:
        sign = 0
    elif family == 2:
        exponent = 119 + (random >> 8) % 17
    elif family == 3:
        exponent = 80 + (random >> 8) % 95
    elif family == 4:
        if k >= 2:
            return 0
        exponent = 115 + sample % 25
        sign = int(right and k == 1)
    elif family == 5:
        sign = 0
        if k:
            exponent = 127 if right else 101 + sample % 5
            mantissa = (sample + row + k) & 127
    elif family in (6, 7):
        exponent = 112 + (random >> 8) % 31
        sign = int(right and k % 2 == 1)
        if family == 7 and right and k % 2:
            mantissa = (mantissa + 1) & 127
    elif family == 8:
        exponent = 127 if not k or right else 96 + sample % 32
        sign = int(right and k > 0 and row % 2)
    elif family == 9:
        exponent = 127 - (k % 4) * (1 + sample % 12)
        sign = int(right and k % 2)
    elif family == 10:
        exponent = 80 if (k + sample) % 2 else 174
    elif family == 11:
        exponent = 112 + (random >> 8) % 31
        sign = int(right)
    return (sign << 15) | (exponent << 7) | mantissa


def words_bytes(words):
    result = array("H", words)
    if sys.byteorder != "little":
        result.byteswap()
    return result.tobytes()


def generate(path, cases=64):
    if not 1 <= cases <= 1024:
        raise ValueError("cases must be within 1..1024")
    tiles = len(FAMILIES) * cases * 4
    header = (INPUT_MAGIC, 1, tiles, len(FAMILIES), cases, 4, 16, 1)
    with Path(path).open("xb") as destination:
        destination.write(HEADER.pack(*header))
        for right in (False, True):
            for family in range(len(FAMILIES)):
                for sample in range(cases):
                    for order in range(4):
                        destination.write(words_bytes(
                            operand(family, sample, row, permute(order, k), right)
                            for row in range(16) for k in range(16)))
    return {"tiles": tiles, "cells": tiles * 256, "input_sha256": digest(path)}


def digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def bf16_integer(word):
    """Value in units of 2**-133, including BF16 subnormals."""
    exponent, fraction = (word >> 7) & 255, word & 127
    if exponent == 255:
        raise ValueError("non-finite BF16")
    magnitude = fraction if not exponent else (128 + fraction) << (exponent - 1)
    return -magnitude if word & 0x8000 else magnitude


def fp32_integer(word):
    """Value in the common product units of 2**-266."""
    exponent, fraction = (word >> 23) & 255, word & 0x7FFFFF
    if exponent == 255:
        raise ValueError("non-finite native result")
    magnitude = fraction << 117 if not exponent else (0x800000 + fraction) << (exponent + 116)
    return -magnitude if word & 0x80000000 else magnitude


def round_fp32(value):
    """Round an exact signed 2**-266 integer to FP32, ties to even."""
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    if not value:
        return sign
    shift = max(117, value.bit_length() - 24)
    significand, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    significand += remainder > half or (remainder == half and significand & 1)
    if significand >= 1 << 24:
        significand >>= 1
        shift += 1
    if significand < 1 << 23:
        return sign | significand
    exponent = shift - 116
    if exponent >= 255:
        return sign | 0x7F800000
    return sign | (exponent << 23) | (significand & 0x7FFFFF)


def compare_dot(left, right, actual):
    products = [bf16_integer(a) * bf16_integer(b) for a, b in zip(left, right)]
    exact, absolute = sum(products), sum(map(abs, products))
    error = abs(fp32_integer(actual) - exact)
    return exact, absolute, error


def analyze(input_path, output_path):
    raw = Path(input_path).read_bytes()
    header = HEADER.unpack_from(raw)
    magic, version, tiles, families, cases, permutations, width, dtype = header
    if (magic, version, families, permutations, width, dtype) != (INPUT_MAGIC, 1, len(FAMILIES), 4, 16, 1) or not 1 <= cases <= 1024 or tiles != families * cases * permutations:
        raise ValueError("input header")
    if len(raw) != HEADER.size + tiles * 1024:
        raise ValueError("input span")
    words = array("H")
    words.frombytes(raw[HEADER.size:])
    if sys.byteorder != "little":
        words.byteswap()
    result = Path(output_path).read_bytes()
    if HEADER.unpack_from(result) != (OUTPUT_MAGIC, *header[1:]) or len(result) != HEADER.size + tiles * 1024:
        raise ValueError("output header or span")
    outputs = array("I")
    outputs.frombytes(result[HEADER.size:])
    if sys.byteorder != "little":
        outputs.byteswap()
    cells = tiles * 256
    reports = []
    for family, name in enumerate(FAMILIES):
        misses = {str(p): 0 for p in POWERS}
        counterexamples = {}
        mismatches = 0
        max_error, max_absolute = 0, 1
        for tile in range(family * cases * 4, (family + 1) * cases * 4):
            # Native output (row, column) is dot(A[row], B[column]).
            left_rows = [words[tile * 256 + row * 16:tile * 256 + (row + 1) * 16] for row in range(16)]
            right_rows = [words[cells + tile * 256 + row * 16:cells + tile * 256 + (row + 1) * 16] for row in range(16)]
            decoded_left = [list(map(bf16_integer, row)) for row in left_rows]
            decoded_right = [list(map(bf16_integer, row)) for row in right_rows]
            for row in range(16):
                for column in range(16):
                    cell = tile * 256 + row * 16 + column
                    products = [a * b for a, b in zip(decoded_left[row], decoded_right[column])]
                    exact, absolute = sum(products), sum(map(abs, products))
                    actual = outputs[cell]
                    error = abs(fp32_integer(actual) - exact)
                    mismatches += fp32_integer(actual) != fp32_integer(round_fp32(exact))
                    if error * max_absolute > max_error * absolute:
                        max_error, max_absolute = error, absolute
                    for p in POWERS:
                        if (error << p) > absolute:
                            misses[str(p)] += 1
                            if str(p) not in counterexamples:
                                counterexamples[str(p)] = {
                                    "cell": cell, "left": list(left_rows[row]), "right": list(right_rows[column]),
                                    "native_bits": actual, "single_round_bits": round_fp32(exact),
                                    "exact_integer": str(exact), "absolute_integer": str(absolute), "error_integer": str(error)}
        reports.append({"family": name, "cells": cases * 4 * 256,
                        "single_round_value_mismatches": mismatches, "bound_misses": misses,
                        "maximum_error_divided_by_l1_diagnostic_only": max_error / max_absolute if max_absolute else None,
                        "counterexamples": counterexamples})
    return {"kind": "exact_integer_native_wmma_error_characterization", "integer_unit_power": -266,
            "input_sha256": digest(input_path), "output_sha256": digest(output_path),
            "cells": cells, "reports": reports, "bound_decisions_use_exact_integers": True,
            "universal_coefficient_proven": False, "product_dispatch_changed": False,
            "inference_acceptance": False, "performance_acceptance": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("path", type=Path)
    generate_parser.add_argument("--cases", type=int, default=64)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument("output", type=Path)
    analyze_parser.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.action == "generate":
        print(json.dumps(generate(args.path, args.cases)))
    else:
        report = analyze(args.input, args.output)
        with args.report.open("x") as destination:
            json.dump(report, destination, indent=2, allow_nan=False)
            destination.write("\n")
        print(json.dumps({"cells": report["cells"], "bound_misses": {
            str(p): sum(row["bound_misses"][str(p)] for row in report["reports"]) for p in POWERS}}))


if __name__ == "__main__":
    main()
