#!/usr/bin/env python3
"""Extract the current provider's normalization into a bounded diagnostic.

The original kernel and its helpers are copied verbatim. Alternative kernels
change only reciprocal-root evaluation or explicit endpoint rounding. This
build step loads no model, captures or numerical reference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re


def definition(source, name):
    pattern = re.compile(r"^__\w+__[^;{]*\b" + re.escape(name)
                         + r"\([^;{]*\)\s*\{", re.MULTILINE)
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        raise ValueError(f"expected one definition of {name}, got {len(matches)}")
    match = matches[0]
    depth, end = 1, match.end()
    while depth and end < len(source):
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    if depth:
        raise ValueError(f"unterminated definition: {name}")
    return source[match.start():end] + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1]
                        / "native/providers/whole_provider.cpp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.read_text()
    names = ["device_bf16_to_float", "device_float_to_bf16",
             "device_bf16_round_to_float", "device_mul_separate",
             "device_add_separate", "device_fma_f32",
             "device_sm121_rsqrt_from_gfx1151", "vllm_triton_lane8_sumsq",
             "vllm_triton_reduce_sumsq", "output_bf16_residual_postnorm_vllm_kernel"]
    functions = [definition(source, name) for name in names]
    original = functions[-1]
    reciprocal = "device_sm121_rsqrt_from_gfx1151(\n            variance,\n            gfx1151_sm121_rsqrt_correction\n        )"
    endpoint = ("values[item] * inv_shared *\n"
                "            (1.0f + device_bf16_to_float(norm_weights[col]))")
    if original.count(reciprocal) != 1 or original.count(endpoint) != 1:
        raise ValueError("live normalization expression changed; review diagnostic variants")
    variants = []
    for table, separate in ((True, False), (False, True), (True, True)):
        name = f"postnorm_table{int(table)}_separate{int(separate)}_kernel"
        variant = original.replace(names[-1], name)
        if table:
            variant = variant.replace(reciprocal,
                                      "qrt_sm121_rsqrt::evaluate(gfx1151_sm121_rsqrt_correction, variance)")
        if separate:
            variant = variant.replace(endpoint,
                                      "device_mul_separate(device_mul_separate(values[item], inv_shared), "
                                      "device_add_separate(1.0f, device_bf16_to_float(norm_weights[col])))")
        variants.append(variant)
    text = ("// Generated from the current whole provider; do not edit.\n"
            "constexpr unsigned int kThreads = 256u;\n"
            "constexpr unsigned int QRT_QWEN36_HIDDEN_SIZE = 2048u;\n"
            "constexpr float QRT_QWEN36_RMS_NORM_EPSILON = 1.0e-6f;\n"
            + "\n".join(functions + variants))
    with args.output.open("x") as stream:
        stream.write(text)
    print(json.dumps({"source": str(args.source),
                      "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
                      "generated_header": str(args.output),
                      "generated_header_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                      "verbatim_functions": names, "diagnostic_variants": 3,
                      "model_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
