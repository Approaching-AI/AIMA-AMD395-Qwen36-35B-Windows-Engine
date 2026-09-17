"""Extract the unchanged production convolution for a standalone HIP comparison."""
import argparse
from pathlib import Path
import re

FUNCTIONS = (
    "device_bf16_to_float", "device_float_to_bf16", "device_bf16_round_to_float",
    "device_silu_bf16_from_f32_acc", "device_triton_silu_bf16_from_f32_acc",
    "cuda_triton_silu_correction_hash", "device_cuda_triton_silu_bf16_from_f32_acc",
    "device_pack_bf16_pair", "device_triton_bf16_product", "device_bf16_product_to_float",
    "selected_conv_qkv_window_kernel",
)


def function(source, name):
    match = re.search(r"^(?:__host__ )?__device__ [^\n]+\b" + re.escape(name) + r"\(", source, re.M)
    if match is None:
        match = re.search(r"^__global__ void " + re.escape(name) + r"\(", source, re.M)
    if match is None:
        raise ValueError(f"Missing function: {name}")
    opening = source.index("{", match.start())
    depth = 1
    position = opening + 1
    while depth:
        depth += (source[position] == "{") - (source[position] == "}")
        position += 1
    return source[match.start():position]


def extract(source):
    constants = []
    for name in ("kQkvRows", "kConvKernel", "kCudaTritonSiluCorrectionTableElements", "kCudaTritonSiluCorrectionTableMask"):
        match = re.search(r"^constexpr unsigned int " + name + r"\s*=.*?;", source, re.M | re.S)
        if match is None:
            raise ValueError(f"Missing constant: {name}")
        constants.append(match.group())
    return "#pragma once\n" + "\n\n".join(constants + [function(source, name) for name in FUNCTIONS]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x", encoding="utf-8") as output:
        output.write(extract(args.source.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
