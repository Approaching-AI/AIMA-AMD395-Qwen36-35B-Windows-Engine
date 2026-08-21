#!/usr/bin/env python3
"""Compile packed conditional-exact routed-down kernels for gfx1151.

The kernel keeps each native WMMA K512 accumulator unless its weighted
contribution is near a BF16 midpoint or the accumulator has a configured low
exponent.  Candidate cells repeat the retained Triton/Hopper-order FP32 dot;
serial output columns only reuse the route activation load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend


TOKENS_VALUE = 8192
TOP_K_VALUE = 8
HIDDEN_VALUE = 2048
INTERMEDIATE_VALUE = 512

HIDDEN = tl.constexpr(HIDDEN_VALUE)
INTERMEDIATE = tl.constexpr(INTERMEDIATE_VALUE)
BLOCK_I = tl.constexpr(INTERMEDIATE_VALUE)


@triton.jit
def _conditional_exact_down_kernel(
    route_outputs_f32,
    topk_weights_f32,
    topk_ids,
    activated_bf16,
    down_bf16,
    midpoint_radius,
    low_exponent_threshold,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_groups = HIDDEN // ROWS_PER_PROGRAM
    route = pid // row_groups
    row_group = pid - route * row_groups
    expert = tl.load(topk_ids + route).to(tl.int64)
    route_weight = tl.load(topk_weights_f32 + route)
    offsets = tl.arange(0, BLOCK_I).to(tl.int64)
    activated = tl.load(
        activated_bf16 + route * INTERMEDIATE + offsets
    ).to(tl.float32)
    expert_base = expert * HIDDEN * INTERMEDIATE
    for local_row in range(ROWS_PER_PROGRAM):
        row = row_group * ROWS_PER_PROGRAM + local_row
        output_index = route * HIDDEN + row
        native_down = tl.load(route_outputs_f32 + output_index)
        contribution = route_weight * native_down
        contribution_bits = contribution.to(tl.uint32, bitcast=True)
        native_bits = native_down.to(tl.uint32, bitcast=True)
        low_bits = contribution_bits & 0xFFFF
        midpoint_distance = tl.abs(low_bits.to(tl.int32) - 0x8000)
        native_exponent = (native_bits >> 23) & 0xFF
        candidate = (
            ((midpoint_radius != 0) &
             (midpoint_distance <= midpoint_radius)) |
            ((low_exponent_threshold != 0) &
             (native_exponent <= low_exponent_threshold))
        )
        down_value = native_down
        if candidate:
            weights = tl.load(
                down_bf16
                + expert_base
                + row * INTERMEDIATE
                + offsets
            ).to(tl.float32)
            down_value = tl.sum(activated * weights, axis=0)
        tl.store(route_outputs_f32 + output_index, down_value)


def compile_variants(
    output_dir: Path,
    metadata_path: Path,
    rows_values: list[int],
) -> None:
    target = GPUTarget("hip", "gfx1151", 32)
    backend = make_backend(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "route_outputs_f32": "*fp32",
        "topk_weights_f32": "*fp32",
        "topk_ids": "*i32",
        "activated_bf16": "*bf16",
        "down_bf16": "*bf16",
        "midpoint_radius": "i32",
        "low_exponent_threshold": "i32",
    }
    launch_options = {
        "num_warps": 1,
        "num_stages": 1,
        "waves_per_eu": 0,
    }
    records = []
    for rows in rows_values:
        if rows < 1 or HIDDEN_VALUE % rows != 0:
            raise ValueError(f"rows must divide {HIDDEN_VALUE}: {rows}")
        options = backend.parse_options(launch_options)
        compiled = triton.compile(
            ASTSource(
                fn=_conditional_exact_down_kernel,
                signature=signature,
                constexprs={"ROWS_PER_PROGRAM": rows},
            ),
            target=target,
            options=options.__dict__,
        )
        binary = compiled.asm[backend.binary_ext]
        output = output_dir / (
            f"q8192_triton_0626_conditional_exact_down_rows{rows}.hsaco"
        )
        output.write_bytes(binary)
        records.append(
            {
                "rows_per_program": rows,
                "file": output.name,
                "symbol": compiled.name,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "bytes": len(binary),
                "grid": [
                    TOKENS_VALUE
                    * TOP_K_VALUE
                    * (HIDDEN_VALUE // rows),
                    1,
                    1,
                ],
                "threads": compiled.metadata.num_warps * 32,
                "dynamic_shared_bytes": compiled.metadata.shared,
                "compiled_hash": compiled.hash,
            }
        )
    metadata = {
        "schema_version": 1,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "arithmetic": "conditional retained K512 FP32 down reduction",
        "shape": {
            "tokens": TOKENS_VALUE,
            "top_k": TOP_K_VALUE,
            "hidden": HIDDEN_VALUE,
            "intermediate": INTERMEDIATE_VALUE,
        },
        "launch_options": launch_options,
        "variants": records,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--rows", type=int, nargs="+", default=[4, 8, 16, 32]
    )
    args = parser.parse_args()
    compile_variants(args.output_dir, args.metadata, args.rows)


if __name__ == "__main__":
    main()
