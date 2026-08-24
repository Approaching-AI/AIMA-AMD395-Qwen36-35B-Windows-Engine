#!/usr/bin/env python3
"""Compile expert-sorted conditional-exact routed gate/up for gfx1151.

The accepted conditional arithmetic is unchanged.  Program order follows the
provider's padded expert-sorted route list so consecutive programs reuse one
expert's weight rows, while every result is stored at its original route
index.  Dynamic launches pass the actual padded-route count on device.
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
BLOCK_M_VALUE = 64
ACTIVATED_ELEMENTS_VALUE = (
    TOKENS_VALUE * TOP_K_VALUE * INTERMEDIATE_VALUE
)

TOP_K = tl.constexpr(TOP_K_VALUE)
ROUTES = tl.constexpr(TOKENS_VALUE * TOP_K_VALUE)
HIDDEN = tl.constexpr(HIDDEN_VALUE)
INTERMEDIATE = tl.constexpr(INTERMEDIATE_VALUE)
GATE_UP_ELEMENTS_PER_EXPERT = tl.constexpr(2 * 512 * 2048)
GATE_ELEMENTS_PER_EXPERT = tl.constexpr(512 * 2048)
BLOCK_H = tl.constexpr(HIDDEN_VALUE)
ACTIVATED_ELEMENTS = tl.constexpr(ACTIVATED_ELEMENTS_VALUE)


@triton.jit
def _sorted_conditional_gate_up_silu_kernel(
    gate_up_native_f32,
    input_bf16,
    gate_up_bf16,
    sorted_routes,
    total_post_pad,
    topk_ids,
    activated_bf16,
    gate_midpoint_radius,
    up_midpoint_radius,
    gate_low_exponent_threshold,
    up_low_exponent_threshold,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    row_groups = INTERMEDIATE // ROWS_PER_PROGRAM
    sorted_position = pid // row_groups
    row_group = pid - sorted_position * row_groups
    padded_routes = tl.load(total_post_pad)
    if sorted_position < padded_routes:
        route_index = tl.load(sorted_routes + sorted_position)
        if route_index >= 0:
            if route_index < ROUTES:
                route_index_i64 = route_index.to(tl.int64)
                token = route_index_i64 // TOP_K
                expert = tl.load(topk_ids + route_index_i64).to(tl.int64)
                offsets = tl.arange(0, BLOCK_H).to(tl.int64)
                inputs = tl.load(
                    input_bf16 + token * HIDDEN + offsets
                ).to(tl.float32)
                expert_base = expert * GATE_UP_ELEMENTS_PER_EXPERT
                for local_row in range(ROWS_PER_PROGRAM):
                    row = row_group * ROWS_PER_PROGRAM + local_row
                    output_index = route_index_i64 * INTERMEDIATE + row
                    native_gate = tl.load(gate_up_native_f32 + output_index)
                    native_up = tl.load(
                        gate_up_native_f32
                        + ACTIVATED_ELEMENTS
                        + output_index
                    )
                    gate_bits = native_gate.to(tl.uint32, bitcast=True)
                    up_bits = native_up.to(tl.uint32, bitcast=True)
                    gate_low_bits = gate_bits & 0xFFFF
                    up_low_bits = up_bits & 0xFFFF
                    gate_distance = tl.abs(
                        gate_low_bits.to(tl.int32) - 0x8000
                    )
                    up_distance = tl.abs(
                        up_low_bits.to(tl.int32) - 0x8000
                    )
                    gate_exponent = (gate_bits >> 23) & 0xFF
                    up_exponent = (up_bits >> 23) & 0xFF
                    gate_candidate = (
                        ((gate_midpoint_radius != 0) &
                         (gate_distance <= gate_midpoint_radius)) |
                        ((gate_low_exponent_threshold != 0) &
                         (gate_exponent <= gate_low_exponent_threshold))
                    )
                    up_candidate = (
                        ((up_midpoint_radius != 0) &
                         (up_distance <= up_midpoint_radius)) |
                        ((up_low_exponent_threshold != 0) &
                         (up_exponent <= up_low_exponent_threshold))
                    )
                    gate_value = native_gate
                    if gate_candidate:
                        gate = tl.load(
                            gate_up_bf16
                            + expert_base
                            + row * HIDDEN
                            + offsets
                        ).to(tl.float32)
                        gate_value = tl.sum(inputs * gate, axis=0)
                    up_value = native_up
                    if up_candidate:
                        up = tl.load(
                            gate_up_bf16
                            + expert_base
                            + GATE_ELEMENTS_PER_EXPERT
                            + row * HIDDEN
                            + offsets
                        ).to(tl.float32)
                        up_value = tl.sum(inputs * up, axis=0)
                    gate_rounded = gate_value.to(tl.bfloat16).to(tl.float32)
                    up_rounded = up_value.to(tl.bfloat16).to(tl.float32)
                    # Match the BF16 tensor boundary produced by F.silu in
                    # Qwen3NextMLP before the BF16 up projection multiply.
                    silu = (
                        gate_rounded / (1.0 + tl.exp(-gate_rounded))
                    )
                    silu_bits = silu.to(tl.uint32, bitcast=True)
                    silu_rounded_bits = (
                        silu_bits
                        + 0x7FFF
                        + ((silu_bits >> 16) & 1)
                    ) & 0xFFFF0000
                    silu = silu_rounded_bits.to(
                        tl.float32, bitcast=True
                    )
                    tl.store(
                        activated_bf16 + output_index,
                        (silu * up_rounded).to(tl.bfloat16),
                    )


def compile_variants(
    output_dir: Path,
    metadata_path: Path,
    rows_values: list[int],
) -> None:
    target = GPUTarget("hip", "gfx1151", 32)
    backend = make_backend(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "gate_up_native_f32": "*fp32",
        "input_bf16": "*bf16",
        "gate_up_bf16": "*bf16",
        "sorted_routes": "*i32",
        "total_post_pad": "*i32",
        "topk_ids": "*i32",
        "activated_bf16": "*bf16",
        "gate_midpoint_radius": "i32",
        "up_midpoint_radius": "i32",
        "gate_low_exponent_threshold": "i32",
        "up_low_exponent_threshold": "i32",
    }
    launch_options = {
        "num_warps": 8,
        "num_stages": 1,
        "waves_per_eu": 0,
    }
    records = []
    for rows in rows_values:
        if rows < 1 or INTERMEDIATE_VALUE % rows != 0:
            raise ValueError(f"rows must divide {INTERMEDIATE_VALUE}: {rows}")
        options = backend.parse_options(launch_options)
        compiled = triton.compile(
            ASTSource(
                fn=_sorted_conditional_gate_up_silu_kernel,
                signature=signature,
                constexprs={"ROWS_PER_PROGRAM": rows},
            ),
            target=target,
            options=options.__dict__,
        )
        binary = compiled.asm[backend.binary_ext]
        output = output_dir / (
            "q8192_triton_0626_sorted_conditional_exact_gate_rows"
            f"{rows}.hsaco"
        )
        output.write_bytes(binary)
        max_sorted_routes = (
            TOKENS_VALUE * TOP_K_VALUE
            + 256 * (BLOCK_M_VALUE - 1)
        )
        grid = (
            (max_sorted_routes + BLOCK_M_VALUE - 1) // BLOCK_M_VALUE
            * BLOCK_M_VALUE
            * (INTERMEDIATE_VALUE // rows)
        )
        records.append(
            {
                "rows_per_program": rows,
                "file": output.name,
                "symbol": compiled.name,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "bytes": len(binary),
                "max_grid": [grid, 1, 1],
                "threads": compiled.metadata.num_warps * 32,
                "dynamic_shared_bytes": compiled.metadata.shared,
                "compiled_hash": compiled.hash,
            }
        )
    metadata = {
        "schema_version": 1,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "arithmetic": "accepted conditional K2048 reductions in sorted route order",
        "shape": {
            "tokens": TOKENS_VALUE,
            "top_k": TOP_K_VALUE,
            "hidden": HIDDEN_VALUE,
            "intermediate": INTERMEDIATE_VALUE,
            "block_m": BLOCK_M_VALUE,
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
    parser.add_argument("--rows", type=int, nargs="+", default=[32, 64, 128])
    args = parser.parse_args()
    compile_variants(args.output_dir, args.metadata, args.rows)


if __name__ == "__main__":
    main()
