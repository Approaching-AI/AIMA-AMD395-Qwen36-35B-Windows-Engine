#!/usr/bin/env python3
"""Compile the retained q8191/q8192 routed gate/up finalizer for gfx1151.

The retained GB10-valid q8192 route rounds the gate and up projections to
BF16, evaluates SiLU and the up multiplication in FP32, and rounds only the
final product to BF16.  This compatibility kernel is selected only for the
logical 8191/8192 product shapes; the arbitrary-length finalizer keeps its
separate model-visible BF16 SiLU endpoint.
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
INTERMEDIATE_VALUE = 512
ACTIVATED_ELEMENTS_VALUE = (
    TOKENS_VALUE * TOP_K_VALUE * INTERMEDIATE_VALUE
)
BLOCK_SIZE_VALUE = 1024

ACTIVATED_ELEMENTS = tl.constexpr(ACTIVATED_ELEMENTS_VALUE)
BLOCK_SIZE = tl.constexpr(BLOCK_SIZE_VALUE)


@triton.jit
def _zero_correction_gate_finalize_kernel(
    gate_up_native_f32,
    activated_bf16,
    element_count,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < element_count
    offsets_i64 = offsets.to(tl.int64)
    native_gate = tl.load(
        gate_up_native_f32 + offsets_i64,
        mask=mask,
        other=0.0,
    )
    native_up = tl.load(
        gate_up_native_f32 + ACTIVATED_ELEMENTS + offsets_i64,
        mask=mask,
        other=0.0,
    )
    gate_rounded = native_gate.to(tl.bfloat16).to(tl.float32)
    up_rounded = native_up.to(tl.bfloat16).to(tl.float32)
    silu = gate_rounded / (1.0 + tl.exp(-gate_rounded))
    tl.store(
        activated_bf16 + offsets_i64,
        (silu * up_rounded).to(tl.bfloat16),
        mask=mask,
    )


def compile_kernel(output_dir: Path, metadata_path: Path) -> None:
    target = GPUTarget("hip", "gfx1151", 32)
    backend = make_backend(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_options = {
        "num_warps": 8,
        "num_stages": 1,
        "waves_per_eu": 0,
    }
    options = backend.parse_options(launch_options)
    compiled = triton.compile(
        ASTSource(
            fn=_zero_correction_gate_finalize_kernel,
            signature={
                "gate_up_native_f32": "*fp32",
                "activated_bf16": "*bf16",
                "element_count": "i32",
            },
        ),
        target=target,
        options=options.__dict__,
    )
    binary = compiled.asm[backend.binary_ext]
    output = (
        output_dir
        / "q8192_triton_0626_zero_correction_gate_finalize_"
        "retained_fused_f32_silu.hsaco"
    )
    output.write_bytes(binary)
    metadata = {
        "schema_version": 1,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "arithmetic": (
            "BF16 gate/up endpoints followed by fused-FP32 SiLU/up "
            "multiplication and a final BF16 endpoint"
        ),
        "shape": {
            "logical_tokens": [TOKENS_VALUE - 1, TOKENS_VALUE],
            "top_k": TOP_K_VALUE,
            "intermediate": INTERMEDIATE_VALUE,
            "max_elements": ACTIVATED_ELEMENTS_VALUE,
            "block_size": BLOCK_SIZE_VALUE,
        },
        "launch_options": launch_options,
        "file": output.name,
        "symbol": compiled.name,
        "sha256": hashlib.sha256(binary).hexdigest(),
        "bytes": len(binary),
        "max_grid": [
            (ACTIVATED_ELEMENTS_VALUE + BLOCK_SIZE_VALUE - 1)
            // BLOCK_SIZE_VALUE,
            1,
            1,
        ],
        "threads": compiled.metadata.num_warps * 32,
        "dynamic_shared_bytes": compiled.metadata.shared,
        "compiled_hash": compiled.hash,
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
    args = parser.parse_args()
    compile_kernel(args.output_dir, args.metadata)


if __name__ == "__main__":
    main()
