#!/usr/bin/env python3
"""AOT-compile a seeded Q1024 recurrent GDN suffix for gfx1151.

This is deliberately separate from ``compile_q8192_aiter_fused_gdn.py`` so
adding the suffix route cannot perturb the retained zero-state code object.
The recurrent body is adapted from AMD AITER's MIT-licensed
``fused_recurrent.py`` kernel.

The input and output recurrent states are F32
``[value_head, key, value]``. This is the key-major layout already retained
by the q1 resident session, so a prefix hit does not transpose state.
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


QKV_ROWS = tl.constexpr(8192)
KEY_FEATURES = tl.constexpr(2048)
GATE_ROWS = tl.constexpr(32)
GATE_OUTPUT_ROWS = tl.constexpr(64)
H = tl.constexpr(16)
HV = tl.constexpr(32)
K = tl.constexpr(128)
V = tl.constexpr(128)
BK = tl.constexpr(128)
BV = tl.constexpr(8)

GRID = [16, 32, 1]


@triton.jit(do_not_specialize=["tokens"])
def _fixed_q1024_seeded_aiter_fused_gdn_kernel(
    postconv_values,
    gate_values,
    initial_state,
    outputs,
    final_state,
    gate_values_are_decay,
    tokens,
):
    """Process a recurrent suffix from resident key-major F32 state."""

    i_v = tl.program_id(0)
    i_hv = tl.program_id(1)
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    p_q = postconv_values + i_h * K + o_k
    p_k = postconv_values + KEY_FEATURES + i_h * K + o_k
    p_v = postconv_values + 2 * KEY_FEATURES + i_hv * V + o_v
    p_g = gate_values + i_hv
    p_beta = gate_values + GATE_ROWS + i_hv
    p_o = outputs + i_hv * V + o_v
    p_hi = (
        initial_state
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )

    b_h = tl.load(p_hi, mask=mask_h, other=0.0).to(tl.float32)

    for _ in range(0, tokens):
        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)
        raw_gate = tl.load(p_g).to(tl.float32)
        decay = tl.where(
            gate_values_are_decay != 0,
            raw_gate,
            tl.exp(raw_gate),
        )

        b_h *= decay
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], axis=0))
        b_h += b_k[:, None] * b_v[None, :]

        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(
            p_o,
            b_o.to(tl.bfloat16).to(tl.float32),
            mask=mask_v,
        )

        p_q += QKV_ROWS
        p_k += QKV_ROWS
        p_v += QKV_ROWS
        p_g += GATE_OUTPUT_ROWS
        p_beta += GATE_OUTPUT_ROWS
        p_o += HV * V

    p_ho = (
        final_state
        + i_hv * K * V
        + o_k[:, None] * V
        + o_v[None, :]
    )
    tl.store(p_ho, b_h, mask=mask_h)


def compile_kernel(output_dir: Path, metadata_path: Path) -> None:
    target = GPUTarget("hip", "gfx1151", 32)
    backend = make_backend(target)
    options = backend.parse_options(
        {
            "num_warps": 1,
            "num_stages": 3,
            "waves_per_eu": 0,
        }
    )
    signature = {
        "postconv_values": "*fp32",
        "gate_values": "*fp32",
        "initial_state": "*fp32",
        "outputs": "*fp32",
        "final_state": "*fp32",
        "gate_values_are_decay": "i32",
        "tokens": "i32",
    }
    compiled = triton.compile(
        ASTSource(
            fn=_fixed_q1024_seeded_aiter_fused_gdn_kernel,
            signature=signature,
        ),
        target=target,
        options=options.__dict__,
    )
    binary = compiled.asm[backend.binary_ext]
    q32768_bf16_signature = {
        "postconv_values": "*bf16",
        "gate_values": "*fp32",
        "initial_state": "*fp32",
        "outputs": "*bf16",
        "final_state": "*fp32",
        "gate_values_are_decay": "i32",
        "tokens": "i32",
    }
    q32768_bf16_compiled = triton.compile(
        ASTSource(
            fn=_fixed_q1024_seeded_aiter_fused_gdn_kernel,
            signature=q32768_bf16_signature,
        ),
        target=target,
        options=options.__dict__,
    )
    q32768_bf16_binary = q32768_bf16_compiled.asm[backend.binary_ext]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "q1024_seeded_aiter_fused_gdn.hsaco"
    output.write_bytes(binary)
    q32768_bf16_output = (
        output_dir / "q32768_seeded_aiter_fused_gdn_bf16.hsaco"
    )
    q32768_bf16_output.write_bytes(q32768_bf16_binary)

    metadata = {
        "schema_version": 1,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "source": "tools/compile_q1024_seeded_aiter_fused_gdn.py",
        "provenance": (
            "seeded direct-boundary adaptation of AMD AITER "
            "fused_recurrent.py (MIT)"
        ),
        "shape": {
            "batch": 1,
            "tokens": 1024,
            "qk_heads": 16,
            "value_heads": 32,
            "key_dim": 128,
            "value_dim": 128,
            "block_k": 128,
            "block_v": 8,
            "postconv_dtype": "float32 cells at native boundary",
            "gate_dtype": "float32",
            "output_dtype": "bfloat16-rounded float32 cells",
            "initial_state_dtype": "float32",
            "initial_state_layout": "value_head_key_value",
            "final_state_dtype": "float32",
            "final_state_layout": "value_head_key_value",
            "query_scale": 1.0,
            "use_g": True,
            "use_gk": False,
            "use_gv": False,
            "use_qk_l2norm_in_kernel": False,
            "is_beta_headwise": True,
            "use_initial_state": True,
            "store_final_state": True,
        },
        "kernel": {
            "file": output.name,
            "symbol": compiled.name,
            "sha256": hashlib.sha256(binary).hexdigest(),
            "bytes": len(binary),
            "grid": GRID,
            "threads": compiled.metadata.num_warps * 32,
            "num_warps": compiled.metadata.num_warps,
            "num_stages": 3,
            "dynamic_shared_bytes": compiled.metadata.shared,
            "abi": list(signature),
            "compiled_hash": compiled.hash,
        },
        "q32768_seeded_bf16_kernel": {
            "file": q32768_bf16_output.name,
            "symbol": q32768_bf16_compiled.name,
            "sha256": hashlib.sha256(q32768_bf16_binary).hexdigest(),
            "bytes": len(q32768_bf16_binary),
            "grid": GRID,
            "threads": q32768_bf16_compiled.metadata.num_warps * 32,
            "num_warps": q32768_bf16_compiled.metadata.num_warps,
            "num_stages": 3,
            "dynamic_shared_bytes": q32768_bf16_compiled.metadata.shared,
            "abi": list(q32768_bf16_signature),
            "compiled_hash": q32768_bf16_compiled.hash,
            "tokens_per_chunk": 32768,
            "correctness_authority": (
                "structural q262144 temporal chunk; preserves F32 state "
                "between exact BF16 chunks"
            ),
        },
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
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
