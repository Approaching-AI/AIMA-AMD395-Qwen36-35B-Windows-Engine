#!/usr/bin/env python3
"""AOT-compile AMD AITER's fused recurrent GDN for the q8192 product shape.

Python and Triton are build-only dependencies.  The resulting gfx1151 code
object is loaded by the Windows HIP module API.  This fixed-shape adaptation
consumes the native runtime's resident F32 boundaries directly:

* postconv: [8192, 8192], q +0, k +2048, v +4096;
* gate: [8192, 64], log-g/decay +0, beta +32;
* output: [8192, 4096], BF16-rounded values stored in F32 cells; and
* final state: [32, 128, 128] in value-head/value/key order.

The recurrent body is adapted from AMD AITER's MIT-licensed
``fused_recurrent.py`` kernel.  Q is already normalized and scaled by the
upstream postconv boundary, K/V and beta are F32 cells holding BF16-rounded
values, so no input conversion or additional query scaling is performed.
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
def _fixed_q8192_aiter_fused_gdn_kernel(
    postconv_values,
    gate_values,
    outputs,
    final_state,
    gate_values_are_decay,
    tokens,
):
    """Fixed B1/T8192/H16/HV32/K128/V128 recurrent forward kernel.

    One program owns eight adjacent value lanes for one value head.  Each
    value head reuses one of the sixteen q/k heads (GVA ratio two).  The state
    is accumulated as [key, value-lane] and stored transposed into the native
    runtime's [value, key] order.
    """

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

    b_h = tl.zeros([BK, BV], dtype=tl.float32)

    # Keep T runtime in the kernel ABI, as in AITER's do_not_specialize=["T"]
    # route.  The provider always supplies the validated fixed value 8192;
    # making it a scalar argument prevents accidental compile-time unrolling.
    for _ in range(0, tokens):
        # The q boundary is already BF16-normalized and F32-scaled by
        # 1/sqrt(128).  Loading the resident F32 cell and using scale=1
        # preserves the current native recurrence input exactly.
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
        # The current core boundary is BF16-rounded but represented as F32.
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

    # AITER's logical state is [HV, K, V].  Store it directly transposed as
    # [HV, V, K], matching state_base = value_index * K in the native core.
    p_ht = (
        final_state
        + (i_hv * V + o_v[None, :]) * K
        + o_k[:, None]
    )
    tl.store(p_ht, b_h, mask=mask_h)


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
        "outputs": "*fp32",
        "final_state": "*fp32",
        "gate_values_are_decay": "i32",
        "tokens": "i32",
    }
    compiled = triton.compile(
        ASTSource(
            fn=_fixed_q8192_aiter_fused_gdn_kernel,
            signature=signature,
        ),
        target=target,
        options=options.__dict__,
    )
    binary = compiled.asm[backend.binary_ext]
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "q8192_aiter_fused_gdn.hsaco"
    output.write_bytes(binary)
    q262144_bf16_signature = {
        "postconv_values": "*bf16",
        "gate_values": "*fp32",
        "outputs": "*bf16",
        "final_state": "*fp32",
        "gate_values_are_decay": "i32",
        "tokens": "i32",
    }
    q262144_bf16_compiled = triton.compile(
        ASTSource(
            fn=_fixed_q8192_aiter_fused_gdn_kernel,
            signature=q262144_bf16_signature,
        ),
        target=target,
        options=options.__dict__,
    )
    q262144_bf16_binary = q262144_bf16_compiled.asm[backend.binary_ext]
    q262144_bf16_output = (
        output_dir / "q262144_aiter_fused_gdn_bf16.hsaco"
    )
    q262144_bf16_output.write_bytes(q262144_bf16_binary)

    metadata = {
        "schema_version": 1,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "source": "tools/compile_q8192_aiter_fused_gdn.py",
        "provenance": (
            "fixed-shape direct-boundary adaptation of AMD AITER "
            "fused_recurrent.py (MIT)"
        ),
        "shape": {
            "batch": 1,
            "tokens": 8192,
            "qk_heads": 16,
            "value_heads": 32,
            "key_dim": 128,
            "value_dim": 128,
            "block_k": 128,
            "block_v": 8,
            "postconv_dtype": "float32 cells at native boundary",
            "gate_dtype": "float32",
            "output_dtype": "bfloat16-rounded float32 cells",
            "final_state_dtype": "float32",
            "final_state_layout": "value_head_value_key",
            "query_scale": 1.0,
            "use_g": True,
            "use_gk": False,
            "use_gv": False,
            "use_qk_l2norm_in_kernel": False,
            "is_beta_headwise": True,
            "use_initial_state": False,
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
        "q262144_structural_bf16_kernel": {
            "file": q262144_bf16_output.name,
            "symbol": q262144_bf16_compiled.name,
            "sha256": hashlib.sha256(q262144_bf16_binary).hexdigest(),
            "bytes": len(q262144_bf16_binary),
            "grid": GRID,
            "threads": q262144_bf16_compiled.metadata.num_warps * 32,
            "num_warps": q262144_bf16_compiled.metadata.num_warps,
            "num_stages": 3,
            "dynamic_shared_bytes": q262144_bf16_compiled.metadata.shared,
            "abi": list(q262144_bf16_signature),
            "compiled_hash": q262144_bf16_compiled.hash,
            "correctness_authority": (
                "structural-only; no q262144 gb10 endpoint is available"
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
