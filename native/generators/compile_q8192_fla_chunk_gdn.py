#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""AOT-compile the service-shaped chunk-64 GDN pipeline for gfx1151.

The kernels are fixed to the Qwen3.6 linear-attention geometry but retain the
sequence length as a runtime ABI argument.  Kernel segments are multiples of
64; the native provider stages only a ragged final chunk with neutral values.
The generated HSACOs
are loaded by a native Windows HIP provider; Python and Triton are build-time
dependencies only.

The arithmetic and decomposition are adapted from the MIT-licensed
flash-linear-attention kernels vendored by vLLM commit 2a69949bd:

* BF16 Q/K L2 normalization with an F32 reduction;
* F32 chunk-local log-decay cumsum;
* beta-scaled K K^T and the 64x64 WY triangular solve;
* BF16 W/U recomputation;
* seeded F32 recurrent state between chunks, with BF16 chunk-start
  publication; and
* the causal chunk output using BF16 tensor-core dot products.

The native input is the runtime's F32 representation of BF16-rounded
post-convolution values.  The provider passes that boundary directly so this
pipeline, rather than the legacy postconv kernel, owns Q/K normalization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource, make_backend


QKV_ROWS = tl.constexpr(8192)
KEY_FEATURES = tl.constexpr(2048)
VALUE_OFFSET = tl.constexpr(4096)
GATE_STRIDE = tl.constexpr(64)
HG = tl.constexpr(16)
H = tl.constexpr(32)
K = tl.constexpr(128)
V = tl.constexpr(128)
BT = tl.constexpr(64)
QK_PREP_ROWS = tl.constexpr(32)
QK_PREP_DIM = tl.constexpr(128)
KKT_BK = tl.constexpr(64)
RECOMPUTE_BK = tl.constexpr(64)
RECOMPUTE_BV = tl.constexpr(64)
STATE_BV = tl.constexpr(16)
OUTPUT_BK = tl.constexpr(64)
OUTPUT_BV = tl.constexpr(32)
Q_SCALE = tl.constexpr(0.08838834764831845)


@triton.jit
def _bf16_rne_f32(value):
    """Publish an explicit BF16 RNE boundary, represented in F32 cells.

    Keep the operands of BF16 elementwise products in F32 before calling this
    helper. A BF16-typed multiply may already have truncated its result before
    a same-dtype .to(bfloat16), which cannot recover the discarded bits.
    This does not imply that all explicit FP32 -> BF16 casts truncate.
    """
    bits = tl.cast(value, tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    # Preserve infinities and signed zero; quiet NaNs instead of rounding a
    # payload into infinity or wrapping it into a finite value.
    rounded = tl.where((bits & 0x7FFFFFFF) > 0x7F800000,
                       (bits | 0x00400000) & 0xFFFF0000, rounded)
    return tl.cast(rounded, tl.float32, bitcast=True)


@triton.jit(do_not_specialize=["tokens"])
def _fla_qk_l2norm_from_native_f32_kernel(
    postconv_raw,
    q,
    k,
    tokens,
):
    """Normalize raw BF16-valued Q/K exactly at the service dtype boundary."""

    rows = tokens * HG
    row = (
        tl.program_id(0) * QK_PREP_ROWS
        + tl.arange(0, QK_PREP_ROWS)[:, None]
    )
    dim = tl.arange(0, QK_PREP_DIM)[None, :]
    row_mask = row < rows
    token = row // HG
    head = row - token * HG
    source_base = token * QKV_ROWS + head * K + dim

    b_q = tl.load(
        postconv_raw + source_base,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    b_k = tl.load(
        postconv_raw + source_base + KEY_FEATURES,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)
    q_square_sum = tl.sum(b_q * b_q, axis=1)[:, None]
    k_square_sum = tl.sum(b_k * b_k, axis=1)[:, None]
    q_normalized = b_q * tl.rsqrt(q_square_sum + 1.0e-6)
    k_normalized = b_k * tl.rsqrt(k_square_sum + 1.0e-6)
    compact_offset = row * K + dim
    tl.store(q + compact_offset, q_normalized, mask=row_mask)
    tl.store(k + compact_offset, k_normalized, mask=row_mask)


@triton.jit(do_not_specialize=["tokens"])
def _fla_v_beta_copy_from_native_f32_kernel(
    postconv_raw,
    gate,
    v,
    beta,
    tokens,
):
    """Publish contiguous BF16 V and beta tensors used by the FLA kernels."""

    row = tl.program_id(0)
    dim = tl.arange(0, V)
    token = row // H
    head = row - token * H
    active = token < tokens
    source = (
        postconv_raw
        + token * QKV_ROWS
        + VALUE_OFFSET
        + head * V
        + dim
    )
    value = tl.load(source, mask=active, other=0.0).to(tl.float32)
    tl.store(v + row * V + dim, value, mask=active)
    if tl.constexpr(V > 0):
        beta_value = tl.load(
            gate + token * GATE_STRIDE + H + head,
            mask=active,
            other=0.0,
        )
        tl.store(beta + token * H + head, beta_value, mask=active)


@triton.jit(do_not_specialize=["tokens"])
def _fla_chunk_gate_cumsum_kernel(
    gate,
    g_cumsum,
    tokens,
):
    """Compute the service's F32 log-decay cumsum independently per chunk."""

    chunk = tl.program_id(0)
    head = tl.program_id(1)
    offsets = chunk * BT + tl.arange(0, BT)
    mask = offsets < tokens
    values = tl.load(
        gate + offsets * GATE_STRIDE + head,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    cumulative = tl.cumsum(values, axis=0)
    tl.store(
        g_cumsum + offsets * H + head,
        cumulative,
        mask=mask,
    )


@triton.jit(do_not_specialize=["tokens"])
def _fla_chunk_scaled_dot_kkt_kernel(
    k,
    beta,
    g_cumsum,
    a_f32,
    tokens,
):
    """Compute the strictly-lower beta K K^T WY matrix in F32."""

    chunk = tl.program_id(0)
    head = tl.program_id(1)
    key_head = head // (H // HG)
    t = chunk * BT + tl.arange(0, BT)
    valid = t < tokens
    b_beta = tl.load(beta + t * H + head, mask=valid, other=0.0)
    b_a = tl.zeros([BT, BT], dtype=tl.float32)

    for key_block in range(0, K, KKT_BK):
        key_dim = key_block + tl.arange(0, KKT_BK)
        b_k = tl.load(
            k
            + t[:, None] * (HG * K)
            + key_head * K
            + key_dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        b_k_beta = _bf16_rne_f32(
            b_k.to(tl.float32) * b_beta[:, None].to(tl.float32)
        )
        b_a += tl.dot(b_k_beta.to(b_k.dtype), tl.trans(b_k))

    b_g = tl.load(
        g_cumsum + t * H + head,
        mask=valid,
        other=0.0,
    )
    b_a *= tl.exp(b_g[:, None] - b_g[None, :])
    local = tl.arange(0, BT)
    lower = (
        (local[:, None] > local[None, :])
        & valid[:, None]
        & valid[None, :]
    )
    b_a = tl.where(lower, b_a, 0.0)
    tl.store(
        a_f32
        + (t[:, None] * H + head) * BT
        + local[None, :],
        b_a,
        mask=valid[:, None],
    )


@triton.jit(do_not_specialize=["tokens"])
def _fla_solve_tril_64_kernel(
    a_f32,
    a_inverse_bf16,
    tokens,
):
    """Compute (I + A)^-1 with the pinned service's 16x16 merge order."""

    chunk = tl.program_id(0)
    head = tl.program_id(1)
    row16 = tl.arange(0, 16)
    lower16 = row16[:, None] > row16[None, :]
    identity16 = row16[:, None] == row16[None, :]
    chunk_row = chunk * BT

    a11 = -tl.where(
        lower16,
        tl.load(
            a_f32
            + ((chunk_row + row16[:, None]) * H + head) * BT
            + row16[None, :]
        ).to(tl.float32),
        0.0,
    )
    a22 = -tl.where(
        lower16,
        tl.load(
            a_f32
            + ((chunk_row + 16 + row16[:, None]) * H + head) * BT
            + 16
            + row16[None, :]
        ).to(tl.float32),
        0.0,
    )
    a33 = -tl.where(
        lower16,
        tl.load(
            a_f32
            + ((chunk_row + 32 + row16[:, None]) * H + head) * BT
            + 32
            + row16[None, :]
        ).to(tl.float32),
        0.0,
    )
    a44 = -tl.where(
        lower16,
        tl.load(
            a_f32
            + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
            + 48
            + row16[None, :]
        ).to(tl.float32),
        0.0,
    )

    for i in range(2, 16):
        source11 = -tl.load(
            a_f32
            + ((chunk_row + i) * H + head) * BT
            + row16
        )
        source11 += tl.sum(source11[:, None] * a11, axis=0)
        a11 = tl.where((row16 == i)[:, None], source11, a11)

        source22 = -tl.load(
            a_f32
            + ((chunk_row + 16 + i) * H + head) * BT
            + 16
            + row16
        )
        source22 += tl.sum(source22[:, None] * a22, axis=0)
        a22 = tl.where((row16 == i)[:, None], source22, a22)

        source33 = -tl.load(
            a_f32
            + ((chunk_row + 32 + i) * H + head) * BT
            + 32
            + row16
        )
        source33 += tl.sum(source33[:, None] * a33, axis=0)
        a33 = tl.where((row16 == i)[:, None], source33, a33)

        source44 = -tl.load(
            a_f32
            + ((chunk_row + 48 + i) * H + head) * BT
            + 48
            + row16
        )
        source44 += tl.sum(source44[:, None] * a44, axis=0)
        a44 = tl.where((row16 == i)[:, None], source44, a44)

    a11 += identity16
    a22 += identity16
    a33 += identity16
    a44 += identity16

    a21 = tl.load(
        a_f32
        + ((chunk_row + 16 + row16[:, None]) * H + head) * BT
        + row16[None, :]
    ).to(tl.float32)
    a31 = tl.load(
        a_f32
        + ((chunk_row + 32 + row16[:, None]) * H + head) * BT
        + row16[None, :]
    ).to(tl.float32)
    a32 = tl.load(
        a_f32
        + ((chunk_row + 32 + row16[:, None]) * H + head) * BT
        + 16
        + row16[None, :]
    ).to(tl.float32)
    a41 = tl.load(
        a_f32
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + row16[None, :]
    ).to(tl.float32)
    a42 = tl.load(
        a_f32
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + 16
        + row16[None, :]
    ).to(tl.float32)
    a43 = tl.load(
        a_f32
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + 32
        + row16[None, :]
    ).to(tl.float32)

    ai21 = -tl.dot(
        tl.dot(a22, a21, input_precision="ieee"),
        a11,
        input_precision="ieee",
    )
    ai32 = -tl.dot(
        tl.dot(a33, a32, input_precision="ieee"),
        a22,
        input_precision="ieee",
    )
    ai43 = -tl.dot(
        tl.dot(a44, a43, input_precision="ieee"),
        a33,
        input_precision="ieee",
    )
    ai31 = -tl.dot(
        a33,
        tl.dot(a31, a11, input_precision="ieee")
        + tl.dot(a32, ai21, input_precision="ieee"),
        input_precision="ieee",
    )
    ai42 = -tl.dot(
        a44,
        tl.dot(a42, a22, input_precision="ieee")
        + tl.dot(a43, ai32, input_precision="ieee"),
        input_precision="ieee",
    )
    ai41 = -tl.dot(
        a44,
        tl.dot(a41, a11, input_precision="ieee")
        + tl.dot(a42, ai21, input_precision="ieee")
        + tl.dot(a43, ai31, input_precision="ieee"),
        input_precision="ieee",
    )

    tl.store(
        a_inverse_bf16
        + ((chunk_row + row16[:, None]) * H + head) * BT
        + row16[None, :],
        a11,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 16 + row16[:, None]) * H + head) * BT
        + 16
        + row16[None, :],
        a22,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 32 + row16[:, None]) * H + head) * BT
        + 32
        + row16[None, :],
        a33,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + 48
        + row16[None, :],
        a44,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 16 + row16[:, None]) * H + head) * BT
        + row16[None, :],
        ai21,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 32 + row16[:, None]) * H + head) * BT
        + row16[None, :],
        ai31,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 32 + row16[:, None]) * H + head) * BT
        + 16
        + row16[None, :],
        ai32,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + row16[None, :],
        ai41,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + 16
        + row16[None, :],
        ai42,
    )
    tl.store(
        a_inverse_bf16
        + ((chunk_row + 48 + row16[:, None]) * H + head) * BT
        + 32
        + row16[None, :],
        ai43,
    )


@triton.jit(do_not_specialize=["tokens"])
def _fla_recompute_w_u_kernel(
    k,
    v,
    beta,
    w,
    u,
    a_inverse_bf16,
    g_cumsum,
    tokens,
):
    """Recompute BF16 W/U from the solved WY matrix."""

    chunk = tl.program_id(0)
    head = tl.program_id(1)
    key_head = head // (H // HG)
    t = chunk * BT + tl.arange(0, BT)
    valid = t < tokens
    local = tl.arange(0, BT)
    b_beta = tl.load(beta + t * H + head, mask=valid, other=0.0)
    b_g = tl.exp(
        tl.load(g_cumsum + t * H + head, mask=valid, other=0.0)
    )
    b_a = tl.load(
        a_inverse_bf16
        + (t[:, None] * H + head) * BT
        + local[None, :],
        mask=valid[:, None],
        other=0.0,
    )

    for value_block in range(0, V, RECOMPUTE_BV):
        value_dim = value_block + tl.arange(0, RECOMPUTE_BV)
        b_v = tl.load(
            v + (t[:, None] * H + head) * V + value_dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        b_v_beta = _bf16_rne_f32(
            b_v.to(tl.float32) * b_beta[:, None].to(tl.float32)
        ).to(b_v.dtype)
        # The captured two-term BF16 midpoint cases distinguish accumulator
        # arithmetic from the already-correct RNE store. Keep the BF16 input
        # boundary, but accumulate U with IEEE F32 instructions, not WMMA.
        b_u = tl.dot(
            b_a.to(tl.float32),
            b_v_beta.to(tl.float32),
            input_precision="ieee",
        )
        tl.store(
            u + (t[:, None] * H + head) * V + value_dim[None, :],
            b_u,
            mask=valid[:, None],
        )

    for key_block in range(0, K, RECOMPUTE_BK):
        key_dim = key_block + tl.arange(0, RECOMPUTE_BK)
        b_k = tl.load(
            k
            + t[:, None] * (HG * K)
            + key_head * K
            + key_dim[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        b_k_beta = _bf16_rne_f32(
            b_k.to(tl.float32) * b_beta[:, None].to(tl.float32)
        )
        b_k_beta_g = _bf16_rne_f32(b_k_beta * b_g[:, None]).to(b_k.dtype)
        # W has the same captured two-term midpoint discrepancy as U. Preserve
        # both BF16 input boundaries and use IEEE F32 accumulation here too.
        b_w = tl.dot(
            b_a.to(tl.float32),
            b_k_beta_g.to(tl.float32),
            input_precision="ieee",
        )
        tl.store(
            w + (t[:, None] * H + head) * K + key_dim[None, :],
            b_w,
            mask=valid[:, None],
        )


@triton.jit(do_not_specialize=["tokens"])
def _fla_chunk_state_kernel(
    k,
    u,
    w,
    v_new,
    g_cumsum,
    initial_state,
    chunk_state,
    final_state,
    tokens,
):
    """Advance a seeded F32 state and publish BF16 chunk boundaries."""

    value_tile = tl.program_id(0)
    head = tl.program_id(1)
    key_head = head // (H // HG)
    value_dim = value_tile * STATE_BV + tl.arange(0, STATE_BV)
    key_dim = tl.arange(0, 64)
    final_base = (head * V + value_dim[:, None]) * K
    h1 = tl.load(
        initial_state + final_base + key_dim[None, :],
    ).to(tl.float32)
    h2 = tl.load(
        initial_state + final_base + 64 + key_dim[None, :],
    ).to(tl.float32)
    chunks = tokens // BT

    for chunk in range(0, chunks):
        chunk_base = chunk * BT
        t = chunk_base + tl.arange(0, BT)

        state_base = (
            ((chunk * H + head) * V + value_dim[:, None]) * K
        )
        tl.store(
            chunk_state + state_base + key_dim[None, :],
            h1,
        )
        tl.store(
            chunk_state + state_base + 64 + key_dim[None, :],
            h2,
        )

        w1 = tl.load(
            w
            + (t[:, None] * H + head) * K
            + key_dim[None, :]
        )
        w2 = tl.load(
            w
            + (t[:, None] * H + head) * K
            + 64
            + key_dim[None, :]
        )
        current_v = tl.dot(w1, tl.trans(h1).to(w1.dtype))
        current_v += tl.dot(w2, tl.trans(h2).to(w2.dtype))
        current_v = tl.load(
            u
            + (t[:, None] * H + head) * V
            + value_dim[None, :]
        ) - current_v
        tl.store(
            v_new
            + (t[:, None] * H + head) * V
            + value_dim[None, :],
            current_v,
        )

        gate = tl.load(g_cumsum + t * H + head)
        gate_last = tl.load(
            g_cumsum + (chunk_base + BT - 1) * H + head
        )
        current_v *= tl.exp(gate_last - gate)[:, None]
        state_decay = tl.exp(gate_last)
        h1 *= state_decay
        h2 *= state_decay
        current_v = current_v.to(k.dtype.element_ty)

        k1 = tl.load(
            k
            + t[None, :] * (HG * K)
            + key_head * K
            + key_dim[:, None]
        )
        k2 = tl.load(
            k
            + t[None, :] * (HG * K)
            + key_head * K
            + 64
            + key_dim[:, None]
        )
        h1 += tl.trans(tl.dot(k1, current_v))
        h2 += tl.trans(tl.dot(k2, current_v))

    tl.store(final_state + final_base + key_dim[None, :], h1)
    tl.store(final_state + final_base + 64 + key_dim[None, :], h2)


@triton.jit(do_not_specialize=["tokens"])
def _fla_chunk_output_kernel(
    q,
    k,
    v_new,
    chunk_state,
    g_cumsum,
    output_f32,
    tokens,
):
    """Compute the service chunk output and publish BF16-valued F32 cells."""

    value_tile = tl.program_id(0)
    chunk = tl.program_id(1)
    head = tl.program_id(2)
    key_head = head // (H // HG)
    t = chunk * BT + tl.arange(0, BT)
    value_dim = value_tile * OUTPUT_BV + tl.arange(0, OUTPUT_BV)
    local = tl.arange(0, BT)
    output = tl.zeros([BT, OUTPUT_BV], dtype=tl.float32)
    qk = tl.zeros([BT, BT], dtype=tl.float32)

    for key_block in range(0, K, OUTPUT_BK):
        key_dim = key_block + tl.arange(0, OUTPUT_BK)
        b_q = tl.load(
            q
            + t[:, None] * (HG * K)
            + key_head * K
            + key_dim[None, :]
        )
        b_k = tl.load(
            k
            + t[None, :] * (HG * K)
            + key_head * K
            + key_dim[:, None]
        )
        b_h = tl.load(
            chunk_state
            + ((chunk * H + head) * V + value_dim[:, None]) * K
            + key_dim[None, :]
        )
        output += tl.dot(b_q, tl.trans(b_h))
        qk += tl.dot(b_q, b_k)

    gate = tl.load(g_cumsum + t * H + head)
    output *= tl.exp(gate)[:, None]
    qk *= tl.exp(gate[:, None] - gate[None, :])
    causal = local[:, None] >= local[None, :]
    qk = tl.where(causal, qk, 0.0)
    values = tl.load(
        v_new
        + (t[:, None] * H + head) * V
        + value_dim[None, :]
    )
    output = output * Q_SCALE + tl.dot(qk.to(values.dtype), values) * Q_SCALE
    output = output.to(tl.bfloat16).to(tl.float32)
    tl.store(
        output_f32
        + (t[:, None] * H + head) * V
        + value_dim[None, :],
        output,
    )


KERNELS: tuple[dict[str, Any], ...] = (
    {
        "name": "qk_l2norm",
        "fn": _fla_qk_l2norm_from_native_f32_kernel,
        "signature": {
            "postconv_raw": "*fp32",
            "q": "*bf16",
            "k": "*bf16",
            "tokens": "i32",
        },
        "grid": ["ceil(tokens*16/32)", 1, 1],
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        # The q262144 streamed runtime keeps its raw post-convolution carrier
        # in BF16.  Compile the identical normalization arithmetic against a
        # BF16 source pointer so the exact FLA path does not require an 8 GiB
        # F32 expansion at maximum context.
        "name": "qk_l2norm_bf16",
        "fn": _fla_qk_l2norm_from_native_f32_kernel,
        "signature": {
            "postconv_raw": "*bf16",
            "q": "*bf16",
            "k": "*bf16",
            "tokens": "i32",
        },
        "grid": ["ceil(tokens*16/32)", 1, 1],
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        "name": "v_beta_copy",
        "fn": _fla_v_beta_copy_from_native_f32_kernel,
        "signature": {
            "postconv_raw": "*fp32",
            "gate": "*fp32",
            "v": "*bf16",
            "beta": "*bf16",
            "tokens": "i32",
        },
        "grid": ["tokens*32", 1, 1],
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        "name": "v_beta_copy_bf16",
        "fn": _fla_v_beta_copy_from_native_f32_kernel,
        "signature": {
            "postconv_raw": "*bf16",
            "gate": "*fp32",
            "v": "*bf16",
            "beta": "*bf16",
            "tokens": "i32",
        },
        "grid": ["tokens*32", 1, 1],
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        "name": "gate_cumsum",
        "fn": _fla_chunk_gate_cumsum_kernel,
        "signature": {
            "gate": "*fp32",
            "g_cumsum": "*fp32",
            "tokens": "i32",
        },
        "grid": ["tokens/64", 32, 1],
        "num_warps": 2,
        "num_stages": 1,
    },
    {
        "name": "scaled_dot_kkt",
        "fn": _fla_chunk_scaled_dot_kkt_kernel,
        "signature": {
            "k": "*bf16",
            "beta": "*bf16",
            "g_cumsum": "*fp32",
            "a_f32": "*fp32",
            "tokens": "i32",
        },
        "grid": ["tokens/64", 32, 1],
        "num_warps": 8,
        "num_stages": 1,
    },
    {
        "name": "solve_tril_64",
        "fn": _fla_solve_tril_64_kernel,
        "signature": {
            "a_f32": "*fp32",
            "a_inverse_bf16": "*bf16",
            "tokens": "i32",
        },
        "grid": ["tokens/64", 32, 1],
        "num_warps": 8,
        "num_stages": 1,
    },
    {
        "name": "recompute_w_u",
        "fn": _fla_recompute_w_u_kernel,
        "signature": {
            "k": "*bf16",
            "v": "*bf16",
            "beta": "*bf16",
            "w": "*bf16",
            "u": "*bf16",
            "a_inverse_bf16": "*bf16",
            "g_cumsum": "*fp32",
            "tokens": "i32",
        },
        "grid": ["tokens/64", 32, 1],
        "num_warps": 8,
        "num_stages": 1,
    },
    {
        "name": "chunk_state",
        "fn": _fla_chunk_state_kernel,
        "signature": {
            "k": "*bf16",
            "u": "*bf16",
            "w": "*bf16",
            "v_new": "*bf16",
            "g_cumsum": "*fp32",
            "initial_state": "*fp32",
            "chunk_state": "*bf16",
            "final_state": "*fp32",
            "tokens": "i32",
        },
        "grid": [8, 32, 1],
        "num_warps": 4,
        "num_stages": 1,
    },
    {
        "name": "chunk_output",
        "fn": _fla_chunk_output_kernel,
        "signature": {
            "q": "*bf16",
            "k": "*bf16",
            "v_new": "*bf16",
            "chunk_state": "*bf16",
            "g_cumsum": "*fp32",
            "output_f32": "*fp32",
            "tokens": "i32",
        },
        "grid": [4, "tokens/64", 32],
        "num_warps": 8,
        "num_stages": 1,
    },
    {
        "name": "chunk_output_bf16",
        "fn": _fla_chunk_output_kernel,
        "signature": {
            "q": "*bf16",
            "k": "*bf16",
            "v_new": "*bf16",
            "chunk_state": "*bf16",
            "g_cumsum": "*fp32",
            "output_f32": "*bf16",
            "tokens": "i32",
        },
        "grid": [4, "tokens/64", 32],
        "num_warps": 8,
        "num_stages": 1,
    },
)


def compile_all(output_dir: Path, metadata_path: Path) -> None:
    target = GPUTarget("hip", "gfx1151", 32)
    backend = make_backend(target)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for kernel in KERNELS:
        options = backend.parse_options(
            {
                "num_warps": kernel["num_warps"],
                "num_stages": kernel["num_stages"],
                "waves_per_eu": 0,
            }
        )
        compiled = triton.compile(
            ASTSource(
                fn=kernel["fn"],
                signature=kernel["signature"],
            ),
            target=target,
            options=options.__dict__,
        )
        binary = compiled.asm[backend.binary_ext]
        output_path = output_dir / f"q8192_fla_chunk_gdn_{kernel['name']}.hsaco"
        output_path.write_bytes(binary)
        records.append(
            {
                "name": kernel["name"],
                "file": output_path.name,
                "symbol": compiled.name,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "bytes": len(binary),
                "grid": kernel["grid"],
                "threads": compiled.metadata.num_warps * 32,
                "num_warps": compiled.metadata.num_warps,
                "num_stages": kernel["num_stages"],
                "dynamic_shared_bytes": compiled.metadata.shared,
                "abi": list(kernel["signature"]),
                "compiled_hash": compiled.hash,
            }
        )

    metadata = {
        "schema_version": 2,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "source": "native/generators/compile_q8192_fla_chunk_gdn.py",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provenance": {
            "algorithm": "vLLM 2a69949bd Triton/FLA chunk gated delta rule",
            "upstream_license": "MIT",
            "chunk_size": 64,
        },
        "shape": {
            "batch": 1,
            "runtime_tokens_multiple": 64,
            "qk_heads": 16,
            "value_heads": 32,
            "key_dim": 128,
            "value_dim": 128,
            "input_dtype": (
                "float32 cells holding BF16 post-convolution values, or "
                "native bfloat16 for the maximum-context streamed ABI"
            ),
            "qkv_compute_dtype": "bfloat16",
            "gate_cumsum_dtype": "float32",
            "chunk_state_dtype": "bfloat16",
            "recurrent_state_dtype": "float32",
            "initial_state_dtype": "float32",
            "output_dtype": (
                "bfloat16-rounded float32 cells, or native bfloat16 for "
                "the maximum-context streamed ABI"
            ),
            "query_scale_stage": "chunk output",
        },
        "streaming": {
            "q65536_segment_tokens": 1024,
            "segments": 64,
            "scratch_reused_on_one_stream": True,
            "state_handoff": "F32 final state seeds the next segment",
            "q262144_chunk_tokens": 32768,
            "q262144_segment_tokens": 1024,
            "q262144_io_dtype": "bfloat16",
            "q262144_state_layout": "value_head_key_value",
        },
        "scratch_bytes_per_token": {
            "compact_qkv_bf16": 16384,
            "gate_and_beta": 192,
            "a_or_w": 8192,
            "ai_or_v_new": 8192,
            "chunk_state": 16384,
            "total": 49344,
        },
        "kernels": records,
    }
    specs_path = output_dir / "qrt_fla_gdn_kernel_specs.inc"
    write_provider_kernel_specs(records, specs_path)
    metadata["provider_specs_sha256"] = hashlib.sha256(specs_path.read_bytes()).hexdigest()
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


def write_provider_kernel_specs(records: list[dict[str, Any]], path: Path) -> None:
    """Bind the native launch resource sizes to this compilation, not old AOTs."""
    names = (
        "qk_l2norm", "v_beta_copy", "gate_cumsum", "scaled_dot_kkt",
        "solve_tril_64", "recompute_w_u", "chunk_state", "chunk_output",
    )
    indexed = {record["name"]: record for record in records}
    lines = [
        "// Generated by compile_q8192_fla_chunk_gdn.py; do not edit.",
        "constexpr std::array<KernelSpec, static_cast<size_t>(KernelIndex::kCount)>",
        "    kKernelSpecs{{",
    ]
    for name in names:
        record = indexed[name]
        threads = int(record["threads"])
        shared = int(record["dynamic_shared_bytes"])
        if threads <= 0 or threads > 1024 or threads % 32 or not 0 <= shared <= 65536:
            raise ValueError(f"unsupported gfx1151 launch resources for {name}")
        lines.append(
            "        {" + json.dumps(record["file"]) + ", "
            + json.dumps(record["symbol"]) + f", {threads}u, {shared}u}},"
        )
    lines.append("    }};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    compile_all(args.output_dir, args.metadata)


if __name__ == "__main__":
    main()
