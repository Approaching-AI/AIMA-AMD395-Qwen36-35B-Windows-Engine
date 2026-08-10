#!/usr/bin/env python3
"""AOT-compile fixed-token Qwen3.6 selected-MoE kernels for gfx1151.

The generated code objects are loaded through the Windows HIP module API.
Python and Triton are build dependencies only.  The kernels are fixed to the
real Qwen3.6-35B-A3B BF16 dimensions: top-k 8, hidden 2048, 256 experts, and
expert intermediate size 512.  The token count defaults to the retained q8192
product shape and may also be fixed at another multiple-of-32 request tile.

The route alignment and matrix kernels are adapted from AMD AITER's MIT
licensed Triton MoE implementation.  This fixed-shape version removes the
Torch-facing wrapper and quantized branches, keeps route-order outputs, and
accepts the engine's resident F32 post-attention surface directly.
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
ROUTES_VALUE = TOKENS_VALUE * 8
ROUTE_PROGRAMS_VALUE = ROUTES_VALUE // 256
TOKENS = tl.constexpr(TOKENS_VALUE)
TOP_K = tl.constexpr(8)
ROUTES = tl.constexpr(ROUTES_VALUE)
ROUTE_PROGRAMS = tl.constexpr(ROUTE_PROGRAMS_VALUE)
EXPERTS = tl.constexpr(256)
HIDDEN = tl.constexpr(2048)
INTERMEDIATE = tl.constexpr(512)
GATE_UP_ROWS = tl.constexpr(1024)

BLOCK_M_VALUE = 64
GATE_BLOCK_N_VALUE = 64
DOWN_BLOCK_N_VALUE = 64
BLOCK_K_VALUE = 64
GATE_BLOCK_K_VALUE = 64
DOWN_BLOCK_K_VALUE = 64
GROUP_M_VALUE = 8
BLOCK_M = tl.constexpr(BLOCK_M_VALUE)
GATE_BLOCK_N = tl.constexpr(GATE_BLOCK_N_VALUE)
DOWN_BLOCK_N = tl.constexpr(DOWN_BLOCK_N_VALUE)
BLOCK_K = tl.constexpr(BLOCK_K_VALUE)
GATE_BLOCK_K = tl.constexpr(GATE_BLOCK_K_VALUE)
DOWN_BLOCK_K = tl.constexpr(DOWN_BLOCK_K_VALUE)
GROUP_M = tl.constexpr(GROUP_M_VALUE)
ROUTE_OUTPUT_BF16_ENDPOINT = tl.constexpr(False)
VLLM_SILU_BF16_INTERMEDIATE = tl.constexpr(False)
ROUTES_PER_SORT_PROGRAM = tl.constexpr(256)
MAX_SORTED_ROUTES = ROUTES_VALUE + 256 * BLOCK_M_VALUE - 8
MAX_ROUTE_BLOCKS = (MAX_SORTED_ROUTES + BLOCK_M_VALUE - 1) // BLOCK_M_VALUE


def configure_shape(
    tokens: int,
    block_m: int,
    gate_block_n: int,
    down_block_n: int,
    gate_block_k: int,
    down_block_k: int,
    group_m: int,
) -> None:
    """Set compile-time tile constants before Triton resolves JIT globals."""

    if tokens <= 0 or (tokens * 8) % 256 != 0:
        raise ValueError("tokens must be positive and tokens * top-k divisible by 256")
    if block_m not in (16, 32, 64, 128):
        raise ValueError("block_m must be one of 16, 32, 64, 128")
    if gate_block_n not in (32, 64, 128, 256):
        raise ValueError("gate_block_n must be one of 32, 64, 128, 256")
    if down_block_n not in (32, 64, 128, 256):
        raise ValueError("down_block_n must be one of 32, 64, 128, 256")
    if gate_block_k not in (32, 64, 128):
        raise ValueError("gate_block_k must be one of 32, 64, 128")
    if down_block_k not in (32, 64, 128):
        raise ValueError("down_block_k must be one of 32, 64, 128")
    if group_m <= 0:
        raise ValueError("group_m must be positive")
    if GATE_UP_ROWS % gate_block_n != 0:
        raise ValueError("gate_block_n must divide gate/up rows")
    if HIDDEN % down_block_n != 0:
        raise ValueError("down_block_n must divide hidden")
    if HIDDEN % gate_block_k != 0:
        raise ValueError("gate_block_k must divide hidden")
    if INTERMEDIATE % down_block_k != 0:
        raise ValueError("down_block_k must divide intermediate")
    if gate_block_n % 2 != 0:
        raise ValueError("gate_block_n must be even for paired gate/up columns")

    global TOKENS_VALUE, ROUTES_VALUE, ROUTE_PROGRAMS_VALUE
    global TOKENS, ROUTES, ROUTE_PROGRAMS
    global BLOCK_M_VALUE, GATE_BLOCK_N_VALUE, DOWN_BLOCK_N_VALUE, BLOCK_K_VALUE
    global GATE_BLOCK_K_VALUE, DOWN_BLOCK_K_VALUE, GROUP_M_VALUE
    global BLOCK_M, GATE_BLOCK_N, DOWN_BLOCK_N
    global BLOCK_K, GATE_BLOCK_K, DOWN_BLOCK_K, GROUP_M
    global MAX_SORTED_ROUTES, MAX_ROUTE_BLOCKS
    TOKENS_VALUE = tokens
    ROUTES_VALUE = tokens * 8
    ROUTE_PROGRAMS_VALUE = ROUTES_VALUE // 256
    TOKENS = tl.constexpr(TOKENS_VALUE)
    ROUTES = tl.constexpr(ROUTES_VALUE)
    ROUTE_PROGRAMS = tl.constexpr(ROUTE_PROGRAMS_VALUE)
    BLOCK_M_VALUE = block_m
    GATE_BLOCK_N_VALUE = gate_block_n
    DOWN_BLOCK_N_VALUE = down_block_n
    BLOCK_K_VALUE = gate_block_k if gate_block_k == down_block_k else 0
    GATE_BLOCK_K_VALUE = gate_block_k
    DOWN_BLOCK_K_VALUE = down_block_k
    GROUP_M_VALUE = group_m
    BLOCK_M = tl.constexpr(block_m)
    GATE_BLOCK_N = tl.constexpr(gate_block_n)
    DOWN_BLOCK_N = tl.constexpr(down_block_n)
    BLOCK_K = tl.constexpr(BLOCK_K_VALUE)
    GATE_BLOCK_K = tl.constexpr(gate_block_k)
    DOWN_BLOCK_K = tl.constexpr(down_block_k)
    GROUP_M = tl.constexpr(group_m)
    MAX_SORTED_ROUTES = ROUTES_VALUE + 256 * block_m - 8
    MAX_ROUTE_BLOCKS = (MAX_SORTED_ROUTES + block_m - 1) // block_m


@triton.jit
def _route_count_kernel(topk_ids, token_counts):
    pid = tl.program_id(0)
    source = pid * ROUTES_PER_SORT_PROGRAM
    count_base = (pid + 1) * EXPERTS
    for offset in range(ROUTES_PER_SORT_PROGRAM):
        expert = tl.load(topk_ids + source + offset)
        count = tl.load(token_counts + count_base + expert)
        tl.store(token_counts + count_base + expert, count + 1)


@triton.jit
def _route_prefix_by_program_kernel(token_counts):
    expert = tl.program_id(0)
    running = 0
    for program in range(1, ROUTE_PROGRAMS + 1):
        index = program * EXPERTS + expert
        running += tl.load(token_counts + index)
        tl.store(token_counts + index, running)


@triton.jit
def _route_padded_prefix_kernel(total_post_pad, token_counts, cumsum):
    running = 0
    final_count_base = ROUTE_PROGRAMS * EXPERTS
    for expert_plus_one in range(1, EXPERTS + 1):
        count = tl.load(token_counts + final_count_base + expert_plus_one - 1)
        running += tl.cdiv(count, BLOCK_M) * BLOCK_M
        tl.store(cumsum + expert_plus_one, running)
    tl.store(total_post_pad, running)


@triton.jit
def _route_scatter_kernel(
    topk_ids,
    sorted_route_ids,
    block_expert_ids,
    token_counts,
    cumsum,
):
    pid = tl.program_id(0)
    if pid < EXPERTS:
        padded_begin = tl.load(cumsum + pid)
        padded_end = tl.load(cumsum + pid + 1)
        for offset in range(padded_begin, padded_end, BLOCK_M):
            tl.store(block_expert_ids + offset // BLOCK_M, pid)

    if pid >= ROUTE_PROGRAMS:
        return

    source = pid * ROUTES_PER_SORT_PROGRAM
    local_count_base = pid * EXPERTS
    for offset in range(ROUTES_PER_SORT_PROGRAM):
        route = source + offset
        expert = tl.load(topk_ids + route)
        rank = tl.load(token_counts + local_count_base + expert)
        destination = rank + tl.load(cumsum + expert)
        tl.store(sorted_route_ids + destination, route)
        tl.store(token_counts + local_count_base + expert, rank + 1)


@triton.jit
def _grouped_pid(pid, num_pid_m, num_pid_n):
    programs_per_group = GROUP_M * num_pid_n
    group = pid // programs_per_group
    first_m = group * GROUP_M
    group_m = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group_m)
    pid_n = (pid % programs_per_group) // group_m
    return pid_m, pid_n


@triton.jit
def _gate_up_silu_kernel(
    post_attention_bf16,
    gate_up_bf16,
    sorted_route_ids,
    block_expert_ids,
    total_post_pad,
    activated_bf16,
):
    pid = tl.program_id(0)
    padded_routes = tl.load(total_post_pad)
    num_pid_m = tl.cdiv(padded_routes, BLOCK_M)
    num_pid_n: tl.constexpr = GATE_UP_ROWS // GATE_BLOCK_N
    grid = num_pid_m * num_pid_n
    if pid >= grid:
        return
    pid_m, pid_n = _grouped_pid(pid, num_pid_m, num_pid_n)

    sorted_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    routes = tl.load(sorted_route_ids + sorted_offsets).to(tl.int64)
    valid = (routes >= 0) & (routes < ROUTES)
    tokens = routes // TOP_K
    expert = tl.load(block_expert_ids + pid_m).to(tl.int64)

    lane = tl.arange(0, GATE_BLOCK_N).to(tl.int64)
    half_lane = lane // 2
    inter = pid_n * (GATE_BLOCK_N // 2) + half_lane
    weight_row = inter + (lane % 2) * INTERMEDIATE
    k_offsets = tl.arange(0, GATE_BLOCK_K).to(tl.int64)
    input_ptrs = (
        post_attention_bf16
        + tokens[:, None] * HIDDEN
        + k_offsets[None, :]
    )
    weight_ptrs = (
        gate_up_bf16
        + expert * GATE_UP_ROWS * HIDDEN
        + weight_row[None, :] * HIDDEN
        + k_offsets[:, None]
    )

    accumulator = tl.zeros((BLOCK_M, GATE_BLOCK_N), tl.float32)
    for _ in range(HIDDEN // GATE_BLOCK_K):
        inputs = tl.load(input_ptrs, mask=valid[:, None], other=0.0)
        weights = tl.load(weight_ptrs)
        accumulator += tl.dot(inputs, weights)
        input_ptrs += GATE_BLOCK_K
        weight_ptrs += GATE_BLOCK_K

    rounded = accumulator.to(tl.bfloat16).to(tl.float32)
    gate, up = rounded.reshape(BLOCK_M, GATE_BLOCK_N // 2, 2).split()
    if VLLM_SILU_BF16_INTERMEDIATE:
        gate = (gate / (1.0 + tl.exp(-gate))).to(
            tl.bfloat16
        ).to(tl.float32)
    else:
        gate = gate / (1.0 + tl.exp2(-(gate * 1.44269504089)))
    output = (gate * up).to(tl.bfloat16)
    output_columns = (
        pid_n * (GATE_BLOCK_N // 2) + tl.arange(0, GATE_BLOCK_N // 2)
    )
    output_ptrs = (
        activated_bf16
        + routes[:, None] * INTERMEDIATE
        + output_columns[None, :]
    )
    tl.store(output_ptrs, output, mask=valid[:, None])


@triton.jit
def _down_kernel(
    activated_bf16,
    down_bf16,
    sorted_route_ids,
    block_expert_ids,
    total_post_pad,
    route_outputs_f32,
):
    pid = tl.program_id(0)
    padded_routes = tl.load(total_post_pad)
    num_pid_m = tl.cdiv(padded_routes, BLOCK_M)
    num_pid_n: tl.constexpr = HIDDEN // DOWN_BLOCK_N
    grid = num_pid_m * num_pid_n
    if pid >= grid:
        return
    pid_m, pid_n = _grouped_pid(pid, num_pid_m, num_pid_n)

    sorted_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    routes = tl.load(sorted_route_ids + sorted_offsets).to(tl.int64)
    valid = (routes >= 0) & (routes < ROUTES)
    expert = tl.load(block_expert_ids + pid_m).to(tl.int64)
    output_columns = (
        pid_n * DOWN_BLOCK_N + tl.arange(0, DOWN_BLOCK_N).to(tl.int64)
    )
    k_offsets = tl.arange(0, DOWN_BLOCK_K).to(tl.int64)
    input_ptrs = (
        activated_bf16
        + routes[:, None] * INTERMEDIATE
        + k_offsets[None, :]
    )
    weight_ptrs = (
        down_bf16
        + expert * HIDDEN * INTERMEDIATE
        + output_columns[None, :] * INTERMEDIATE
        + k_offsets[:, None]
    )

    accumulator = tl.zeros((BLOCK_M, DOWN_BLOCK_N), tl.float32)
    for _ in range(INTERMEDIATE // DOWN_BLOCK_K):
        inputs = tl.load(input_ptrs, mask=valid[:, None], other=0.0)
        weights = tl.load(weight_ptrs)
        accumulator += tl.dot(inputs, weights)
        input_ptrs += DOWN_BLOCK_K
        weight_ptrs += DOWN_BLOCK_K

    output_ptrs = (
        route_outputs_f32
        + routes[:, None] * HIDDEN
        + output_columns[None, :]
    )
    # q8192 retains F32 through route-order combination.  The q1024 owner can
    # instead request the q1-compatible BF16 per-route endpoint while keeping
    # the same grouped tensor-core matrix shape.
    output = accumulator
    if ROUTE_OUTPUT_BF16_ENDPOINT:
        output = accumulator.to(tl.bfloat16).to(tl.float32)
    tl.store(output_ptrs, output, mask=valid[:, None])


def kernel_specs():
    return (
        (
            "route_count",
            _route_count_kernel,
            {"topk_ids": "*i32", "token_counts": "*i32"},
            [ROUTE_PROGRAMS_VALUE, 1, 1],
        ),
        (
            "route_prefix_by_program",
            _route_prefix_by_program_kernel,
            {"token_counts": "*i32"},
            [256, 1, 1],
        ),
        (
            "route_padded_prefix",
            _route_padded_prefix_kernel,
            {
                "total_post_pad": "*i32",
                "token_counts": "*i32",
                "cumsum": "*i32",
            },
            [1, 1, 1],
        ),
        (
            "route_scatter",
            _route_scatter_kernel,
            {
                "topk_ids": "*i32",
                "sorted_route_ids": "*i32",
                "block_expert_ids": "*i32",
                "token_counts": "*i32",
                "cumsum": "*i32",
            },
            [max(ROUTE_PROGRAMS_VALUE, 256), 1, 1],
        ),
        (
            "gate_up_silu",
            _gate_up_silu_kernel,
            {
                "post_attention_bf16": "*bf16",
                "gate_up_bf16": "*bf16",
                "sorted_route_ids": "*i32",
                "block_expert_ids": "*i32",
                "total_post_pad": "*i32",
                "activated_bf16": "*bf16",
            },
            [MAX_ROUTE_BLOCKS * (1024 // GATE_BLOCK_N_VALUE), 1, 1],
        ),
        (
            "down",
            _down_kernel,
            {
                "activated_bf16": "*bf16",
                "down_bf16": "*bf16",
                "sorted_route_ids": "*i32",
                "block_expert_ids": "*i32",
                "total_post_pad": "*i32",
                "route_outputs_f32": "*fp32",
            },
            [MAX_ROUTE_BLOCKS * (2048 // DOWN_BLOCK_N_VALUE), 1, 1],
        ),
    )


def compile_all(
    output_dir: Path,
    metadata_path: Path,
    *,
    route_launch_options: dict[str, int],
    gate_launch_options: dict[str, int],
    down_launch_options: dict[str, int],
) -> None:
    target = GPUTarget("hip", "gfx1151", 32)
    backend = make_backend(target)
    launch_options = {
        "route": route_launch_options,
        "gate_up_silu": gate_launch_options,
        "down": down_launch_options,
    }
    compiled_options = {
        family: backend.parse_options(values)
        for family, values in launch_options.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for stem, fn, signature, grid in kernel_specs():
        family = stem if stem in ("gate_up_silu", "down") else "route"
        options = compiled_options[family]
        compiled = triton.compile(
            ASTSource(fn=fn, signature=signature),
            target=target,
            options=options.__dict__,
        )
        binary = compiled.asm[backend.binary_ext]
        output = output_dir / f"q{TOKENS_VALUE}_selected_moe_{stem}.hsaco"
        output.write_bytes(binary)
        records.append(
            {
                "name": stem,
                "file": output.name,
                "symbol": compiled.name,
                "sha256": hashlib.sha256(binary).hexdigest(),
                "bytes": len(binary),
                "grid": grid,
                "threads": compiled.metadata.num_warps * 32,
                "dynamic_shared_bytes": compiled.metadata.shared,
                "launch_options": launch_options[family],
                "abi": list(signature),
                "compiled_hash": compiled.hash,
            }
        )

    metadata = {
        "schema_version": 1,
        "target": "gfx1151",
        "compiler": f"Triton {triton.__version__} HIP backend",
        "source": "tools/compile_q8192_triton_selected_moe.py",
        "provenance": "fixed-shape adaptation of AMD AITER Triton MoE (MIT)",
        "shape": {
            "tokens": TOKENS_VALUE,
            "top_k": 8,
            "routes": ROUTES_VALUE,
            "route_programs": ROUTE_PROGRAMS_VALUE,
            "experts": 256,
            "hidden": 2048,
            "intermediate": 512,
            "source_input_dtype": "float32",
            "kernel_input_dtype": "bfloat16",
            "weight_dtype": "bfloat16",
            "intermediate_dtype": "bfloat16",
            "route_output_dtype": "float32",
            "route_output_bf16_endpoint": bool(
                ROUTE_OUTPUT_BF16_ENDPOINT.value
            ),
            "vllm_silu_bf16_intermediate": bool(
                VLLM_SILU_BF16_INTERMEDIATE.value
            ),
            "block_m": BLOCK_M_VALUE,
            "block_n": (
                GATE_BLOCK_N_VALUE
                if GATE_BLOCK_N_VALUE == DOWN_BLOCK_N_VALUE
                else 0
            ),
            "gate_block_n": GATE_BLOCK_N_VALUE,
            "down_block_n": DOWN_BLOCK_N_VALUE,
            "block_k": BLOCK_K_VALUE,
            "gate_block_k": GATE_BLOCK_K_VALUE,
            "down_block_k": DOWN_BLOCK_K_VALUE,
            "group_m": GROUP_M_VALUE,
            "max_sorted_routes": MAX_SORTED_ROUTES,
            "max_route_blocks": MAX_ROUTE_BLOCKS,
        },
        # Preserve the original flat route/default alias for existing metadata
        # readers while recording the independently compiled matrix kernels.
        "launch_options": route_launch_options,
        "kernel_launch_options": launch_options,
        "kernels": records,
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
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--gate-block-n", type=int)
    parser.add_argument("--down-block-n", type=int)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--gate-block-k", type=int)
    parser.add_argument("--down-block-k", type=int)
    parser.add_argument("--group-m", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=1)
    parser.add_argument("--waves-per-eu", type=int, default=0)
    parser.add_argument("--gate-num-warps", type=int)
    parser.add_argument("--gate-num-stages", type=int)
    parser.add_argument("--gate-waves-per-eu", type=int)
    parser.add_argument("--down-num-warps", type=int)
    parser.add_argument("--down-num-stages", type=int)
    parser.add_argument("--down-waves-per-eu", type=int)
    parser.add_argument(
        "--route-output-bf16-endpoint",
        action="store_true",
    )
    parser.add_argument(
        "--vllm-silu-bf16-intermediate",
        action="store_true",
    )
    args = parser.parse_args()
    global ROUTE_OUTPUT_BF16_ENDPOINT, VLLM_SILU_BF16_INTERMEDIATE
    ROUTE_OUTPUT_BF16_ENDPOINT = tl.constexpr(
        args.route_output_bf16_endpoint
    )
    VLLM_SILU_BF16_INTERMEDIATE = tl.constexpr(
        args.vllm_silu_bf16_intermediate
    )
    configure_shape(
        args.tokens,
        args.block_m,
        args.gate_block_n if args.gate_block_n is not None else args.block_n,
        args.down_block_n if args.down_block_n is not None else args.block_n,
        args.gate_block_k if args.gate_block_k is not None else args.block_k,
        args.down_block_k if args.down_block_k is not None else args.block_k,
        args.group_m,
    )
    compile_all(
        args.output_dir,
        args.metadata,
        route_launch_options={
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "waves_per_eu": args.waves_per_eu,
        },
        gate_launch_options={
            "num_warps": (
                args.gate_num_warps
                if args.gate_num_warps is not None
                else args.num_warps
            ),
            "num_stages": (
                args.gate_num_stages
                if args.gate_num_stages is not None
                else args.num_stages
            ),
            "waves_per_eu": (
                args.gate_waves_per_eu
                if args.gate_waves_per_eu is not None
                else args.waves_per_eu
            ),
        },
        down_launch_options={
            "num_warps": (
                args.down_num_warps
                if args.down_num_warps is not None
                else args.num_warps
            ),
            "num_stages": (
                args.down_num_stages
                if args.down_num_stages is not None
                else args.num_stages
            ),
            "waves_per_eu": (
                args.down_waves_per_eu
                if args.down_waves_per_eu is not None
                else args.waves_per_eu
            ),
        },
    )


if __name__ == "__main__":
    main()
