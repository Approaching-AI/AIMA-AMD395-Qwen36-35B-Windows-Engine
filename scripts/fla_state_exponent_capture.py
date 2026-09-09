#!/usr/bin/env python3
"""Bounded SM121 exponent attribution, separate from the reference state kernel.

Import and argument preparation are CPU-only. GPU dispatch is called only by
the guarded prefix sampler after its BF16 and raw-state controls pass. Running
this file directly only compiles an offline CUDA target with all GPUs hidden.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import sys
import types

BLOCK = 256
LOG2E_BITS = 0x3FB8AA3B


def fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def exponent_arguments(gates: bytes, tokens: int) -> bytes:
    """Gate differences first [T,32], then old-state decay [T/64,32]."""
    if type(tokens) is not int or not 64 <= tokens <= 1024 or tokens % 64:
        raise ValueError("exponent capture requires complete 64-token chunks, at most 1024 tokens")
    if len(gates) != tokens * 32 * 4:
        raise ValueError("gate byte count mismatch")
    values = struct.unpack(f"<{tokens * 32}f", gates)
    if any(not math.isfinite(x) or x > 0 for x in values):
        raise ValueError("state gate must be finite and nonpositive")
    log2e = struct.unpack("<f", struct.pack("<I", LOG2E_BITS))[0]
    result = []
    for t in range(tokens):
        last = (t // 64 + 1) * 64 - 1
        for head in range(32):
            if t % 64 and values[t * 32 + head] > values[(t - 1) * 32 + head]:
                raise ValueError("state gate must be nonincreasing within each chunk")
            # Preserve BOTH F32 rounding points from the inspected state PTX.
            result.append(f32(f32(values[last * 32 + head] - values[t * 32 + head]) * log2e))
    for chunk in range(tokens // 64):
        for head in range(32):
            result.append(f32(values[((chunk + 1) * 64 - 1) * 32 + head] * log2e))
    if any(not math.isfinite(x) or x > 0 for x in result):
        raise ValueError("exponent argument must be finite and nonpositive")
    return struct.pack(f"<{len(result)}f", *result)


def validate_samples(expected_arguments: bytes, actual_arguments: bytes, values: bytes) -> dict:
    if not expected_arguments or len(expected_arguments) % 4:
        raise ValueError("invalid exponent argument byte count")
    if actual_arguments != expected_arguments or len(values) != len(expected_arguments):
        raise ValueError("exponent argument or output byte count mismatch")
    observed = {}
    for (argument,), (bits,) in zip(struct.iter_unpack("<I", actual_arguments), struct.iter_unpack("<I", values)):
        output = struct.unpack("<f", struct.pack("<I", bits))[0]
        if not math.isfinite(output) or not 0 <= output <= 1 or (argument & 0x7FFFFFFF == 0 and bits != 0x3F800000):
            raise ValueError("invalid or unwritten exponent output")
        # Normalize signed zero only in the input key; retain raw output bits.
        key = argument if argument & 0x7FFFFFFF else 0
        if key in observed and observed[key] != bits:
            raise ValueError("identical exponent inputs produced inconsistent output bits")
        observed[key] = bits
    return dict(elements=len(values) // 4, unique_inputs=len(observed), arguments_exact=True,
                duplicate_inputs_consistent=True, all_finite=True)


# This function is extracted, not called as Python. Its two stores are outside
# the original state kernel and cannot change that kernel's register lifetime.
def state_exponent_kernel(g, arguments, values, T: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    count = (T + T // 64) * 32
    is_gate = index < T * 32
    chunk = tl.where(is_gate, index // (64 * 32), (index - T * 32) // 32)
    head = index % 32
    last = (chunk + 1) * 64 - 1
    last_gate = tl.load(g + last * 32 + head, index < count, other=0)
    current = tl.load(g + index, is_gate & (index < count), other=0)
    argument, output = tl.inline_asm_elementwise(
        "{ .reg .f32 delta; sub.rn.f32 delta, $2, $3; "
        "mul.rn.f32 $0, delta, 0f3FB8AA3B; ex2.approx.f32 $1, $0; }",
        constraints="=f,=f,f,f", args=[last_gate, current],
        dtype=(tl.float32, tl.float32), is_pure=True, pack=1)
    tl.store(arguments + index, argument, index < count)
    tl.store(values + index, output, index < count)


def load_exponent_kernel(triton, tl):
    source = Path(__file__)
    tree = ast.parse(source.read_text(), filename=str(source))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "state_exponent_kernel")
    module = types.ModuleType("_qrt_fla_state_exponent_kernel")
    module.__file__ = str(source)
    module.__dict__["tl"] = tl
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), module.__dict__)
    return triton.jit(module.__dict__[function.name])


def capture_exponents(*, tokens, gates, device_gates, output_dir, torch, triton, tl, progress, device_limit):
    expected = exponent_arguments(gates, tokens)
    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("exponent probe requires one SM121 CUDA device")
    kernel = load_exponent_kernel(triton, tl)
    storage, outputs = {}, {}
    for name in ("arguments", "values"):
        raw = b"\x5a" * 256 + b"\x00\x00\xc0\x7f" * (len(expected) // 4) + b"\x5a" * 256
        storage[name] = torch.frombuffer(bytearray(raw), dtype=torch.float32).to(device="cuda")
        outputs[name] = storage[name][64:-64]
    if torch.cuda.max_memory_allocated() > device_limit:
        raise ValueError("device allocation ceiling exceeded before exponent compilation")
    arguments = (device_gates, outputs["arguments"], outputs["values"])
    options = dict(T=tokens, BLOCK=BLOCK, num_warps=4, num_stages=1, enable_fp_fusion=False)
    grid = (triton.cdiv(len(expected) // 4, BLOCK),)
    progress("compile", component="exponent_probe")
    prepared = kernel.warmup(*arguments, grid=grid, **options)
    ptx = prepared.asm["ptx"].encode()
    (output_dir / "state-exponents.ptx").write_bytes(ptx)
    progress("compiled", component="exponent_probe", ptx_sha256=fingerprint(ptx), options=options)
    if torch.cuda.max_memory_allocated() > device_limit:
        raise ValueError("device allocation ceiling exceeded before exponent dispatch")
    progress("dispatch", component="exponent_probe", elements=len(expected) // 4)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    compiled = kernel[grid](*arguments, **options)
    end.record()
    end.synchronize()
    elapsed = start.elapsed_time(end)
    progress("completed_dispatch", component="exponent_probe", elapsed_ms=elapsed)
    if elapsed > 100:
        raise ValueError("100 ms exponent post-dispatch admission exceeded; no further launch")
    if fingerprint(compiled.asm["ptx"].encode()) != fingerprint(ptx):
        raise ValueError("exponent PTX differs from the warmup artifact")
    result = {}
    for name, tensor in storage.items():
        raw = tensor.view(torch.uint8).cpu().numpy().tobytes()
        if raw[:256] != b"\x5a" * 256 or raw[-256:] != b"\x5a" * 256:
            raise ValueError("exponent output redzone changed")
        result[name] = raw[256:-256]
    validation = validate_samples(expected, result["arguments"], result["values"])
    if device_gates.view(torch.uint8).cpu().numpy().tobytes() != gates:
        raise ValueError("gate input changed during exponent probe")
    files = {}
    for name, data in result.items():
        filename = f"state-exponent-{name}-f32.bin"
        (output_dir / filename).write_bytes(data)
        files[name] = dict(file=filename, bytes=len(data), sha256=fingerprint(data))
    return dict(kind="separate_sm121_state_exp2_capture", validation=validation, files=files,
                elapsed_ms=elapsed, launches=1, ptx_sha256=fingerprint(ptx), gate_input_sha256=fingerprint(gates),
                layout=dict(gate_shape=[tokens, 32], decay_shape=[tokens // 64, 32], decay_offset_elements=tokens * 32),
                kernel_executed=True, state_kernel_modified=False, original_worker_register_capture=False,
                production_implementation=False, inference_acceptance=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=384)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    exponent_arguments(b"\0" * (max(0, min(args.tokens, 1024)) * 32 * 4), args.tokens)
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(name) != "-1":
            raise ValueError(f"CPU-only compiler requires {name}=-1")
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    kernel = load_exponent_kernel(triton, tl)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    options = dict(num_warps=4, num_stages=1, enable_fp_fusion=False)
    compiled = triton.compile(ASTSource(kernel, signature=dict(g="*fp32", arguments="*fp32", values="*fp32"),
                                       constexprs=dict(T=args.tokens, BLOCK=BLOCK)),
                              target=GPUTarget("cuda", 121, 32), options=options)
    artifacts = []
    for name in ("ttir", "ptx"):
        data = compiled.asm[name].encode()
        filename = "state-exponents." + name
        (args.output_dir / filename).write_bytes(data)
        artifacts.append(dict(file=filename, bytes=len(data), sha256=fingerprint(data)))
    record = dict(kind="cpu_only_state_exponent_ir_audit", host=socket.gethostname(), command=sys.argv,
                  command_source_sha256=fingerprint(Path(__file__).read_bytes()), tokens=args.tokens,
                  triton_version=triton.__version__, target="cuda-sm121", options=options,
                  artifacts=artifacts, kernel_executed=False, inference_acceptance=False)
    (args.output_dir / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
