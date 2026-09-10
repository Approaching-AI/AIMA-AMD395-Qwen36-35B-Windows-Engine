#!/usr/bin/env python3
"""Enumerate model-independent SM121 reciprocal deltas for attention sums."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

ENTRIES = 1 << 23
MAX_EXPONENT = 18
HEADER = struct.Struct("<8s6I")


def reciprocal_kernel(table, errors, start, exponent, N: tl.constexpr,
                      WRITE: tl.constexpr, BLOCK: tl.constexpr):
    local = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    index = start + local
    base_bits = (0x3F800000 + index).to(tl.uint32)
    base = base_bits.to(tl.float32, bitcast=True)
    x = (base_bits + (exponent.to(tl.uint32) << 23)).to(tl.float32, bitcast=True)
    actual = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
                                      args=[x], dtype=tl.float32, is_pure=True, pack=1).to(tl.int32, bitcast=True)
    rn = tl.inline_asm_elementwise("rcp.rn.f32 $0, $1;", constraints="=f,f",
                                  args=[base], dtype=tl.float32, is_pure=True, pack=1).to(tl.int32, bitcast=True)
    if WRITE:
        delta = actual - rn
        wrong = (local < N) & ((delta < -127) | (delta > 127))
        tl.store(table + index, delta, local < N)
    else:
        delta = tl.load(table + index, local < N, other=0).to(tl.int32)
        expected = rn + delta - (exponent << 23)
        wrong = (local < N) & (actual != expected)
    tl.store(errors + tl.program_id(0), tl.sum(wrong.to(tl.int32), 0))


def execute(args):
    global tl
    import torch
    import triton
    import triton.language as tl

    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("SM121 required")
    table = torch.empty(ENTRIES, dtype=torch.int8, device="cuda")
    batch = 1 << 20
    errors = torch.empty(batch // 256, dtype=torch.int32, device="cuda")
    kernel = triton.jit(reciprocal_kernel)
    maximum_ms = 0.0
    dispatches = 0
    for write, exponents in ((True, [0]), (False, range(MAX_EXPONENT + 1))):
        for exponent in exponents:
            for start in range(0, ENTRIES, batch):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # Import, JIT and lazy launcher setup are excluded from dispatch timing.
                compiled = kernel[(batch // 256,)](table, errors, start, exponent, batch, write, 256)
                compiled.run
                begin.record()
                kernel[(batch // 256,)](table, errors, start, exponent, batch, write, 256)
                end.record(); end.synchronize()
                elapsed = begin.elapsed_time(end)
                maximum_ms = max(maximum_ms, elapsed)
                dispatches += 1
                if elapsed > 1000 or int(errors.sum().item()):
                    raise ValueError("reciprocal delta or scale control failed")
    host = table.cpu().numpy()
    payload = HEADER.pack(b"QRCPTB01", 1, ENTRIES, 0, MAX_EXPONENT, 1, 0) + host.tobytes()
    path = args.output_dir / "sm121-attention-rcp-delta-i8.bin"
    with path.open("xb") as stream:
        stream.write(payload)
    if torch.cuda.max_memory_allocated() > 32 << 20:
        raise ValueError("primitive builder exceeds its device limit")
    return dict(completed=True, entries=ENTRIES, minimum_exponent=0, maximum_exponent=MAX_EXPONENT,
                verified_inputs=ENTRIES * (MAX_EXPONENT + 1), bit_mismatches=0,
                minimum_delta=int(host.min()), maximum_delta=int(host.max()),
                dispatches=dispatches, maximum_dispatch_ms=maximum_ms,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                torch_version=torch.__version__, triton_version=triton.__version__,
                artifact=dict(file=path.name, bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.timeout_seconds <= 60 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit) or args.output_dir.exists()):
        raise ValueError("invalid commit, deadline or existing output")
    if args.worker and not args.execute:
        raise ValueError("worker requires explicit execution")
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname() != args.expected_host:
            raise ValueError("execution host mismatch before GPU import")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="sm121_model_independent_attention_reciprocal", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)), completed=False,
                  model_inputs=False, prompt_inputs=False, inference_acceptance=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
