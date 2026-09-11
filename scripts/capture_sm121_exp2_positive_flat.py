#!/usr/bin/env python3
"""Verify the complete tiny-positive FP32 exp2 interval on the GB10 authority."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

END = 0x33000000  # Exclusive: 2**-25. Every result must be exactly FP32 one.
BATCH = 1 << 20


def verify_kernel(output, start, count, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = start.to(tl.uint32) + index.to(tl.uint32)
    argument = bits.to(tl.float32, bitcast=True)
    result = tl.inline_asm_elementwise("ex2.approx.f32 $0, $1;", constraints="=f,f",
                                      args=[argument], dtype=tl.float32,
                                      is_pure=True, pack=1).to(tl.uint32, bitcast=True)
    wrong = tl.sum(((index < count) & (result != 0x3F800000)).to(tl.int32), 0)
    tl.store(output + tl.program_id(0), wrong)


def execute(output):
    import torch
    import triton
    import triton.language as tl

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("one original SM121 CUDA device is required")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < 512 << 20:
        raise ValueError("device reserve unavailable")
    globals()["tl"] = tl
    kernel = triton.jit(verify_kernel)
    counts = torch.empty(BATCH // 1024, device="cuda", dtype=torch.int32)
    compiled = kernel[(BATCH // 1024,)](counts, 0, BATCH, BLOCK=1024, num_warps=4)
    torch.cuda.synchronize()
    ptx = compiled.asm["ptx"]
    (output / "verify.ptx").write_text(ptx)
    started = time.monotonic()
    verified, maximum_ms = 0, 0.0
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for begin in range(0, END, BATCH):
        if time.monotonic() - started > 45:
            raise ValueError("positive-domain enumeration exceeded its deadline")
        size = min(BATCH, END - begin)
        blocks = triton.cdiv(size, 1024)
        start.record()
        kernel[(blocks,)](counts, begin, size, BLOCK=1024, num_warps=4)
        stop.record()
        stop.synchronize()
        ms = start.elapsed_time(stop)
        maximum_ms = max(maximum_ms, ms)
        if ms > 1000 or int(counts[:blocks].sum().item()):
            raise ValueError("positive-domain exp2 result or dispatch bound failed")
        verified += size
    return dict(completed=True, verified_inputs=verified, mismatches=0,
                output_bits=0x3F800000, maximum_dispatch_ms=maximum_ms,
                enumeration_seconds=time.monotonic() - started,
                torch_version=torch.__version__, triton_version=triton.__version__,
                ptx_sha256=file_sha(output / "verify.ptx"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.output_dir.exists() or not 1 <= args.timeout_seconds <= 120 or
            len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("existing output or invalid source/deadline")
    if args.worker and not args.execute:
        raise ValueError("worker requires explicit execution")
    if args.execute:
        if sys.platform != "linux" or socket.gethostname() != args.expected_host:
            raise ValueError("explicit GB10 execution host required")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            return supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                              "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds)
    args.output_dir.mkdir(parents=True)
    record = dict(kind="sm121_exp2_tiny_positive_exhaustive", host=socket.gethostname(),
                  command=sys.argv, source_commit=args.source_commit,
                  source_sha256=file_sha(Path(__file__)), model=None,
                  input_bits_begin=0, input_bits_end_exclusive=END,
                  completed=False, model_independent=True, inference_acceptance=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        record.update(execute(args.output_dir))
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
