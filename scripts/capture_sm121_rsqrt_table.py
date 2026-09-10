#!/usr/bin/env python3
"""Build a model-independent SM121 rsqrt table and exhaustively verify scaling.

Only the mantissas of [1,4) are stored. Every nonnegative finite FP32 input,
positive infinity and the FTZ subnormal range must match exponent scaling of
this table before the artifact is published. No model or prompt input exists.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha, pack_pages

BEGIN, END = 0x3F800000, 0x40800000
DOMAIN_END = 0x7F800001
PAGE, BATCH = 256, 1 << 20
DEVICE_LIMIT = 96 << 20
HEADER = struct.Struct("<8s8IQ")


def scaled_result(input_bits: int, base_bits: int) -> int:
    """CPU definition of the independent exponent rule checked on the GPU."""
    if not 0 <= input_bits <= 0x7F800000:
        raise ValueError("rsqrt domain is nonnegative finite FP32 and infinity")
    if input_bits < 0x00800000:
        return 0x7F800000
    if input_bits == 0x7F800000:
        return 0
    exponent = (input_bits >> 23) - 127
    return base_bits - ((exponent // 2) * (1 << 23))


def rsqrt_table_kernel(table, failures, start, count, VERIFY: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = start.to(tl.uint32) + index.to(tl.uint32)
    argument = bits.to(tl.float32, bitcast=True)
    actual = tl.inline_asm_elementwise("rsqrt.approx.ftz.f32 $0, $1;", constraints="=f,f", args=[argument],
                                       dtype=tl.float32, is_pure=True, pack=1).to(tl.uint32, bitcast=True)
    if VERIFY:
        exponent = (bits >> 23).to(tl.int32) - 127
        base_index = (bits & 0x7FFFFF) | ((exponent & 1).to(tl.uint32) << 23)
        normal = (bits >= 0x00800000) & (bits < 0x7F800000) & (index < count)
        base = tl.load(table + base_index, normal, other=0)
        expected = base - ((exponent >> 1) * 0x800000).to(tl.uint32)
        expected = tl.where(bits < 0x00800000, 0x7F800000, tl.where(bits == 0x7F800000, 0, expected))
        wrong = tl.sum(((index < count) & (actual != expected)).to(tl.int32), 0)
        tl.store(failures + tl.program_id(0), wrong)
    else:
        tl.store(table + (bits - 0x3F800000), actual, index < count)


def gpu_build(args, record):
    import fcntl
    import tempfile
    import numpy as np
    import torch
    import triton
    global tl
    import triton.language as tl
    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("requires exactly one SM121 CUDA device")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < (1 << 30) + DEVICE_LIMIT:
        raise ValueError("rsqrt device reserve unavailable")
    with (Path(tempfile.gettempdir()) / "qrt-sm121-rsqrt-table.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.cuda.reset_peak_memory_stats()
        table = torch.empty(END - BEGIN, dtype=torch.uint32, device="cuda")
        table.view(torch.uint8).fill_(0x5A)
        failures = torch.empty(BATCH // 512, dtype=torch.int32, device="cuda")
        kernel = triton.jit(rsqrt_table_kernel, do_not_specialize=["start", "count"])
        ptx_hashes = {}
        for verify in (False, True):
            compiled = kernel.warmup(table, failures, BEGIN, BATCH, VERIFY=verify, BLOCK=512,
                                     grid=(BATCH // 512,), num_warps=4, enable_fp_fusion=False)
            if not callable(compiled.run):
                raise ValueError("rsqrt launcher unavailable")
            path = args.output_dir / ("verify.ptx" if verify else "base.ptx")
            path.write_text(compiled.asm["ptx"])
            ptx_hashes[str(verify)] = file_sha(path)
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        calls, maximum_ms, verified = 0, 0.0, 0
        directory = bytearray((END - BEGIN) // PAGE * 8)
        payload_bytes = 0
        partial = args.output_dir / "sm121-rsqrt-positive.bin.part"
        started = time.monotonic()

        def launch(first, count, verify):
            nonlocal calls, maximum_ms
            if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
                raise ValueError("rsqrt allocation ceiling exceeded")
            start_event.record()
            actual = kernel[(triton.cdiv(count, 512),)](table, failures, first, count, VERIFY=verify, BLOCK=512,
                                                        num_warps=4, enable_fp_fusion=False)
            end_event.record(); end_event.synchronize()
            elapsed = start_event.elapsed_time(end_event)
            calls += 1
            maximum_ms = max(maximum_ms, elapsed)
            import hashlib
            if elapsed > 100 or hashlib.sha256(actual.asm["ptx"].encode()).hexdigest() != ptx_hashes[str(verify)]:
                raise ValueError("rsqrt dispatch admission or PTX changed; no further submission")
            if verify and int(failures[:triton.cdiv(count, 512)].cpu().numpy().sum()) != 0:
                raise ValueError(f"rsqrt exponent scaling fails at input batch {first:#x}")

        with partial.open("w+b") as output:
            output.seek(HEADER.size + len(directory))
            for first in range(BEGIN, END, BATCH):
                count = min(BATCH, END - first)
                launch(first, count, False)
                values = table[first - BEGIN:first - BEGIN + count].cpu().numpy()
                if np.any(values < 0x3F000000) or np.any(values > 0x3F800000):
                    raise ValueError("rsqrt base table invalid or unwritten")
                packed, pieces, payload_bytes = pack_pages(values, payload_bytes)
                offset = (first - BEGIN) // PAGE * 8
                directory[offset:offset + len(packed)] = packed
                for piece in pieces:
                    output.write(piece)
            for first in range(0, DOMAIN_END, BATCH):
                count = min(BATCH, DOMAIN_END - first)
                launch(first, count, True)
                verified += count
                if calls % 64 == 0 or verified == DOMAIN_END:
                    with (args.output_dir / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(calls=calls, verified_inputs=verified,
                                                     maximum_dispatch_ms=maximum_ms,
                                                     wall_ms=(time.monotonic() - started) * 1000)) + "\n")
            if verified != DOMAIN_END:
                raise ValueError("incomplete rsqrt domain")
            output.seek(0)
            output.write(HEADER.pack(b"QRSQTBL1", 1, 8, BEGIN, END, 0x7F800000, 0, len(directory) // 8, 1, payload_bytes))
            output.write(directory)
        final = partial.with_suffix("")
        partial.rename(final)
        record.update(completed=True, kernel_executed=True, verified_inputs=verified, launches=calls,
                      maximum_dispatch_ms=maximum_ms, payload_bytes=payload_bytes, directory_bytes=len(directory),
                      table_file=final.name, table_bytes=final.stat().st_size, table_sha256=file_sha(final),
                      wall_ms=(time.monotonic() - started) * 1000, ptx_sha256=ptx_hashes,
                      device=torch.cuda.get_device_name(0), torch_version=torch.__version__, triton_version=triton.__version__,
                      peak_device_bytes=torch.cuda.max_memory_allocated(), exponent_scaling_exhaustively_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 300 or len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("invalid rsqrt deadline or source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires explicit execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists():
        raise ValueError("output exists; never overwrite a previous capture")
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("rsqrt execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="exhaustive_sm121_nonnegative_rsqrt_table", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)),
                  helper_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
                                 ("capture_sm121_exp2_table.py", "capture_fla_state_prefix.py",
                                  "audit_fla_reference_state_ir.py", "fla_state_exponent_capture.py", "prepare_fla_state_prefix.py")},
                  model_loaded=False, model_or_prompt_inputs=False, inference_acceptance=False, kernel_executed=False,
                  input_domain_end=DOMAIN_END, base_begin=BEGIN, base_end=END,
                  timeout_seconds=args.timeout_seconds, maximum_device_bytes=DEVICE_LIMIT,
                  per_dispatch_admission_ms=100, completed=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            gpu_build(args, record)
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error), completed=False)) + "\n")
            raise
        (args.output_dir / "table.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
