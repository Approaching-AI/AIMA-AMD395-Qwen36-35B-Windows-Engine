#!/usr/bin/env python3
"""Exhaustive, model-independent nonpositive FP32 ex2.approx compatibility table.

No prompt, token or model input is accepted. Constant exterior ranges are
exhaustively checked on the same SM121 as the packed interior table. This is
an offline artifact builder, never an inference or performance acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import time

from capture_fla_state_prefix import arm_parent_death, supervise

BEGIN = 0x2F800000  # abs(x) = 2**-32; lower magnitudes must all return 1.
END = 0x43180000    # abs(x) = 152; larger finite magnitudes must all return 0.
FINITE_END = 0x7F800001  # Includes negative infinity, excludes NaNs.
PAGE_BITS = 8
PAGE = 1 << PAGE_BITS
BATCH = 1 << 20
HEADER = struct.Struct("<8s8IQ")


def exp2_table_kernel(output, start, count, expected, FLAT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    magnitude = start.to(tl.uint32) + index.to(tl.uint32)
    argument = (magnitude | 0x80000000).to(tl.float32, bitcast=True)
    result = tl.inline_asm_elementwise("ex2.approx.f32 $0, $1;", constraints="=f,f", args=[argument],
                                       dtype=tl.float32, is_pure=True, pack=1).to(tl.uint32, bitcast=True)
    if FLAT:
        wrong = tl.sum(((index < count) & (result != expected)).to(tl.int32), 0)
        tl.store(output + tl.program_id(0), wrong)
    else:
        tl.store(output + index, result, index < count)


def pack_pages(values, offset):
    import numpy as np
    pages = np.asarray(values, dtype=np.uint32).reshape(-1, PAGE)
    base, top = pages.min(axis=1), pages.max(axis=1)
    span = top - base
    entries = np.zeros((len(pages), 2), dtype="<u4")
    entries[:, 0] = base
    pieces = []
    for kind, low, high, dtype in ((1, 1, 255, "u1"), (2, 256, 65535, "<u2"), (3, 65536, 0xFFFFFFFF, "<u4")):
        selected = (span >= low) & (span <= high)
        count = int(selected.sum())
        width = np.dtype(dtype).itemsize
        if offset + count * PAGE * width >= 1 << 30:
            raise ValueError("table payload exceeds its 30-bit byte offset")
        entries[selected, 1] = (np.arange(count, dtype=np.uint32) * PAGE * width + offset) | (kind << 30)
        raw = (pages[selected] - base[selected, None]).astype(dtype).tobytes()
        pieces.append(raw)
        offset += len(raw)
    return entries.tobytes(), pieces, offset


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(4 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


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
    if not 1 <= args.timeout_seconds <= 300:
        raise ValueError("invalid supervisor timeout")
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("source commit must be a full SHA")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists():
        raise ValueError("output exists; inspect the existing run before further work")
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("explicit Linux execution host mismatch")
        if not args.worker:
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--supervisor-pid", str(os.getpid())]
            raise SystemExit(supervise(command, args.timeout_seconds))
    pages = (END - BEGIN) // PAGE
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="exhaustive_sm121_nonpositive_exp2_table", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)), model_loaded=False,
                  model_or_prompt_inputs=False, inference_acceptance=False, kernel_executed=False,
                  magnitude_begin=BEGIN, magnitude_end=END, page_bits=PAGE_BITS, page_count=pages,
                  exhaustive_input_count=FINITE_END, timeout_seconds=args.timeout_seconds,
                  maximum_device_bytes=64 << 20, per_dispatch_admission_ms=100, completed=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(record))
        return
    import fcntl
    import tempfile
    with (Path(tempfile.gettempdir()) / "qrt-sm121-exp2-table.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch
        import triton
        global tl
        import triton.language as tl
        if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
            raise ValueError("requires one visible SM121 CUDA device")
        torch.set_num_threads(2)
        if torch.cuda.mem_get_info()[0] < (1 << 30) + (64 << 20):
            raise ValueError("insufficient device reserve")
        torch.cuda.reset_peak_memory_stats()

        buffer = torch.empty(BATCH, device="cuda", dtype=torch.uint32)
        # Fix every runtime scalar specialization; load modules outside timing.
        probe = triton.jit(exp2_table_kernel, do_not_specialize=["start", "count", "expected"])
        ptx_hashes = {}
        for flat in (True, False):
            compiled = probe.warmup(buffer, BEGIN, BATCH, 0, FLAT=flat, BLOCK=512, grid=(BATCH // 512,),
                                    num_warps=4, enable_fp_fusion=False)
            if not callable(compiled.run):
                raise ValueError("compiled exp2 launcher unavailable")
            data = compiled.asm["ptx"].encode()
            name = "constant.ptx" if flat else "table.ptx"
            (args.output_dir / name).write_bytes(data)
            ptx_hashes[str(flat)] = hashlib.sha256(data).hexdigest()
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        directory = bytearray(pages * 8)
        payload_bytes, calls, maximum_ms, verified = 0, 0, 0.0, 0
        table = args.output_dir / "sm121-ex2-negative.bin.part"
        started = time.monotonic()
        with table.open("w+b") as output:
            output.seek(HEADER.size + len(directory))
            for lower, upper, flat, expected in ((0, BEGIN, True, 0x3F800000), (BEGIN, END, False, 0),
                                                 (END, FINITE_END, True, 0)):
                for first in range(lower, upper, BATCH):
                    count = min(BATCH, upper - first)
                    start_event.record()
                    actual = probe[(triton.cdiv(count, 512),)](buffer, first, count, expected, FLAT=flat, BLOCK=512,
                                                              num_warps=4, enable_fp_fusion=False)
                    end_event.record(); end_event.synchronize()
                    elapsed = start_event.elapsed_time(end_event)
                    maximum_ms = max(maximum_ms, elapsed)
                    calls += 1
                    if elapsed > 100 or torch.cuda.max_memory_allocated() > 64 << 20:
                        raise ValueError("dispatch or memory admission exceeded; no further submission")
                    if hashlib.sha256(actual.asm["ptx"].encode()).hexdigest() != ptx_hashes[str(flat)]:
                        raise ValueError("unexpected code specialization in table sweep")
                    if flat:
                        if int(buffer[:triton.cdiv(count, 512)].cpu().numpy().sum()) != 0:
                            raise ValueError(f"exterior is not constant at magnitude {first:#x}")
                    else:
                        values = buffer[:count].cpu().numpy()
                        if np.any(values > 0x3F800000):
                            raise ValueError("nonfinite or out-of-range negative exponential")
                        packed, pieces, payload_bytes = pack_pages(values, payload_bytes)
                        offset = (first - BEGIN) // PAGE * 8
                        directory[offset:offset + len(packed)] = packed
                        for piece in pieces:
                            output.write(piece)
                    verified += count
                    if calls % 64 == 0 or first + count == upper:
                        progress = dict(calls=calls, verified_inputs=verified, phase="constant" if flat else "table",
                                        magnitude_end=first + count, maximum_dispatch_ms=maximum_ms,
                                        payload_bytes=payload_bytes, wall_ms=(time.monotonic() - started) * 1000)
                        with (args.output_dir / "progress.jsonl").open("a") as stream:
                            stream.write(json.dumps(progress) + "\n")
            if verified != FINITE_END:
                raise ValueError("incomplete input domain")
            output.seek(0)
            output.write(HEADER.pack(b"QEX2TBL1", 1, PAGE_BITS, BEGIN, END, 0x3F800000, 0, pages, 1, payload_bytes))
            output.write(directory)
        final = table.with_suffix("")
        table.rename(final)
        record.update(completed=True, kernel_executed=True, verified_inputs=verified, launches=calls,
                      maximum_dispatch_ms=maximum_ms, payload_bytes=payload_bytes, directory_bytes=len(directory),
                      table_file=final.name, table_bytes=final.stat().st_size, table_sha256=file_sha(final),
                      wall_ms=(time.monotonic() - started) * 1000, ptx_sha256=ptx_hashes,
                      device=torch.cuda.get_device_name(0), torch_version=torch.__version__, triton_version=triton.__version__,
                      peak_device_bytes=torch.cuda.max_memory_allocated(), constant_exteriors_exhaustively_verified=True)
        (args.output_dir / "table.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
