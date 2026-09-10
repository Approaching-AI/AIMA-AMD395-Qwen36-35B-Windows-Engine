#!/usr/bin/env python3
"""Enumerate the SM121 Triton SiLU BF16 endpoint over every finite FP32 input.

A real convolution control is required before enumeration. Its expected values
only validate the expression; every table entry comes from the entire numeric
domain. Store every output transition, including local nonmonotonic transitions,
with a page directory for bounded native lookup. No model is loaded.
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

FINITE_END = 0x7F800000
BATCH = 1 << 22
CAPACITY = 1 << 16
HEADER = struct.Struct("<8s6I4Q")


def silu_control_kernel(arguments, output, count, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(arguments + i, i < count, other=0)
    y = (x / (1.0 + tl.exp(-x))).to(tl.bfloat16)
    tl.store(output + i, y, i < count)


def silu_transition_kernel(indices, values, counter, start, count,
                           NEGATIVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    magnitude = start.to(tl.uint32) + i.to(tl.uint32)
    previous = tl.maximum(magnitude, 1) - 1
    if NEGATIVE:
        raw = magnitude | 0x80000000
        previous_raw = previous | 0x80000000
    else:
        raw = magnitude
        previous_raw = previous
    x = raw.to(tl.float32, bitcast=True)
    before = previous_raw.to(tl.float32, bitcast=True)
    y = (x / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    prior_y = (before / (1.0 + tl.exp(-before))).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    changed = (i < count) & ((magnitude == 0) | (y != prior_y))
    # One bounded index reservation per CTA; every store still checks capacity.
    prefix = tl.cumsum(changed.to(tl.int32), 0)
    total = tl.sum(changed.to(tl.int32), 0)
    base = tl.atomic_add(counter, total)
    offset = base + prefix - 1
    tl.store(indices + offset, raw, changed & (offset < 65536))
    tl.store(values + offset, y, changed & (offset < 65536))


def silu_verify_kernel(directory, keys, endpoints, errors, start, count,
                       NEGATIVE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    magnitude = start.to(tl.uint32) + i.to(tl.uint32)
    raw = magnitude | 0x80000000 if NEGATIVE else magnitude
    page = raw >> 16
    lower = tl.load(directory + page)
    upper = tl.load(directory + page + 1) + 1
    while tl.sum((lower < upper).to(tl.int32), 0) > 0:
        active = lower < upper
        middle = (lower + upper) // 2
        key = tl.load(keys + middle, active, other=0xFFFFFFFF)
        right = active & (key <= raw)
        lower = tl.where(right, middle + 1, lower)
        upper = tl.where(active & ~right, middle, upper)
    found = tl.load(endpoints + lower - 1)
    x = raw.to(tl.float32, bitcast=True)
    expected = (x / (1.0 + tl.exp(-x))).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    wrong = tl.sum(((i < count) & (found != expected)).to(tl.int32), 0)
    tl.store(errors + tl.program_id(0), wrong)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("source_commit", "control_sha256"):
        digest = getattr(args, name)
        if len(digest) != (40 if name == "source_commit" else 64) or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid source/control fingerprint")
    if not 1 <= args.timeout_seconds <= 300:
        raise ValueError("invalid deadline")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists():
        raise ValueError("output exists; inspect it before further work")
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("explicit Linux execution host mismatch")
        if not args.worker:
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--supervisor-pid", str(os.getpid())]
            raise SystemExit(supervise(command, args.timeout_seconds))
    control_path = args.control_dir / "control.json"
    if sha(control_path) != args.control_sha256:
        raise ValueError("control manifest fingerprint mismatch")
    control = json.loads(control_path.read_text())
    arguments_path = args.control_dir / "arguments-f32.bin"
    expected_path = args.control_dir / "expected-bf16.bin"
    if sha(arguments_path) != control["arguments_sha256"] or sha(expected_path) != control["expected_sha256"]:
        raise ValueError("control tensor fingerprint mismatch")
    count = arguments_path.stat().st_size // 4
    if not 1 <= count <= 65536 or arguments_path.stat().st_size != count * 4 or expected_path.stat().st_size != count * 2:
        raise ValueError("control span mismatch")
    record = dict(kind="exhaustive_sm121_triton_silu_bf16", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=sha(Path(__file__)), control_sha256=args.control_sha256,
                  model_loaded=False, table_entries_model_independent=True, inference_acceptance=False,
                  finite_inputs=FINITE_END * 2, batch_inputs=BATCH, maximum_device_bytes=16 << 20,
                  per_dispatch_admission_ms=100, timeout_seconds=args.timeout_seconds, completed=False)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if not args.execute:
        print(json.dumps(record))
        return
    import fcntl
    import tempfile
    with (Path(tempfile.gettempdir()) / "qrt-sm121-silu-table.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import numpy as np
        import torch
        import triton
        global tl
        import triton.language as tl
        if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
            raise ValueError("requires one visible SM121 CUDA device")
        torch.set_num_threads(2)
        if torch.cuda.mem_get_info()[0] < (1 << 30) + (16 << 20):
            raise ValueError("insufficient device reserve")
        torch.cuda.reset_peak_memory_stats()
        inputs = torch.from_numpy(np.fromfile(arguments_path, dtype="<f4")).cuda()
        result = torch.empty(count, dtype=torch.bfloat16, device="cuda")
        probe = triton.jit(silu_control_kernel)
        compiled = probe.warmup(inputs, result, count, BLOCK=256, grid=(triton.cdiv(count, 256),), num_warps=4)
        if not callable(compiled.run):
            raise ValueError("control launcher unavailable")
        compiled = probe[(triton.cdiv(count, 256),)](inputs, result, count, BLOCK=256, num_warps=4)
        torch.cuda.synchronize()
        expected = np.fromfile(expected_path, dtype="<u2")
        actual = result.view(torch.uint16).cpu().numpy()
        differences = int(np.count_nonzero(actual != expected))
        (args.output_dir / "control.ptx").write_text(compiled.asm["ptx"])
        control_result = dict(elements=count, bf16_differences=differences,
                              ptx_sha256=sha(args.output_dir / "control.ptx"),
                              actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                              first_differences=np.flatnonzero(actual != expected)[:16].tolist())
        (args.output_dir / "control.json").write_text(json.dumps(control_result, indent=2) + "\n")
        if differences:
            raise ValueError("SiLU expression differs from the real convolution reference")

        indices = torch.empty(CAPACITY, dtype=torch.uint32, device="cuda")
        values = torch.empty(CAPACITY, dtype=torch.uint16, device="cuda")
        counter = torch.empty((), dtype=torch.int32, device="cuda")
        sweep = triton.jit(silu_transition_kernel, do_not_specialize=["start", "count"])
        ptx_hashes = {}
        for negative in (False, True):
            compiled = sweep.warmup(indices, values, counter, 0, BATCH, NEGATIVE=negative, BLOCK=256,
                                    grid=(BATCH // 256,), num_warps=4)
            if not callable(compiled.run):
                raise ValueError("transition launcher unavailable")
            name = "negative.ptx" if negative else "positive.ptx"
            (args.output_dir / name).write_text(compiled.asm["ptx"])
            ptx_hashes[negative] = sha(args.output_dir / name)
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        keys_parts, value_parts = [], []
        launches, verified, total_transitions, maximum_ms = 0, 0, 0, 0.0
        started = time.monotonic()
        for negative in (False, True):
            for lower in range(0, FINITE_END, BATCH):
                count = min(BATCH, FINITE_END - lower)
                counter.zero_()
                start_event.record()
                compiled = sweep[(triton.cdiv(count, 256),)](indices, values, counter, lower, count,
                                                            NEGATIVE=negative, BLOCK=256, num_warps=4)
                end_event.record(); end_event.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                maximum_ms = max(maximum_ms, elapsed)
                launches += 1
                if elapsed > 100 or torch.cuda.max_memory_allocated() > 16 << 20:
                    raise ValueError("dispatch or memory admission exceeded")
                if hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest() != ptx_hashes[negative]:
                    raise ValueError("unexpected sweep specialization")
                transitions = int(counter.item())
                if not 0 <= transitions <= CAPACITY:
                    raise ValueError("transition capacity exceeded; no out-of-bounds stores allowed")
                if transitions:
                    keys = indices[:transitions].cpu().numpy().copy()
                    outputs = values[:transitions].cpu().numpy().copy()
                    order = np.argsort(keys)
                    keys_parts.append(keys[order]); value_parts.append(outputs[order])
                    total_transitions += transitions
                    if total_transitions > 1 << 20:
                        raise ValueError("table exceeds the bounded host representation")
                verified += count
                if launches % 64 == 0 or lower + count == FINITE_END:
                    with (args.output_dir / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(launches=launches, verified_inputs=verified,
                            transitions=total_transitions, maximum_dispatch_ms=maximum_ms,
                            wall_ms=(time.monotonic() - started) * 1000)) + "\n")
        keys = np.concatenate(keys_parts).astype("<u4")
        endpoints = np.concatenate(value_parts).astype("<u2")
        if verified != FINITE_END * 2 or keys[0] != 0 or not np.all(keys[1:] > keys[:-1]):
            raise ValueError("incomplete or unordered numeric domain")
        boundaries = np.arange(65537, dtype=np.uint64) << 16
        directory = (np.searchsorted(keys, boundaries, side="right") - 1).astype("<u4")
        key_start = HEADER.size + directory.nbytes
        value_start = key_start + keys.nbytes
        size = value_start + endpoints.nbytes
        table = args.output_dir / "sm121-silu-bf16.bin.part"
        with table.open("wb") as output:
            output.write(HEADER.pack(b"QSLUTB1\0", 1, 16, len(directory), len(keys), FINITE_END, 0,
                                     key_start, value_start, size, verified))
            output.write(directory.tobytes()); output.write(keys.tobytes()); output.write(endpoints.tobytes())
        # Independent lookup of the held-out real control after construction.
        control_bits = np.fromfile(arguments_path, dtype="<u4")
        found = endpoints[np.searchsorted(keys, control_bits, side="right") - 1]
        if np.any(found != expected):
            raise ValueError("constructed lookup failed real control")
        device_directory = torch.from_numpy(directory).cuda()
        device_keys = torch.from_numpy(keys).cuda()
        device_endpoints = torch.from_numpy(endpoints).cuda()
        errors = torch.empty(BATCH // 256, device="cuda", dtype=torch.int32)
        verify = triton.jit(silu_verify_kernel, do_not_specialize=["start", "count"])
        for negative in (False, True):
            compiled = verify.warmup(device_directory, device_keys, device_endpoints, errors, 0, BATCH,
                                      NEGATIVE=negative, BLOCK=256, grid=(BATCH // 256,), num_warps=4)
            if not callable(compiled.run):
                raise ValueError("verification launcher unavailable")
            name = "verify-negative.ptx" if negative else "verify-positive.ptx"
            (args.output_dir / name).write_text(compiled.asm["ptx"])
            ptx_hashes["verify_" + str(negative)] = sha(args.output_dir / name)
        torch.cuda.synchronize()
        lookup_verified = 0
        for negative in (False, True):
            for lower in range(0, FINITE_END, BATCH):
                count = min(BATCH, FINITE_END - lower)
                start_event.record()
                compiled = verify[(triton.cdiv(count, 256),)](
                    device_directory, device_keys, device_endpoints, errors, lower, count,
                    NEGATIVE=negative, BLOCK=256, num_warps=4)
                end_event.record(); end_event.synchronize()
                elapsed = start_event.elapsed_time(end_event)
                maximum_ms = max(maximum_ms, elapsed)
                launches += 1
                if elapsed > 100 or torch.cuda.max_memory_allocated() > 16 << 20:
                    raise ValueError("lookup verification dispatch or memory admission exceeded")
                if hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest() != ptx_hashes["verify_" + str(negative)]:
                    raise ValueError("unexpected verification specialization")
                if int(errors[:triton.cdiv(count, 256)].cpu().numpy().sum()):
                    raise ValueError("packed lookup differs from the SM121 expression")
                lookup_verified += count
                if launches % 64 == 0 or lower + count == FINITE_END:
                    with (args.output_dir / "progress.jsonl").open("a") as stream:
                        stream.write(json.dumps(dict(launches=launches, lookup_verified_inputs=lookup_verified,
                            maximum_dispatch_ms=maximum_ms, wall_ms=(time.monotonic() - started) * 1000)) + "\n")
        if lookup_verified != verified:
            raise ValueError("incomplete packed lookup verification")
        final = table.with_suffix("")
        table.rename(final)
        record.update(completed=True, verified_inputs=verified, launches=launches, transitions=len(keys),
                      lookup_verified_inputs=lookup_verified,
                      maximum_dispatch_ms=maximum_ms, peak_device_bytes=torch.cuda.max_memory_allocated(),
                      table_bytes=final.stat().st_size, table_sha256=sha(final), table_file=final.name,
                      wall_ms=(time.monotonic() - started) * 1000, ptx_sha256=ptx_hashes,
                      control=control_result, final_control_differences=0,
                      device=torch.cuda.get_device_name(0), torch_version=torch.__version__, triton_version=triton.__version__)
        (args.output_dir / "table.json").write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
