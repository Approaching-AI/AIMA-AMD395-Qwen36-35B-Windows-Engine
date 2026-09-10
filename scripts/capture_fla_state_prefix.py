#!/usr/bin/env python3
"""Guarded, model-free CUDA state capture from a fingerprinted saved prefix.

Default mode validates on CPU without importing torch/triton. Explicit execution
runs a BF16 control first; raw F32 checkpoints are retained only after saved
BF16 and same-run raw-state parity. No reference checkpoint feeds recurrence.
"""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import sys
import time

from audit_fla_reference_state_ir import load_state_kernel
from fla_state_exponent_capture import capture_exponents, exponent_arguments
from prepare_fla_state_prefix import layouts

DEVICE_LIMIT = 256 << 20
DEVICE_RESERVE = 1 << 30
FRAME = 32 * 128 * 128


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def words(data: bytes, dtype: str) -> array:
    values = array("H" if dtype == "bf16" else "I")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return values


def finite(data: bytes, dtype: str) -> bool:
    mask = 0x7F80 if dtype == "bf16" else 0x7F800000
    return all(value & mask != mask for value in words(data, dtype))


def rounded_bf16(data: bytes) -> bytes:
    if not finite(data, "f32"):
        raise ValueError("nonfinite F32 state cannot be a trusted trace")
    values = array("H", (((value + 0x7FFF + ((value >> 16) & 1)) >> 16) & 0xFFFF
                         for value in words(data, "f32")))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def compare(left: bytes, right: bytes, dtype: str) -> dict:
    if len(left) != len(right):
        raise ValueError("comparison byte count mismatch")
    count, first = 0, None
    for index, (a, b) in enumerate(zip(words(left, dtype), words(right, dtype))):
        if a != b:
            count += 1
            if first is None:
                first = dict(index=index, actual_bits=a, expected_bits=b)
    all_finite = finite(left, dtype) and finite(right, dtype)
    return dict(bit_mismatch_count=count, first_mismatch=first, all_finite=all_finite,
                exact=count == 0 and all_finite, actual_sha256=sha256(left), expected_sha256=sha256(right))


def validate_prefix(directory: Path, expected_sha256: str) -> tuple[dict, dict]:
    if (directory / "manifest.json").stat().st_size > 1 << 20:
        raise ValueError("prefix manifest exceeds 1 MiB")
    raw = (directory / "manifest.json").read_bytes()
    if sha256(raw) != expected_sha256:
        raise ValueError("prefix manifest fingerprint mismatch")
    manifest = json.loads(raw)
    tokens = manifest["tokens"]
    if (manifest["kind"] != "cpu_prepared_gdn_state_prefix" or type(tokens) is not int
            or not 64 <= tokens <= 1024 or tokens % 64 or manifest["chunks"] != tokens // 64
            or manifest["reference_checkpoints_are_recurrent_inputs"] is not False
            or manifest["future_raw_trace_requires_saved_bf16_parity"] is not True):
        raise ValueError("invalid prefix contract")
    expected = {name + "-" + suffix: (dtype, shape, name in ("v-new", "chunk-state"))
                for name, (suffix, dtype, shape) in layouts(tokens).items()}
    expected["next-chunk-state-bf16"] = ("bfloat16", [1, 32, 128, 128], True)
    if set(manifest["files"]) != set(expected):
        raise ValueError("prefix file set mismatch")
    payloads = {}
    for name, (dtype, shape, reference_only) in expected.items():
        item = manifest["files"][name]
        suffix = "bf16" if dtype == "bfloat16" else "f32"
        size = math.prod(shape) * (2 if suffix == "bf16" else 4)
        if (item["file"] != name + ".bin" or item["dtype"] != dtype or item["shape"] != shape
                or item["bytes"] != size or item["reference_only"] is not reference_only):
            raise ValueError(f"prefix file layout mismatch: {name}")
        path = directory / item["file"]
        if path.stat().st_size != size:
            raise ValueError(f"prefix file size mismatch: {name}")
        data = path.read_bytes()
        if sha256(data) != item["sha256"] or not finite(data, suffix):
            raise ValueError(f"prefix fingerprint or finite-value check failed: {name}")
        payloads[name] = data
    if sum(map(len, payloads.values())) != manifest["total_bytes"]:
        raise ValueError("prefix total byte count mismatch")
    return manifest, payloads


def baseline_comparisons(result: dict, inputs: dict) -> dict:
    return {"v_new": compare(result["v_new"], inputs["v-new-bf16"], "bf16"),
            "checkpoints": compare(result["h"], inputs["chunk-state-bf16"], "bf16"),
            "terminal_bf16": compare(rounded_bf16(result["final"]), inputs["next-chunk-state-bf16"], "bf16")}


def trace_comparisons(result: dict, baseline: dict) -> dict:
    return {"v_new": compare(result["v_new"], baseline["v_new"], "bf16"),
            "checkpoints_bf16": compare(rounded_bf16(result["h"]), baseline["h"], "bf16"),
            "terminal_f32": compare(result["final"], baseline["final"], "f32")}


def run_pair(launch, inputs: dict, exponent_probe=None) -> tuple[dict, dict | None]:
    baseline = launch("bf16")
    record = dict(baseline=baseline_comparisons(baseline, inputs), state_kernel_launches=1,
                  raw_trace_valid=False)
    if not all(item["exact"] for item in record["baseline"].values()):
        return record, None
    raw = launch("f32")
    record.update(trace=trace_comparisons(raw, baseline), state_kernel_launches=2)
    record["raw_trace_valid"] = all(item["exact"] for item in record["trace"].values())
    if record["raw_trace_valid"] and exponent_probe is not None:
        record["exponent_capture"] = exponent_probe()
    return record, raw if record["raw_trace_valid"] else None


def arm_parent_death(supervisor_pid: int) -> None:
    """Linux kills this owned GPU worker even if its supervisor is killed."""
    if sys.platform != "linux" or supervisor_pid <= 1 or os.getppid() != supervisor_pid:
        raise ValueError("worker has no live matching supervisor")
    import ctypes
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot arm parent-death signal")
    if os.getppid() != supervisor_pid:
        raise ValueError("supervisor disappeared while arming worker")


def supervise(command: list[str], timeout: float) -> int:
    child = subprocess.Popen(command, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("owned capture deadline exceeded; inspect existing output, never automatically resubmit", file=sys.stderr)
        return 124
    finally:
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait(timeout=5)


def gpu_capture(args, manifest: dict, payloads: dict) -> dict:
    if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
        raise ValueError("GPU capture requires Linux and an explicitly matching expected host")
    import fcntl
    import tempfile
    # Only this diagnostic's lock is touched; no service/container is stopped.
    with (Path(tempfile.gettempdir()) / "qrt-fla-state-capture.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        import triton
        import triton.language as tl
        if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
            raise ValueError("requires one visible SM121 CUDA device, never an AMD/ROCm device")
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < DEVICE_LIMIT + DEVICE_RESERVE:
            raise ValueError("insufficient device memory reserve")
        torch.cuda.reset_peak_memory_stats()
        kernel = load_state_kernel(args.source.resolve(), triton, tl)
        tokens, chunks = manifest["tokens"], manifest["chunks"]
        names = ("k-normalized-bf16", "u-bf16", "w-bf16", "g-cumsum-f32", "initial_state-f32")
        def upload(data: bytes, dtype):
            return torch.frombuffer(bytearray(data), dtype=dtype).to(device="cuda")
        device = {name: upload(payloads[name], torch.bfloat16 if name.endswith("bf16") else torch.float32)
                  for name in names}
        seq = upload(struct.pack("<qq", 0, tokens), torch.int64)
        offsets = upload(struct.pack("<i", 0), torch.int32)
        launches = []
        def progress(phase: str, **fields):
            with (args.output_dir / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(phase=phase, pid=os.getpid(), **fields)) + "\n")
        def launch(dtype: str) -> dict:
            if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
                raise ValueError("device allocation ceiling exceeded before dispatch")
            sizes = {"v_new": tokens * 32 * 128 * 2,
                     "h": chunks * FRAME * (2 if dtype == "bf16" else 4), "final": FRAME * 4}
            output, storage = {}, {}
            for name, size in sizes.items():
                width = 2 if name == "v_new" or (name == "h" and dtype == "bf16") else 4
                marker = b"\xc1\x7f" if width == 2 else b"\x00\x00\xc0\x7f"
                data = b"\x5a" * 256 + marker * (size // width) + b"\x5a" * 256
                tensor = upload(data, torch.bfloat16 if width == 2 else torch.float32)
                storage[name] = tensor
                output[name] = tensor[256 // width:(256 + size) // width]
            arguments = (
                device[names[0]], device[names[1]], device[names[2]], output["v_new"], device[names[3]], None,
                output["h"], device[names[4]], output["final"], seq, offsets, tokens)
            options = dict(H=32, Hg=16, K=128, V=128, BT=64, BV=args.value_block,
                           USE_G=True, USE_GK=False, USE_INITIAL_STATE=True, STORE_FINAL_STATE=True,
                           SAVE_NEW_VALUE=True, IS_VARLEN=True, num_warps=4, num_stages=2, enable_fp_fusion=True)
            grid = (triton.cdiv(128, args.value_block), 32)
            # Compile without dispatch before timing. A first-JIT host pause
            # between CUDA events is not a slow GPU kernel.
            progress("compile", checkpoint_dtype=dtype)
            prepared = kernel.warmup(*arguments, grid=grid, **options)
            ptx = prepared.asm["ptx"].encode()
            (args.output_dir / f"state-{dtype}.ptx").write_bytes(ptx)
            progress("compiled", checkpoint_dtype=dtype, ptx_sha256=sha256(ptx), options=options)
            # Triton warmup compiles the code but leaves the launcher/module
            # lazy. The public run property initializes them without dispatch.
            # Loading between CUDA events counts host idle time as kernel time.
            load_started = time.monotonic()
            if not callable(prepared.run):
                raise ValueError("compiled state launcher is unavailable")
            torch.cuda.synchronize()
            progress("loaded", checkpoint_dtype=dtype,
                     load_wall_ms=(time.monotonic() - load_started) * 1000)
            if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
                raise ValueError("device allocation ceiling exceeded before dispatch")
            progress("dispatch", checkpoint_dtype=dtype, tokens=tokens)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            compiled = kernel[grid](*arguments, **options)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            progress("completed_dispatch", checkpoint_dtype=dtype, elapsed_ms=elapsed)
            if sha256(compiled.asm["ptx"].encode()) != sha256(ptx):
                raise ValueError("dispatched PTX differs from the timed warmup artifact")
            launches.append(dict(checkpoint_dtype=dtype, elapsed_ms=elapsed,
                                 ptx_sha256=sha256(compiled.asm["ptx"].encode())))
            if elapsed > 100:
                raise ValueError("100 ms post-dispatch admission exceeded; no further launch")
            result = {}
            for name, tensor in storage.items():
                data = tensor.view(torch.uint8).cpu().numpy().tobytes()
                if data[:256] != b"\x5a" * 256 or data[-256:] != b"\x5a" * 256:
                    raise ValueError("output redzone changed")
                result[name] = data[256:-256]
                if not finite(result[name], "bf16" if name == "v_new" or (name == "h" and dtype == "bf16") else "f32"):
                    raise ValueError("state output is nonfinite or not fully written")
            for name in names:
                actual = device[name].view(torch.uint8).cpu().numpy().tobytes()
                if actual != payloads[name]:
                    raise ValueError("an input tensor was modified by the state kernel")
            return result
        # The separate instruction probe runs only after BOTH controls pass.
        # It cannot alter the reference state kernel's register lifetime.
        comparisons, raw = run_pair(launch, payloads, lambda: capture_exponents(
            tokens=tokens, gates=payloads["g-cumsum-f32"], device_gates=device["g-cumsum-f32"],
            output_dir=args.output_dir, torch=torch, triton=triton, tl=tl, progress=progress, device_limit=DEVICE_LIMIT))
        files, exponents = {}, comparisons.pop("exponent_capture", None)
        if raw is not None:
            for key, filename in (("h", "checkpoint-f32.bin"), ("final", "terminal-state-f32.bin")):
                data = raw[key]
                (args.output_dir / filename).write_bytes(data)
                files[key] = dict(file=filename, bytes=len(data), sha256=sha256(data))
        return dict(comparisons=comparisons, launches=launches, files=files, exponent_capture=exponents,
                    device=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                    triton_version=triton.__version__, peak_device_bytes=torch.cuda.max_memory_allocated(),
                    memory_reserve_bytes=DEVICE_RESERVE, kernel_executed=True,
                    kernel_body_modified=False, inputs_unchanged=True, exponent_binding="tl.exp",
                    live_autotune_config_verified=False, worker_parent_death_sigkill=True)


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--value-block", type=int, choices=(32, 64), default=32)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 120 or (args.worker and not args.execute):
        raise ValueError("invalid execution deadline or worker mode")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("source commit must be a full 40-character SHA")
    if args.worker:
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists():
        raise ValueError("output already exists; never overwrite or resubmit a capture")
    if args.source.stat().st_size > 1 << 20:
        raise ValueError("reference source exceeds 1 MiB")
    if sha256(args.source.read_bytes()) != args.source_sha256:
        raise ValueError("reference kernel source fingerprint mismatch")
    manifest, payloads = validate_prefix(args.prefix_dir, args.manifest_sha256)
    planned_exponents = exponent_arguments(payloads["g-cumsum-f32"], manifest["tokens"])
    validation_wall_ms = (time.monotonic() - started) * 1000
    if args.execute and not args.worker:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("execution host mismatch before importing GPU libraries")
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--supervisor-pid", str(os.getpid())]
        raise SystemExit(supervise(command, args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="gdn_state_prefix_capture", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_source_sha256=sha256(Path(__file__).read_bytes()),
                  helper_sha256={name: sha256(Path(__file__).with_name(name).read_bytes()) for name in
                                 ("audit_fla_reference_state_ir.py", "prepare_fla_state_prefix.py", "fla_state_exponent_capture.py")},
                  reference_source=str(args.source), reference_source_sha256=args.source_sha256,
                  prefix_manifest_sha256=args.manifest_sha256, tokens=manifest["tokens"],
                  validated_files=len(payloads), validated_bytes=manifest["total_bytes"],
                  validation_wall_ms=validation_wall_ms,
                  model_loaded=False, kernel_executed=False, inference_acceptance=False,
                  reference_service_executed=False,
                  raw_state_semantics="fixed-config isolated replay, not original worker register capture",
                  exponent_probe_plan=dict(elements=len(planned_exponents) // 4, output_bytes=2 * len(planned_exponents),
                                           host_argument_sha256=sha256(planned_exponents), requires_raw_state_parity=True),
                  reference_checkpoints_are_recurrent_inputs=False, timeout_seconds=args.timeout_seconds)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    execution_started = time.monotonic()
    if args.execute:
        try:
            record.update(gpu_capture(args, manifest, payloads))
        except Exception as error:
            failure = dict(error=str(error), inspect_progress_for_dispatch_state=True,
                           inference_acceptance=False, wall_ms=(time.monotonic() - started) * 1000)
            (args.output_dir / "failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            raise
    record["execution_wall_ms"] = (time.monotonic() - execution_started) * 1000
    record["wall_ms"] = (time.monotonic() - started) * 1000
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    if args.execute and not record["comparisons"]["raw_trace_valid"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
