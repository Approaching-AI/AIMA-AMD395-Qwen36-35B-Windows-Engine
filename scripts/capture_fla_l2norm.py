#!/usr/bin/env python3
"""Fingerprint-checked isolated GB10 L2 normalization, CPU preflight by default.

The unmodified reference kernel must reproduce every saved BF16 output before
an instrumented copy may expose square sums and reciprocal square roots.
This loads no model and never feeds expected outputs into an operator.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import types

from capture_fla_state_prefix import arm_parent_death, finite, supervise

DEVICE_LIMIT = 256 << 20
DEVICE_RESERVE = 1 << 30


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(directory: Path, manifest_sha256: str) -> tuple[dict, dict]:
    if (directory / "manifest.json").stat().st_size > 1 << 20:
        raise ValueError("normalization manifest exceeds 1 MiB")
    data = (directory / "manifest.json").read_bytes()
    if digest(data) != manifest_sha256:
        raise ValueError("normalization manifest fingerprint mismatch")
    manifest = json.loads(data)
    tokens = manifest.get("tokens")
    if type(tokens) is not int or not 1 <= tokens <= 8192:
        raise ValueError("normalization tokens must be 1..8192")
    expected = {name + suffix for name in ("q", "k") for suffix in ("-input", "-reference")}
    if set(manifest.get("files", {})) != expected:
        raise ValueError("normalization requires exactly four input/reference files")
    payloads = {}
    for name, meta in manifest["files"].items():
        if meta["file"] != name + ".bin":
            raise ValueError("invalid normalization basename")
        if meta.get("bytes") != tokens * 16 * 128 * 2 or meta.get("dtype") != "bfloat16" or meta.get("shape") != [1, tokens, 16, 128]:
            raise ValueError("normalization input shape/type mismatch")
        path = directory / meta["file"]
        if path.stat().st_size != meta["bytes"]:
            raise ValueError("normalization input size mismatch")
        payload = path.read_bytes()
        if digest(payload) != meta["sha256"] or not finite(payload, "bf16"):
            raise ValueError("normalization input fingerprint/nonfinite mismatch")
        payloads[name] = payload
    return manifest, payloads


def kernel_source(source: Path, trace: bool) -> str:
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "l2norm_fwd_kernel2")
    function.decorator_list = []
    if trace:
        function.name += "_trace"
        function.args.args.extend([ast.arg(arg="RawSum"), ast.arg(arg="RawRsqrt")])
        function.body.extend(ast.parse("tl.store(RawSum + row_idx, square_sum, xmask)\ntl.store(RawRsqrt + row_idx, rsqrt, xmask)").body)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))) + "\n"


def load_kernel(source: Path, trace: bool, directory: Path, triton, tl):
    text = kernel_source(source, trace)
    name = "_qrt_l2norm_trace" if trace else "_qrt_l2norm_baseline"
    path = directory / (name + ".py")
    path.write_text(text)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__dict__["tl"] = tl
    sys.modules[name] = module
    # Triton uses inspect.getsource. A real generated file keeps that source
    # identical to the function being compiled, including the added stores.
    exec(compile(text, str(path), "exec", dont_inherit=True), module.__dict__)
    function_name = "l2norm_fwd_kernel2_trace" if trace else "l2norm_fwd_kernel2"
    kernel = triton.jit(module.__dict__[function_name])
    return kernel, digest(text.encode())


def run_controls(launch, payloads: dict) -> tuple[dict, dict]:
    baselines, traces = {}, {}
    for name in ("q", "k"):
        baselines[name] = launch(name, False)
        if baselines[name]["normalized"] != payloads[name + "-reference"]:
            raise ValueError(f"{name} BF16 baseline differs from saved reference; no trace dispatch")
    for name in ("q", "k"):
        traces[name] = launch(name, True)
        if traces[name]["normalized"] != baselines[name]["normalized"]:
            raise ValueError(f"{name} trace changes BF16 normalization; no further dispatch")
    return baselines, traces


def gpu_capture(args, manifest, payloads):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("requires exactly one visible SM121 CUDA device")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + DEVICE_RESERVE:
        raise ValueError("normalization device reserve unavailable")
    torch.cuda.reset_peak_memory_stats()
    rows = manifest["tokens"] * 16
    kernels = {trace: load_kernel(args.source, trace, args.output_dir, triton, tl) for trace in (False, True)}
    device = {name: torch.frombuffer(bytearray(payloads[name + "-input"]), dtype=torch.bfloat16).clone().cuda() for name in ("q", "k")}
    launches = []

    def launch(name, trace):
        storage, outputs = {}, {}
        layout = {"normalized": (rows * 128 * 2, torch.bfloat16)}
        if trace:
            layout.update(square_sum=(rows * 4, torch.float32), rsqrt=(rows * 4, torch.float32))
        for key, (size, dtype) in layout.items():
            width = 2 if dtype == torch.bfloat16 else 4
            data = torch.empty((size + 512) // width, dtype=dtype, device="cuda")
            data.view(torch.uint8).fill_(0x5a)
            storage[key] = data
            outputs[key] = data[256 // width:(256 + size) // width]
        arguments = [device[name], outputs["normalized"], 1.0e-6, rows]
        options = dict(N=128, BD=128, MBLOCK=32, num_warps=4, num_stages=1, enable_fp_fusion=True)
        if trace:
            options.update(RawSum=outputs["square_sum"], RawRsqrt=outputs["rsqrt"])
        kernel, source_sha = kernels[trace]
        prepared = kernel.warmup(*arguments, grid=(triton.cdiv(rows, 32),), **options)
        ptx = prepared.asm["ptx"].encode()
        label = name + ("-trace" if trace else "-baseline")
        (args.output_dir / (label + ".ptx")).write_bytes(ptx)
        if not callable(prepared.run):
            raise ValueError("normalization launcher unavailable")
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("normalization allocation ceiling exceeded before dispatch")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        actual = kernel[(triton.cdiv(rows, 32),)](*arguments, **options)
        end.record(); end.synchronize()
        elapsed = start.elapsed_time(end)
        progress = dict(label=label, elapsed_ms=elapsed, kernel_source_sha256=source_sha, ptx_sha256=digest(ptx))
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(progress) + "\n")
        launches.append(progress)
        if elapsed > 100 or digest(actual.asm["ptx"].encode()) != digest(ptx):
            raise ValueError("normalization dispatch admission/PTX changed; no further submission")
        result = {}
        for key, data in storage.items():
            raw = data.view(torch.uint8).cpu().numpy().tobytes()
            if raw[:256] != b"\x5a" * 256 or raw[-256:] != b"\x5a" * 256:
                raise ValueError("normalization output redzone changed")
            result[key] = raw[256:-256]
            if not finite(result[key], "bf16" if key == "normalized" else "f32"):
                raise ValueError("normalization output nonfinite")
            if key != "normalized":
                values = np.frombuffer(result[key], dtype="<f4")
                if np.any(values < 0) or np.any(values.view("<u4") == 0x5a5a5a5a) or (key == "rsqrt" and np.any(values == 0)):
                    raise ValueError("normalization trace invalid or unwritten")
        if device[name].view(torch.uint8).cpu().numpy().tobytes() != payloads[name + "-input"]:
            raise ValueError("normalization kernel changed input")
        return result

    _, traces = run_controls(launch, payloads)
    files = {}
    for name, values in traces.items():
        for key in ("square_sum", "rsqrt"):
            data = values[key]
            filename = name + "-" + key + "-f32.bin"
            (args.output_dir / filename).write_bytes(data)
            files[filename] = dict(bytes=len(data), sha256=digest(data))
    return dict(kernel_executed=True, baseline_bf16_exact=True, trace_bf16_exact=True,
                reference_service_executed=False, original_worker_register_capture=False,
                files=files, launches=launches, device=torch.cuda.get_device_name(0),
                torch_version=torch.__version__, triton_version=triton.__version__,
                peak_device_bytes=torch.cuda.max_memory_allocated())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 120 or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("invalid normalization deadline or source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires explicit execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists() or args.source.stat().st_size > 1 << 20 or digest(args.source.read_bytes()) != args.source_sha256:
        raise ValueError("existing output or reference source fingerprint mismatch")
    manifest, payloads = validate(args.input_dir, args.manifest_sha256)
    kernel_source(args.source, False); kernel_source(args.source, True)
    if args.execute and not args.worker:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("normalization execution host mismatch before GPU import")
        raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="isolated_reference_l2norm_capture", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_source_sha256=digest(Path(__file__).read_bytes()),
                  helper_sha256={name: digest(Path(__file__).with_name(name).read_bytes()) for name in
                                 ("capture_fla_state_prefix.py", "audit_fla_reference_state_ir.py",
                                  "fla_state_exponent_capture.py", "prepare_fla_state_prefix.py")},
                  source_sha256=args.source_sha256, manifest_sha256=args.manifest_sha256, tokens=manifest["tokens"],
                  input_bytes=sum(len(p) for p in payloads.values()), timeout_seconds=args.timeout_seconds,
                  model_loaded=False, inference_acceptance=False, kernel_executed=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    started = time.monotonic()
    if args.execute:
        try:
            record.update(gpu_capture(args, manifest, payloads))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error), inference_acceptance=False)) + "\n")
            raise
    record["execution_wall_ms"] = (time.monotonic() - started) * 1000
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
