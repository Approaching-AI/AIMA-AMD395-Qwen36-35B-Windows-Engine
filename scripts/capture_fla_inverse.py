#!/usr/bin/env python3
"""Bounded original-reference triangular inverse replay from a saved q64 view.

Three declared warp configurations expose generated PTX and BF16 comparisons.
Only configurations reproducing every saved BF16 cell qualify as comparable
controls. No expected value is passed to an operator or used by inference.
"""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import socket
import sys
import types

from capture_fla_state_prefix import arm_parent_death, compare, finite, supervise
from capture_sm121_exp2_table import file_sha

DEVICE_LIMIT = 64 << 20
FUNCTION = "merge_16x16_to_64x64_inverse_kernel"


def validate(directory, expected_sha):
    manifest_path = directory / "manifest.json"
    if manifest_path.stat().st_size > 1 << 20 or file_sha(manifest_path) != expected_sha:
        raise ValueError("inverse manifest fingerprint/size mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("tokens") != 64 or set(manifest.get("files", {})) != {"a-input", "a-reference"}:
        raise ValueError("inverse replay requires a declared q64 input/reference pair")
    payloads = {}
    for name, meta in manifest["files"].items():
        width, dtype = (4, "float32") if name == "a-input" else (2, "bfloat16")
        if meta["file"] != name + ".bin" or meta["shape"] != [1, 64, 32, 64] or meta["dtype"] != dtype or meta["bytes"] != 64 * 32 * 64 * width:
            raise ValueError("inverse file layout mismatch")
        path = directory / meta["file"]
        if path.stat().st_size != meta["bytes"] or file_sha(path) != meta["sha256"]:
            raise ValueError("inverse file fingerprint/size mismatch")
        payload = path.read_bytes()
        if not finite(payload, "f32" if width == 4 else "bf16"):
            raise ValueError("inverse input/reference is nonfinite")
        payloads[name] = payload
    return manifest, payloads


def extract(source):
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    function.decorator_list = []
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))) + "\n"


def load_kernel(source, directory, triton, tl):
    path = directory / "reference-inverse-extracted.py"
    path.write_text(extract(source))
    module = types.ModuleType("_qrt_reference_inverse")
    module.__file__ = str(path)
    module.__dict__["tl"] = tl
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), str(path), "exec", dont_inherit=True), module.__dict__)
    return triton.jit(module.__dict__[FUNCTION], do_not_specialize=["T"])


def gpu_capture(args, payloads):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("inverse capture requires exactly one SM121 CUDA device")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("inverse device reserve unavailable")
    torch.cuda.reset_peak_memory_stats()
    kernel = load_kernel(args.source, args.output_dir, triton, tl)
    source = torch.frombuffer(bytearray(payloads["a-input"]), dtype=torch.float32).clone().cuda()
    seq = torch.tensor([0, 64], dtype=torch.int64, device="cuda")
    indices = torch.tensor([[0, 0]], dtype=torch.int32, device="cuda")
    records = []
    for warps in (2, 4, 8):
        # The service creates Ai with zeros, including unwritten upper blocks.
        storage = torch.empty(64 * 32 * 64 * 2 + 512, dtype=torch.uint8, device="cuda")
        storage.fill_(0x5A)
        output = storage[256:-256].view(torch.bfloat16)
        output.zero_()
        arguments = [source, output, seq, indices, 64]
        options = dict(H=32, BT=64, USE_TMA=False, IS_VARLEN=True, DOT_PRECISION="ieee",
                       num_warps=warps, num_stages=2, enable_fp_fusion=True)
        prepared = kernel.warmup(*arguments, grid=(1, 32), **options)
        ptx_path = args.output_dir / f"inverse-w{warps}.ptx"
        ptx_path.write_text(prepared.asm["ptx"])
        if not callable(prepared.run):
            raise ValueError("inverse launcher unavailable")
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("inverse allocation ceiling exceeded")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); actual = kernel[(1, 32)](*arguments, **options); end.record(); end.synchronize()
        elapsed = start.elapsed_time(end)
        import hashlib
        progress = dict(warps=warps, options=options, elapsed_ms=elapsed, ptx_sha256=file_sha(ptx_path))
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(progress) + "\n")
        if elapsed > 100 or hashlib.sha256(actual.asm["ptx"].encode()).hexdigest() != progress["ptx_sha256"]:
            raise ValueError("inverse dispatch admission or PTX changed; no further submission")
        raw = storage.cpu().numpy().tobytes()
        if raw[:256] != b"\x5a" * 256 or raw[-256:] != b"\x5a" * 256 or not finite(raw[256:-256], "bf16"):
            raise ValueError("inverse output bounds/finiteness failure; no further submission")
        if source.view(torch.uint8).cpu().numpy().tobytes() != payloads["a-input"] or not np.array_equal(seq.cpu().numpy(), [0, 64]) or not np.array_equal(indices.cpu().numpy(), [[0, 0]]):
            raise ValueError("inverse input changed; no further submission")
        output_path = args.output_dir / f"inverse-w{warps}-bf16.bin"
        output_path.write_bytes(raw[256:-256])
        progress.update(comparison=compare(raw[256:-256], payloads["a-reference"], "bf16"),
                        file=output_path.name, output_sha256=file_sha(output_path))
        records.append(progress)
    return dict(kernel_executed=True, configurations=records,
                comparable_warps=[r["warps"] for r in records if r["comparison"]["exact"]],
                device=torch.cuda.get_device_name(0), torch_version=torch.__version__, triton_version=triton.__version__,
                peak_device_bytes=torch.cuda.max_memory_allocated(), original_worker_register_capture=False,
                reference_service_executed=False)


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
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 120 or len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("invalid inverse deadline or source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("inverse worker requires explicit execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists() or args.source.stat().st_size > 1 << 20 or file_sha(args.source) != args.source_sha256:
        raise ValueError("existing output or inverse source fingerprint mismatch")
    manifest, payloads = validate(args.input_dir, args.manifest_sha256)
    extract(args.source)
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("inverse execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="isolated_reference_inverse_q64", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_source_sha256=file_sha(Path(__file__)),
                  source_sha256=args.source_sha256, manifest_sha256=args.manifest_sha256,
                  helper_sha256={name: file_sha(Path(__file__).with_name(name)) for name in
                                 ("capture_sm121_exp2_table.py", "capture_fla_state_prefix.py",
                                  "audit_fla_reference_state_ir.py", "fla_state_exponent_capture.py", "prepare_fla_state_prefix.py")},
                  tokens=manifest["tokens"], model_loaded=False, inference_acceptance=False, kernel_executed=False,
                  timeout_seconds=args.timeout_seconds)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(gpu_capture(args, payloads))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error), inference_acceptance=False)) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
