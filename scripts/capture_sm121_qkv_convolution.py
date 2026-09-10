#!/usr/bin/env python3
"""Replay real BF16 QKV projections and the original convolution on SM121.

Inputs are fingerprinted native RMSNorm rows and model weights. Frozen GB10
outputs are comparison targets only. This component probe never loads a model
or claims inference acceptance.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import types

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

FUNCTION = "_causal_conv1d_fwd_kernel"
TOKENS = 7169
# Full-shape CUDA BLAS can allocate an FP32 projection workspace in addition
# to the BF16 result. Include that transient allocation in the bounded replay.
DEVICE_LIMIT = 1 << 30
LAYOUTS = {"input": (TOKENS, 2048), "qkv_weight": (8192, 2048),
           "z_weight": (4096, 2048), "conv_weight": (8192, 4),
           "terminal_qkv": (8192,), "q": (TOKENS, 2048),
           "k": (TOKENS, 2048), "v": (TOKENS, 4096)}


def validate(directory, fingerprint):
    path = directory / "manifest.json"
    if path.stat().st_size > 1 << 20 or file_sha(path) != fingerprint:
        raise ValueError("projection manifest fingerprint/size mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("tokens") != TOKENS or set(manifest.get("files", {})) != set(LAYOUTS):
        raise ValueError("projection file set or product shape mismatch")
    for name, shape in LAYOUTS.items():
        meta = manifest["files"][name]
        path = directory / (name + ".bin")
        count = 1
        for size in shape:
            count *= size
        if (meta.get("file") != path.name or meta.get("dtype") != "bf16" or
                meta.get("shape") != list(shape) or path.stat().st_size != count * 2 or
                file_sha(path) != meta.get("sha256")):
            raise ValueError("projection input layout/fingerprint mismatch: " + name)
    return manifest


def extracted(source):
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    node.decorator_list = []
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))) + "\n"


def execute(args):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("requires one SM121 CUDA device")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("device memory reserve unavailable")
    arrays = {name: np.fromfile(args.input_dir / (name + ".bin"), dtype=np.uint16).reshape(shape)
              for name, shape in LAYOUTS.items()}
    if any(np.any((a & 0x7f80) == 0x7f80) for a in arrays.values()):
        raise ValueError("nonfinite input/reference")

    def upload(name):
        return torch.from_numpy(arrays[name]).view(torch.bfloat16).cuda()

    def compare(actual, expected):
        different = actual != expected
        indices = np.flatnonzero(different)
        return dict(elements=actual.size, bit_mismatch_count=int(indices.size),
                    first_mismatch=None if not indices.size else int(indices[0]),
                    all_finite=bool(np.all((actual & 0x7f80) != 0x7f80)),
                    actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest())

    extracted_path = args.output_dir / "reference-convolution-extracted.py"
    extracted_path.write_text(extracted(args.source))
    module = types.ModuleType("_qrt_reference_convolution")
    module.__file__ = str(extracted_path)
    module.tl = tl
    sys.modules[module.__name__] = module
    exec(compile(extracted_path.read_text(), str(extracted_path), "exec", dont_inherit=True), module.__dict__)
    kernel = triton.jit(module.__dict__[FUNCTION])
    x, w, z, cw = (upload(name) for name in ("input", "qkv_weight", "z_weight", "conv_weight"))
    # Initialize the CUDA BLAS handle before stage timing.
    torch.mm(x[:1, :1], w[:1, :1].T)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    result = dict(torch_version=torch.__version__, triton_version=triton.__version__,
                  device=torch.cuda.get_device_name(),
                  allow_bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                  cases=[])
    for fused in (True, False):
        name = "fused_qkvz" if fused else "separate_qkv"
        weights = torch.cat((w, z), dim=0) if fused else w
        projected = torch.empty((TOKENS, weights.shape[0]), device="cuda", dtype=torch.bfloat16)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        torch.mm(x, weights.T, out=projected)
        end.record(); end.synchronize()
        projection_ms = start.elapsed_time(end)
        with (args.output_dir / "allocation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(case=name, stage="projection", elapsed_ms=projection_ms,
                                         current_bytes=torch.cuda.memory_allocated(),
                                         peak_bytes=torch.cuda.max_memory_allocated())) + "\n")
        if projection_ms > 100:
            raise ValueError("projection dispatch exceeded 100 ms")
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("projection device allocation ceiling exceeded")
        qkv = projected[:, :8192]
        qkv_cpu = qkv.view(torch.uint16).cpu().numpy()
        qkv_path = args.output_dir / (name + "-qkv-bf16.bin")
        qkv_path.write_bytes(qkv_cpu.tobytes())
        state = torch.zeros((1, 3, 8192), dtype=torch.bfloat16, device="cuda").transpose(1, 2)
        cache = torch.zeros(1, dtype=torch.int32, device="cuda")
        has_initial = torch.zeros(1, dtype=torch.bool, device="cuda")
        query = torch.tensor([0, TOKENS], dtype=torch.int32, device="cuda")
        chunks = (TOKENS + 7) // 8
        batch = torch.zeros(chunks, dtype=torch.int32, device="cuda")
        offsets = torch.arange(chunks, dtype=torch.int32, device="cuda")
        output = torch.empty((TOKENS, 8192), dtype=torch.bfloat16, device="cuda")
        arguments = [qkv, cw, None, state, cache, has_initial, query, batch, offsets,
                     None, None, None, None, output, 8192, TOKENS, 1,
                     1, projected.shape[1], 4, 1, 24576, 1, 8192, 1, 1, 8192, 1, -1]
        options = dict(HAS_BIAS=False, KERNEL_WIDTH=4, SILU_ACTIVATION=True,
                       IS_APC_ENABLED=False, USE_PAD_SLOT=True, NP2_STATELEN=4,
                       BLOCK_M=8, BLOCK_N=256, num_stages=2, num_warps=4)
        grid = (chunks, 32)
        compiled = kernel.warmup(*arguments, grid=grid, **options)
        if not callable(compiled.run):
            raise ValueError("convolution launcher unavailable")
        (args.output_dir / (name + "-conv.ptx")).write_text(compiled.asm["ptx"])
        torch.cuda.synchronize()
        with (args.output_dir / "allocation.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(case=name, stage="convolution_ready",
                                         current_bytes=torch.cuda.memory_allocated(),
                                         peak_bytes=torch.cuda.max_memory_allocated())) + "\n")
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("device allocation ceiling exceeded")
        start.record()
        kernel[grid](*arguments, **options)
        end.record(); end.synchronize()
        convolution_ms = start.elapsed_time(end)
        if convolution_ms > 100:
            raise ValueError("convolution dispatch exceeded 100 ms")
        actual = output.view(torch.uint16).cpu().numpy()
        comparisons = {"terminal_qkv": compare(qkv_cpu[-1], arrays["terminal_qkv"])}
        for surface, lo, hi in (("q", 0, 2048), ("k", 2048, 4096), ("v", 4096, 8192)):
            comparisons[surface] = compare(actual[:, lo:hi], arrays[surface])
        if not np.array_equal(qkv.view(torch.uint16).cpu().numpy(), qkv_cpu):
            raise ValueError("convolution modified projection input")
        case = dict(name=name, projection_ms=projection_ms, convolution_ms=convolution_ms,
                    comparisons=comparisons, projection_file=qkv_path.name,
                    projection_sha256=file_sha(qkv_path), convolution_options=options,
                    convolution_ptx_sha256=hashlib.sha256(compiled.asm["ptx"].encode()).hexdigest())
        result["cases"].append(case)
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(case) + "\n")
        del arguments, projected, qkv, output, state, weights
    for name, tensor in (("input", x), ("qkv_weight", w), ("z_weight", z), ("conv_weight", cw)):
        if not np.array_equal(tensor.view(torch.uint16).cpu().numpy(), arrays[name]):
            raise ValueError("projection input or model weight changed")
    result["peak_device_bytes"] = torch.cuda.max_memory_allocated()
    return result


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
    if (not 1 <= args.timeout_seconds <= 120 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("invalid deadline or source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires explicit execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists() or args.source.stat().st_size > 1 << 20 or file_sha(args.source) != args.source_sha256:
        raise ValueError("existing output or convolution source mismatch")
    validate(args.input_dir, args.manifest_sha256)
    extracted(args.source)
    if args.execute:
        if sys.platform != "linux" or socket.gethostname().lower() != (args.expected_host or "").lower():
            raise ValueError("execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                       "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="sm121_real_qkv_convolution_replay", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_sha256=file_sha(Path(__file__)),
                  manifest_sha256=args.manifest_sha256, convolution_source_sha256=args.source_sha256,
                  model_loaded=False, inference_acceptance=False, kernel_executed=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args))
            record["kernel_executed"] = True
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
