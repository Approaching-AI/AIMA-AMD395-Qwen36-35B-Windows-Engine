#!/usr/bin/env python3
"""Replay the real layer-zero Z/gated/output/postnorm boundaries on SM121.

Frozen GDN output is an explicitly declared component input. Native captures
and authority terminal values are comparison targets only. This diagnostic
does not load an inference engine or establish token/performance acceptance.
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
import time
import types

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

TOKENS = 7169
DEVICE_LIMIT = 1536 << 20
LAYOUTS = {
    "input": (TOKENS, 2048), "core": (TOKENS, 4096),
    "residual": (TOKENS, 2048), "qkv_weight": (8192, 2048),
    "z_weight": (4096, 2048), "gated_weight": (128,),
    "out_weight": (2048, 4096), "post_weight": (2048,),
    "native_z": (TOKENS, 4096), "native_gated": (TOKENS, 4096),
    "native_out": (TOKENS, 2048), "native_postnorm": (TOKENS, 2048),
    "terminal_z": (4096,), "terminal_gated": (4096,),
    "terminal_out": (2048,), "terminal_postnorm": (2048,),
}


def validate(args):
    path = args.input_dir / "manifest.json"
    if path.stat().st_size > 1 << 20 or file_sha(path) != args.manifest_sha256:
        raise ValueError("manifest fingerprint/size mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("tokens") != TOKENS or set(manifest.get("files", {})) != set(LAYOUTS):
        raise ValueError("product shape or file set mismatch")
    for name, shape in LAYOUTS.items():
        meta = manifest["files"][name]
        count = 1
        for dim in shape:
            count *= dim
        path = args.input_dir / (name + ".bin")
        if (meta.get("file") != path.name or meta.get("dtype") != "bf16" or
                meta.get("shape") != list(shape) or path.stat().st_size != count * 2 or
                file_sha(path) != meta.get("sha256")):
            raise ValueError("tensor layout/fingerprint mismatch: " + name)
    for path, fingerprint in ((args.gated_source, args.gated_source_sha256),
                              (args.norm_source, args.norm_source_sha256)):
        if path.stat().st_size > 1 << 20 or file_sha(path) != fingerprint:
            raise ValueError("reference source fingerprint/size mismatch")
    return manifest


def extracted(path, function, owner=None):
    tree = ast.parse(path.read_text())
    if owner:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    node.decorator_list = []
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))) + "\n"


def execute(args, manifest):
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
    arrays = {name: np.memmap(args.input_dir / (name + ".bin"), dtype=np.uint16,
                             mode="r", shape=shape) for name, shape in LAYOUTS.items()}
    for a in arrays.values():
        if np.any((a & 0x7f80) == 0x7f80):
            raise ValueError("nonfinite input/reference")

    def upload(name):
        return torch.from_numpy(np.array(arrays[name])).view(torch.bfloat16).cuda()

    def cpu(tensor):
        return tensor.contiguous().view(torch.uint16).cpu().numpy()

    def compare(actual, expected):
        positions = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
        return dict(elements=actual.size, bf16_differences=int(positions.size),
                    first_differences=[int(i) for i in positions[:64]],
                    actual_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    expected_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    nonfinite=int(np.count_nonzero((actual & 0x7f80) == 0x7f80)))

    def progress(record):
        record["peak_device_bytes"] = torch.cuda.max_memory_allocated()
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if record["peak_device_bytes"] > DEVICE_LIMIT:
            raise ValueError("device allocation ceiling exceeded")

    modules = []
    for owner, function, source, name in (
        (None, "layer_norm_fwd_kernel", args.gated_source, "gated"),
        ("GemmaRMSNorm", "_forward_static_with_residual", args.norm_source, "postnorm"),
    ):
        code = extracted(source, function, owner)
        path = args.output_dir / ("original-" + name + ".py")
        path.write_text(code)
        module = types.ModuleType("_qrt_original_post_gdn_" + name)
        module.__file__ = str(path)
        module.__dict__.update(tl=tl, torch=torch)
        sys.modules[module.__name__] = module
        exec(compile(code, str(path), "exec", dont_inherit=True), module.__dict__)
        modules.append(module.__dict__[function])
    gated_kernel = triton.jit(modules[0])
    postnorm = torch.compile(modules[1], fullgraph=True)
    torch.cuda.reset_peak_memory_stats()
    x, qw, zw = upload("input"), upload("qkv_weight"), upload("z_weight")
    fused_weight = torch.cat((qw, zw), dim=0)
    projected = torch.matmul(x, fused_weight.T)
    torch.cuda.synchronize()
    qkv_hash = hashlib.sha256(cpu(projected[:, :8192]).tobytes()).hexdigest()
    if qkv_hash != manifest["qkv_control_sha256"]:
        raise ValueError("fused projection does not reproduce the complete QKV control")
    z = projected[:, 8192:].contiguous()
    progress(dict(stage="fused_qkvz", qkv_sha256=qkv_hash))
    del projected, fused_weight, qw, zw, x
    core, weight = upload("core"), upload("gated_weight")
    options = dict(BLOCK_N=128, ROWS_PER_BLOCK=4, HAS_BIAS=False, HAS_Z=True,
                   NORM_BEFORE_GATE=True, IS_RMS_NORM=True, num_warps=1, ACTIVATION="swish")

    def gated(z_value, name):
        result = torch.empty_like(core)
        rstd = torch.empty(TOKENS * 32, dtype=torch.float32, device="cuda")
        arguments = [core, result, weight, None, z_value, None, rstd,
                     128, 128, 128, TOKENS * 32, 128, 1e-6]
        grid = (triton.cdiv(TOKENS * 32, 4), 1)
        compiled = gated_kernel.warmup(*arguments, **options, grid=grid)
        load_started = time.monotonic()
        if not callable(compiled.run):
            raise ValueError("gated normalization launcher unavailable")
        torch.cuda.synchronize()
        progress(dict(stage=name + "_loaded", load_wall_ms=(time.monotonic() - load_started) * 1000))
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        gated_kernel[grid](*arguments, **options)
        end.record(); end.synchronize()
        ms = start.elapsed_time(end)
        progress(dict(stage=name + "_completed_dispatch", dispatch_ms=ms))
        if ms > 100:
            raise ValueError("gated normalization dispatch exceeded 100 ms")
        ptx = compiled.asm["ptx"]
        (args.output_dir / (name + ".ptx")).write_text(ptx)
        progress(dict(stage=name, dispatch_ms=ms, options=options,
                      ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest()))
        return result, rstd

    gated_output, rstd = gated(z, "gated_reference")
    ow, pw, residual = upload("out_weight"), upload("post_weight"), upload("residual")
    out = torch.matmul(gated_output, ow.T)
    normalized, combined = postnorm(pw, 1e-6, out, residual)
    torch.cuda.synchronize()
    comparisons = {}
    controls = {}
    for name, tensor in (("z", z), ("gated", gated_output), ("out", out), ("postnorm", normalized)):
        actual = cpu(tensor)
        controls[name] = compare(actual[-1], arrays["terminal_" + name])
        comparisons[name] = compare(actual, arrays["native_" + name])
        if controls[name]["bf16_differences"] or controls[name]["nonfinite"]:
            progress(dict(stage="terminal_control_failure", surface=name, comparison=controls[name]))
            raise ValueError("original terminal reference control failed: " + name)
        (args.output_dir / (name + "-bf16.bin")).write_bytes(actual.tobytes())
    (args.output_dir / "gated-rstd-f32.bin").write_bytes(rstd.cpu().numpy().tobytes())
    progress(dict(stage="full_reference", controls=controls, comparisons=comparisons))
    # Hold each native predecessor fixed to separate propagation from local math.
    native_z = upload("native_z")
    replay_gated, _ = gated(native_z, "gated_native_z")
    native_gated = upload("native_gated")
    replay_out = torch.matmul(native_gated, ow.T)
    native_out = upload("native_out")
    replay_postnorm, _ = postnorm(pw, 1e-6, native_out, residual)
    torch.cuda.synchronize()
    isolated = {name: compare(cpu(tensor), arrays["native_" + name]) for name, tensor in
                (("gated", replay_gated), ("out", replay_out), ("postnorm", replay_postnorm))}
    for name, tensor in (("core", core), ("gated_weight", weight), ("out_weight", ow),
                         ("post_weight", pw), ("residual", residual), ("native_z", native_z),
                         ("native_gated", native_gated), ("native_out", native_out)):
        if not np.array_equal(cpu(tensor), arrays[name]):
            raise ValueError("reference replay modified its input: " + name)
    progress(dict(stage="fixed_native_predecessor", comparisons=isolated))
    return dict(completed=True, controls=controls, comparisons=comparisons,
                isolated_comparisons=isolated, qkv_control_sha256=qkv_hash,
                inputs_immutable=True,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                torch_version=torch.__version__, triton_version=triton.__version__)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--manifest-sha256", required=True)
    p.add_argument("--gated-source", type=Path, required=True)
    p.add_argument("--gated-source-sha256", required=True)
    p.add_argument("--norm-source", type=Path, required=True)
    p.add_argument("--norm-source-sha256", required=True)
    p.add_argument("--source-commit", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--expected-host")
    p.add_argument("--timeout-seconds", type=int, default=90)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = p.parse_args()
    if (not 1 <= args.timeout_seconds <= 120 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("invalid deadline or source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires explicit execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists():
        raise ValueError("output already exists")
    manifest = validate(args)
    extracted(args.gated_source, "layer_norm_fwd_kernel")
    extracted(args.norm_source, "_forward_static_with_residual", "GemmaRMSNorm")
    if args.execute:
        if sys.platform != "linux" or socket.gethostname().lower() != (args.expected_host or "").lower():
            raise ValueError("execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                       "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="sm121_real_post_gdn_replay", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_sha256=file_sha(Path(__file__)),
                  manifest_sha256=args.manifest_sha256, manifest=manifest,
                  gated_source_sha256=args.gated_source_sha256, norm_source_sha256=args.norm_source_sha256,
                  model_loaded=False, inference_acceptance=False, completed=False)
    if args.execute:
        record.update(execute(args, manifest))
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
