#!/usr/bin/env python3
"""Observe the pinned dynamic-width GemmaRMSNorm inverse for every embedding.

The original compiled module and a correctness-qualified real-token capture
are required. The observer preserves the i64 width argument and checks every
vocabulary row against the unmodified module before publishing the table.
This reference-side tool adds no dependency to the Windows runtime.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import struct
import sys

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_embedding_scales import tensor_location
from capture_sm121_exp2_table import file_sha

FUNCTION = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0"
MODULE_SHA = "2a871a26924143ed071e3546647c796cf9b81f3f2f546a585a5ee3e18211ba87"
VOCAB, WIDTH, CHUNK = 248320, 2048, 8192
DEVICE_LIMIT = 256 << 20


def import_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def observer_source(source):
    tree = ast.parse(source)
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    if function.name != FUNCTION or [a.arg for a in function.args.args] != [
        "in_ptr0", "in_ptr1", "out_ptr1", "ks0", "xnumel", "r0_numel", "XBLOCK", "R0_BLOCK"
    ] or not isinstance(function.body[-1], ast.For):
        raise ValueError("original dynamic norm signature changed")
    function.decorator_list = []
    function.args.args.insert(3, ast.arg(arg="inverse_ptr"))
    function.body[-1].body.extend(ast.parse("tl.store(inverse_ptr + xindex, tmp13, xmask)").body)
    return ("import triton\nimport triton.language as tl\n"
            "from torch._inductor.runtime.triton_helpers import libdevice\n@triton.jit\n" +
            ast.unparse(ast.fix_missing_locations(function)) + "\n")


def preflight(args):
    if args.output_dir.exists() or file_sha(args.norm_module) != MODULE_SHA:
        raise ValueError("existing output or unqualified original norm module")
    if args.reference_capture.stat().st_size > 8 << 20 or file_sha(args.reference_capture) != args.reference_capture_sha256:
        raise ValueError("reference capture fingerprint mismatch")
    capture = json.loads(args.reference_capture.read_text())
    if not capture.get("completed") or not capture.get("controls_qualified"):
        raise ValueError("reference controls are not qualified")
    if file_sha(args.prompt_cases) != capture["additional_cases_sha256"]:
        raise ValueError("actual prompt inputs changed")
    prompts = json.loads(args.prompt_cases.read_text())
    if not 1 <= len(prompts) <= 16:
        raise ValueError("control case bound exceeded")
    controls = []
    for prompt in prompts:
        name, ids = prompt["name"], prompt["prompt_token_ids"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not 1 <= len(ids) <= 384 or any(type(t) is not int or not 0 <= t < VOCAB for t in ids):
            raise ValueError("invalid control tokens")
        case = next(c for c in capture["cases"] if c["name"] == name)
        packed = struct.pack("<" + "I" * len(ids), *ids)
        if len(ids) != case["prompt"]["token_count"] or hashlib.sha256(packed).hexdigest() != case["prompt"]["u32le_sha256"]:
            raise ValueError("actual tokens differ from the captured prompt")
        meta = next(e for e in case["worker"]["runtime_boundaries"]["files"].values()
                    if e["transaction"] == 0 and e["label"] == "layer-00-input-rmsnorm")
        if Path(meta["file"]).name != meta["file"] or meta["dtype"] != "bf16" or meta["shape"] != [len(ids), WIDTH]:
            raise ValueError("invalid captured normalization layout")
        path = args.reference_capture.parent / name / meta["file"]
        if path.stat().st_size != len(ids) * WIDTH * 2 or file_sha(path) != meta["sha256"]:
            raise ValueError("captured normalization changed")
        controls.append(dict(name=name, ids=ids, reference_path=path, reference_sha256=meta["sha256"]))
    index = args.model_dir / "model.safetensors.index.json"
    if file_sha(index) != capture["model_evidence"]["index_sha256"]:
        raise ValueError("model index differs from the reference")
    return capture, controls, json.loads(index.read_text())


def execute(args, capture, controls, index):
    import numpy as np
    import torch
    import triton
    from triton.compiler import ASTSource

    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("requires one SM121 CUDA device")
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("device reserve unavailable")
    torch.set_num_threads(2)
    locations = [tensor_location(args.model_dir, index, "model.language_model.embed_tokens.weight", [VOCAB, WIDTH]),
                 tensor_location(args.model_dir, index, "model.language_model.layers.0.input_layernorm.weight", [WIDTH])]
    embedding, weight_bits = [np.memmap(args.model_dir / e["shard"], dtype=np.uint16, mode="r",
                                      offset=e["offset"], shape=tuple(e["shape"])) for e in locations]
    weight = torch.from_numpy(np.asarray(weight_bits).copy()).view(torch.bfloat16).cuda()
    original = import_module("_qrt_original_dynamic_norm", args.norm_module)
    kernel = getattr(original, FUNCTION)
    source = args.output_dir / "dynamic-norm-with-inverse.py"
    source.write_text(observer_source(kernel.fn.src))
    observer = getattr(import_module("_qrt_dynamic_norm_inverse", source), FUNCTION)
    signature = dict(in_ptr0="*bf16", in_ptr1="*bf16", out_ptr1="*bf16", inverse_ptr="*fp32",
                     ks0="i64", xnumel="i32", r0_numel="i32")
    options = dict(num_warps=16, num_stages=1, enable_fp_fusion=True, enable_reflect_ftz=True)
    # JIT argument inference makes the literal width i32 and changes rounding.
    compiled = triton.compile(ASTSource(observer, signature,
        constexprs=dict(XBLOCK=2, R0_BLOCK=WIDTH),
        attrs={(i,): [["tt.divisibility", 16]] for i in range(4)}), options=options)
    (args.output_dir / "observer.ptx").write_text(compiled.asm["ptx"])
    launches, maximum_ms = 0, 0.0
    maximum_device_bytes, preparation_peak_bytes = 0, 0

    def launch(values):
        nonlocal launches, maximum_ms, maximum_device_bytes, preparation_peak_bytes
        if np.any((values & 0x7f80) == 0x7f80):
            raise ValueError("nonfinite embedding")
        x = torch.from_numpy(values.copy()).view(torch.bfloat16).cuda()
        rows = len(values)
        # Prepare the original launcher outside the measured observation.
        original.call([rows, WIDTH, x, weight])
        torch.cuda.synchronize()
        out = torch.empty_like(x)
        inv = torch.empty(rows, dtype=torch.float32, device="cuda")
        # CompiledKernel loads its CUDA module on first invocation. Prepare
        # both launchers before timing GPU execution and retain preparation
        # memory separately from the bounded observation's live tensors.
        compiled[((rows + 1) // 2, 1, 1)](x, weight, out, inv, WIDTH, rows, WIDTH)
        torch.cuda.synchronize()
        preparation_peak_bytes = max(preparation_peak_bytes, torch.cuda.max_memory_allocated())
        torch.cuda.reset_peak_memory_stats()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        expected = original.call([rows, WIDTH, x, weight])[0]
        compiled[((rows + 1) // 2, 1, 1)](x, weight, out, inv, WIDTH, rows, WIDTH)
        end.record(); end.synchronize()
        ms = start.elapsed_time(end)
        launches += 1; maximum_ms = max(maximum_ms, ms)
        peak_bytes = torch.cuda.max_memory_allocated()
        maximum_device_bytes = max(maximum_device_bytes, peak_bytes)
        with (args.output_dir / "dispatches.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(rows=rows, elapsed_ms=ms, peak_device_bytes=peak_bytes,
                                         preparation_peak_device_bytes=preparation_peak_bytes)) + "\n")
        if ms > 100 or peak_bytes > DEVICE_LIMIT:
            raise ValueError(f"observation dispatch/memory bound exceeded: ms={ms}, peak_bytes={peak_bytes}")
        if not torch.equal(expected, out) or not torch.isfinite(inv).all() or not (inv > 0).all():
            raise ValueError("inverse observer changed original normalization")
        return out.view(torch.uint16).cpu().numpy(), inv.cpu().numpy()

    def apply(values, inverse):
        x = (values.astype(np.uint32) << 16).view(np.float32)
        w = (np.asarray(weight_bits).astype(np.uint32) << 16).view(np.float32)
        bits = ((x * inverse[:, None]) * (np.float32(1) + w)).view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

    torch.cuda.reset_peak_memory_stats()
    checks = []
    for control in controls:
        values = np.asarray(embedding[control["ids"]]).copy()
        expected = np.fromfile(control["reference_path"], dtype=np.uint16).reshape(-1, WIDTH)
        output, inv = launch(values)
        if not np.array_equal(output, expected) or not np.array_equal(apply(values, inv), expected):
            raise ValueError("complete real-token norm control failed")
        checks.append(dict(name=control["name"], elements=expected.size,
                           reference_sha256=control["reference_sha256"], mismatches=0))
    table = np.empty(VOCAB, dtype=np.float32)
    embedding_hash = hashlib.sha256()
    for offset in range(0, VOCAB, CHUNK):
        values = np.asarray(embedding[offset:offset + CHUNK]).copy()
        embedding_hash.update(values.tobytes())
        _, inv = launch(values)
        table[offset:offset + len(values)] = inv
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(verified_vocabulary_rows=offset + len(values))) + "\n")
    for control in controls:
        expected = np.fromfile(control["reference_path"], dtype=np.uint16).reshape(-1, WIDTH)
        if not np.array_equal(apply(np.asarray(embedding[control["ids"]]), table[control["ids"]]), expected):
            raise ValueError("enumerated table differs from the real-token control")
    path = args.output_dir / "layer0-embedding-dynamic-inverse-f32.bin"
    path.write_bytes(table.tobytes())
    locations[0]["sha256"] = embedding_hash.hexdigest()
    locations[1]["sha256"] = hashlib.sha256(weight_bits.tobytes()).hexdigest()
    return dict(completed=True, entries=VOCAB, all_original_vocabulary_norm_outputs_unchanged=True,
                model_tensors=locations, controls=checks, table_file=path.name,
                table_bytes=path.stat().st_size, table_sha256=file_sha(path),
                observer_source_sha256=file_sha(source), observer_signature=signature, options=options,
                observer_ptx_sha256=file_sha(args.output_dir / "observer.ptx"), launches=launches,
                maximum_dispatch_ms=maximum_ms, peak_device_bytes=maximum_device_bytes,
                preparation_peak_device_bytes=preparation_peak_bytes,
                torch_version=torch.__version__, triton_version=triton.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "norm-module", "reference-capture", "prompt-cases", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("reference-capture-sha256", "source-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 120 or not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise ValueError("invalid deadline/source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires execution")
        arm_parent_death(args.supervisor_pid)
    capture, controls, index = preflight(args)
    if args.execute:
        if sys.platform != "linux" or socket.gethostname().lower() != (args.expected_host or "").lower():
            raise ValueError("execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                       "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(host=socket.gethostname(), command=sys.argv, source_commit=args.source_commit,
                  source_sha256=file_sha(Path(__file__)), original_norm_module_sha256=MODULE_SHA,
                  reference_capture_sha256=args.reference_capture_sha256,
                  model_index_sha256=capture["model_evidence"]["index_sha256"],
                  model_path=str(args.model_dir), model_loaded=False, inference_acceptance=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args, capture, controls, index))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
