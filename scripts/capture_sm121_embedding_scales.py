#!/usr/bin/env python3
"""Precompute original SM121 GemmaRMSNorm inverse scales for every embedding.

The table depends only on immutable model embeddings and the norm epsilon.
A complete real-token normalization control must pass before enumeration.
Reference outputs are never used to calculate table entries.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import sys
import types

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha
from capture_sm121_qkv_convolution import embedding_manifest

FUNCTION = "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0"
VOCAB = 248320
HIDDEN = 2048
CHUNK = 2048
DEVICE_LIMIT = 128 << 20


def tensor_location(model, index, name, shape):
    shard = index["weight_map"][name]
    if Path(shard).name != shard:
        raise ValueError("invalid model shard path")
    path = model / shard
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("short model header")
        length, = struct.unpack("<Q", prefix)
        if length > 16 << 20:
            raise ValueError("model header bound exceeded")
        meta = json.loads(stream.read(length))[name]
    count = 1
    for size in shape:
        count *= size
    lo, hi = meta["data_offsets"]
    offset = 8 + length + lo
    if (meta["dtype"] != "BF16" or meta["shape"] != shape or lo < 0 or
            hi - lo != count * 2 or offset + count * 2 > path.stat().st_size):
        raise ValueError("model tensor layout/span mismatch")
    return dict(name=name, shard=shard, offset=offset, bytes=count * 2, dtype="bf16", shape=shape)


def scale_source(path):
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    function.decorator_list = []
    function.args.args.insert(3, ast.arg(arg="inverse_ptr"))
    fixed_rows = [n for n in function.body if isinstance(n, ast.Assign) and
                  any(isinstance(t, ast.Name) and t.id == "xnumel" for t in n.targets)]
    if len(fixed_rows) != 1 or not isinstance(fixed_rows[0].value, ast.Constant) or fixed_rows[0].value.value != 7169:
        raise ValueError("original normalization row contract changed")
    function.body.remove(fixed_rows[0])
    function.body.extend(ast.parse("tl.store(inverse_ptr + xindex, libdevice.rsqrt(tmp4 / 2048.0 + 1e-6), xmask)").body)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))) + "\n"


def execute(args, control):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice
    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("requires one SM121 CUDA device")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("device reserve unavailable")
    index_path = args.model_dir / "model.safetensors.index.json"
    if file_sha(index_path) != args.index_sha256:
        raise ValueError("model index fingerprint mismatch")
    index = json.loads(index_path.read_text())
    locations = [tensor_location(args.model_dir, index, "model.language_model.embed_tokens.weight", [VOCAB, HIDDEN]),
                 tensor_location(args.model_dir, index, "model.language_model.layers.0.input_layernorm.weight", [HIDDEN])]
    embeddings, norm_weight = [np.memmap(args.model_dir / m["shard"], dtype=np.uint16, mode="r",
                                       offset=m["offset"], shape=tuple(m["shape"])) for m in locations]
    data = {name: np.fromfile(args.embedding_input_dir / meta["file"], dtype=np.uint32 if meta["dtype"] == "u32" else np.uint16).reshape(meta["shape"])
            for name, meta in control["files"].items()}
    if (np.any(data["token_ids"] >= VOCAB) or np.any(data["prompt"] >= VOCAB) or
            not np.array_equal(embeddings[data["token_ids"]], data["embedding"]) or
            not np.array_equal(norm_weight, data["weight"])):
        raise ValueError("GB10 model weights differ from native embedding control")
    expected = np.fromfile(args.reference_norm, dtype=np.uint16).reshape(7169, HIDDEN)
    selected = np.asarray(embeddings[data["prompt"]]).copy()
    weight = torch.from_numpy(np.asarray(norm_weight).copy()).view(torch.bfloat16).cuda()
    source_path = args.output_dir / "original-norm-with-inverse.py"
    source_path.write_text(scale_source(args.source))
    module = types.ModuleType("_qrt_original_norm_inverse")
    module.__file__ = str(source_path)
    module.__dict__.update(tl=tl, libdevice=libdevice)
    sys.modules[module.__name__] = module
    exec(compile(source_path.read_text(), str(source_path), "exec", dont_inherit=True), module.__dict__)
    kernel = triton.jit(module.__dict__[FUNCTION])
    options = dict(num_warps=16, num_stages=1, enable_fp_fusion=True, enable_reflect_ftz=True)
    torch.cuda.reset_peak_memory_stats()
    launches = 0
    maximum_ms = 0.0
    ptx_hashes = set()

    def launch(values):
        nonlocal launches, maximum_ms
        if np.any((values & 0x7f80) == 0x7f80):
            raise ValueError("nonfinite embedding")
        count = values.shape[0]
        x = torch.from_numpy(values).view(torch.bfloat16).cuda()
        output = torch.empty_like(x)
        inverse = torch.empty(count, dtype=torch.float32, device="cuda")
        arguments = [x, weight, output, inverse, count, HIDDEN, 2, HIDDEN]
        grid = ((count + 1) // 2,)
        prepared = kernel.warmup(*arguments, grid=grid, **options)
        if not callable(prepared.run):
            raise ValueError("normalization inverse launcher unavailable")
        ptx_sha = hashlib.sha256(prepared.asm["ptx"].encode()).hexdigest()
        if ptx_sha not in ptx_hashes:
            (args.output_dir / (ptx_sha + ".ptx")).write_text(prepared.asm["ptx"])
            ptx_hashes.add(ptx_sha)
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("embedding inverse device bound exceeded")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); kernel[grid](*arguments, **options); end.record(); end.synchronize()
        ms = start.elapsed_time(end)
        launches += 1; maximum_ms = max(maximum_ms, ms)
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(launch=launches, rows=count, elapsed_ms=ms)) + "\n")
        if ms > 100:
            raise ValueError("embedding inverse dispatch deadline exceeded")
        actual = output.view(torch.uint16).cpu().numpy()
        scales = inverse.cpu().numpy()
        if not np.all(np.isfinite(scales) & (scales > 0)) or not np.array_equal(x.view(torch.uint16).cpu().numpy(), values):
            raise ValueError("nonfinite inverse or changed embedding input")
        return actual, scales

    def apply(values, scales):
        x = (values.astype(np.uint32) << 16).view(np.float32)
        w = (np.asarray(norm_weight).astype(np.uint32) << 16).view(np.float32)
        output = (x * scales[:, None]) * (np.float32(1) + w)
        bits = output.view(np.uint32)
        return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)

    actual, control_scales = launch(selected)
    control_result = dict(original_norm_differences=int(np.count_nonzero(actual != expected)),
                          inverse_application_differences=int(np.count_nonzero(apply(selected, control_scales) != expected)))
    (args.output_dir / "control.json").write_text(json.dumps(control_result, indent=2) + "\n")
    if any(control_result.values()):
        raise ValueError("inverse observation does not reproduce the complete real-token norm control")
    table = np.empty(VOCAB, dtype=np.float32)
    embedding_hash = hashlib.sha256()
    for offset in range(0, VOCAB, CHUNK):
        values = np.asarray(embeddings[offset:offset + CHUNK]).copy()
        embedding_hash.update(values.tobytes())
        unused, scales = launch(values)
        table[offset:offset + len(scales)] = scales
    selected_scales = table[data["prompt"]]
    table_control = dict(inverse_bits_different=int(np.count_nonzero(selected_scales.view(np.uint32) != control_scales.view(np.uint32))),
                         output_bits_different=int(np.count_nonzero(apply(selected, selected_scales) != expected)))
    if any(table_control.values()):
        raise ValueError("vocabulary enumeration differs from real-token control")
    if not np.array_equal(weight.view(torch.uint16).cpu().numpy(), norm_weight):
        raise ValueError("normalization weight changed")
    table_path = args.output_dir / "layer0-embedding-inverse-f32.bin"
    table_path.write_bytes(table.tobytes())
    locations[0]["sha256"] = embedding_hash.hexdigest()
    locations[1]["sha256"] = hashlib.sha256(norm_weight.tobytes()).hexdigest()
    return dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__, triton_version=triton.__version__,
                model_tensors=locations, entries=VOCAB, epsilon=1e-6, control=control_result, table_control=table_control,
                table_file=table_path.name, table_bytes=table_path.stat().st_size, table_sha256=file_sha(table_path),
                options=options, xblock=2, rblock=HIDDEN, launches=launches, maximum_dispatch_ms=maximum_ms,
                peak_device_bytes=torch.cuda.max_memory_allocated(), ptx_sha256=sorted(ptx_hashes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "embedding-input-dir", "norm-source", "source", "reference-norm", "reference-capture", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("index-sha256", "embedding-manifest-sha256", "norm-source-sha256", "source-sha256", "reference-norm-sha256", "reference-capture-sha256", "source-commit"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.timeout_seconds <= 120 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("invalid deadline/source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("worker requires execution")
        arm_parent_death(args.supervisor_pid)
    if (args.output_dir.exists() or args.source.stat().st_size > 1 << 20 or file_sha(args.source) != args.source_sha256 or
            args.reference_norm.stat().st_size != 7169 * HIDDEN * 2 or file_sha(args.reference_norm) != args.reference_norm_sha256):
        raise ValueError("existing output or original kernel/norm control mismatch")
    if args.reference_capture.stat().st_size > 1 << 20 or file_sha(args.reference_capture) != args.reference_capture_sha256:
        raise ValueError("normalization reference capture fingerprint mismatch")
    capture = json.loads(args.reference_capture.read_text())
    if (not capture.get("kernel_executed") or
            not any(c.get("mode") == "compiled" and c["native_comparison"]["actual_sha256"] == args.reference_norm_sha256
                    for c in capture.get("normalization", {}).get("cases", [])) or
            not any(all(c["comparisons"][s]["bit_mismatch_count"] == 0 and c["comparisons"][s]["all_finite"]
                        for s in ("terminal_qkv", "q", "k", "v")) for c in capture.get("cases", []))):
        raise ValueError("normalization capture lacks the complete GB10 projection/convolution boundary")
    control = embedding_manifest(args)
    scale_source(args.source)
    if args.execute:
        if sys.platform != "linux" or socket.gethostname().lower() != (args.expected_host or "").lower():
            raise ValueError("execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                       "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="sm121_model_embedding_inverse_table", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_sha256=file_sha(Path(__file__)),
                  original_kernel_sha256=args.source_sha256, control_norm_sha256=args.reference_norm_sha256,
                  reference_capture_sha256=args.reference_capture_sha256, model_path=str(args.model_dir),
                  model_index_sha256=args.index_sha256, model_loaded=False, inference_acceptance=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args, control))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
