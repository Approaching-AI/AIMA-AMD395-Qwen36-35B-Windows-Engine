#!/usr/bin/env python3
"""Replay the real q7169 first-layer MoE and next normalization on SM121.

The complete post-attention input is GB10-validated. Expected tensors only
compare outputs. This bounded component replay does not establish inference
or product performance acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import types

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha
from capture_sm121_post_gdn import extracted

TOKENS = 7169
DEVICE_LIMIT = 4 << 30
SHAPES = {
    "input": ("bf16", [TOKENS, 2048]),
    "residual": ("bf16", [TOKENS, 2048]),
    "native_unrounded": ("f32", [TOKENS, 2048]),
    "native_nextnorm": ("bf16", [TOKENS, 2048]),
    "next_weight": ("bf16", [2048]),
}
CONTROL_SHAPES = {
    "input": ("bf16", [2048]), "router": ("bf16", [256]),
    "topk_ids": ("i32", [8]), "topk_weights": ("f32", [8]),
    "shared_gate_up": ("bf16", [1024]), "shared_activated": ("bf16", [512]),
    "shared_down": ("bf16", [2048]), "shared_gate": ("bf16", [1]),
    "shared": ("bf16", [2048]), "routed": ("bf16", [2048]),
    "moe": ("bf16", [2048]), "nextnorm": ("bf16", [2048]),
    "routed_gate_up": ("bf16", [8, 1024]),
    "routed_activated": ("bf16", [8, 512]),
    "routed_weighted": ("bf16", [8, 2048]),
}
WEIGHT_SHAPES = {
    "router": [256, 2048], "routed_gate_up": [256, 1024, 2048],
    "routed_down": [256, 2048, 512], "shared_gate_up_gate": [512, 2048],
    "shared_gate_up_up": [512, 2048], "shared_down": [2048, 512],
    "shared_gate": [1, 2048],
}
WEIGHT_NAMES = {
    "router": "mlp.gate.weight", "routed_gate_up": "mlp.experts.gate_up_proj",
    "routed_down": "mlp.experts.down_proj",
    "shared_gate_up_gate": "mlp.shared_expert.gate_proj.weight",
    "shared_gate_up_up": "mlp.shared_expert.up_proj.weight",
    "shared_down": "mlp.shared_expert.down_proj.weight",
    "shared_gate": "mlp.shared_expert_gate.weight",
}


def validate(args):
    path = args.input_dir / "manifest.json"
    if path.stat().st_size > 1 << 20 or file_sha(path) != args.manifest_sha256:
        raise ValueError("MoE manifest fingerprint or size mismatch")
    manifest = json.loads(path.read_text())
    layouts = dict(SHAPES)
    layouts.update({"terminal_" + k: v for k, v in CONTROL_SHAPES.items()})
    if manifest.get("tokens") != TOKENS or set(manifest.get("files", {})) != set(layouts):
        raise ValueError("MoE replay requires the complete q7169 input/control set")
    for name, (dtype, shape) in layouts.items():
        meta = manifest["files"][name]
        path = args.input_dir / (name + ".bin")
        if (meta.get("file") != path.name or meta.get("dtype") != dtype or
                meta.get("shape") != shape or
                path.stat().st_size != math.prod(shape) * (2 if dtype == "bf16" else 4) or
                file_sha(path) != meta.get("sha256")):
            raise ValueError("MoE tensor layout/fingerprint mismatch: " + name)
    weights = manifest.get("weights", [])
    if {m.get("key") for m in weights} != set(WEIGHT_SHAPES) or len(weights) != len(WEIGHT_SHAPES):
        raise ValueError("MoE model weight set mismatch")
    for meta in weights:
        if (meta.get("dtype") != "BF16" or meta.get("shape") != WEIGHT_SHAPES[meta["key"]] or
                meta.get("name") != "model.language_model.layers.0." + WEIGHT_NAMES[meta["key"]] or
                meta.get("bytes") != math.prod(meta["shape"]) * 2 or
                Path(meta["shard"]).name != meta["shard"] or meta.get("offset", -1) < 0):
            raise ValueError("invalid native model tensor span")
    norm_source = args.input_dir / "reference-norm.py"
    if file_sha(norm_source) != manifest.get("norm_source_sha256"):
        raise ValueError("original residual normalization source changed")
    extracted(norm_source, "_forward_static_with_residual", "GemmaRMSNorm")
    return manifest


def execute(args, manifest):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import triton
    from safetensors import safe_open
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("MoE reference requires one SM121 CUDA device")
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("MoE reference device reserve unavailable")
    torch.set_num_threads(2)
    for meta in manifest["reference_sources"]:
        module = importlib.import_module(meta["module"])
        if file_sha(Path(module.__file__)) != meta["sha256"]:
            raise ValueError("installed reference source differs: " + meta["module"])
    norm_source = args.input_dir / "reference-norm.py"
    if file_sha(norm_source) != manifest["norm_source_sha256"]:
        raise ValueError("original residual normalization source changed")
    index_path = args.model_root / "model.safetensors.index.json"
    if file_sha(index_path) != manifest["model_index_sha256"]:
        raise ValueError("GB10 and native model index differ")
    weight_map = json.loads(index_path.read_text())["weight_map"]
    arrays = {}
    for name, meta in manifest["files"].items():
        dtype = {"bf16": np.uint16, "f32": np.float32, "i32": np.int32}[meta["dtype"]]
        array = np.memmap(args.input_dir / meta["file"], dtype=dtype, mode="r", shape=meta["shape"])
        if meta["dtype"] == "bf16" and np.any((array & 0x7f80) == 0x7f80):
            raise ValueError("nonfinite BF16 input/control")
        if meta["dtype"] == "f32" and not np.all(np.isfinite(array)):
            raise ValueError("nonfinite FP32 input/control")
        arrays[name] = array

    def upload(name):
        t = torch.from_numpy(np.array(arrays[name]))
        if manifest["files"][name]["dtype"] == "bf16":
            t = t.view(torch.bfloat16)
        return t.cuda()

    def cpu(t):
        if t.dtype == torch.bfloat16:
            return t.contiguous().view(torch.uint16).cpu().numpy()
        return t.contiguous().cpu().numpy()

    def comparison(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError("comparison type or shape mismatch")
        av, bv = a.reshape(-1), b.reshape(-1)
        bits_a = av.view(np.uint32) if a.dtype == np.float32 else av
        bits_b = bv.view(np.uint32) if b.dtype == np.float32 else bv
        bad = np.flatnonzero(bits_a != bits_b)
        return dict(elements=a.size, bit_mismatches=int(bad.size),
                    first_differences=[dict(index=int(i), actual_bits=int(bits_a[i]),
                                            expected_bits=int(bits_b[i])) for i in bad[:64]],
                    actual_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                    expected_sha256=hashlib.sha256(b.tobytes()).hexdigest())

    def progress(stage):
        torch.cuda.synchronize()
        record = dict(stage=stage, wall_seconds=time.monotonic() - started,
                      peak_device_bytes=torch.cuda.max_memory_allocated())
        with (args.output_dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if record["peak_device_bytes"] > DEVICE_LIMIT:
            raise ValueError("MoE replay device ceiling exceeded")

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    weights = {}
    for meta in manifest["weights"]:
        if weight_map.get(meta["name"]) != meta["shard"]:
            raise ValueError("model tensor shard changed")
        with safe_open(args.model_root / meta["shard"], framework="pt", device="cpu") as checkpoint:
            t = checkpoint.get_tensor(meta["name"])
        if t.dtype != torch.bfloat16 or list(t.shape) != meta["shape"]:
            raise ValueError("model weight dtype or shape mismatch")
        view = t.contiguous().view(torch.uint8).numpy()
        if hashlib.sha256(memoryview(view).cast("B")).hexdigest() != meta["sha256"]:
            raise ValueError("GB10/native model weight bytes differ: " + meta["key"])
        weights[meta["key"]] = t.cuda()
        del t, view
    next_key = "model.language_model.layers.1.input_layernorm.weight"
    next_shard = weight_map[next_key]
    if Path(next_shard).name != next_shard:
        raise ValueError("invalid next normalization model shard")
    with safe_open(args.model_root / next_shard, framework="pt", device="cpu") as checkpoint:
        t = checkpoint.get_tensor(next_key)
    if (t.dtype != torch.bfloat16 or list(t.shape) != [2048] or
            hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest() !=
            manifest["files"]["next_weight"]["sha256"]):
        raise ValueError("GB10/native next normalization weight differs")
    del t
    progress("model_weights_verified")
    x, residual = upload("input"), upload("residual")
    logits = F.linear(x, weights["router"])
    topk_weights, topk_ids, _ = fused_topk(hidden_states=x, gating_output=logits,
                                         topk=8, renormalize=True)
    shared_w = torch.cat((weights["shared_gate_up_gate"], weights["shared_gate_up_up"]), dim=0)
    shared_gate_up = F.linear(x, shared_w)
    shared_activated = torch.empty((TOKENS, 512), dtype=torch.bfloat16, device="cuda")
    torch.ops._C.silu_and_mul(shared_activated, shared_gate_up)
    shared_down = F.linear(shared_activated, weights["shared_down"])
    shared_gate = F.linear(x, weights["shared_gate"])
    shared = torch.sigmoid(shared_gate) * shared_down
    progress("router_and_shared")
    module = importlib.import_module("vllm.model_executor.layers.fused_moe.fused_moe")
    original_dispatch, original_activation = module.dispatch_fused_moe_kernel, module.apply_moe_activation
    stages = {}
    dispatch_count = 0

    def dispatch(*pos, **kw):
        nonlocal dispatch_count
        result = original_dispatch(*pos, **kw)
        if dispatch_count >= 2:
            raise ValueError("unexpected extra MoE projection dispatch")
        name = "routed_gate_up" if dispatch_count == 0 else "routed_weighted"
        stages[name] = pos[2][-1].detach().clone()
        dispatch_count += 1
        return result

    def activation(kind, out, inp):
        result = original_activation(kind, out, inp)
        stages["routed_activated"] = out[-8:].detach().clone()
        return result

    module.dispatch_fused_moe_kernel, module.apply_moe_activation = dispatch, activation
    try:
        routed = fused_experts(x, weights["routed_gate_up"], weights["routed_down"],
                               topk_weights, topk_ids, inplace=False)
        torch.cuda.synchronize()
    finally:
        module.dispatch_fused_moe_kernel, module.apply_moe_activation = original_dispatch, original_activation
    if dispatch_count != 2 or len(stages) != 3:
        raise ValueError("incomplete original MoE stage capture")
    progress("routed_experts")
    moe = shared + routed
    unrounded = residual.float() + moe.float()
    code = extracted(norm_source, "_forward_static_with_residual", "GemmaRMSNorm")
    path = args.output_dir / "original-nextnorm.py"
    path.write_text(code)
    scope = types.ModuleType("_qrt_moe_nextnorm")
    scope.__file__ = str(path)
    scope.torch = torch
    sys.modules[scope.__name__] = scope
    exec(compile(code, str(path), "exec", dont_inherit=True), scope.__dict__)
    normalize = torch.compile(scope._forward_static_with_residual, fullgraph=True)
    nextnorm, rounded = normalize(upload("next_weight"), 1e-6, moe, residual)
    progress("next_normalization")
    terminal = dict(input=x[-1], router=logits[-1], topk_ids=topk_ids[-1].to(torch.int32),
                    topk_weights=topk_weights[-1], shared_gate_up=shared_gate_up[-1],
                    shared_activated=shared_activated[-1], shared_down=shared_down[-1],
                    shared_gate=shared_gate[-1], shared=shared[-1], routed=routed[-1],
                    moe=moe[-1], nextnorm=nextnorm[-1], **stages)
    controls = {name: comparison(cpu(t).reshape(CONTROL_SHAPES[name][1]), arrays["terminal_" + name])
                for name, t in terminal.items()}
    (args.output_dir / "controls.json").write_text(json.dumps(controls, indent=2) + "\n")
    if any(v["bit_mismatches"] for v in controls.values()):
        raise ValueError("full-shape MoE replay does not reproduce all independent terminal controls")
    comparisons = dict(unrounded=comparison(cpu(unrounded), arrays["native_unrounded"]),
                       nextnorm=comparison(cpu(nextnorm), arrays["native_nextnorm"]))
    files = []
    for name, t in dict(router=logits, topk_ids=topk_ids, topk_weights=topk_weights,
                        shared=shared, routed=routed, moe=moe, unrounded=unrounded,
                        rounded=rounded, nextnorm=nextnorm).items():
        a = cpu(t)
        path = args.output_dir / (name + ".bin")
        path.write_bytes(a.tobytes())
        files.append(dict(file=path.name, dtype=str(t.dtype), shape=list(t.shape),
                          bytes=path.stat().st_size, sha256=file_sha(path)))
    for name, meta in manifest["files"].items():
        if file_sha(args.input_dir / meta["file"]) != meta["sha256"]:
            raise ValueError("input/control file changed during replay")
    return dict(completed=True, controls=controls, native_comparisons=comparisons,
                files=files, peak_device_bytes=torch.cuda.max_memory_allocated(),
                torch_version=torch.__version__, triton_version=triton.__version__,
                wall_seconds=time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.timeout_seconds <= 120 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit) or args.output_dir.exists()):
        raise ValueError("invalid source, deadline or existing output")
    if args.worker and not args.execute:
        raise ValueError("worker requires explicit execution")
    manifest = validate(args)
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname() != args.expected_host:
            raise ValueError("MoE replay host mismatch before GPU import")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="sm121_real_q7169_moe_component", host=socket.gethostname(),
                  command=sys.argv, source_commit=args.source_commit,
                  command_source_sha256=file_sha(Path(__file__)), manifest_sha256=args.manifest_sha256,
                  manifest=manifest,
                  model_root=str(args.model_root), model_index_sha256=manifest["model_index_sha256"],
                  model_engine_loaded=False, inference_acceptance=False, completed=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args, manifest))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
