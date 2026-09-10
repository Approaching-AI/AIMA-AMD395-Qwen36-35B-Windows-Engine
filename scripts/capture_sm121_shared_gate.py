#!/usr/bin/env python3
"""Replay all shared scalar gates from qualified GB10 q7169 layer inputs.

The original full-model layer-26 gate is an independent full-shape control.
CPU reductions are diagnostics; only the unchanged CUDA linear supplies the
reference output. This component capture does not qualify native inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

ALL_NORM_SHA = "f17592ae9d7d332386eb4b2a3f001f73e497aab94463588b4252e2f23eebdcc0"
MOE_SHA = "fb2e1a4c20b852b966c83c4853b0f75b0476fc0fd64650a0841ba2e9b8883c2c"
INDEX_SHA = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
ORACLE_SHA = "7fa645e8111932279e71ad20b9a1117b5f5f4f26074fdd43c7b2d274a86ac121"


def qualified(root, expected):
    path = root / "capture.json"
    if file_sha(path) != expected:
        raise ValueError("reference capture fingerprint changed")
    m = json.loads(path.read_text())
    if not (m["completed"] and m["oracle_qualified"] and
            m["oracle_sha256"] == ORACLE_SHA and
            m["raw_logit_argmax_token"] == 82 and m["raw_logit"] == 9.25 and
            len(m["output_token_ids"]) == 32):
        raise ValueError("qualified real-token reference required")
    return m


def execute(args):
    import numpy as np
    import torch
    from safetensors import safe_open

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("one original SM121 CUDA device required")
    torch.set_num_threads(2)
    norms = qualified(args.norm_capture, ALL_NORM_SHA)
    moe = qualified(args.moe_capture, MOE_SHA)
    if norms["output_token_ids"] != moe["output_token_ids"]:
        raise ValueError("reference token boundaries differ")
    index_path = args.model_root / "model.safetensors.index.json"
    if file_sha(index_path) != INDEX_SHA:
        raise ValueError("original/native model index mismatch")
    weights = json.loads(index_path.read_text())["weight_map"]

    def read_tensor(root, meta, shape):
        path = root / "tensors" / meta["file"]
        if (Path(meta["file"]).name != meta["file"] or meta["shape"] != shape or
                path.stat().st_size != 2 * int(np.prod(shape)) or
                file_sha(path) != meta["sha256"]):
            raise ValueError("reference tensor layout/fingerprint mismatch")
        a = np.fromfile(path, dtype=np.uint16).reshape(shape)
        if np.any((a & 0x7f80) == 0x7f80):
            raise ValueError("nonfinite reference input")
        return a

    def bf(a):
        u = np.asarray(a, dtype=np.float32).view(np.uint32)
        return ((u + 0x7fff + ((u >> 16) & 1)) >> 16).astype(np.uint16)

    def compare(a, ref):
        bad = np.flatnonzero(a != ref)
        return dict(mismatches=int(bad.size), first=[dict(position=int(i),
                    actual_bits=int(a[i]), expected_bits=int(ref[i])) for i in bad[:64]])

    records = []
    started = time.monotonic()
    for layer in range(40):
        if time.monotonic() - started > 150:
            raise ValueError("shared gate replay exceeded bounded work time")
        meta = norms["worker"]["files"][f"layer-{layer:02d}-post-attention-rmsnorm"]
        a = read_tensor(args.norm_capture, meta, [7169, 2048])
        name = f"model.language_model.layers.{layer}.mlp.shared_expert_gate.weight"
        shard = weights[name]
        if Path(shard).name != shard:
            raise ValueError("invalid model shard")
        with safe_open(args.model_root / shard, framework="pt", device="cpu") as f:
            w = f.get_tensor(name)
        if w.dtype != torch.bfloat16 or list(w.shape) != [1, 2048]:
            raise ValueError("shared scalar gate model layout changed")
        raw_weight = w.view(torch.uint8).numpy().tobytes()
        xg, wg = torch.from_numpy(a).view(torch.bfloat16).cuda(), w.cuda()
        with torch.inference_mode():
            result = torch.nn.functional.linear(xg, wg)
        ref = result.view(torch.uint16).cpu().numpy().reshape(-1)
        control = None
        if layer == 26:
            if meta["sha256"] != moe["worker"]["files"]["moe-input"]["sha256"]:
                raise ValueError("full-model MoE input control differs")
            expected = read_tensor(args.moe_capture,
                                   moe["worker"]["files"]["moe-shared-gate"], [7169, 1])
            control = compare(ref, expected.reshape(-1))
            if control["mismatches"]:
                raise ValueError("original CUDA linear does not reproduce full-model gates")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                torch.nn.functional.linear(xg, wg)
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(args.output_dir / "linear26-profile.json"))
        xf = (a.astype(np.uint32) << 16).view(np.float32)
        wf = w.float().numpy()
        # BF16 products are exactly representable in FP32 for these finite inputs.
        products = xf * wf
        exact = np.sum(products.astype(np.float64), axis=1)
        tree = products.copy()
        while tree.shape[1] > 1:
            tree = tree[:, 0::2] + tree[:, 1::2]
        balanced = tree[:, 0]
        variants = dict(adjacent_f32=bf(balanced), fp64=bf(exact))
        comparisons = {k: compare(v, ref) for k, v in variants.items()}
        suspect = np.flatnonzero(np.any(np.stack([v != ref for v in variants.values()]), axis=0))
        prefix = f"layer-{layer:02d}"
        (args.output_dir / (prefix + "-gate-bf16.bin")).write_bytes(ref.tobytes())
        (args.output_dir / (prefix + "-weight-bf16.bin")).write_bytes(raw_weight)
        (args.output_dir / (prefix + "-suspect-input-bf16.bin")).write_bytes(a[suspect].tobytes())
        record = dict(layer=layer, input_sha256=meta["sha256"], weight=name, shard=shard,
                      weight_sha256=hashlib.sha256(raw_weight).hexdigest(),
                      output_sha256=hashlib.sha256(ref.tobytes()).hexdigest(),
                      comparisons=comparisons, full_model_control=control,
                      suspects=[dict(position=int(i), expected_bits=int(ref[i]),
                                     fp64=float(exact[i]), adjacent_f32=float(balanced[i]))
                                for i in suspect])
        records.append(record)
        with (args.output_dir / "progress.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        del xg, wg, result, products, xf, tree
    return dict(completed=True, component_qualified=True, layers=records,
                wall_seconds=time.monotonic() - started, torch_version=torch.__version__,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                norm_capture_sha256=ALL_NORM_SHA, moe_capture_sha256=MOE_SHA,
                model_index_sha256=INDEX_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "norm-capture", "moe-capture", "model-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.output_dir.exists() or not 1 <= args.timeout_seconds <= 210 or
            len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("existing output or invalid source/deadline")
    if args.worker and not args.execute:
        raise ValueError("worker requires explicit execution")
    if args.execute:
        if sys.platform != "linux" or socket.gethostname() != args.expected_host:
            raise ValueError("explicit GB10 execution host required")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            return supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                              "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds)
    args.output_dir.mkdir(parents=True)
    record = dict(kind="sm121_shared_gate_full_shape_replay", host=socket.gethostname(),
                  command=sys.argv, source_commit=args.source_commit,
                  source_sha256=file_sha(Path(__file__)), model=str(args.model_root),
                  completed=False, component_qualified=False, inference_acceptance=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        record.update(execute(args))
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "layers"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
