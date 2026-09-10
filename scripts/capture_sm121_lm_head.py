#!/usr/bin/env python3
"""Replay the original CUDA BF16 output head on a qualified real model input.

All vocabulary scores are computed before selecting a token. CPU BF16 and
unrounded FP64 scores are diagnostics; neither supplies CUDA compute inputs.
This component replay does not establish native inference acceptance.
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
from capture_sm121_shared_gate import ALL_NORM_SHA, ORACLE_SHA, qualified


def execute(args):
    import torch
    from safetensors import safe_open

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("one original SM121 CUDA device required")
    torch.set_num_threads(2)
    capture = qualified(args.norm_capture, ALL_NORM_SHA)
    if file_sha(args.oracle) != ORACLE_SHA:
        raise ValueError("frozen oracle changed")
    oracle = json.loads(args.oracle.read_text())
    evidence = oracle["model_evidence"]
    for leaf, key in (("config.json", "config_sha256"),
                      ("model.safetensors.index.json", "index_sha256")):
        if file_sha(args.model_root / leaf) != evidence[key]:
            raise ValueError("model configuration/index changed")
    meta = capture["worker"]["files"]["final-norm"]
    if (Path(meta["file"]).name != meta["file"] or
            meta["shape"] != [1, 2048] or meta["bytes"] != 4096):
        raise ValueError("invalid final normalization layout")
    hidden_path = args.norm_capture / "tensors" / meta["file"]
    if hidden_path.stat().st_size != 4096 or file_sha(hidden_path) != meta["sha256"]:
        raise ValueError("qualified final normalization changed")
    shard = args.model_root / evidence["weight_shard"]
    if (Path(evidence["weight_shard"]).name != evidence["weight_shard"] or
            file_sha(shard) != evidence["weight_shard_sha256"]):
        raise ValueError("original LM-head shard changed")
    hidden = torch.frombuffer(bytearray(hidden_path.read_bytes()),
                              dtype=torch.bfloat16).reshape(1, 2048)
    with safe_open(shard, framework="pt", device="cpu") as stream:
        weight = stream.get_tensor(evidence["weight_tensor_key"])
    if (weight.dtype != torch.bfloat16 or list(weight.shape) != [248320, 2048] or
            not torch.isfinite(hidden).all()):
        raise ValueError("original LM-head layout or finite input required")
    weight_sha = hashlib.sha256(weight.view(torch.uint8).numpy()).hexdigest()
    started = time.monotonic()
    xg, wg = hidden.cuda(), weight.cuda()
    with torch.inference_mode():
        result = torch.nn.functional.linear(xg, wg)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) as prof:
            repeated = torch.nn.functional.linear(xg, wg)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(args.output_dir / "linear-profile.json"))
        if not torch.equal(result, repeated):
            raise ValueError("original CUDA output head is not repeatable")
        cpu_result = torch.nn.functional.linear(hidden, weight)
    scores = result.cpu()[0]
    raw = scores.view(torch.uint8).numpy().tobytes()
    (args.output_dir / "logits-bf16.bin").write_bytes(raw)
    (args.output_dir / "input-bf16.bin").write_bytes(hidden_path.read_bytes())
    winner = int(scores.argmax())
    expected = oracle["expected"]
    component_qualified = (winner == expected["first_token_id"] and
                          abs(float(scores[winner]) - expected["first_token_raw_logit"])
                          <= expected["first_token_raw_logit_tolerance"])
    order = torch.argsort(scores.float(), descending=True, stable=True)[:32]
    selected_weight = weight[order].contiguous()
    (args.output_dir / "top32-weight-bf16.bin").write_bytes(
        selected_weight.view(torch.uint8).numpy().tobytes())
    unrounded = torch.nn.functional.linear(hidden.double(), selected_weight.double())[0]
    top = [dict(token_id=int(i), bf16_logit=float(scores[i]),
                bf16_bits=int(scores.view(torch.uint16)[i]),
                cpu_bf16_logit=float(cpu_result[0, i]),
                fp64_dot=float(unrounded[j])) for j, i in enumerate(order)]
    return dict(completed=True, component_qualified=component_qualified,
                norm_capture_sha256=ALL_NORM_SHA, oracle_sha256=ORACLE_SHA,
                model_evidence=evidence, weight_sha256=weight_sha,
                input_sha256=meta["sha256"], logits_sha256=hashlib.sha256(raw).hexdigest(),
                elements=scores.numel(), raw_logit_argmax_token=winner,
                raw_logit=float(scores[winner]),
                cpu_argmax_token=int(cpu_result.argmax()),
                cpu_bf16_mismatches=int((scores != cpu_result[0]).sum()),
                maximum_tie_token_ids=torch.where(scores == scores.max())[0].tolist(),
                top32=top, wall_seconds=time.monotonic() - started,
                torch_version=torch.__version__,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                reference_logits_are_compute_input=False, captured_real_input=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "norm-capture", "oracle", "model-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-host")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.output_dir.exists() or not 1 <= args.timeout_seconds <= 120 or
            len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit)):
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
    record = dict(kind="sm121_lm_head_full_vocabulary_replay", host=socket.gethostname(),
                  command=sys.argv, source_commit=args.source_commit,
                  source_sha256=file_sha(Path(__file__)), model=str(args.model_root),
                  completed=False, component_qualified=False, inference_acceptance=False)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        record.update(execute(args))
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "top32"}))
    return 0 if not args.execute or record["component_qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
