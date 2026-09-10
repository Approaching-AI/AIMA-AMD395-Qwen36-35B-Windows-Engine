#!/usr/bin/env python3
"""Observe q7169 full-prefix attention in an owned, bounded GB10 engine.

The existing oracle is an immutable qualification input. Native captures are
never inputs. Read-only module hooks are installed after engine startup and
removed after one request. Full artifacts remain diagnostic until the same
request passes the existing 32-token and raw-logit boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import struct
import sys
import time

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

TOKENS = 7169
MAXIMUM_CAPTURE_BYTES = 768 << 20
ORACLE_SHA = "7fa645e8111932279e71ad20b9a1117b5f5f4f26074fdd43c7b2d274a86ac121"


def prompt_and_oracle(path):
    if file_sha(path) != ORACLE_SHA:
        raise ValueError("frozen q7169 oracle changed")
    oracle = json.loads(path.read_text())
    source = oracle["prompt"]
    generator = random.Random(source["seed"])
    prompt = [source["first_token_id"]]
    prompt.extend(32 + generator.randrange(256) for _ in range(TOKENS - 1))
    packed = struct.pack(f"<{TOKENS}I", *prompt)
    if hashlib.sha256(packed).hexdigest() != source["u32le_sha256"]:
        raise ValueError("real-token prompt fingerprint mismatch")
    return prompt, oracle


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


class FullAttentionCapture:
    """vLLM worker extension; observes the existing model without replacing ops."""

    def qrt_arm_full_attention(self, directory):
        import torch

        if hasattr(self, "_qrt_handles"):
            raise ValueError("worker capture already initialized")
        root = Path(directory)
        root.mkdir(exist_ok=False)
        self._qrt_root, self._qrt_files = root, {}
        self._qrt_bytes, self._qrt_started = 0, time.monotonic()
        self._qrt_handles = []
        model = self.model_runner.model
        containers = [(name, module) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.ModuleList) and len(module) == 40
                      and hasattr(module[3], "self_attn")]
        if len(containers) != 1:
            raise ValueError("ambiguous forty-layer model")
        name, layers = containers[0]
        parent = model.get_submodule(name.rsplit(".", 1)[0])

        def save(label, tensor, terminal=False):
            if label in self._qrt_files or tensor.shape[0] != TOKENS:
                return
            if tensor.dtype != torch.bfloat16 or time.monotonic() - self._qrt_started > 90:
                raise ValueError("capture dtype or observation deadline changed")
            value = tensor[-1:] if terminal else tensor
            size = value.numel() * 2
            if self._qrt_bytes + size > MAXIMUM_CAPTURE_BYTES:
                raise ValueError("full-prefix capture byte ceiling exceeded")
            payload = value.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
            path = root / (label + "-bf16.bin")
            with path.open("xb") as stream:
                stream.write(payload)
            self._qrt_bytes += size
            self._qrt_files[label] = dict(file=path.name, shape=list(value.shape),
                                         bytes=size, sha256=file_sha(path), terminal=terminal)

        def first(value):
            return value[0] if isinstance(value, tuple) else value

        def output_hook(label):
            return lambda module, args, output: save(label, first(output))

        def input_hook(label):
            return lambda module, args: save(label, args[0])

        def attach(module, callback, pre=False):
            method = module.register_forward_pre_hook if pre else module.register_forward_hook
            self._qrt_handles.append(method(callback))

        def layer_hook(layer):
            def observe(module, args, output):
                hidden, residual = output
                if hidden.shape[0] != TOKENS:
                    return
                save(f"layer-{layer:02d}-hidden", hidden, terminal=True)
                save(f"layer-{layer:02d}-residual", residual, terminal=True)
                # Both operands are BF16. CPU addition of the copied row keeps
                # the observer from adding a kernel to the original GPU stream.
                if f"layer-{layer:02d}-combined" not in self._qrt_files:
                    combined = hidden[-1:].detach().cpu() + residual[-1:].detach().cpu()
                    path = root / f"layer-{layer:02d}-combined-bf16.bin"
                    payload = combined.contiguous().view(torch.uint16).numpy().tobytes()
                    with path.open("xb") as stream:
                        stream.write(payload)
                    self._qrt_bytes += len(payload)
                    self._qrt_files[f"layer-{layer:02d}-combined"] = dict(
                        file=path.name, shape=list(combined.shape), bytes=len(payload),
                        sha256=file_sha(path), terminal=True)
            return observe

        for index, layer in enumerate(layers):
            attach(layer, layer_hook(index))
            if index <= 3:
                attach(layer.input_layernorm, output_hook(f"layer-{index:02d}-input-rmsnorm"))
        attention = layers[3].self_attn
        attach(attention.qkv_proj, output_hook("full-attention-qkv"))
        attach(attention.q_norm, output_hook("full-attention-q-norm"))
        attach(attention.k_norm, output_hook("full-attention-k-norm"))

        def attention_inputs(module, args):
            for label, tensor in zip(("q-rope", "k-rope", "v"), args[:3]):
                save("full-attention-" + label, tensor)

        attach(attention.attn, attention_inputs, pre=True)
        attach(attention.attn, output_hook("full-attention-context"))
        attach(attention.o_proj, input_hook("full-attention-gated-context"), pre=True)
        attach(attention.o_proj, output_hook("full-attention-o-projection"))

        def postnorm(module, args, output):
            save("full-attention-postnorm", output[0])
            save("full-attention-residual", output[1])

        attach(layers[3].post_attention_layernorm, postnorm)
        attach(parent.norm, lambda module, args, output: save("final-norm", first(output), terminal=True))
        return dict(model_type=type(model).__name__, layer_container=name,
                    hooks=len(self._qrt_handles), maximum_bytes=MAXIMUM_CAPTURE_BYTES)

    def qrt_finish_full_attention(self):
        for handle in self._qrt_handles:
            handle.remove()
        required = {f"layer-{i:02d}-combined" for i in range(40)} | {
            "full-attention-qkv", "full-attention-q-norm", "full-attention-k-norm",
            "full-attention-q-rope", "full-attention-k-rope", "full-attention-v",
            "full-attention-context", "full-attention-gated-context",
            "full-attention-o-projection", "full-attention-postnorm",
            "full-attention-residual", "final-norm"}
        record = dict(files=self._qrt_files, bytes=self._qrt_bytes,
                      complete=required <= self._qrt_files.keys(),
                      missing=sorted(required - self._qrt_files.keys()),
                      observation_seconds=time.monotonic() - self._qrt_started)
        write_json(self._qrt_root / "worker-capture.json", record)
        return record


def execute(args, prompt, oracle):
    import torch
    from safetensors import safe_open
    from vllm import LLM, SamplingParams

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("full model reference requires one SM121 CUDA device")
    torch.set_num_threads(4)
    for filename, key in (("config.json", "config_sha256"),
                          ("model.safetensors.index.json", "index_sha256")):
        if file_sha(args.model_root / filename) != oracle["model_evidence"][key]:
            raise ValueError("reference model layout or configuration changed")
    started = time.monotonic()
    llm = LLM(model=str(args.model_root), dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.8, max_model_len=263680,
              max_num_batched_tokens=8192, max_num_seqs=4, skip_mm_profiling=True,
              async_scheduling=False, attention_config={"backend": "TRITON_ATTN"},
              mm_encoder_attn_backend="TORCH_SDPA",
              speculative_config={"method": "mtp", "num_speculative_tokens": 1},
              worker_extension_cls="capture_gb10_full_attention.FullAttentionCapture")
    load_seconds = time.monotonic() - started
    write_json(args.output_dir / "ready.json", dict(load_seconds=load_seconds))
    armed = llm.collective_rpc("qrt_arm_full_attention", args=(str(args.output_dir / "tensors"),))
    requested = time.monotonic()
    outputs = llm.generate([dict(prompt_token_ids=prompt)],
                           SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
                           use_tqdm=False)
    request_seconds = time.monotonic() - requested
    worker = llm.collective_rpc("qrt_finish_full_attention")[0]
    tokens = list(outputs[0].outputs[0].token_ids)
    if not worker["complete"]:
        raise ValueError("full-prefix module observations are incomplete")
    hidden_file = args.output_dir / "tensors" / worker["files"]["final-norm"]["file"]
    hidden = torch.frombuffer(bytearray(hidden_file.read_bytes()), dtype=torch.bfloat16).reshape(1, 2048)
    evidence = oracle["model_evidence"]
    shard = args.model_root / evidence["weight_shard"]
    if file_sha(shard) != evidence["weight_shard_sha256"]:
        raise ValueError("independent LM-head weight fingerprint changed")
    with safe_open(shard, framework="pt", device="cpu") as stream:
        weight = stream.get_tensor(evidence["weight_tensor_key"])
        logits = torch.nn.functional.linear(hidden, weight)[0]
    selected = int(logits.argmax().item())
    raw_logit = float(logits[selected].item())
    expected = oracle["expected"]
    qualified = (tokens == expected["output_token_ids"] and selected == tokens[0] and
                 abs(raw_logit - expected["first_token_raw_logit"]) <= expected["first_token_raw_logit_tolerance"])
    return dict(completed=True, oracle_qualified=qualified, output_token_ids=tokens,
                raw_logit_argmax_token=selected, raw_logit=raw_logit,
                final_norm_sha256=file_sha(hidden_file), load_seconds=load_seconds,
                request_seconds=request_seconds, worker=worker, armed=armed,
                torch_version=torch.__version__, raw_logit_method="original CPU BF16 full-vocabulary LM-head replay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=420)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    prompt, oracle = prompt_and_oracle(args.oracle)
    if (args.output_dir.exists() or not 1 <= args.timeout_seconds <= 480 or
            len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit)):
        raise ValueError("existing output, invalid source or invalid deadline")
    if args.worker and not args.execute:
        raise ValueError("worker requires execution")
    if args.execute:
        if sys.platform != "linux" or socket.gethostname() != args.expected_host:
            raise ValueError("explicit GB10 host required before GPU import")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            return supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                              "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="gb10_q7169_full_attention_capture", host=socket.gethostname(),
                  command=sys.argv, source_commit=args.source_commit,
                  source_sha256=file_sha(Path(__file__)), model=str(args.model_root),
                  oracle_sha256=file_sha(args.oracle), completed=False, oracle_qualified=False,
                  native_tensor_inputs=False, windows_acceptance=False)
    write_json(args.output_dir / "preflight.json", record)
    if args.execute:
        try:
            record.update(execute(args, prompt, oracle))
        except Exception as error:
            write_json(args.output_dir / "failure.json", dict(error=str(error)))
            raise
    write_json(args.output_dir / "capture.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "worker"}))
    return 0 if not args.execute or record["oracle_qualified"] else 6


if __name__ == "__main__":
    raise SystemExit(main())
