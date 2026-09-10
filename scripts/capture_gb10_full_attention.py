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
ALL_NORM_CAPTURE_BYTES = 3 << 30
LINEAR_CAPTURE_BYTES = 1536 << 20
MOE_CAPTURE_BYTES = 3 << 30
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

    def qrt_arm_full_attention(self, directory, attention_layer=3, all_layer_norms=False,
                              linear_layer=None, moe_layer=None):
        import torch

        if attention_layer not in range(3, 40, 4) or not isinstance(all_layer_norms, bool):
            raise ValueError("invalid full-attention observation scope")
        if linear_layer is not None and (linear_layer not in range(40) or
                                         linear_layer % 4 == 3 or all_layer_norms):
            raise ValueError("select one linear layer without the all-normalization scope")
        if moe_layer is not None and (moe_layer not in range(39) or all_layer_norms):
            raise ValueError("select one MoE and its next normalization without the all-normalization scope")
        if hasattr(self, "_qrt_handles"):
            raise ValueError("worker capture already initialized")
        root = Path(directory)
        root.mkdir(exist_ok=False)
        self._qrt_root, self._qrt_files = root, {}
        self._qrt_bytes, self._qrt_started = 0, time.monotonic()
        self._qrt_maximum_bytes = ALL_NORM_CAPTURE_BYTES if all_layer_norms else MAXIMUM_CAPTURE_BYTES
        if linear_layer is not None:
            self._qrt_maximum_bytes = LINEAR_CAPTURE_BYTES
        if moe_layer is not None:
            self._qrt_maximum_bytes = MOE_CAPTURE_BYTES
        self._qrt_observation_seconds = 180 if all_layer_norms else 90
        self._qrt_norm_labels = set()
        self._qrt_handles = []
        model = self.model_runner.model
        containers = [(name, module) for name, module in model.named_modules()
                      if isinstance(module, torch.nn.ModuleList) and len(module) == 40
                      and hasattr(module[3], "self_attn")]
        if len(containers) != 1:
            raise ValueError("ambiguous forty-layer model")
        name, layers = containers[0]
        parent = model.get_submodule(name.rsplit(".", 1)[0])

        def save(label, tensor, terminal=False, allow_f32=False):
            if label in self._qrt_files or tensor.shape[0] != TOKENS:
                return
            if (tensor.dtype not in ((torch.bfloat16, torch.float32) if allow_f32 else (torch.bfloat16,)) or
                    time.monotonic() - self._qrt_started > self._qrt_observation_seconds):
                raise ValueError("capture dtype or observation deadline changed")
            value = tensor[-1:] if terminal else tensor
            size = value.numel() * value.element_size()
            if self._qrt_bytes + size > self._qrt_maximum_bytes:
                raise ValueError("full-prefix capture byte ceiling exceeded")
            payload = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            dtype = "bf16" if tensor.dtype == torch.bfloat16 else "f32"
            path = root / (label + "-" + dtype + ".bin")
            with path.open("xb") as stream:
                stream.write(payload)
            self._qrt_bytes += size
            self._qrt_files[label] = dict(file=path.name, shape=list(value.shape),
                                         bytes=size, sha256=file_sha(path), terminal=terminal,
                                         dtype=dtype)

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
            if (index <= 3 or all_layer_norms or index == linear_layer or
                    (moe_layer is not None and index in (moe_layer, moe_layer + 1))):
                label = f"layer-{index:02d}-input-rmsnorm"
                self._qrt_norm_labels.add(label)
                attach(layer.input_layernorm, output_hook(label))
            if all_layer_norms or index == linear_layer or index == moe_layer:
                label = f"layer-{index:02d}-post-attention-rmsnorm"
                self._qrt_norm_labels.add(label)
                attach(layer.post_attention_layernorm, output_hook(label))
        attention = layers[attention_layer].self_attn
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

        attach(layers[attention_layer].post_attention_layernorm, postnorm)
        self._qrt_linear_labels = set()
        linear_source = None
        if linear_layer is not None:
            import inspect

            linear = layers[linear_layer].linear_attn
            if linear.gqa_interleaved_layout or linear.tp_size != 1:
                raise ValueError("linear observation requires the original single-device Qwen3.5 layout")
            source_path = Path(inspect.getsourcefile(type(linear)))
            linear_source = dict(file=str(source_path), sha256=file_sha(source_path))
            self._qrt_linear_labels = {
                "linear-qkvz", "linear-ba", "linear-q-postconv", "linear-k-postconv",
                "linear-v-postconv", "linear-g", "linear-beta", "linear-core", "linear-z",
                "linear-gated", "linear-o-projection", "linear-seed", "linear-residual"}
            attach(linear.in_proj_qkvz, output_hook("linear-qkvz"))
            attach(linear.in_proj_ba, output_hook("linear-ba"))

            def core_inputs(module, args, kwargs):
                q = kwargs.get("q")
                if q is None or tuple(q.shape[:2]) != (1, TOKENS):
                    return
                for key, width in (("q", 2048), ("k", 2048), ("v", 4096),
                                   ("g", 32), ("beta", 32)):
                    value = kwargs[key]
                    if value.numel() != TOKENS * width:
                        raise ValueError("linear input shape changed: " + key)
                    label = "linear-" + key + ("-postconv" if key in ("q", "k", "v") else "")
                    save(label, value.reshape(TOKENS, width), allow_f32=key in ("g", "beta"))

            self._qrt_handles.append(linear.chunk_gated_delta_rule.register_forward_pre_hook(
                core_inputs, with_kwargs=True))

            def gated_inputs(module, args):
                if args[0].numel() != TOKENS * 4096:
                    return
                for label, value in zip(("linear-core", "linear-z"), args[:2]):
                    save(label, value.reshape(TOKENS, 4096))

            attach(linear.norm, gated_inputs, pre=True)
            attach(linear.out_proj, input_hook("linear-gated"), pre=True)
            attach(linear.out_proj, output_hook("linear-o-projection"))
            attach(layers[linear_layer].input_layernorm,
                   lambda module, args, output: save("linear-seed", output[1] if isinstance(output, tuple) else args[0]))
            attach(layers[linear_layer].post_attention_layernorm,
                   lambda module, args, output: save("linear-residual", output[1]))
        self._qrt_moe_labels = set()
        moe_source = None
        if moe_layer is not None:
            import inspect

            mlp = layers[moe_layer].mlp
            if mlp.experts.is_internal_router or mlp.tp_size != 1 or mlp.shared_expert is None:
                raise ValueError("MoE observation requires the original single-device external router and shared expert")
            source_path = Path(inspect.getsourcefile(type(mlp)))
            moe_source = dict(file=str(source_path), sha256=file_sha(source_path))
            self._qrt_moe_labels = {
                "moe-input", "moe-router", "moe-output", "moe-shared",
                "moe-expert-part-0", "moe-expert-part-1", "moe-next-hidden",
                "moe-residual"}
            attach(mlp, input_hook("moe-input"), pre=True)
            attach(mlp.gate, output_hook("moe-router"))
            attach(mlp.shared_expert, output_hook("moe-shared"))
            attach(mlp, output_hook("moe-output"))

            def expert_outputs(module, args, output):
                if not isinstance(output, tuple) or len(output) != 2:
                    raise ValueError("original shared/routed expert tuple changed")
                for index, value in enumerate(output):
                    save(f"moe-expert-part-{index}", value)

            def next_inputs(module, args):
                if len(args) != 2:
                    raise ValueError("original residual normalization arguments changed")
                save("moe-next-hidden", args[0])
                save("moe-residual", args[1])

            attach(mlp.experts, expert_outputs)
            attach(layers[moe_layer + 1].input_layernorm, next_inputs, pre=True)
        attach(parent.norm, lambda module, args, output: save("final-norm", first(output), terminal=True))
        return dict(model_type=type(model).__name__, layer_container=name,
                    hooks=len(self._qrt_handles), maximum_bytes=self._qrt_maximum_bytes,
                    attention_layer=attention_layer, all_layer_norms=all_layer_norms,
                    linear_layer=linear_layer, linear_source=linear_source,
                    moe_layer=moe_layer, moe_source=moe_source,
                    maximum_observation_seconds=self._qrt_observation_seconds)

    def qrt_finish_full_attention(self):
        for handle in self._qrt_handles:
            handle.remove()
        required = {f"layer-{i:02d}-combined" for i in range(40)} | {
            "full-attention-qkv", "full-attention-q-norm", "full-attention-k-norm",
            "full-attention-q-rope", "full-attention-k-rope", "full-attention-v",
            "full-attention-context", "full-attention-gated-context",
            "full-attention-o-projection", "full-attention-postnorm",
            "full-attention-residual", "final-norm"}
        required |= self._qrt_norm_labels
        required |= self._qrt_linear_labels
        required |= self._qrt_moe_labels
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
    armed = llm.collective_rpc("qrt_arm_full_attention", args=(
        str(args.output_dir / "tensors"), args.attention_layer, args.all_layer_norms,
        args.linear_layer, args.moe_layer))
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
    parser.add_argument("--attention-layer", type=int, choices=range(3, 40, 4), default=3)
    parser.add_argument("--all-layer-norms", action="store_true",
                        help="observe all 80 complete normalization boundaries with a three-GiB ceiling")
    parser.add_argument("--linear-layer", type=int, choices=[i for i in range(40) if i % 4 != 3],
                        help="also observe one original linear-attention path with a 1.5-GiB total ceiling")
    parser.add_argument("--moe-layer", type=int, choices=range(39),
                        help="also observe one original MoE and next norm with a three-GiB total ceiling")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.linear_layer is not None or args.moe_layer is not None) and args.all_layer_norms:
        raise ValueError("selected-operator and all-normalization capture scopes are separately bounded")
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
                  native_tensor_inputs=False, windows_acceptance=False,
                  attention_layer=args.attention_layer, all_layer_norms=args.all_layer_norms,
                  linear_layer=args.linear_layer, moe_layer=args.moe_layer,
                  maximum_capture_bytes=(MOE_CAPTURE_BYTES if args.moe_layer is not None else
                                         LINEAR_CAPTURE_BYTES if args.linear_layer is not None else
                                         ALL_NORM_CAPTURE_BYTES if args.all_layer_norms else MAXIMUM_CAPTURE_BYTES))
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
