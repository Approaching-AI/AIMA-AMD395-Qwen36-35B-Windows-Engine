#!/usr/bin/env python3
"""Capture adjacent prompt lengths and 512-token continuations on owned GB10.

Two immutable 32-token oracles qualify the same loaded BF16 reference engine.
The observer copies its original first logits without replacing model math.
Every request disables prefix caching; new captures are reference evidence,
not Windows acceptance or changes to either existing oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import socket
import struct
import sys
import time

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha
from generate_random_token_prompt import fnv1a64


ORACLE_SHAS = {
    7169: "7fa645e8111932279e71ad20b9a1117b5f5f4f26074fdd43c7b2d274a86ac121",
    8192: "e0323f365318dca622c100f0269a22edcef4e632f78b8b6f9d62476d43834890",
}
GPU_MODEL_RUNNER_SHA = "3afc290d3df1be3df1b89b9b35942695f5896c7729f608d6c44580899212c301"


def sampled_prefill_boundary(requests, prompt_tokens, processed_tokens, discarded, expected_prompt):
    """Use the same CPU discard mask as the pinned runner's sampler bookkeeping."""
    if requests != 1 or prompt_tokens != expected_prompt or processed_tokens < 1:
        raise ValueError("unexpected first-sample request metadata")
    if bool(discarded) != (processed_tokens < prompt_tokens):
        raise ValueError("first-sample discard mask disagrees with prompt progress")
    return not discarded


def fingerprints(tokens):
    packed = struct.pack(f"<{len(tokens)}I", *tokens)
    # The engine's historical offset basis is part of both frozen contracts.
    return dict(u32le_sha256=hashlib.sha256(packed).hexdigest(), u32le_fnv1a64=f"{fnv1a64(packed):016x}")


def fixtures(paths):
    oracles = {}
    for length, path in paths.items():
        if file_sha(path) != ORACLE_SHAS[length]:
            raise ValueError("immutable control oracle changed")
        oracle = json.loads(path.read_text())
        if oracle["prompt"]["token_count"] != length or len(oracle["expected"]["output_token_ids"]) != 32:
            raise ValueError("control shape changed")
        oracles[length] = oracle
    result = []
    for length, family, outputs in ((7169, 7169, 32), (8192, 8192, 32),
                                    (7168, 7169, 32), (7170, 7169, 32),
                                    (8191, 8192, 32), (8193, 8192, 32),
                                    (7169, 7169, 512), (8192, 8192, 512)):
        source = oracles[family]["prompt"]
        generator = random.Random(source["seed"])
        prompt = [source["first_token_id"]] + [32 + generator.randrange(256) for _ in range(length - 1)]
        digest = fingerprints(prompt)
        control = length == family
        if control and any(digest[key] != source[key] for key in digest):
            raise ValueError("control token materialization changed")
        result.append(dict(name=f"q{length}-out{outputs}", prompt_token_ids=prompt,
                           prompt=dict(token_count=length, seed=source["seed"],
                                       first_token_id=source["first_token_id"], **digest),
                           output_count=outputs, control=control, family=family))
    return result, oracles


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class TokenMatrixCapture:
    """Read-only first-logit observer in the existing vLLM model worker."""

    def qrt_arm_token_matrix(self, directory, prompt_tokens):
        import inspect
        import torch

        if getattr(self, "_qrt_token_capture", None) is not None:
            raise ValueError("previous token observer is still armed")
        root = Path(directory)
        root.mkdir(exist_ok=False)
        model = self.model_runner.model
        original = model.compute_logits
        source = Path(inspect.getsourcefile(original))
        runner_source = Path(inspect.getsourcefile(type(self.model_runner)))
        if file_sha(runner_source) != GPU_MODEL_RUNNER_SHA:
            raise ValueError("sampler discard-mask source contract changed")
        self._qrt_token_capture = dict(complete=False, discarded_prefill_calls=0)
        self._qrt_token_owner, self._qrt_token_original = model, original

        def observe(*args, **kwargs):
            output = original(*args, **kwargs)
            if not self._qrt_token_capture["complete"]:
                runner = self.model_runner
                boundary = dict(requests=runner.input_batch.num_reqs,
                                prompt_tokens=int(runner.input_batch.num_prompt_tokens[0]),
                                processed_tokens=int(runner.optimistic_seq_lens_cpu[0]),
                                discarded=bool(runner.discard_request_mask.np[0]),
                                expected_prompt=prompt_tokens)
                if not sampled_prefill_boundary(**boundary):
                    self._qrt_token_capture["discarded_prefill_calls"] += 1
                    return output
                if (output is None or output.ndim != 2 or output.shape[0] != 1 or
                        not 1 <= output.shape[1] <= 262144 or
                        output.dtype not in (torch.bfloat16, torch.float32)):
                    raise ValueError("unexpected original first-logit layout")
                copied = output.detach().contiguous().cpu()
                values = copied[0].float()
                if torch.isnan(values).any() or torch.isposinf(values).any():
                    raise ValueError("invalid original logits")
                selected = int(values.argmax().item())
                raw_logit = float(values[selected].item())
                if not math.isfinite(raw_logit):
                    raise ValueError("nonfinite original argmax")
                payload = copied.view(torch.uint8).numpy().tobytes()
                path = root / "first-logits.bin"
                with path.open("xb") as stream:
                    stream.write(payload)
                self._qrt_token_capture = dict(
                    complete=True, raw_argmax_token=selected, raw_logit=raw_logit,
                    sampling_boundary=boundary,
                    discarded_prefill_calls=self._qrt_token_capture["discarded_prefill_calls"],
                    dtype=str(copied.dtype), shape=list(copied.shape), bytes=len(payload),
                    file=path.name, sha256=hashlib.sha256(payload).hexdigest(),
                    negative_infinity_count=int(torch.isneginf(values).sum().item()),
                    observer_modifies_output=False)
                write_json(root / "worker-capture.json", self._qrt_token_capture)
            return output

        model.compute_logits = observe
        return dict(compute_logits_source=dict(file=str(source), sha256=file_sha(source)),
                    model_runner_source=dict(file=str(runner_source), sha256=GPU_MODEL_RUNNER_SHA),
                    model_class=type(model).__name__, observer_modifies_output=False)

    def qrt_finish_token_matrix(self):
        self._qrt_token_owner.compute_logits = self._qrt_token_original
        record = self._qrt_token_capture
        self._qrt_token_capture = None
        self._qrt_token_owner = self._qrt_token_original = None
        return record


def execute(args, cases, oracles):
    import torch
    from vllm import LLM, SamplingParams

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("reference capture requires one SM121 CUDA device")
    torch.set_num_threads(4)
    evidence = oracles[8192]["model_evidence"]
    for filename, key in (("config.json", "config_sha256"),
                          ("model.safetensors.index.json", "index_sha256")):
        if file_sha(args.model_root / filename) != evidence[key]:
            raise ValueError("reference model configuration changed")
    if file_sha(args.model_root / evidence["weight_shard"]) != evidence["weight_shard_sha256"]:
        raise ValueError("reference LM-head weight shard changed")
    matrix = None
    if args.runtime_boundaries:
        path = args.oracle_q8192.parent / "gb10_cold_token_matrix_20260911_oracle.json"
        if file_sha(path) != "7a6feb488f4dd136e49c4d7227c1fc325e3216af3dae3e9b60aa0b3025f3d1b1":
            raise ValueError("frozen complete matrix changed")
        matrix = {case["name"]: case for case in json.loads(path.read_text())["cases"]}
    begun = time.monotonic()
    llm = LLM(model=str(args.model_root), dtype="bfloat16", trust_remote_code=True,
              enforce_eager=True, gpu_memory_utilization=0.8, max_model_len=263680,
              max_num_batched_tokens=8192, max_num_seqs=4, skip_mm_profiling=True,
              async_scheduling=False, enable_prefix_caching=False,
              attention_config={"backend": "TRITON_ATTN"}, mm_encoder_attn_backend="TORCH_SDPA",
              speculative_config={"method": "mtp", "num_speculative_tokens": 1},
              worker_extension_cls=("capture_gb10_runtime_boundaries.RuntimeBoundaryCapture"
                                    if args.runtime_boundaries else
                                    "capture_gb10_token_matrix.TokenMatrixCapture"))
    load_seconds = time.monotonic() - begun
    write_json(args.output_dir / "ready.json", dict(load_seconds=load_seconds,
               torch_version=torch.__version__, model_evidence=evidence))
    results = []
    for case in cases:
        armed = llm.collective_rpc("qrt_arm_token_matrix", args=(
            str(args.output_dir / case["name"]), case["prompt"]["token_count"]))
        requested = time.monotonic()
        try:
            outputs = llm.generate([dict(prompt_token_ids=case["prompt_token_ids"])],
                                   SamplingParams(temperature=0, max_tokens=case["output_count"],
                                                  ignore_eos=True), use_tqdm=False)
        finally:
            captured = llm.collective_rpc("qrt_finish_token_matrix")
        request_seconds = time.monotonic() - requested
        if len(armed) != 1 or len(captured) != 1 or not captured[0]["complete"]:
            raise ValueError("first-logit observation is incomplete")
        worker = captured[0]
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError("batch-one output shape changed")
        generated = outputs[0].outputs[0]
        tokens = list(generated.token_ids)
        if len(tokens) != case["output_count"] or tokens[0] != worker["raw_argmax_token"]:
            write_json(args.output_dir / (case["name"] + "-rejected.json"),
                       dict(prompt=case["prompt"], requested_outputs=case["output_count"],
                            output_token_ids=tokens, worker=worker, armed=armed,
                            request_seconds=request_seconds, windows_acceptance=False))
            raise ValueError("generation differs from original greedy logits or requested length")
        expected = oracles[case["family"]]["expected"]
        control_pass = not case["control"] or (
            tokens[:32] == expected["output_token_ids"] and
            abs(worker["raw_logit"] - expected["first_token_raw_logit"]) <=
            expected["first_token_raw_logit_tolerance"])
        record = dict(name=case["name"], prompt=case["prompt"], output_token_ids=tokens,
                      output_fingerprints=fingerprints(tokens), first_token_id=tokens[0],
                      first_token_raw_logit=worker["raw_logit"], first_token_raw_logit_tolerance=0.125,
                      finish_reason=generated.finish_reason, request_seconds=request_seconds,
                      control=case["control"], control_pass=control_pass,
                      control_oracle_sha256=ORACLE_SHAS[case["family"]] if case["control"] else None,
                      worker=worker, armed=armed[0], prefix_caching=False,
                      native_tensor_inputs=False, windows_acceptance=False)
        if matrix is not None:
            from capture_gb10_runtime_boundaries import full_cache_row_is_qualified, qualify_transaction

            frozen = matrix[case["name"]]["expected"]
            record["full_matrix_case_pass"] = (tokens == frozen["output_token_ids"] and
                abs(worker["raw_logit"] - frozen["first_token_raw_logit"]) <= 0.125)
            history = case["prompt_token_ids"] + tokens
            qualified_positions = set()
            for transaction in worker["runtime_boundaries"]["transactions"]:
                transaction["qualified_rows"] = qualify_transaction(transaction, history)
                qualified_positions.update(row["position"] for row in transaction["qualified_rows"]
                                           if row["matches_generated_history"])
            if qualified_positions != set(worker["runtime_boundaries"]["selected_positions"]):
                raise ValueError("selected boundaries lack matching generated histories")
            if not full_cache_row_is_qualified(
                    worker['runtime_boundaries']['full_attention_cache'],
                    worker['runtime_boundaries']['transactions']):
                raise ValueError('full-attention cache lacks a matching generated history')
        write_json(args.output_dir / (case["name"] + ".json"), record)
        results.append(record)
        print(json.dumps(dict(case=case["name"], first_token=tokens[0], raw_logit=worker["raw_logit"],
                              outputs=len(tokens), control_pass=control_pass, request_seconds=request_seconds)),
              flush=True)
        if not control_pass:
            raise ValueError("same-engine immutable control failed")
        if matrix is not None and not record["full_matrix_case_pass"]:
            raise ValueError("observed engine differs from the frozen complete token matrix")
        if len(results) == 2:
            write_json(args.output_dir / "controls-qualified.json", dict(
                controls_qualified=True, oracle_sha256=ORACLE_SHAS,
                cases=[dict(name=item["name"], sha256=file_sha(args.output_dir / (item["name"] + ".json")))
                       for item in results], model_evidence=evidence, torch_version=torch.__version__,
                windows_acceptance=False))
    return dict(completed=True, controls_qualified=True, cases=results, load_seconds=load_seconds,
                model_evidence=evidence, torch_version=torch.__version__,
                raw_logit_method="observer of original GPU model.compute_logits output; returned unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-q7169", type=Path, required=True)
    parser.add_argument("--oracle-q8192", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=590)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--runtime-boundaries", action="store_true",
                        help="observe pinned target rows and require the complete frozen matrix")
    parser.add_argument("--expected-host")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cases, oracles = fixtures({7169: args.oracle_q7169, 8192: args.oracle_q8192})
    if (args.output_dir.exists() or not 1 <= args.timeout_seconds <= 600 or
            len(args.source_commit) != 40 or any(x not in "0123456789abcdef" for x in args.source_commit)):
        raise ValueError("existing output, invalid source or invalid deadline")
    if args.worker and not args.execute:
        raise ValueError("worker requires execution")
    if args.execute:
        if sys.platform != "linux" or socket.gethostname() != args.expected_host:
            raise ValueError("explicit matching GB10 host required before GPU import")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            return supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                              "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="gb10_bf16_token_matrix_capture", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)),
                  model=str(args.model_root), oracle_sha256=ORACLE_SHAS,
                  fixtures=[{k: v for k, v in case.items() if k != "prompt_token_ids"} for case in cases],
                  completed=False, controls_qualified=False, windows_acceptance=False,
                  prefix_caching=False, native_tensor_inputs=False,
                  runtime_boundaries=args.runtime_boundaries)
    write_json(args.output_dir / "preflight.json", record)
    if args.execute:
        try:
            record.update(execute(args, cases, oracles))
        except Exception as error:
            write_json(args.output_dir / "failure.json", dict(error=str(error)))
            raise
    write_json(args.output_dir / "capture.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in ("cases", "fixtures")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
