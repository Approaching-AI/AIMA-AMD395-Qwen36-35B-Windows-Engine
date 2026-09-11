"""Replay the original fused decode operator from qualified real-model states.

The original call is compared to the saved first row and state. A separately
instrumented copy must reproduce both before its internal values are used.
This is an operator diagnostic, never a complete inference acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import socket
import sys

from capture_sm121_exp2_table import arm_parent_death, file_sha, supervise

SOURCE_SHA = "2df5092c250c2046e070af9e219046e9bc97e2ebf8ca82eb0e7c4201d4b771e7"


def instrument(source):
    edits = [
        ("    A_log,\n    a,\n", "    qrt_observed,\n    A_log,\n    a,\n"),
        ("        b_q = b_q * scale\n", """        b_q = b_q * scale
        diagnostic = qrt_observed + i_hv * (2 * K + 2 * V + 3)
        if i_v == 0:
            tl.store(diagnostic + o_k, b_q, mask=mask_k)
            tl.store(diagnostic + K + o_k, b_k, mask=mask_k)
            tl.store(diagnostic + 2 * K + 2 * V, b_g)
            tl.store(diagnostic + 2 * K + 2 * V + 1, tl.exp(b_g))
            tl.store(diagnostic + 2 * K + 2 * V + 2, b_beta)
"""),
        ("        b_v -= tl.sum(b_h * b_k[None, :], 1)\n", """        projected = tl.sum(b_h * b_k[None, :], 1)
        b_v -= projected
        tl.store(diagnostic + 2 * K + o_v, projected, mask=mask_v)
"""),
        ("        b_v *= b_beta\n", """        b_v *= b_beta
        tl.store(diagnostic + 2 * K + V + o_v, b_v, mask=mask_v)
"""),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise ValueError("pinned decode instrumentation point changed")
        source = source.replace(old, new)
    return source


def execute(args, manifest):
    import torch
    import triton
    from vllm.model_executor.layers.fla.ops import fused_sigmoid_gating as original

    source = Path(inspect.getsourcefile(original))
    if file_sha(source) != SOURCE_SHA:
        raise ValueError("original fused decode source changed")
    output = args.output_dir
    output.mkdir(exist_ok=False)
    instrumented_source = output / "observed_decode.py"
    instrumented_source.write_text(instrument(source.read_text()))
    spec = importlib.util.spec_from_file_location("qrt_observed_decode", instrumented_source)
    observed = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = observed
    spec.loader.exec_module(observed)

    def payload(tensor):
        return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

    def save(name, tensor):
        raw = payload(tensor)
        path = output / name
        path.write_bytes(raw)
        return dict(file=name, shape=list(tensor.shape), dtype=str(tensor.dtype),
                    bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

    class KernelObserver:
        def __init__(self, kernel, name, diagnostic=None):
            self.kernel, self.name, self.diagnostic = kernel, name, diagnostic

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if self.diagnostic is not None:
                    kwargs["qrt_observed"] = self.diagnostic
                compiled = self.kernel[grid](*args, **kwargs)
                for kind in ("ptx", "ttir", "ttgir"):
                    (output / (self.name + "." + kind)).write_text(compiled.asm[kind])
                return compiled
            return launch

    original_kernel = original.fused_sigmoid_gating_delta_rule_update_kernel
    observed_kernel = observed.fused_sigmoid_gating_delta_rule_update_kernel
    parameters = {}
    for name in ("a_log", "dt_bias"):
        item = manifest["parameters"][name]
        path = args.parameter_dir / item["file"]
        if file_sha(path) != item["sha256"] or path.stat().st_size != 64:
            raise ValueError("model parameter changed")
        parameters[name] = torch.frombuffer(bytearray(path.read_bytes()),
                                           dtype=torch.bfloat16).clone().float().cuda()
    records = []
    beta_controls = []
    for case in manifest["cases"]:
        path = args.reference_dir / (case["name"] + ".json")
        if file_sha(path) != case["sha256"]:
            raise ValueError("qualified reference case changed")
        capture = json.loads(path.read_text())
        if not capture["full_matrix_case_pass"]:
            raise ValueError("reference case did not pass its complete frozen token gate")
        boundaries = capture["worker"]["runtime_boundaries"]
        transaction = next(x for x in boundaries["transactions"] if x["ordinal"] == 1)
        if not transaction["qualified_rows"][0]["matches_generated_history"]:
            raise ValueError("decode input row does not match original generated history")

        def tensor(label):
            item = next(v for v in boundaries["files"].values()
                        if v["transaction"] == 1 and v["label"] == "linear-00-" + label)
            path = args.reference_dir / case["name"] / item["file"]
            if file_sha(path) != item["sha256"] or item["bytes"] > 8 << 20:
                raise ValueError("reference tensor changed or exceeds diagnostic bound")
            return torch.frombuffer(bytearray(path.read_bytes()),
                dtype={"bf16": torch.bfloat16, "f32": torch.float32}[item["dtype"]]
            ).clone().reshape(item["shape"])

        arguments = dict(A_log=parameters["a_log"], dt_bias=parameters["dt_bias"],
            a=tensor("a-projection")[:1].cuda(), b=tensor("b-projection")[:1].cuda(),
            q=tensor("q-decode-input")[:1].reshape(1, 1, 16, 128).cuda(),
            k=tensor("k-decode-input")[:1].reshape(1, 1, 16, 128).cuda(),
            v=tensor("v-decode-input")[:1].reshape(1, 1, 32, 128).cuda(),
            inplace_final_state=True, cu_seqlens=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            ssm_state_indices=torch.tensor([[0]], dtype=torch.int32, device="cuda"),
            num_accepted_tokens=torch.tensor([1], dtype=torch.int32, device="cuda"),
            use_qk_l2norm_in_kernel=True)
        initial = tensor("decode-state-before").cuda()
        expected_core = payload(tensor("core")[:1])
        expected_state = payload(tensor("decode-state-after")[:1])
        original.fused_sigmoid_gating_delta_rule_update_kernel = KernelObserver(
            original_kernel, case["name"] + "-original")
        core, state = original.fused_sigmoid_gating_delta_rule_update(
            initial_state=initial.clone(), **arguments)
        original_match = payload(core) == expected_core and payload(state) == expected_state
        if not original_match:
            raise ValueError("original one-row replay differs from the real target forward")
        diagnostic = torch.full((32, 515), float("nan"), dtype=torch.float32, device="cuda")
        observed.fused_sigmoid_gating_delta_rule_update_kernel = KernelObserver(
            observed_kernel, case["name"] + "-observed", diagnostic)
        replay_core, replay_state = observed.fused_sigmoid_gating_delta_rule_update(
            initial_state=initial.clone(), **arguments)
        if payload(replay_core) != expected_core or payload(replay_state) != expected_state:
            raise ValueError("instrumentation changed original decode arithmetic")
        if not bool(torch.isfinite(diagnostic).all().item()):
            raise ValueError("incomplete internal operator observation")
        beta_controls.append((arguments["b"].detach().cpu().view(torch.uint16).reshape(-1).to(torch.int64),
                              diagnostic[:, 514].detach().cpu().clone()))
        records.append(dict(name=case["name"], reference_case_sha256=case["sha256"],
            original_replay_exact=True, instrumented_replay_exact=True,
            core=save(case["name"] + "-core-bf16.bin", replay_core),
            state=save(case["name"] + "-state-f32.bin", replay_state),
            diagnostic=save(case["name"] + "-diagnostic-f32.bin", diagnostic)))

    original.fused_sigmoid_gating_delta_rule_update_kernel = original_kernel
    # The fused kernel consumes BF16 B but keeps sigmoid(B) in FP32. Enumerate
    # that model-independent 16-bit domain and bind real internal beta values
    # to it before exporting the lookup artifact.
    beta_source = output / "beta_domain.py"
    beta_source.write_text('''import triton
import triton.language as tl
@triton.jit
def beta_domain(B, O, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(B + i).to(tl.float32)
    tl.store(O + i, tl.sigmoid(b))
''')
    beta_spec = importlib.util.spec_from_file_location("qrt_beta_domain", beta_source)
    beta_module = importlib.util.module_from_spec(beta_spec)
    sys.modules[beta_spec.name] = beta_module
    beta_spec.loader.exec_module(beta_module)
    encodings = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    beta_table = torch.empty(65536, dtype=torch.float32, device="cuda")
    compiled = beta_module.beta_domain[(256,)](encodings, beta_table, BLOCK=256, num_warps=4)
    (output / "beta-domain.ptx").write_text(compiled.asm["ptx"])
    beta_cpu = beta_table.cpu()
    for indices, values in beta_controls:
        if not torch.equal(beta_cpu[indices].view(torch.int32), values.view(torch.int32)):
            raise ValueError("FP32 sigmoid domain differs from the actual fused operator")
    beta_file = save("sigmoid-beta-f32.bin", beta_table)
    record = dict(kind="original_fused_decode_same_input_replay", host=socket.gethostname(),
        command=sys.argv, command_source_sha256=file_sha(Path(__file__)),
        source_commit=args.source_commit, original_source_sha256=SOURCE_SHA,
        manifest_sha256=args.manifest_sha256, cases=records,
        diagnostic_layout="per value head: Q128, K128, projected128, residual128, g, decay, beta",
        torch_version=torch.__version__, triton_version=triton.__version__,
        model_loaded=False, reference_service_forward=False, inference_acceptance=False,
        beta_domain=dict(input_encodings=65536, original_internal_controls=32 * len(beta_controls),
                         controls_exact=True, artifact=beta_file),
        files=[dict(file=p.name, bytes=p.stat().st_size, sha256=file_sha(p))
               for p in sorted(output.iterdir()) if p.is_file()])
    (output / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-dir", "parameter-dir", "output-dir", "manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("manifest-sha256", "source-commit", "expected-host"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not args.execute or sys.platform != "linux" or socket.gethostname() != args.expected_host or
            not 1 <= args.timeout_seconds <= 90 or len(args.source_commit) != 40 or
            args.output_dir.exists() or file_sha(args.manifest) != args.manifest_sha256):
        raise ValueError("invalid explicit replay request or provenance")
    if not args.worker:
        raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
            "--worker", "--supervisor-pid", str(__import__("os").getpid())], args.timeout_seconds))
    arm_parent_death(args.supervisor_pid)
    execute(args, json.loads(args.manifest.read_text()))


if __name__ == "__main__":
    main()
