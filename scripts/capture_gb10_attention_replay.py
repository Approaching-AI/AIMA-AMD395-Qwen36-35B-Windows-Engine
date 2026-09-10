#!/usr/bin/env python3
"""Replay original GB10 attention with qualified Q/K/V and expose its FP32 output."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import socket
import sys

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

SOURCE_SHA = "5a8af26832bd0b23604fefa4a0876e12b39101b85431c368649fbb7f779e8921"
DEVICE_LIMIT = 1 << 30


def division_kernel(numerator, denominator, output, N: tl.constexpr,
                    MODE: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = tl.load(numerator + index, index < N, other=0.0)
    b = tl.load(denominator + index // 256, index < N, other=1.0)
    if MODE == 0:
        result = tl.inline_asm_elementwise("div.full.f32 $0, $1, $2;", constraints="=f,f,f",
                                          args=[a, b], dtype=tl.float32, is_pure=True, pack=1)
    elif MODE == 1:
        result = tl.inline_asm_elementwise("div.rn.f32 $0, $1, $2;", constraints="=f,f,f",
                                          args=[a, b], dtype=tl.float32, is_pure=True, pack=1)
    elif MODE == 2:
        inverse = tl.inline_asm_elementwise("rcp.approx.ftz.f32 $0, $1;", constraints="=f,f",
                                           args=[b], dtype=tl.float32, is_pure=True, pack=1)
        result = a * inverse
    else:
        inverse = tl.inline_asm_elementwise("rcp.rn.f32 $0, $1;", constraints="=f,f",
                                           args=[b], dtype=tl.float32, is_pure=True, pack=1)
        result = a * inverse
    tl.store(output + index, result, index < N)


def division_controls(args, expected):
    global tl
    import torch
    import triton
    import triton.language as tl

    manifest = json.loads(args.native_division_manifest.read_text())
    if manifest["query_start"] != 0 or manifest["query_count"] != 7169:
        raise ValueError("division controls require the full qualified geometry")
    loaded = []
    for key, count in (("accumulator", 7169 * 4096), ("denominator", 7169 * 16)):
        item = manifest[key]
        path = args.native_division_manifest.parent / item["file"]
        if Path(item["file"]).name != item["file"] or path.stat().st_size != count * 4 or file_sha(path) != item["sha256"]:
            raise ValueError("native division input fingerprint changed")
        host = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32)
        if not bool(torch.isfinite(host).all().item()):
            raise ValueError("nonfinite native division input")
        if key == "denominator" and (float(host.min()) < 1.0 or float(host.max()) > 8192.0):
            raise ValueError("native denominator lies outside the attention bound")
        loaded.append(host.cuda())
    output = torch.empty_like(loaded[0])
    reference = expected.reshape(-1).contiguous()
    kernel = triton.jit(division_kernel)
    rows = []
    artifacts = []
    for mode, name in enumerate(("div-full", "div-rn", "rcp-approx-multiply", "rcp-rn-multiply")):
        compiled = kernel[(triton.cdiv(output.numel(), 256),)](*loaded, output, output.numel(), mode, 256)
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("division control exceeds the device ceiling")
        actual = output.cpu()
        differences = actual.view(torch.int32) != reference.view(torch.int32)
        indices = torch.nonzero(differences).reshape(-1)[:8]
        row = dict(mode=name, elements=output.numel(), f32_mismatches=int(differences.sum()),
                   bf16_mismatches=int((actual.to(torch.bfloat16).view(torch.uint16) != reference.to(torch.bfloat16).view(torch.uint16)).sum()),
                   first_indices=indices.tolist())
        rows.append(row)
        print(json.dumps(row), flush=True)
        path = args.output_dir / f"division-{name}.cubin"
        with path.open("xb") as stream:
            stream.write(compiled.asm["cubin"])
        artifacts.append(dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path)))
    return dict(native_manifest_sha256=file_sha(args.native_division_manifest),
                native_manifest=manifest, rows=rows, artifacts=artifacts,
                reference_is_compute_input=False)


def execute(args, reference):
    import torch
    import triton

    module = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    if file_sha(Path(module.__file__)) != SOURCE_SHA:
        raise ValueError("original attention implementation changed")
    if torch.cuda.get_device_capability() != (12, 1):
        raise ValueError("SM121 is required")
    files = reference["worker"]["files"]

    def load(name):
        item = files[name]
        path = args.tensor_dir / item["file"]
        if Path(item["file"]).name != item["file"] or path.stat().st_size != item["bytes"] or file_sha(path) != item["sha256"]:
            raise ValueError("qualified input fingerprint changed")
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.uint16).view(torch.bfloat16).reshape(item["shape"])

    q = load("full-attention-q-rope").reshape(7169, 16, 256).cuda()
    k = load("full-attention-k-rope").reshape(7169, 2, 256).cuda()
    v = load("full-attention-v").reshape(7169, 2, 256).cuda()
    expected = load("full-attention-context").reshape(7169, 16, 256)
    blocks = (7169 + 15) // 16
    kc = torch.zeros((blocks, 16, 2, 256), dtype=torch.bfloat16, device="cuda")
    vc = torch.zeros_like(kc)
    kc.view(-1, 2, 256)[:7169].copy_(k)
    vc.view(-1, 2, 256)[:7169].copy_(v)
    block_table = torch.arange(blocks, dtype=torch.int32, device="cuda").reshape(1, -1)
    cu = torch.tensor([0, 7169], dtype=torch.int32, device="cuda")
    used = torch.tensor([7169], dtype=torch.int32, device="cuda")
    scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    compiled = []
    original = module.kernel_unified_attention_2d

    class ObserveCompilation:
        def __getitem__(self, grid):
            def launch(*a, **kw):
                kernel = original[grid](*a, **kw)
                compiled.append(kernel)
                return kernel
            return launch

    controls = []
    artifacts = []
    module.kernel_unified_attention_2d = ObserveCompilation()
    try:
        for dtype, name in ((torch.bfloat16, "bf16"), (torch.float32, "f32")):
            out = torch.empty((7169, 16, 256), dtype=dtype, device="cuda")
            module.unified_attention(
                q=q, k=kc, v=vc, out=out, cu_seqlens_q=cu, max_seqlen_q=7169,
                seqused_k=used, max_seqlen_k=7169, softmax_scale=0.0625,
                causal=True, window_size=(-1, -1), block_table=block_table,
                softcap=0.0, q_descale=None, k_descale=scale, v_descale=scale)
            torch.cuda.synchronize()
            if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
                raise ValueError("device allocation ceiling exceeded")
            host = out.cpu()
            if not bool(torch.isfinite(host).all().item()):
                raise ValueError("nonfinite attention output")
            differences = int((host.to(torch.bfloat16).view(torch.uint16) != expected.view(torch.uint16)).sum().item())
            control = dict(output_dtype=name, elements=expected.numel(), bf16_mismatches=differences)
            controls.append(control)
            print(json.dumps(control), flush=True)
            if differences:
                raise ValueError("replay does not reproduce qualified original attention")
            if name == "f32":
                path = args.output_dir / "attention-f32.bin"
                with path.open("xb") as stream:
                    stream.write(host.numpy().tobytes())
                artifacts.append(dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path)))
            for kind in ("ptx", "ttir"):
                text = compiled[-1].asm[kind]
                if len(text.encode()) > 16 << 20:
                    raise ValueError("compiler artifact ceiling exceeded")
                path = args.output_dir / f"attention-{name}.{kind}"
                with path.open("x") as stream:
                    stream.write(text)
                artifacts.append(dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path)))
    finally:
        module.kernel_unified_attention_2d = original
    division = division_controls(args, host) if args.native_division_manifest else None
    return dict(completed=True, controls=controls, artifacts=artifacts,
                division_controls=division,
                maximum_device_bytes=torch.cuda.max_memory_allocated(),
                torch_version=torch.__version__, triton_version=triton.__version__,
                original_source_path=str(module.__file__), original_source_sha256=SOURCE_SHA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--reference-capture", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--tensor-dir", type=Path, required=True)
    parser.add_argument("--native-division-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.timeout_seconds <= 90 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit) or args.output_dir.exists()):
        raise ValueError("invalid commit, deadline or existing output")
    if file_sha(args.reference_capture) != args.expected_reference_sha256:
        raise ValueError("reference capture fingerprint mismatch")
    reference = json.loads(args.reference_capture.read_text())
    if not reference["completed"] or not reference["oracle_qualified"] or reference["native_tensor_inputs"]:
        raise ValueError("reference is not an independently qualified full-model capture")
    if args.worker and not args.execute:
        raise ValueError("worker requires explicit execution")
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname() != args.expected_host:
            raise ValueError("execution host mismatch before GPU import")
        if args.worker:
            arm_parent_death(args.supervisor_pid)
        else:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="gb10_full_attention_fp32_replay", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)),
                  reference_capture_sha256=file_sha(args.reference_capture),
                  model=reference["model"], model_weights_loaded=False, inference_acceptance=False,
                  completed=False, device_limit_bytes=DEVICE_LIMIT)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args, reference))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
