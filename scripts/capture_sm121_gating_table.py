#!/usr/bin/env python3
"""Characterize the original SM121 GDN gate over every BF16 input encoding.

A real-token terminal control must match before enumeration. The table itself
uses only all BF16 encodings and the declared model parameters; no expected
model output or prompt is used to generate a table entry.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import types

from capture_fla_state_prefix import arm_parent_death, compare, finite, supervise
from capture_sm121_exp2_table import file_sha

FUNCTION = "fused_gdn_gating_kernel"
DEVICE_LIMIT = 32 << 20
CHUNK = 4096


def validate(directory, expected_sha):
    path = directory / "manifest.json"
    if path.stat().st_size > 1 << 20 or file_sha(path) != expected_sha:
        raise ValueError("gate manifest fingerprint or size mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("heads") != 32 or manifest.get("tokens") != 1 or not isinstance(manifest.get("layer"), int) or not 0 <= manifest["layer"] < 40:
        raise ValueError("gate control requires one terminal token and 32 heads")
    if set(manifest.get("files", {})) != {"a_log", "dt_bias", "a", "b", "g", "beta"}:
        raise ValueError("gate input/control file set mismatch")
    payloads = {}
    for name, meta in manifest["files"].items():
        dtype, width = ("f32", 4) if name == "g" else ("bf16", 2)
        path = directory / (name + ".bin")
        if meta.get("file") != path.name or meta.get("dtype") != dtype or meta.get("elements") != 32 or path.stat().st_size != 32 * width or file_sha(path) != meta["sha256"]:
            raise ValueError("gate file layout or fingerprint mismatch: " + name)
        payloads[name] = path.read_bytes()
        if not finite(payloads[name], dtype):
            raise ValueError("gate control must be finite: " + name)
    return manifest, payloads


def extract(source):
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == FUNCTION)
    function.decorator_list = []
    return ast.unparse(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))) + "\n"


def validate_model_parameters(directory, expected_sha, primary_layer, primary):
    """Bind every table to actual model tensors; optional controls only compare."""
    path = directory / "parameters.json"
    if path.stat().st_size > 1 << 20 or file_sha(path) != expected_sha:
        raise ValueError("model gate parameter manifest fingerprint or size mismatch")
    record = json.loads(path.read_text())
    layers = record.get("layers", [])
    expected_layers = [i for i in range(40) if (i + 1) % 4]
    if [x.get("layer") for x in layers] != expected_layers:
        raise ValueError("model gate parameters require all 30 linear-attention layers")
    result = []
    for layer in layers:
        payloads = {}
        for key in ("a_log", "dt_bias"):
            meta = layer[key]
            name = f"layer{layer['layer']}-{key}.bin"
            tensor = directory / name
            if (meta.get("file") != name or meta.get("dtype") != "BF16" or
                    meta.get("shape") != [32] or meta.get("bytes") != 64 or
                    tensor.stat().st_size != 64 or file_sha(tensor) != meta.get("sha256")):
                raise ValueError("model gate parameter fingerprint or span mismatch")
            payloads[key] = tensor.read_bytes()
            if not finite(payloads[key], "bf16"):
                raise ValueError("nonfinite model gate parameter")
        control = None
        if "control_manifest_sha256" in layer:
            control_manifest, control = validate(
                directory / f"layer{layer['layer']}-control",
                layer["control_manifest_sha256"],
            )
            if control_manifest["layer"] != layer["layer"] or any(
                control[k] != payloads[k] for k in payloads
            ):
                raise ValueError("model gate control parameter binding mismatch")
        if layer["layer"] == primary_layer:
            if any(primary[k] != payloads[k] for k in payloads):
                raise ValueError("primary gate control is for different model parameters")
            control = primary
        result.append((layer["layer"], payloads, control))
    if primary_layer not in expected_layers:
        raise ValueError("primary gate control must select a linear-attention layer")
    return record, result


def gpu_capture(args, manifest, payloads):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    if torch.version.hip is not None or torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0) != (12, 1):
        raise ValueError("gating capture requires one SM121 CUDA device")
    torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("gating device reserve unavailable")
    torch.cuda.reset_peak_memory_stats()
    path = args.output_dir / "reference-gating-extracted.py"
    path.write_text(extract(args.source))
    module = types.ModuleType("_qrt_reference_gating")
    module.__file__ = str(path)
    module.__dict__["tl"] = tl
    sys.modules[module.__name__] = module
    exec(compile(path.read_text(), str(path), "exec", dont_inherit=True), module.__dict__)
    kernel = triton.jit(module.__dict__[FUNCTION])
    inputs = {name: torch.frombuffer(bytearray(payloads[name]), dtype=torch.bfloat16).clone().cuda()
              for name in ("a_log", "dt_bias", "a", "b")}
    g = torch.empty(32, dtype=torch.float32, device="cuda")
    beta = torch.empty(32, dtype=torch.bfloat16, device="cuda")
    parameters = [inputs["a_log"], inputs["dt_bias"]]
    options = dict(num_warps=1)
    arguments = [g, beta, parameters[0], inputs["a"], inputs["b"], parameters[1], 1, 32, 1.0, 20.0, 8]
    prepared = kernel.warmup(*arguments, grid=(1, 1, 4), **options)
    # Triton's lazy driver/module initialization must complete before the CUDA
    # timing events. Merely compiling with warmup does not load the executable.
    if not callable(prepared.run):
        raise ValueError("gating launcher unavailable")
    ptx_path = args.output_dir / "gating.ptx"
    ptx_path.write_text(prepared.asm["ptx"])
    ptx_sha = file_sha(ptx_path)
    maximum_ms = 0.0
    launches = 0

    def launch(arguments, rows):
        nonlocal maximum_ms, launches
        torch.cuda.synchronize()
        if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
            raise ValueError("gating allocation ceiling exceeded")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        actual = kernel[(rows, 1, 4)](*arguments, **options)
        end.record()
        end.synchronize()
        elapsed = start.elapsed_time(end)
        launches += 1
        maximum_ms = max(maximum_ms, elapsed)
        with (args.output_dir / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(rows=rows, elapsed_ms=elapsed, launch=launches)) + "\n")
        if elapsed > 100 or hashlib.sha256(actual.asm["ptx"].encode()).hexdigest() != ptx_sha:
            raise ValueError("gating dispatch limit or PTX mismatch; no further submission")

    launch(arguments, 1)
    comparisons = dict(g=compare(g.cpu().numpy().tobytes(), payloads["g"], "f32"),
                       beta=compare(beta.view(torch.uint16).cpu().numpy().tobytes(), payloads["beta"], "bf16"))
    (args.output_dir / "control.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    if not all(value["exact"] for value in comparisons.values()):
        raise ValueError("original gating source does not reproduce the real-token control")
    # These host arrays contain the entire BF16 domain, including infinities and
    # NaNs. Nonfinite input encodings are legitimate table entries, not control
    # cases; model execution still owns its ordinary input validation.
    gs = torch.full((CHUNK * 32 * 4 + 512,), 0x5A, dtype=torch.uint8, device="cuda")
    bs = torch.full((CHUNK * 32 * 2 + 512,), 0x5A, dtype=torch.uint8, device="cuda")
    go = gs[256:-256].view(torch.float32)
    bo = bs[256:-256].view(torch.bfloat16)
    layer_specs = args.model_parameters or [
        (manifest["layer"], {k: payloads[k] for k in ("a_log", "dt_bias")}, payloads)
    ]
    files, layer_controls = [], {}
    shared_beta = None
    for layer, parameter_bytes, control in layer_specs:
        parameters = [torch.frombuffer(bytearray(parameter_bytes[k]), dtype=torch.bfloat16).clone().cuda()
                      for k in ("a_log", "dt_bias")]
        if control is not None:
            control_ab = [torch.frombuffer(bytearray(control[k]), dtype=torch.bfloat16).clone().cuda()
                          for k in ("a", "b")]
            launch([g, beta, parameters[0], control_ab[0], control_ab[1], parameters[1],
                    1, 32, 1.0, 20.0, 8], 1)
            direct = dict(g=compare(g.cpu().numpy().tobytes(), control["g"], "f32"),
                          beta=compare(beta.view(torch.uint16).cpu().numpy().tobytes(), control["beta"], "bf16"))
            if not all(v["exact"] for v in direct.values()):
                raise ValueError(f"original gating control failed for layer {layer}")
            if any(t.view(torch.uint8).cpu().numpy().tobytes() != control[k]
                   for k, t in zip(("a", "b"), control_ab)):
                raise ValueError("additional gating control input changed")
            layer_controls[str(layer)] = dict(direct=direct)
        g_table = np.empty((32, 65536), dtype=np.float32)
        beta_table = np.empty(65536, dtype=np.uint16)
        for offset in range(0, 65536, CHUNK):
            values = torch.arange(offset, offset + CHUNK, dtype=torch.int32, device="cuda").to(torch.int16)
            ab = values.view(torch.bfloat16)[:, None].expand(-1, 32).contiguous()
            arguments = [go, bo, parameters[0], ab, ab, parameters[1], 1, 32, 1.0, 20.0, 8]
            launch(arguments, CHUNK)
            gr = gs.cpu().numpy().tobytes()
            br = bs.cpu().numpy().tobytes()
            if any(raw[:256] != b"\x5a" * 256 or raw[-256:] != b"\x5a" * 256 for raw in (gr, br)):
                raise ValueError("gate output redzone changed")
            g_table[:, offset:offset + CHUNK] = np.frombuffer(gr[256:-256], dtype=np.float32).reshape(CHUNK, 32).T
            beta_rows = np.frombuffer(br[256:-256], dtype=np.uint16).reshape(CHUNK, 32)
            if not np.all(beta_rows == beta_rows[:, :1]):
                raise ValueError("sigmoid depends on head; shared-table contract invalid")
            beta_table[offset:offset + CHUNK] = beta_rows[:, 0]
            expected_ab = np.arange(offset, offset + CHUNK, dtype=np.uint16)[:, None].repeat(32, axis=1)
            if not np.array_equal(ab.view(torch.uint16).cpu().numpy(), expected_ab):
                raise ValueError("enumerated input changed")
        for k, tensor in zip(("a_log", "dt_bias"), parameters):
            if tensor.view(torch.uint8).cpu().numpy().tobytes() != parameter_bytes[k]:
                raise ValueError("model gating parameters changed during enumeration")
        if control is not None:
            a_bits = np.frombuffer(control["a"], dtype=np.uint16)
            b_bits = np.frombuffer(control["b"], dtype=np.uint16)
            checked = dict(g=compare(g_table[np.arange(32), a_bits].tobytes(), control["g"], "f32"),
                           beta=compare(beta_table[b_bits].tobytes(), control["beta"], "bf16"))
            if not all(v["exact"] for v in checked.values()):
                raise ValueError(f"table lookup control failed for layer {layer}")
            layer_controls[str(layer)]["lookup"] = checked
        path = args.output_dir / f"layer{layer}-g-f32-head-major.bin"
        path.write_bytes(g_table.tobytes())
        files.append(dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path)))
        if shared_beta is None:
            shared_beta = beta_table.copy()
        elif not np.array_equal(shared_beta, beta_table):
            raise ValueError("sigmoid table depends on model layer")
    for name, tensor in inputs.items():
        if tensor.view(torch.uint8).cpu().numpy().tobytes() != payloads[name]:
            raise ValueError("gating parameters/control input changed")
    path = args.output_dir / "sigmoid-beta-bf16.bin"
    path.write_bytes(shared_beta.tobytes())
    files.append(dict(file=path.name, bytes=path.stat().st_size, sha256=file_sha(path)))
    table_control = layer_controls[str(manifest["layer"])]["lookup"]
    return dict(kernel_executed=True, device=torch.cuda.get_device_name(0),
                torch_version=torch.__version__, triton_version=triton.__version__,
                control=comparisons, table_control=table_control, ptx_sha256=ptx_sha,
                options=dict(num_warps=1, block_heads=8, beta=1.0, threshold=20.0),
                input_encodings=65536, entries=len(layer_specs) * 32 * 65536, launches=launches,
                layers=[x[0] for x in layer_specs], layer_controls=layer_controls,
                maximum_dispatch_ms=maximum_ms, peak_device_bytes=torch.cuda.max_memory_allocated(),
                files=files, original_worker_register_capture=False, reference_service_executed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--model-parameter-dir", type=Path)
    parser.add_argument("--model-parameter-sha256")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.timeout_seconds <= 120 or len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise ValueError("invalid gating deadline or source commit")
    if args.worker:
        if not args.execute:
            raise ValueError("gating worker requires explicit execution")
        arm_parent_death(args.supervisor_pid)
    if args.output_dir.exists() or args.source.stat().st_size > 1 << 20 or file_sha(args.source) != args.source_sha256:
        raise ValueError("existing output or gating source fingerprint mismatch")
    manifest, payloads = validate(args.input_dir, args.manifest_sha256)
    args.model_parameters = None
    parameter_record = None
    if bool(args.model_parameter_dir) != bool(args.model_parameter_sha256):
        raise ValueError("model parameter directory and fingerprint must be supplied together")
    if args.model_parameter_dir:
        parameter_record, args.model_parameters = validate_model_parameters(
            args.model_parameter_dir, args.model_parameter_sha256, manifest["layer"], payloads
        )
    extract(args.source)
    if args.execute:
        if sys.platform != "linux" or not args.expected_host or socket.gethostname().lower() != args.expected_host.lower():
            raise ValueError("gating execution host mismatch before GPU import")
        if not args.worker:
            raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:],
                                        "--worker", "--supervisor-pid", str(os.getpid())], args.timeout_seconds))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    record = dict(kind="sm121_model_parameter_gating_table", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, command_source_sha256=file_sha(Path(__file__)),
                  source_sha256=args.source_sha256, manifest_sha256=args.manifest_sha256,
                  layer=manifest["layer"], model_loaded=False, inference_acceptance=False,
                  kernel_executed=False, timeout_seconds=args.timeout_seconds)
    if parameter_record is not None:
        record["model_parameters"] = parameter_record
        record["model_parameter_manifest_sha256"] = args.model_parameter_sha256
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(gpu_capture(args, manifest, payloads))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error), inference_acceptance=False)) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
