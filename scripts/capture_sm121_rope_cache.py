#!/usr/bin/env python3
"""Capture the original SM121 BF16 rotary cache from model configuration.

The payload covers every native context position. It contains no prompt IDs,
weights, hidden states, logits or generated tokens. The original MRoPE class
builds four times the configured context; a separate base-cache construction
checks every emitted value against that full constructor's prefix.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

DEVICE_LIMIT = 1 << 30
ROWS = 262144
COLUMNS = 64
PARAMETERS = dict(mrope_interleaved=True, mrope_section=[11, 11, 10],
                  partial_rotary_factor=0.25, rope_theta=10000000, rope_type="default")


def configuration(path, expected_sha):
    if file_sha(path) != expected_sha:
        raise ValueError("model configuration fingerprint changed")
    value = json.loads(path.read_text())
    value = value.get("text_config", value)
    if (value.get("head_dim") != 256 or value.get("max_position_embeddings") != ROWS or
            value.get("rope_parameters") != PARAMETERS):
        raise ValueError("model rotary geometry does not match the native cache layout")
    return value


def execute(args, config):
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    import vllm.model_executor.layers.rotary_embedding as rotary
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    root = Path(rotary.__file__).parent
    expected = json.loads(args.source_manifest.read_text())
    if not {"__init__.py", "base.py", "mrope.py"} <= expected.keys():
        raise ValueError("missing original rotary source binding")
    for name, digest in expected.items():
        if Path(name).name != name or not name.endswith(".py") or file_sha(root / name) != digest:
            raise ValueError("installed rotary source differs from the original reference")
    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("rotary cache requires one SM121 CUDA device")
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("device reserve unavailable")
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    # CustomOp construction reads the active compilation configuration even
    # though cache generation itself is eager and does not load a model.
    runtime_config = VllmConfig()
    with set_current_vllm_config(runtime_config), torch.device("cuda"):
        original = rotary.get_rope(head_size=config["head_dim"], max_position=ROWS,
                                   rope_parameters=config["rope_parameters"], dtype=torch.bfloat16)
    full = original.cos_sin_cache
    if full.dtype != torch.bfloat16 or tuple(full.shape) != (4 * ROWS, COLUMNS):
        raise ValueError("original MRoPE cache layout changed")
    torch.cuda.synchronize()
    full_peak = torch.cuda.max_memory_allocated()
    if full_peak > DEVICE_LIMIT:
        raise ValueError("original cache construction exceeded the device ceiling")
    # This shape changes allocation only; compare the complete payload before
    # writing it, rather than assuming shape-independent pointwise arithmetic.
    with set_current_vllm_config(runtime_config), torch.device("cuda"):
        control = RotaryEmbedding(256, COLUMNS, ROWS, PARAMETERS["rope_theta"],
                                  True, torch.bfloat16)
    expected_prefix = full[:ROWS]
    differences = int(torch.count_nonzero(expected_prefix.view(torch.int16) !=
                                          control.cos_sin_cache.view(torch.int16)).item())
    if differences:
        raise ValueError("cache prefix depends on constructor extent")
    # Verify the inverse frequencies separately: these FP32 constants explain
    # phase drift and remain useful when replacing the cache with native code.
    with torch.device("cuda"):
        inverse = original._compute_inv_freq(PARAMETERS["rope_theta"])
    if tuple(inverse.shape) != (COLUMNS // 2,) or not bool(torch.isfinite(full).all().item()):
        raise ValueError("nonfinite or invalid rotary table")
    if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
        raise ValueError("rotary controls exceeded the device ceiling")
    files = []
    for name, tensor in (("sm121-rope-bf16.bin", expected_prefix),
                         ("sm121-rope-inverse-f32.bin", inverse)):
        path = args.output_dir / name
        with path.open("xb") as stream:
            stream.write(tensor.contiguous().cpu().numpy().tobytes() if tensor.dtype != torch.bfloat16
                         else tensor.contiguous().view(torch.uint16).cpu().numpy().tobytes())
        files.append(dict(file=name, bytes=path.stat().st_size, sha256=file_sha(path)))
    return dict(completed=True, rows=ROWS, columns=COLUMNS, dtype="bf16",
                layout="position_cos32_sin32", original_cache_rows=4 * ROWS,
                extent_control_elements=ROWS * COLUMNS, extent_control_bit_mismatches=differences,
                original_constructor_peak_device_bytes=full_peak,
                peak_device_bytes=torch.cuda.max_memory_allocated(),
                torch_version=torch.__version__, rotary_class=type(original).__name__,
                reference_sources=expected, files=files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-host")
    parser.add_argument("--timeout-seconds", type=int, default=45)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--supervisor-pid", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (not 1 <= args.timeout_seconds <= 60 or len(args.source_commit) != 40 or
            any(c not in "0123456789abcdef" for c in args.source_commit) or args.output_dir.exists()):
        raise ValueError("invalid source, deadline or existing output")
    config = configuration(args.model_config, args.expected_config_sha256)
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
    record = dict(kind="sm121_model_rotary_cache", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)),
                  model_config_sha256=file_sha(args.model_config),
                  source_manifest_sha256=file_sha(args.source_manifest),
                  model_weights_loaded=False, prompt_inputs=False, inference_acceptance=False,
                  completed=False, maximum_device_bytes=DEVICE_LIMIT, rope_parameters=PARAMETERS)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args, config))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
