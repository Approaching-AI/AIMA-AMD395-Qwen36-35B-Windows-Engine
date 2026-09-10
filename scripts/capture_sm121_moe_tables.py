#!/usr/bin/env python3
"""Capture model-independent SM121 primitives consumed by the native MoE.

The fraction table covers every 23-bit fraction used by the CUDA expf range
reducer. The SiLU table covers the BF16 input domain. Neither contains model
weights, token IDs or reference inference outputs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import struct
import sys
import types

from capture_fla_state_prefix import arm_parent_death, supervise
from capture_sm121_exp2_table import file_sha

ENTRIES = 1 << 23
CHUNK = 1 << 20
DEVICE_LIMIT = 64 << 20
KERNEL = '''def fraction(Output, Begin: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    argument = (index + Begin).to(tl.float32) * (1.0 / 8388608.0)
    result = tl.inline_asm_elementwise("ex2.approx.ftz.f32 $0, $1;",
        constraints="=f,f", args=[argument], dtype=tl.float32, is_pure=True, pack=1)
    tl.store(Output + index, result)
'''


def execute(args):
    import numpy as np
    import torch
    import triton
    import triton.language as tl
    import vllm._custom_ops  # Registers the original CUDA SiLU-and-multiply op.

    if (torch.version.hip is not None or torch.cuda.device_count() != 1 or
            torch.cuda.get_device_capability(0) != (12, 1)):
        raise ValueError("MoE tables require one SM121 CUDA device")
    if torch.cuda.mem_get_info()[0] < DEVICE_LIMIT + (1 << 30):
        raise ValueError("device reserve unavailable")
    torch.set_num_threads(2)
    torch.cuda.reset_peak_memory_stats()
    path = args.output_dir / "fraction-kernel.py"
    path.write_text(KERNEL)
    module = types.ModuleType("_qrt_sm121_moe_fraction")
    module.__file__ = str(path)
    module.tl = tl
    sys.modules[module.__name__] = module
    exec(compile(KERNEL, str(path), "exec", dont_inherit=True), module.__dict__)
    kernel = triton.jit(module.fraction)
    output = torch.empty(CHUNK, dtype=torch.float32, device="cuda")
    maximum_ms = 0.0
    launches = 0
    ptx = []
    fraction_path = args.output_dir / "cuda-router-ex2-fraction-f32.bin"
    with fraction_path.open("xb") as stream:
        for begin in range(0, ENTRIES, CHUNK):
            compiled = kernel.warmup(output, begin, BLOCK=1024, grid=(CHUNK // 1024,))
            compiled._init_handles()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            kernel[(CHUNK // 1024,)](output, begin, BLOCK=1024)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if elapsed > 100.0:
                raise ValueError("fraction dispatch exceeded 100 ms")
            maximum_ms = max(maximum_ms, elapsed)
            launches += 1
            array = output.cpu().numpy()
            if not np.all(np.isfinite(array)) or np.any(array < 1.0) or np.any(array > 2.0):
                raise ValueError("invalid fraction output")
            stream.write(array.tobytes())
            asm = args.output_dir / ("fraction-" + str(begin) + ".ptx")
            asm.write_text(compiled.asm["ptx"])
            ptx.append(dict(file=asm.name, sha256=file_sha(asm)))

    raw = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
    inputs = torch.stack((raw, torch.ones_like(raw)), dim=1)
    silu = torch.empty_like(raw)
    # Initialize the installed CUDA module before measuring the full capture.
    torch.ops._C.silu_and_mul(silu[:1], inputs[:1])
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    torch.ops._C.silu_and_mul(silu, inputs)
    end.record()
    end.synchronize()
    silu_ms = start.elapsed_time(end)
    if silu_ms > 100.0:
        raise ValueError("SiLU capture exceeded 100 ms")
    # Check the BF16 materialization boundary with non-unit up projections.
    up = ((torch.arange(65536, device="cuda", dtype=torch.float32) % 2049 - 1024) / 512).to(torch.bfloat16)
    inputs[:, 1] = up
    actual = torch.empty_like(raw)
    torch.ops._C.silu_and_mul(actual, inputs)
    expected = silu * up
    finite = torch.isfinite(raw)
    differences = int(torch.count_nonzero(actual.view(torch.int16)[finite] !=
                                          expected.view(torch.int16)[finite]).item())
    if differences:
        raise ValueError("original CUDA SiLU does not obey its BF16 table boundary")
    silu_path = args.output_dir / "cuda-vllm-silu-bf16-domain.bin"
    with silu_path.open("xb") as stream:
        stream.write(struct.pack("<8s4I", b"QRTSBF1\0", 1, 65536, 2, 0))
        stream.write(silu.view(torch.uint16).cpu().numpy().tobytes())
    if torch.cuda.max_memory_allocated() > DEVICE_LIMIT:
        raise ValueError("MoE table device ceiling exceeded")
    return dict(completed=True, fraction_entries=ENTRIES, fraction_launches=launches,
                maximum_fraction_dispatch_ms=maximum_ms, silu_dispatch_ms=silu_ms,
                silu_entries=65536, finite_silu_control_inputs=int(finite.sum().item()),
                silu_product_control_bit_mismatches=differences,
                peak_device_bytes=torch.cuda.max_memory_allocated(), ptx=ptx,
                files=[dict(file=p.name, bytes=p.stat().st_size, sha256=file_sha(p))
                       for p in (fraction_path, silu_path)],
                torch_version=torch.__version__, triton_version=triton.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
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
    record = dict(kind="sm121_moe_primitive_tables", host=socket.gethostname(), command=sys.argv,
                  source_commit=args.source_commit, source_sha256=file_sha(Path(__file__)),
                  model_loaded=False, model_or_prompt_inputs=False, inference_acceptance=False,
                  completed=False, maximum_device_bytes=DEVICE_LIMIT, per_dispatch_admission_ms=100)
    (args.output_dir / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if args.execute:
        try:
            record.update(execute(args))
        except Exception as error:
            (args.output_dir / "failure.json").write_text(json.dumps(dict(error=str(error))) + "\n")
            raise
    (args.output_dir / "capture.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
