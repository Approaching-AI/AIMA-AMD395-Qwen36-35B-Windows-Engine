#!/usr/bin/env python3
"""CPU-only CUDA IR audit of a supplied vendored FLA state kernel.

This does not import vLLM, select a live GPU, run autotuning, or prove the
configuration used by the reference worker. It compiles an explicit target.
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
import time
import types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--value-block", type=int, choices=(32, 64), default=32)
    args = parser.parse_args()
    for key in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(key) != "-1":
            raise ValueError(f"CPU-only compiler requires {key}=-1")
    import triton
    import triton.language as tl
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    source = args.source.resolve()
    content = source.read_text()
    tree = ast.parse(content, filename=str(source))
    kernel_name = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == kernel_name)
    function.decorator_list = []  # Do not invoke the autotuner or heuristics.
    module = types.ModuleType("_qrt_fla_reference_state_ir")
    module.__file__ = str(source)
    module.__dict__.update(tl=tl, exp=tl.exp)
    sys.modules[module.__name__] = module
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), module.__dict__)
    kernel = triton.jit(module.__dict__[kernel_name], do_not_specialize=["T"])
    signature = dict(k="*bf16", v="*bf16", w="*bf16", v_new="*bf16", g="*fp32", gk="*fp32",
                     h="*bf16", h0="*fp32", ht="*fp32", cu_seqlens="*i64", chunk_offsets="*i32", T="i32")
    constants = dict(H=32, Hg=16, K=128, V=128, BT=64, BV=args.value_block, USE_G=True, USE_GK=False,
                     USE_INITIAL_STATE=True, STORE_FINAL_STATE=True, SAVE_NEW_VALUE=True, IS_VARLEN=True)
    target = GPUTarget("cuda", 121, 32)
    options = dict(num_warps=4, num_stages=2, enable_fp_fusion=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = triton.compile(ASTSource(kernel, signature=signature, constexprs=constants), target=target, options=options)
    artifacts = []
    for name in ("ttir", "ttgir", "llir", "ptx"):
        data = result.asm[name].encode()
        path = args.output_dir / ("state." + name)
        path.write_bytes(data)
        artifacts.append(dict(file=path.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    record = dict(kind="cpu_only_reference_state_ir_audit", host=socket.gethostname(), triton_version=triton.__version__,
                  source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  command=sys.argv, command_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  target=dict(backend="cuda", arch=121, warp_size=32), signature=signature, constants=constants,
                  options=options, wall_ms=(time.monotonic()-started)*1000, artifacts=artifacts,
                  exponent_binding="tl.exp", live_reference_autotune_config_verified=False,
                  kernel_executed=False, model_loaded=False, inference_acceptance=False)
    (args.output_dir / "audit.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
