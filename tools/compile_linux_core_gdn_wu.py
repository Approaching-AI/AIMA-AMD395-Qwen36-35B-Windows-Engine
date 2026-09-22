#!/usr/bin/env python3
"""Offline gfx1151 compilation of the native GDN W/U BF16 boundary repair.

Run in the pinned Triton 3.6.0 build environment with CUDA_VISIBLE_DEVICES,
HIP_VISIBLE_DEVICES and ROCR_VISIBLE_DEVICES all set to -1. The generated
include embeds the image in C++; the Windows runtime does not require Python.
Code object hashes can include compiler/debug-path differences. Any newly
compiled image requires its own native numerical and token qualification.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        if os.environ.get(name) != "-1":
            raise ValueError("Offline compilation requires " + name + "=-1")
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    if triton.__version__ != "3.6.0":
        raise ValueError("This artifact requires the pinned Triton 3.6.0 build environment")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = ROOT / "native/linux_core_port/gb10_gdn_wu.py"
    spec = importlib.util.spec_from_file_location("qrt_native_wu", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = {name: "*bf16" for name in ("k", "v", "w", "u", "A")}
    signature.update(beta="*fp32", g="*fp32", T="i32")
    constants = dict(cu_seqlens=None, chunk_indices=None, H=32, Hg=16, K=128,
                     V=128, BT=64, BK=64, BV=64, IS_VARLEN=False)
    options = dict(num_warps=2, num_stages=2, enable_fp_fusion=True)
    start = time.monotonic()
    kernel = triton.compile(ASTSource(module.recompute_w_u_fwd_kernel,
        signature=signature, constexprs=constants),
        target=GPUTarget("hip", "gfx1151", 32), options=options)
    artifacts = []
    for kind in ("hsaco", "ttir", "ttgir", "llir", "amdgcn"):
        data = kernel.asm[kind]
        data = data.encode() if isinstance(data, str) else data
        path = out / ("kernel." + kind)
        path.write_bytes(data)
        artifacts.append(dict(file=path.name, bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest()))
    data = kernel.asm["hsaco"]
    lines = ["// Generated from gb10_gdn_wu.py; Apache-2.0, with bundled vLLM/FLA notices.",
             "// Triton 3.6.0 CPU-only compilation for gfx1151; no runtime Python dependency.",
             'static constexpr const char* gdn_wu_image_sha256 = "' + hashlib.sha256(data).hexdigest() + '";',
             "alignas(256) static const unsigned char gdn_wu_image[] = {"]
    lines += ["  " + ", ".join("0x%02x" % value for value in data[i:i + 16]) + ","
              for i in range(0, len(data), 16)]
    lines.append("};")
    include = out / "gb10_gdn_wu_image.inc"
    include.write_text("\n".join(lines) + "\n")
    report = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        compiler_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        triton_version=triton.__version__, signature=signature, constants=constants,
        options=options, target=dict(backend="hip", arch="gfx1151", warp_size=32),
        metadata=kernel.metadata._asdict(), kernel_hash=kernel.hash, files=artifacts,
        embedded_image_sha256=hashlib.sha256(include.read_bytes()).hexdigest(),
        wall_ms=(time.monotonic() - start) * 1000, cpu_only=True,
        gpu_executed=False, inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
