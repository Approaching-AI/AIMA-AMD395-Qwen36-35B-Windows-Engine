#!/usr/bin/env python3
"""Prepare a small, fingerprinted prefix of saved GDN state inputs on CPU.

No GPU libraries, model loading, network access or numerical transformations.
Saved checkpoint/V-new surfaces are comparison-only, never recurrent inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import socket
import sys


def layouts(tokens: int) -> dict:
    return {
        "k-normalized": ("bf16", "bfloat16", [1, tokens, 16, 128]),
        "u": ("bf16", "bfloat16", [1, tokens, 32, 128]),
        "w": ("bf16", "bfloat16", [1, tokens, 32, 128]),
        "g-cumsum": ("f32", "float32", [1, tokens, 32]),
        "initial_state": ("f32", "float32", [1, 32, 128, 128]),
        "v-new": ("bf16", "bfloat16", [1, tokens, 32, 128]),
        "chunk-state": ("bf16", "bfloat16", [1, (tokens + 63) // 64, 32, 128, 128]),
    }


def verified_prefix(path: Path, expected: dict, size: int) -> bytes:
    if size > expected["bytes"] or path.stat().st_size != expected["bytes"]:
        raise ValueError(f"source byte count mismatch: {path.name}")
    digest, copied, count = hashlib.sha256(), [], 0
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
            if count < size:
                copied.append(block[:size - count])
            count += len(block)
    if count != expected["bytes"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"source fingerprint mismatch: {path.name}")
    return b"".join(copied)


def prepare(reference: Path, output: Path, tokens: int) -> dict:
    if tokens < 64 or tokens > 1024 or tokens % 64:
        raise ValueError("prefix must contain 64..1024 tokens in complete 64-token chunks")
    if output.exists():
        raise ValueError("output already exists; refusing overwrite")
    descriptor = json.loads((reference / "detail-full-w.json").read_bytes())
    source_tokens = descriptor["file"]["shape"][1]
    if type(source_tokens) is not int or not tokens < source_tokens <= 8192:
        raise ValueError("source must include a later BF16 checkpoint and at most 8192 tokens")
    source_layout, prefix_layout = layouts(source_tokens), layouts(tokens)
    parent_files, payloads = {}, {}
    frame_bytes = 32 * 128 * 128 * 2
    for name, (suffix, dtype, shape) in source_layout.items():
        descriptor_path = reference / f"detail-full-{name}.json"
        raw = descriptor_path.read_bytes()
        descriptor = json.loads(raw)
        parent = descriptor["file"]
        filename = f"full-{name}-{suffix}.bin"
        if (descriptor["surface"] != f"full-{name}" or parent["dtype"] != dtype
                or parent["shape"] != shape or Path(parent["path"]).name != filename
                or parent["bytes"] != math.prod(shape) * (2 if suffix == "bf16" else 4)):
            raise ValueError(f"source layout mismatch: {name}")
        prefix_shape = prefix_layout[name][2]
        size = math.prod(prefix_shape) * (2 if suffix == "bf16" else 4)
        # Preserve the checkpoint immediately following the complete prefix.
        data = verified_prefix(reference / filename, parent,
                               size + (frame_bytes if name == "chunk-state" else 0))
        parent_files[name] = dict(parent, descriptor_sha256=hashlib.sha256(raw).hexdigest())
        payloads[name + "-" + suffix] = (data[:size], prefix_shape, dtype,
                                         name in ("v-new", "chunk-state"))
        if name == "chunk-state":
            payloads["next-chunk-state-bf16"] = (data[size:], [1, 32, 128, 128], dtype, True)
    total_bytes = sum(len(item[0]) for item in payloads.values())
    if total_bytes > 64 << 20:
        raise ValueError("prefix artifact size exceeds 64 MiB")
    # Validate every complete parent before creating any output file.
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, (data, shape, dtype, reference_only) in payloads.items():
        (output / (name + ".bin")).write_bytes(data)
        files[name] = dict(file=name + ".bin", bytes=len(data), shape=shape, dtype=dtype,
                           sha256=hashlib.sha256(data).hexdigest(), reference_only=reference_only)
    record = dict(kind="cpu_prepared_gdn_state_prefix", schema_version=1,
                  host=socket.gethostname(), command=sys.argv,
                  command_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_tokens=source_tokens, tokens=tokens, chunks=tokens // 64,
                  parent_files=parent_files, files=files, total_bytes=total_bytes,
                  kernel_executed=False, model_loaded=False, inference_acceptance=False,
                  reference_raw_f32_checkpoints_available=False,
                  reference_checkpoints_are_recurrent_inputs=False,
                  future_raw_trace_requires_saved_bf16_parity=True)
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=384)
    args = parser.parse_args()
    print(json.dumps(prepare(args.reference, args.output_dir, args.tokens), indent=2))


if __name__ == "__main__":
    main()
