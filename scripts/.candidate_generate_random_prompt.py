#!/usr/bin/env python3
"""Generate the deterministic random-token prompt used by native/GB10 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
from pathlib import Path


def fnv1a64(payload: bytes) -> int:
    # QRT's published digest contract intentionally retains its historical
    # seed, which differs from the canonical FNV-1a offset basis.
    digest = 1_469_598_103_934_665_603
    for value in payload:
        digest ^= value
        digest = (digest * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--first-token", type=int, required=True)
    args = parser.parse_args()
    if args.tokens < 1:
        parser.error("--tokens must be positive")
    if not 0 <= args.first_token < 248_320:
        parser.error("--first-token is outside the Qwen3.6 vocabulary")
    if args.output.exists():
        parser.error(f"refusing to overwrite prompt: {args.output}")

    generator = random.Random(args.seed)
    prompt = [args.first_token]
    prompt.extend(32 + generator.randrange(256) for _ in range(args.tokens - 1))
    packed = b"".join(struct.pack("<I", token) for token in prompt)
    args.output.write_text(
        json.dumps(prompt, separators=(",", ":")), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tokens": len(prompt),
                "seed": args.seed,
                "first_token": args.first_token,
                "u32le_sha256": hashlib.sha256(packed).hexdigest(),
                "u32le_fnv1a64": f"{fnv1a64(packed):016x}",
                "output": str(args.output),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
