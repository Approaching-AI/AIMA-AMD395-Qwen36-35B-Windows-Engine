#!/usr/bin/env python3
"""Capture the deterministic 32-token GB10 continuation for the q8192 gate.

Run this file inside the same unmodified BF16 vLLM service container used by
``capture_gb10_q8192_oracle.py``.  This capture does not arm the LM-head hook;
the first-token raw logit remains owned by that independent capture.  The
complete returned token-ID sequence is the decode correctness authority.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import random
import socket
import struct
import time
import urllib.error
import urllib.request
from typing import Any


PROMPT_TOKENS = 8192
PROMPT_SEED = 395_518
PROMPT_FIRST_TOKEN = 84_411
PROMPT_SHA256 = "dda20edc609f935f34d3d41ca4a84ffefa66726676756fe26ff3bef4fbff0b96"
PROMPT_FNV1A64 = "1584e34d56e5d78b"
CONTINUATION_TOKENS = 32
FNV_SEED = 1_469_598_103_934_665_603
FNV_PRIME = 0x100000001B3


class CaptureError(RuntimeError):
    """Raised when the response cannot become authoritative evidence."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fnv1a64(payload: bytes) -> str:
    digest = FNV_SEED
    for value in payload:
        digest ^= value
        digest = (digest * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return f"{digest:016x}"


def build_prompt() -> tuple[list[int], bytes]:
    generator = random.Random(PROMPT_SEED)
    prompt = [PROMPT_FIRST_TOKEN]
    prompt.extend(32 + generator.randrange(256) for _ in range(PROMPT_TOKENS - 1))
    packed = b"".join(struct.pack("<I", token) for token in prompt)
    actual_sha256 = sha256_bytes(packed)
    actual_fnv1a64 = fnv1a64(packed)
    if actual_sha256 != PROMPT_SHA256 or actual_fnv1a64 != PROMPT_FNV1A64:
        raise CaptureError(
            "generated prompt differs from the immutable q8192 contract: "
            f"sha256={actual_sha256} fnv1a64={actual_fnv1a64}"
        )
    return prompt, packed


def request_json(
    url: str, payload: dict[str, Any], timeout_seconds: float
) -> tuple[int, bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=canonical_json(payload),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "qrt-gb10-q8192-continuation-capture/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read()
        status = error.code
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureError(f"HTTP {status} returned non-JSON: {error}") from error
    if not isinstance(value, dict):
        raise CaptureError(f"HTTP {status} did not return a JSON object")
    return status, body, value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="vLLM base URL"
    )
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    prompt, packed_prompt = build_prompt()
    payload = {
        "ignore_eos": True,
        "max_tokens": CONTINUATION_TOKENS,
        "model": args.model,
        "n": 1,
        "prompt": prompt,
        "return_token_ids": True,
        "stream": False,
        "temperature": 0,
        "top_p": 1,
    }
    request_body = canonical_json(payload)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    status, response_body, response = request_json(
        f"{args.base_url.rstrip('/')}/v1/completions",
        payload,
        args.timeout_seconds,
    )
    elapsed_ms = (time.monotonic() - started) * 1000.0
    completed_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    if status != 200:
        raise CaptureError(f"completion returned HTTP {status}: {response}")

    choices = response.get("choices")
    usage = response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1:
        raise CaptureError("completion response did not contain one choice")
    if (
        not isinstance(usage, dict)
        or usage.get("prompt_tokens") != PROMPT_TOKENS
        or usage.get("completion_tokens") != CONTINUATION_TOKENS
    ):
        raise CaptureError("completion response did not attest q8192 plus 32 tokens")
    output_token_ids = choices[0].get("token_ids")
    if (
        not isinstance(output_token_ids, list)
        or len(output_token_ids) != CONTINUATION_TOKENS
        or any(not isinstance(token, int) for token in output_token_ids)
    ):
        raise CaptureError("completion response did not expose 32 output token IDs")
    packed_output = b"".join(
        struct.pack("<I", token) for token in output_token_ids
    )
    record = {
        "schema_version": 1,
        "record_type": "gb10_q8192_continuation_capture",
        "authority": {
            "host": "gb10-4t",
            "service_container_hostname": socket.gethostname(),
            "model": args.model,
            "model_reference": "/models",
            "dtype": "bfloat16",
        },
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/completions",
        "request_policy": {
            key: value for key, value in payload.items() if key != "prompt"
        },
        "prompt": {
            "token_count": len(prompt),
            "first_token_id": prompt[0],
            "seed": PROMPT_SEED,
            "u32le_sha256": sha256_bytes(packed_prompt),
            "u32le_fnv1a64": fnv1a64(packed_prompt),
        },
        "request_sha256": sha256_bytes(request_body),
        "response_sha256": sha256_bytes(response_body),
        "http_status": status,
        "elapsed_ms": elapsed_ms,
        "usage": usage,
        "output_token_ids": output_token_ids,
        "output_token_ids_u32le_sha256": sha256_bytes(packed_output),
        "output_token_ids_u32le_fnv1a64": fnv1a64(packed_output),
    }
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
