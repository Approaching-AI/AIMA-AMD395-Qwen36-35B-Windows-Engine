#!/usr/bin/env python3
"""Capture the real-token q8192 GB10 first-token/raw-logit oracle.

Run this file inside the diagnostic vLLM container whose opt-in
``sitecustomize.py`` hook writes LM-head input rows while ``/capture/ARM``
exists.  The hook observes the unmodified BF16 service.  This script keeps the
arm window around one request, identifies the base-model row independently of
the MTP row, and replays the exact service LM-head weight in BF16.

Example (from a machine with the repository and SSH access to gb10-4t)::

    ssh gb10-4t \
      'docker exec -i qwen36-lm-head-capture python3 -' \
      < scripts/capture_gb10_q8192_oracle.py

The canonical JSON written to stdout is capture evidence.  OpenAI logprobs are
retained as diagnostics only; ``selected_token_raw_logit_bf16`` is the raw
logit used by the product gate.
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
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


PROMPT_TOKENS = 8192
PROMPT_SEED = 395_518
PROMPT_FIRST_TOKEN = 84_411
PROMPT_SHA256 = "dda20edc609f935f34d3d41ca4a84ffefa66726676756fe26ff3bef4fbff0b96"
PROMPT_FNV1A64 = "1584e34d56e5d78b"
FNV_SEED = 1_469_598_103_934_665_603
FNV_PRIME = 0x100000001B3


class CaptureError(RuntimeError):
    """Raised when the response cannot become authoritative evidence."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def snapshot_capture_files(capture_dir: Path) -> dict[str, tuple[int, str]]:
    return {
        path.name: (path.stat().st_mtime_ns, sha256_file(path))
        for path in capture_dir.glob("lm_head_hidden_*.pt")
    }


def request_json(
    url: str, payload: dict[str, Any], timeout_seconds: float
) -> tuple[int, bytes, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=canonical_json(payload),
        headers={
            "Content-Type": "application/json",
            "User-Agent": "qrt-gb10-q8192-oracle-capture/1",
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


def bf16_rounding_margin(value: float) -> dict[str, float]:
    center = torch.tensor(value, dtype=torch.bfloat16)
    negative = torch.tensor(float("-inf"), dtype=torch.bfloat16)
    positive = torch.tensor(float("inf"), dtype=torch.bfloat16)
    lower = torch.nextafter(center, negative).float().item()
    upper = torch.nextafter(center, positive).float().item()
    center_value = center.float().item()
    lower_midpoint = (lower + center_value) * 0.5
    upper_midpoint = (center_value + upper) * 0.5
    return {
        "rounded_bf16": center_value,
        "lower_neighbor_bf16": lower,
        "upper_neighbor_bf16": upper,
        "lower_rounding_midpoint": lower_midpoint,
        "upper_rounding_midpoint": upper_midpoint,
        "nearest_midpoint_distance": min(
            value - lower_midpoint, upper_midpoint - value
        ),
    }


def replay_capture(
    capture_path: Path, prior_sha256: str | None, weight: torch.Tensor
) -> dict[str, Any]:
    package = torch.load(capture_path, map_location="cpu", weights_only=True)
    hidden = package.get("hidden")
    if not isinstance(hidden, torch.Tensor):
        raise CaptureError(f"capture has no hidden tensor: {capture_path}")
    if hidden.shape != (1, 2048) or hidden.dtype != torch.bfloat16:
        raise CaptureError(
            f"capture has unexpected hidden contract: {hidden.shape} {hidden.dtype}"
        )
    logits = torch.nn.functional.linear(hidden, weight)[0]
    if logits.shape != (248_320,) or logits.dtype != torch.bfloat16:
        raise CaptureError(
            f"LM-head replay has unexpected contract: {logits.shape} {logits.dtype}"
        )
    top_values, top_ids = torch.topk(logits.float(), 5)
    return {
        "name": capture_path.name,
        "mtime_ns": capture_path.stat().st_mtime_ns,
        "file_sha256": sha256_file(capture_path),
        "prior_file_sha256": prior_sha256,
        "shape": list(hidden.shape),
        "dtype": str(hidden.dtype),
        "hidden_bf16_sha256": sha256_bytes(
            hidden.view(torch.uint8).numpy().tobytes()
        ),
        "argmax_token_id": int(torch.argmax(logits).item()),
        "argmax_raw_logit": float(logits.max().item()),
        "top5": [
            {"token_id": int(token_id), "raw_logit": float(raw_logit)}
            for raw_logit, token_id in zip(top_values.tolist(), top_ids.tolist())
        ],
        "_logits": logits,
        "_hidden": hidden,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url", default="http://127.0.0.1:8000", help="vLLM base URL"
    )
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--model-root", type=Path, default=Path("/models"))
    parser.add_argument("--capture-dir", type=Path, default=Path("/capture"))
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    capture_dir = args.capture_dir.resolve(strict=True)
    model_root = args.model_root.resolve(strict=True)
    arm_path = capture_dir / "ARM"
    if arm_path.exists():
        raise CaptureError(f"refusing to reuse armed capture directory: {arm_path}")

    prompt, packed_prompt = build_prompt()
    payload = {
        "ignore_eos": True,
        "logprobs": 20,
        "max_tokens": 1,
        "model": args.model,
        "n": 1,
        "prompt": prompt,
        "return_token_ids": True,
        "stream": False,
        "temperature": 0,
        "top_p": 1,
    }
    request_body = canonical_json(payload)
    before = snapshot_capture_files(capture_dir)
    started_at = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    try:
        arm_path.write_text(
            "qrt-gb10-q8192-oracle-capture-v1\n", encoding="utf-8"
        )
        status, response_body, response = request_json(
            f"{args.base_url.rstrip('/')}/v1/completions",
            payload,
            args.timeout_seconds,
        )
    finally:
        arm_path.unlink(missing_ok=True)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    completed_at = (
        dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    )
    if status != 200:
        raise CaptureError(f"completion returned HTTP {status}: {response}")

    choices = response.get("choices")
    usage = response.get("usage")
    if not isinstance(choices, list) or len(choices) != 1:
        raise CaptureError("completion response did not contain one choice")
    if not isinstance(usage, dict) or usage.get("prompt_tokens") != PROMPT_TOKENS:
        raise CaptureError("completion response did not attest the q8192 prompt")
    choice = choices[0]
    output_token_ids = choice.get("token_ids")
    if (
        not isinstance(output_token_ids, list)
        or len(output_token_ids) != 1
        or not isinstance(output_token_ids[0], int)
    ):
        raise CaptureError("completion response did not expose one output token ID")

    after = snapshot_capture_files(capture_dir)
    changed: list[tuple[Path, str | None]] = []
    for name, (mtime_ns, _sha256) in sorted(after.items()):
        prior = before.get(name)
        if prior is None or prior[0] != mtime_ns:
            changed.append((capture_dir / name, None if prior is None else prior[1]))
    if not changed:
        raise CaptureError("the armed request did not update an LM-head capture")

    index_path = model_root / "model.safetensors.index.json"
    config_path = model_root / "config.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_key = "lm_head.weight"
    weight_shard_name = index["weight_map"].get(weight_key)
    if not isinstance(weight_shard_name, str):
        raise CaptureError(f"model index does not contain {weight_key}")
    weight_shard = model_root / weight_shard_name
    with safe_open(weight_shard, framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(weight_key)
    if weight.shape != (248_320, 2048) or weight.dtype != torch.bfloat16:
        raise CaptureError(
            f"LM-head weight has unexpected contract: {weight.shape} {weight.dtype}"
        )

    replays = [
        replay_capture(capture_path, prior_sha256, weight)
        for capture_path, prior_sha256 in changed
    ]
    selected_token_id = output_token_ids[0]
    base = [item for item in replays if item["argmax_token_id"] == selected_token_id]
    if len(base) != 1:
        raise CaptureError(
            "expected one base LM-head row matching the service token, found "
            f"{len(base)}"
        )
    base_capture = base[0]
    logits = base_capture["_logits"]
    hidden = base_capture["_hidden"]
    f32_dot = float(torch.dot(weight[selected_token_id].float(), hidden[0].float()))
    selected_raw_logit = float(logits[selected_token_id])
    rounding = bf16_rounding_margin(f32_dot)
    if rounding["rounded_bf16"] != selected_raw_logit:
        raise CaptureError("BF16 full-vocabulary replay differs from rounded F32 dot")

    for item in replays:
        item.pop("_logits")
        item.pop("_hidden")
    base_capture["selected_token_id"] = selected_token_id
    base_capture["selected_token_raw_logit_bf16"] = selected_raw_logit
    base_capture["selected_token_f32_dot"] = f32_dot
    base_capture["bf16_rounding"] = rounding

    logprobs = choice.get("logprobs")
    token_logprobs = logprobs.get("token_logprobs") if isinstance(logprobs, dict) else None
    openai_logprob = (
        token_logprobs[0]
        if isinstance(token_logprobs, list) and len(token_logprobs) == 1
        else None
    )
    record = {
        "schema_version": 1,
        "record_type": "gb10_q8192_first_token_raw_logit_capture",
        "authority": {
            "host": "gb10-4t",
            "service_container_hostname": socket.gethostname(),
            "model": args.model,
            "model_reference": str(model_root),
            "dtype": "bfloat16",
        },
        "started_at_utc": started_at,
        "completed_at_utc": completed_at,
        "endpoint": f"{args.base_url.rstrip('/')}/v1/completions",
        "request_policy": {key: value for key, value in payload.items() if key != "prompt"},
        "prompt": {
            "token_count": len(prompt),
            "first_token_id": prompt[0],
            "seed": PROMPT_SEED,
            "random_token_min_inclusive": 32,
            "random_token_max_inclusive": 287,
            "u32le_sha256": sha256_bytes(packed_prompt),
            "u32le_fnv1a64": fnv1a64(packed_prompt),
        },
        "request_sha256": sha256_bytes(request_body),
        "response_sha256": sha256_bytes(response_body),
        "http_status": status,
        "elapsed_ms": elapsed_ms,
        "usage": usage,
        "output_token_ids": output_token_ids,
        "selected_token_openai_logprob_diagnostic": openai_logprob,
        "capture_files": replays,
        "base_lm_head_capture": base_capture,
        "raw_logit_method": (
            "captured GB10 BF16 service LM-head input replayed with the exact "
            "BF16 lm_head.weight using torch.nn.functional.linear"
        ),
        "model_evidence": {
            "config_file": config_path.name,
            "config_sha256": sha256_file(config_path),
            "index_file": index_path.name,
            "index_sha256": sha256_file(index_path),
            "weight_tensor_key": weight_key,
            "weight_shape": list(weight.shape),
            "weight_dtype": str(weight.dtype),
            "weight_shard": weight_shard.name,
            "weight_shard_bytes": weight_shard.stat().st_size,
            "weight_shard_sha256": sha256_file(weight_shard),
        },
    }
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
