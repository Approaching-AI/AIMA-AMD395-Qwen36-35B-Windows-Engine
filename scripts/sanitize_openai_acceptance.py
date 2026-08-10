#!/usr/bin/env python3
"""Remove deployment-local fields from an OpenAI product acceptance report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_BLOCKED_KEYS = {
    "address",
    "arbitrary_moe_kernel_dir",
    "arbitrary_moe_provider_dll",
    "base_url",
    "command_file",
    "endpoint",
    "executable",
    "model_path",
    "path",
    "pid",
    "provider_dll",
    "service_log",
}
_PRIVATE_TEXT = re.compile(
    r"(?:[A-Za-z]:\\|/mnt/[a-z]/|/(?:Users|home|data/home)/|"
    r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.|"
    r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.))",
    re.IGNORECASE,
)


def _sanitize(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        clean: dict[str, Any] = {}
        for child_key, child_value in value.items():
            lowered = child_key.lower()
            if (
                lowered in _BLOCKED_KEYS
                or lowered.endswith("_path")
                or lowered.endswith("_dir")
            ):
                continue
            clean[child_key] = _sanitize(child_value, child_key)
        return clean
    if isinstance(value, list):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, str) and _PRIVATE_TEXT.search(value):
        if key == "name":
            return re.split(r"[\\/]", value)[-1]
        raise ValueError(f"deployment-local text remained in field {key!r}")
    return value


def sanitize_report(payload: bytes) -> dict[str, Any]:
    source = json.loads(payload)
    if not isinstance(source, dict):
        raise ValueError("acceptance report must be a JSON object")
    clean = _sanitize(source)
    clean["record_type"] = "openai_http_product_acceptance_publication"
    clean["source_report"] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    encoded = json.dumps(clean, sort_keys=True)
    if _PRIVATE_TEXT.search(encoded):
        raise ValueError("sanitized report still contains deployment-local text")
    return clean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    payload = args.input.read_bytes()
    clean = sanitize_report(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(clean, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(clean["source_report"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
