#!/usr/bin/env python3
"""Create a public JSON view without personal home-directory paths.

Embedded hashes continue to identify the original, unmodified evidence.
The public view records its own transformation and the original JSON digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

HOME = re.compile(r"/(?:Users|home|data/home)/[A-Za-z0-9._-]+(?=/|\b)")
METADATA = "public_evidence_view"


def redact_evidence(raw: bytes, *, source_revision: str, source_path: str) -> bytes:
    """Only replace home prefixes; preserve numbers, tokens and artifact hashes."""
    if HOME.search(source_revision) or HOME.search(source_path):
        raise ValueError("public provenance must use a revision and repository-relative path")
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("evidence must be a JSON object")
    prefixes = set()

    def inspect(value):
        if isinstance(value, str):
            prefixes.update(HOME.findall(value))
        elif isinstance(value, dict):
            for key, item in value.items():
                inspect(key)
                inspect(item)
        elif isinstance(value, list):
            for item in value:
                inspect(item)

    inspect(document)
    if not prefixes:
        return raw
    if METADATA in document:
        raise ValueError("an already redacted view contains a new private path")
    aliases = {prefix: f"<evidence-home-{index}>" for index, prefix in enumerate(sorted(prefixes), 1)}
    counts = dict.fromkeys(aliases.values(), 0)

    def replace(match):
        alias = aliases[match.group()]
        counts[alias] += 1
        return alias

    def transform(value):
        if isinstance(value, str):
            return HOME.sub(replace, value)
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, dict):
            result = {}
            for key, item in value.items():
                public_key = transform(key)
                if public_key in result:
                    raise ValueError("path redaction would merge distinct object keys")
                result[public_key] = transform(item)
            return result
        return value

    public = transform(document)
    public[METADATA] = {
        "schema": "qrt-public-evidence-path-view-v1",
        "original_json_sha256": hashlib.sha256(raw).hexdigest(),
        "original_json_bytes": len(raw),
        "source_revision": source_revision,
        "source_path": source_path,
        "home_path_alias_counts": counts,
        "embedded_artifact_hashes_changed": False,
        "numeric_results_changed": False,
        "scope": (
            "Public view with personal home prefixes replaced in text and object keys. "
            "Embedded hashes identify unmodified original artifacts. "
            "Commands containing aliases require local path substitution."
        ),
    }
    return (json.dumps(public, ensure_ascii=True, indent=2, allow_nan=False) + "\n").encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-path", required=True)
    args = parser.parse_args()
    result = redact_evidence(args.input.read_bytes(), source_revision=args.source_revision, source_path=args.source_path)
    # Preserve the original and never silently replace another public artifact.
    with args.output.open("xb") as output:
        output.write(result)
    print(json.dumps({"bytes": len(result), "sha256": hashlib.sha256(result).hexdigest()}))


if __name__ == "__main__":
    main()
