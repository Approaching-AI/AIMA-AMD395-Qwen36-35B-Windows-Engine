#!/usr/bin/env python3
"""Capture or verify cold q8191/q8192/q8193 correctness and TTFT continuity."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import platform
import random
import statistics
import struct
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


PROMPT_LENGTHS = (8191, 8192, 8193)
MAX_TOKEN_COUNTS = (1, 2)
REFERENCE_RECORD_TYPE = "gb10_q8192_neighbor_continuation_reference"
VERIFICATION_RECORD_TYPE = "qrt_q8192_neighbor_continuity_verification"
NEIGHBOR_TO_CENTER_RATIO_MAX = 1.10
NEIGHBOR_POSITIVE_RESIDUAL_MAX_MS = 500.0


class VerificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def redacted_command(argv: list[str]) -> list[str]:
    command = []
    redact_next = False
    for argument in argv:
        if redact_next:
            command.append("<redacted>")
            redact_next = False
        elif argument == "--api-key":
            command.append(argument)
            redact_next = True
        elif argument.startswith("--api-key="):
            command.append("--api-key=<redacted>")
        else:
            command.append(argument)
    return command


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fnv1a64_u32le(values: list[int]) -> str:
    digest = 0xCBF29CE484222325
    for value in values:
        for byte in struct.pack("<I", value):
            digest ^= byte
            digest = (digest * 0x100000001B3) & 0xFFFF_FFFF_FFFF_FFFF
    return f"{digest:016x}"


def prompt_sha256(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<I", value))
    return digest.hexdigest()


def build_cases(repetitions: int, base_seed: int) -> list[dict[str, int]]:
    cases = []
    ordinal = 0
    for max_tokens in MAX_TOKEN_COUNTS:
        for repetition in range(repetitions):
            for prompt_tokens in PROMPT_LENGTHS:
                cases.append(
                    {
                        "ordinal": ordinal,
                        "prompt_tokens": prompt_tokens,
                        "max_tokens": max_tokens,
                        "repetition": repetition,
                        "prompt_seed": base_seed + ordinal * 1_000_003,
                        # Every request has a distinct first token, so no two
                        # cases can reuse a non-empty prefix-cache entry.
                        "first_token_id": 1024 + ordinal,
                    }
                )
                ordinal += 1
    random.Random(base_seed ^ 0x8192_8193).shuffle(cases)
    return cases


def build_prompt(case: dict[str, int]) -> list[int]:
    length = case["prompt_tokens"]
    generator = random.Random(case["prompt_seed"])
    prompt = [case["first_token_id"]]
    prompt.extend(32 + generator.randrange(256) for _ in range(length - 1))
    return prompt


def request_completion(
    base_url: str,
    model: str,
    api_key: str | None,
    timeout_seconds: float,
    prompt: list[int],
    max_tokens: int,
) -> tuple[bytes, dict[str, Any], float]:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/v1"):
        normalized_base_url = normalized_base_url[: -len("/v1")]
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "ignore_eos": True,
        "stream": False,
        # vLLM's explicit extension keeps token IDs independent of text
        # detokenization. The IDs, not detokenized text, are the oracle.
        "return_token_ids": True,
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "qrt-q8192-neighbor-continuity/1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{normalized_base_url}/v1/completions",
        data=canonical_json(payload),
        headers=headers,
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        body = error.read()
        status = error.code
    elapsed_ms = (time.monotonic() - started) * 1000.0
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"completion returned non-JSON status {status}: {error}"
        ) from error
    if not 200 <= status < 300:
        raise VerificationError(
            f"completion returned status {status}: {value}"
        )
    return body, value, elapsed_ms


def response_record(
    case: dict[str, int],
    prompt: list[int],
    body: bytes,
    value: dict[str, Any],
    elapsed_ms: float,
) -> dict[str, Any]:
    choices = value.get("choices")
    usage = value.get("usage")
    if not isinstance(choices, list) or len(choices) != 1:
        raise VerificationError("completion response does not contain one choice")
    if not isinstance(usage, dict):
        raise VerificationError("completion response has no usage object")
    choice = choices[0]
    text = choice.get("text")
    if not isinstance(text, str):
        raise VerificationError("completion choice text is absent")
    token_ids = choice.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != case["max_tokens"]
        or any(not isinstance(token_id, int) or token_id < 0 for token_id in token_ids)
    ):
        raise VerificationError("completion choice token IDs are absent or invalid")
    if usage.get("prompt_tokens") != case["prompt_tokens"]:
        raise VerificationError("completion prompt-token count differs")
    if usage.get("completion_tokens") != case["max_tokens"]:
        raise VerificationError("completion output-token count differs")
    record = {
        **case,
        "prompt_token_ids_fnv1a64": fnv1a64_u32le(prompt),
        "prompt_token_ids_sha256": prompt_sha256(prompt),
        "token_ids": token_ids,
        "token_ids_fnv1a64": fnv1a64_u32le(token_ids),
        "token_ids_sha256": prompt_sha256(token_ids),
        # Text is a transport diagnostic. GB10's vLLM service can return an
        # anomalously large detokenized string for an integer-token prompt,
        # while its explicit token_ids field remains correct. Keep the report
        # compact and make token IDs the cross-runtime correctness authority.
        "text_char_count": len(text),
        "text_utf8_bytes": len(text.encode("utf-8")),
        "text_utf8_sha256": sha256_bytes(text.encode("utf-8")),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage["completion_tokens"],
        "elapsed_ms": elapsed_ms,
        "response_sha256": sha256_bytes(body),
    }
    metrics = value.get("qrt_metrics")
    if isinstance(metrics, dict):
        record["ttft_ms"] = metrics.get("ttft_ms")
        record["request_ms"] = metrics.get("request_ms")
    return record


def reference_index(
    reference: dict[str, Any],
    expected_model: str,
    expected_repetitions: int,
    expected_base_seed: int,
) -> dict[int, dict[str, Any]]:
    if reference.get("record_type") != REFERENCE_RECORD_TYPE:
        raise VerificationError("reference record type differs")
    authority = reference.get("authority")
    if not isinstance(authority, dict) or str(
        authority.get("host", "")
    ).casefold() != "gb10-4t":
        raise VerificationError("reference authority is not gb10-4t")
    if authority.get("model") != expected_model:
        raise VerificationError("reference authority model differs")
    if reference.get("model") != expected_model:
        raise VerificationError("reference model differs")
    request_policy = reference.get("request_policy")
    expected_policy = {
        "prompt_lengths": list(PROMPT_LENGTHS),
        "max_token_counts": list(MAX_TOKEN_COUNTS),
        "repetitions": expected_repetitions,
        "base_seed": expected_base_seed,
        "temperature": 0,
        "top_p": 1,
        "ignore_eos": True,
        "return_token_ids": True,
        "cold_prefix_contract": "globally_unique_first_token_id",
    }
    if request_policy != expected_policy:
        raise VerificationError("reference request policy differs")
    cases = reference.get("cases")
    if not isinstance(cases, list) or not cases:
        raise VerificationError("reference has no cases")
    indexed = {}
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("ordinal"), int):
            raise VerificationError("reference case is invalid")
        if case["ordinal"] in indexed:
            raise VerificationError("reference duplicates a case ordinal")
        indexed[case["ordinal"]] = case
    return indexed


def continuity_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    cohorts: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in records:
        ttft_ms = record.get("ttft_ms")
        if not isinstance(ttft_ms, (int, float)) or ttft_ms < 0:
            raise VerificationError("verification response has no valid TTFT")
        cohorts[(record["max_tokens"], record["prompt_tokens"])].append(
            float(ttft_ms)
        )
    cohort_rows = []
    medians = {}
    for max_tokens in MAX_TOKEN_COUNTS:
        for prompt_tokens in PROMPT_LENGTHS:
            values = cohorts[(max_tokens, prompt_tokens)]
            if not values:
                raise VerificationError("verification TTFT cohort is empty")
            median = statistics.median(values)
            medians[(max_tokens, prompt_tokens)] = median
            cohort_rows.append(
                {
                    "max_tokens": max_tokens,
                    "prompt_tokens": prompt_tokens,
                    "samples_ms": values,
                    "median_ttft_ms": median,
                }
            )
    gates = []
    for max_tokens in MAX_TOKEN_COUNTS:
        center = medians[(max_tokens, 8192)]
        for neighbor in (8191, 8193):
            value = medians[(max_tokens, neighbor)]
            ratio = value / center if center > 0 else float("inf")
            residual_ms = value - center
            gates.append(
                {
                    "max_tokens": max_tokens,
                    "neighbor_prompt_tokens": neighbor,
                    "center_prompt_tokens": 8192,
                    "neighbor_to_center_ratio": ratio,
                    "positive_residual_ms": max(0.0, residual_ms),
                    "neighbor_to_center_ratio_max":
                        NEIGHBOR_TO_CENTER_RATIO_MAX,
                    "positive_residual_max_ms":
                        NEIGHBOR_POSITIVE_RESIDUAL_MAX_MS,
                    "ratio_passed":
                        ratio <= NEIGHBOR_TO_CENTER_RATIO_MAX,
                    "positive_residual_passed":
                        residual_ms <= NEIGHBOR_POSITIVE_RESIDUAL_MAX_MS,
                    "passed":
                        ratio <= NEIGHBOR_TO_CENTER_RATIO_MAX and
                        residual_ms <= NEIGHBOR_POSITIVE_RESIDUAL_MAX_MS,
                }
            )
    return {
        "cohorts": cohort_rows,
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates),
    }


def write_report(path: Path, value: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise VerificationError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("capture-reference", "verify"), required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=81928193)
    parser.add_argument("--authority-host", default=platform.node())
    parser.add_argument("--repo-commit")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    started_at_utc = utc_now()
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if not 1 <= args.repetitions <= 20:
        raise SystemExit("--repetitions must be between 1 and 20")
    if args.mode == "verify" and args.reference is None:
        raise SystemExit("--verify requires --reference")
    if args.mode == "capture-reference" and args.reference is not None:
        raise SystemExit("--capture-reference does not accept --reference")

    expected_cases = build_cases(args.repetitions, args.base_seed)
    expected_reference = None
    reference_sha256 = None
    if args.reference is not None:
        reference_bytes = args.reference.read_bytes()
        expected_reference = reference_index(
            json.loads(reference_bytes),
            args.model,
            args.repetitions,
            args.base_seed,
        )
        reference_sha256 = sha256_bytes(reference_bytes)
        if set(expected_reference) != {case["ordinal"] for case in expected_cases}:
            raise VerificationError("reference case set differs")

    records = []
    correctness_passed = True
    for request_index, case in enumerate(expected_cases):
        prompt = build_prompt(case)
        body, value, elapsed_ms = request_completion(
            args.base_url,
            args.model,
            args.api_key,
            args.timeout_seconds,
            prompt,
            case["max_tokens"],
        )
        record = response_record(case, prompt, body, value, elapsed_ms)
        record["request_index"] = request_index
        if expected_reference is not None:
            reference_case = expected_reference[case["ordinal"]]
            matches_reference = all(
                record.get(field) == reference_case.get(field)
                for field in (
                    "prompt_tokens",
                    "max_tokens",
                    "repetition",
                    "prompt_seed",
                    "first_token_id",
                    "prompt_token_ids_fnv1a64",
                    "prompt_token_ids_sha256",
                    "token_ids",
                    "token_ids_fnv1a64",
                    "token_ids_sha256",
                    "finish_reason",
                    "completion_tokens",
                )
            )
            record["gb10_match"] = matches_reference
            correctness_passed = correctness_passed and matches_reference
        records.append(record)
        print(
            f"[{request_index + 1:02d}/{len(expected_cases)}] "
            f"q={case['prompt_tokens']} max_tokens={case['max_tokens']} "
            f"rep={case['repetition']} ttft_ms={record.get('ttft_ms')} "
            f"gb10_match={record.get('gb10_match')}",
            flush=True,
        )

    common = {
        "schema_version": 1,
        "started_at_utc": started_at_utc,
        "completed_at_utc": utc_now(),
        "host": platform.node(),
        "model": args.model,
        "endpoint": args.base_url,
        "repo_commit": args.repo_commit,
        "command": redacted_command([sys.executable, *sys.argv]),
        "python_version": platform.python_version(),
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "request_policy": {
            "prompt_lengths": list(PROMPT_LENGTHS),
            "max_token_counts": list(MAX_TOKEN_COUNTS),
            "repetitions": args.repetitions,
            "base_seed": args.base_seed,
            "temperature": 0,
            "top_p": 1,
            "ignore_eos": True,
            "return_token_ids": True,
            "cold_prefix_contract": "globally_unique_first_token_id",
        },
        "cases": records,
    }
    if args.mode == "capture-reference":
        report = {
            **common,
            "record_type": REFERENCE_RECORD_TYPE,
            "authority": {
                "host": args.authority_host,
                "model": args.model,
            },
        }
        exit_code = 0
    else:
        summary = continuity_summary(records)
        passed = correctness_passed and summary["all_gates_passed"]
        report = {
            **common,
            "record_type": VERIFICATION_RECORD_TYPE,
            "gb10_reference": {
                "path": str(args.reference.resolve()),
                "sha256": reference_sha256,
            },
            "correctness_passed": correctness_passed,
            "continuity": summary,
            "passed": passed,
        }
        exit_code = 0 if passed else 3
    write_report(args.output, report, args.force)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "record_type": report["record_type"],
                "passed": report.get("passed"),
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
