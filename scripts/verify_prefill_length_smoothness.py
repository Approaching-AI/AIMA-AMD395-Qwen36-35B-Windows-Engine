#!/usr/bin/env python3
"""Verify cold-prefill continuity across multiple prompt-length boundaries."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_CENTERS = (4096, 6144, 8192, 9216, 10240, 12288, 14336, 16384)
LOCAL_TTFT_RATIO_MAX = 1.10
LOCAL_POSITIVE_RESIDUAL_MAX_MS = 500.0
GLOBAL_THROUGHPUT_MAX_TO_MIN_RATIO_MAX = 1.30


class VerificationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def build_cases(
    centers: tuple[int, ...], repetitions: int, base_seed: int
) -> list[dict[str, int]]:
    cases = []
    ordinal = 0
    for repetition in range(repetitions):
        for group, center in enumerate(centers):
            content_seed = base_seed + repetition * 10_000_019 + group * 1_000_003
            for delta in (-1, 0, 1):
                cases.append(
                    {
                        "ordinal": ordinal,
                        "center_prompt_tokens": center,
                        "prompt_tokens": center + delta,
                        "delta": delta,
                        "repetition": repetition,
                        "content_seed": content_seed,
                        "first_token_id": 1024 + ordinal,
                    }
                )
                ordinal += 1
    random.Random(base_seed ^ 0x395_3600).shuffle(cases)
    return cases


def build_prompt(case: dict[str, int]) -> list[int]:
    generator = random.Random(case["content_seed"])
    return [case["first_token_id"]] + [
        32 + generator.randrange(256)
        for _ in range(case["prompt_tokens"] - 1)
    ]


def request_completion(
    base_url: str,
    model: str,
    prompt: list[int],
    timeout_seconds: float,
) -> tuple[list[int], float, float | None]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "ignore_eos": True,
            "stream": False,
            "return_token_ids": True,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    normalized = base_url.rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    request = urllib.request.Request(
        f"{normalized}/v1/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "qrt-prefill-length-smoothness/1",
        },
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
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
        raise VerificationError(f"completion returned status {status}: {value}")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise VerificationError("completion response does not contain one choice")
    token_ids = choices[0].get("token_ids")
    if (
        not isinstance(token_ids, list)
        or len(token_ids) != 1
        or not isinstance(token_ids[0], int)
    ):
        raise VerificationError("completion response has no first token ID")
    metrics = value.get("qrt_metrics")
    ttft_ms = metrics.get("ttft_ms") if isinstance(metrics, dict) else None
    if ttft_ms is not None and not isinstance(ttft_ms, (int, float)):
        raise VerificationError("completion response has an invalid TTFT")
    return token_ids, elapsed_ms, float(ttft_ms) if ttft_ms is not None else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    samples: dict[int, list[float]] = defaultdict(list)
    for record in records:
        samples[record["prompt_tokens"]].append(record["ttft_ms"])
    cohorts = []
    medians: dict[int, float] = {}
    throughputs: dict[int, float] = {}
    for prompt_tokens in sorted(samples):
        median_ttft_ms = statistics.median(samples[prompt_tokens])
        medians[prompt_tokens] = median_ttft_ms
        throughputs[prompt_tokens] = prompt_tokens * 1000.0 / median_ttft_ms
        cohorts.append(
            {
                "prompt_tokens": prompt_tokens,
                "samples_ms": samples[prompt_tokens],
                "median_ttft_ms": median_ttft_ms,
                "median_prefill_tokens_per_second": throughputs[prompt_tokens],
            }
        )

    centers = sorted({record["center_prompt_tokens"] for record in records})
    local_gates = []
    for center in centers:
        triplet = [medians[center - 1], medians[center], medians[center + 1]]
        ttft_ratio = max(triplet) / min(triplet)
        positive_residual_ms = max(
            0.0,
            medians[center - 1] - medians[center],
            medians[center + 1] - medians[center],
        )
        ratio_passed = ttft_ratio <= LOCAL_TTFT_RATIO_MAX
        residual_passed = (
            positive_residual_ms <= LOCAL_POSITIVE_RESIDUAL_MAX_MS
        )
        local_gates.append(
            {
                "center_prompt_tokens": center,
                "neighbor_triplet": [center - 1, center, center + 1],
                "ttft_max_to_min_ratio": ttft_ratio,
                "ttft_max_to_min_ratio_max": LOCAL_TTFT_RATIO_MAX,
                "positive_residual_ms": positive_residual_ms,
                "positive_residual_max_ms": LOCAL_POSITIVE_RESIDUAL_MAX_MS,
                "ratio_passed": ratio_passed,
                "positive_residual_passed": residual_passed,
                "passed": ratio_passed and residual_passed,
            }
        )
    throughput_values = list(throughputs.values())
    global_ratio = max(throughput_values) / min(throughput_values)
    global_gate = {
        "median_throughput_max_to_min_ratio": global_ratio,
        "median_throughput_max_to_min_ratio_max": (
            GLOBAL_THROUGHPUT_MAX_TO_MIN_RATIO_MAX
        ),
        "minimum_median_prefill_tokens_per_second": min(throughput_values),
        "maximum_median_prefill_tokens_per_second": max(throughput_values),
        "passed": global_ratio <= GLOBAL_THROUGHPUT_MAX_TO_MIN_RATIO_MAX,
    }
    return {
        "cohorts": cohorts,
        "local_gates": local_gates,
        "global_throughput_gate": global_gate,
        "all_local_gates_passed": all(gate["passed"] for gate in local_gates),
        "all_gates_passed": (
            all(gate["passed"] for gate in local_gates)
            and global_gate["passed"]
        ),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amd-base-url", required=True)
    parser.add_argument("--gb10-base-url", required=True)
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=39_536_000)
    parser.add_argument("--centers", type=int, nargs="+", default=DEFAULT_CENTERS)
    parser.add_argument("--repo-commit", default="unknown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.output.exists() and not args.force:
        raise VerificationError(f"refusing to overwrite {args.output}")
    if args.repetitions < 1:
        raise VerificationError("--repetitions must be positive")
    centers = tuple(args.centers)
    if not centers or len(set(centers)) != len(centers) or min(centers) < 2:
        raise VerificationError("--centers must be unique integers above one")

    started_at = utc_now()
    cases = build_cases(centers, args.repetitions, args.base_seed)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for position, case in enumerate(cases, 1):
            prompt = build_prompt(case)
            amd_future = pool.submit(
                request_completion,
                args.amd_base_url,
                args.model,
                prompt,
                args.timeout_seconds,
            )
            gb10_future = pool.submit(
                request_completion,
                args.gb10_base_url,
                args.model,
                prompt,
                args.timeout_seconds,
            )
            amd_tokens, amd_elapsed_ms, ttft_ms = amd_future.result()
            gb10_tokens, gb10_elapsed_ms, _ = gb10_future.result()
            if ttft_ms is None:
                raise VerificationError("AMD response has no qrt_metrics.ttft_ms")
            record = {
                **case,
                "amd_token_ids": amd_tokens,
                "gb10_token_ids": gb10_tokens,
                "gb10_match": amd_tokens == gb10_tokens,
                "ttft_ms": ttft_ms,
                "prefill_tokens_per_second": (
                    case["prompt_tokens"] * 1000.0 / ttft_ms
                ),
                "amd_elapsed_ms": amd_elapsed_ms,
                "gb10_elapsed_ms": gb10_elapsed_ms,
            }
            records.append(record)
            print(
                f"[{position:02d}/{len(cases)}] "
                f"q={case['prompt_tokens']} ttft_ms={ttft_ms:.3f} "
                f"tok_s={record['prefill_tokens_per_second']:.3f} "
                f"gb10_match={record['gb10_match']}",
                flush=True,
            )

    records.sort(key=lambda record: record["ordinal"])
    continuity = summarize(records)
    correctness_passed = all(record["gb10_match"] for record in records)
    report = {
        "schema_version": 1,
        "record_type": "qrt_prefill_length_smoothness_verification",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "source_commit": args.repo_commit,
        "model": args.model,
        "amd_endpoint": args.amd_base_url,
        "gb10_endpoint": args.gb10_base_url,
        "request_policy": {
            "centers": list(centers),
            "neighbor_delta": [-1, 0, 1],
            "repetitions": args.repetitions,
            "base_seed": args.base_seed,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "ignore_eos": True,
            "return_token_ids": True,
            "cold_prefix_contract": "globally_unique_first_token_id",
            "controlled_content_contract": (
                "same generated content prefix within each center/repetition triplet"
            ),
        },
        "correctness_passed": correctness_passed,
        "continuity": continuity,
        "passed": correctness_passed and continuity["all_gates_passed"],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["passed"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
