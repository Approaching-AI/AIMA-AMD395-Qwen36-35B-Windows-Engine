#!/usr/bin/env python3
"""Publish aligned MMLU-Pro results without prompts or deployment details."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import zip_longest
import json
from pathlib import Path
from typing import Any, TextIO


@dataclass(frozen=True)
class Summary:
    rows: int
    candidate_correct: int
    reference_correct: int
    candidate_parsed: int
    reference_parsed: int
    prediction_agreement: int
    correctness_agreement: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "candidate_correct": self.candidate_correct,
            "reference_correct": self.reference_correct,
            "candidate_parsed": self.candidate_parsed,
            "reference_parsed": self.reference_parsed,
            "prediction_agreement": self.prediction_agreement,
            "correctness_agreement": self.correctness_agreement,
        }


def _load(line: str, path: Path, number: int) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{number}: row is not an object")
    return value


def public_row(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    binding_fields = (
        "sequence",
        "question_id",
        "category",
        "source",
        "answer",
        "prompt_sha256",
        "messages_sha256",
    )
    for field in binding_fields:
        if candidate.get(field) != reference.get(field):
            raise ValueError(f"candidate/reference {field} mismatch")

    def side(value: dict[str, Any], include_timing: bool) -> dict[str, Any]:
        result = {
            "prediction": value.get("prediction"),
            "parsed": value.get("parsed"),
            "correct": value.get("correct"),
            "finish_reason": value.get("finish_reason"),
            "usage": value.get("usage"),
        }
        if include_timing:
            result["elapsed_ms"] = value.get("elapsed_ms")
            result["qrt_metrics"] = value.get("qrt_metrics")
        return result

    return {
        "schema_version": 1,
        "record_type": "mmlu_pro_public_parity",
        **{field: candidate.get(field) for field in binding_fields},
        "candidate": side(candidate, include_timing=True),
        "reference": side(reference, include_timing=False),
        "agreement": {
            "prediction": candidate.get("prediction") == reference.get("prediction"),
            "correctness": candidate.get("correct") == reference.get("correct"),
        },
    }


def sanitize(
    candidate_path: Path,
    reference_path: Path,
    output: TextIO,
    expected_rows: int | None = None,
) -> Summary:
    counters = {
        "rows": 0,
        "candidate_correct": 0,
        "reference_correct": 0,
        "candidate_parsed": 0,
        "reference_parsed": 0,
        "prediction_agreement": 0,
        "correctness_agreement": 0,
    }
    with ExitStack() as stack:
        candidate_file = stack.enter_context(candidate_path.open(encoding="utf-8"))
        reference_file = stack.enter_context(reference_path.open(encoding="utf-8"))
        for number, lines in enumerate(
            zip_longest(candidate_file, reference_file), start=1
        ):
            candidate_line, reference_line = lines
            if candidate_line is None or reference_line is None:
                raise ValueError("candidate/reference row counts differ")
            candidate = _load(candidate_line, candidate_path, number)
            reference = _load(reference_line, reference_path, number)
            row = public_row(candidate, reference)
            output.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
            output.write("\n")
            counters["rows"] += 1
            counters["candidate_correct"] += int(candidate.get("correct") is True)
            counters["reference_correct"] += int(reference.get("correct") is True)
            counters["candidate_parsed"] += int(candidate.get("parsed") is True)
            counters["reference_parsed"] += int(reference.get("parsed") is True)
            counters["prediction_agreement"] += int(
                candidate.get("prediction") == reference.get("prediction")
            )
            counters["correctness_agreement"] += int(
                candidate.get("correct") == reference.get("correct")
            )
    if expected_rows is not None and counters["rows"] != expected_rows:
        raise ValueError(
            f"expected {expected_rows} rows, received {counters['rows']}"
        )
    return Summary(**counters)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        summary = sanitize(
            args.candidate,
            args.reference,
            output,
            expected_rows=args.expected_rows,
        )
    print(json.dumps(summary.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
