#!/usr/bin/env python3
"""Project a targeted ULP-window replay without treating it as formal acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import analyze_mmlu_inverse_window as inverse_window


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class JsonlSnapshot:
    path: Path
    rows: tuple[dict[str, Any], ...]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class JsonObjectSnapshot:
    path: Path
    value: dict[str, Any]
    sha256: str
    size_bytes: int


def read_jsonl_snapshot(path: Path) -> JsonlSnapshot:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        try:
            row = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AnalysisError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise AnalysisError(f"row at {path}:{line_number} is not an object")
        rows.append(row)
    return JsonlSnapshot(
        path=path,
        rows=tuple(rows),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def read_json_object_snapshot(path: Path) -> JsonObjectSnapshot:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AnalysisError(f"invalid JSON object at {path}: {error}") from error
    if not isinstance(value, dict):
        raise AnalysisError(f"JSON input is not an object: {path}")
    return JsonObjectSnapshot(
        path=path,
        value=value,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def validate_rows(
    rows: tuple[dict[str, Any], ...],
    label: str,
    config_id: str | None = None,
) -> tuple[dict[Any, dict[str, Any]], str]:
    by_id: dict[Any, dict[str, Any]] = {}
    inferred_config = config_id
    for expected_sequence, row in enumerate(rows):
        question_id = row.get("question_id")
        if question_id is None:
            raise AnalysisError(f"{label} row {expected_sequence} has no question ID")
        if question_id in by_id:
            raise AnalysisError(f"duplicate {label} question ID: {question_id}")
        if row.get("sequence") != expected_sequence:
            raise AnalysisError(
                f"{label} sequence is not dense at {row.get('sequence')}; "
                f"expected {expected_sequence}"
            )
        if row.get("ok") is not True:
            raise AnalysisError(f"{label} question {question_id} is not successful")
        if not isinstance(row.get("correct"), bool):
            raise AnalysisError(
                f"{label} question {question_id} has no Boolean correctness"
            )
        row_config = row.get("config_id")
        if not isinstance(row_config, str) or not row_config:
            raise AnalysisError(f"{label} question {question_id} has no config ID")
        if inferred_config is None:
            inferred_config = row_config
        if row_config != inferred_config:
            raise AnalysisError(
                f"{label} question {question_id} has config {row_config}; "
                f"expected {inferred_config}"
            )
        by_id[question_id] = row
    if inferred_config is None:
        raise AnalysisError(f"{label} has no rows from which to infer a config ID")
    return by_id, inferred_config


def expected_replay_ids(
    report: dict[str, Any],
    target_max_ulps: int,
    base_order: tuple[Any, ...],
) -> tuple[Any, ...]:
    active_max_ulps = report.get("active_max_ulps")
    analysis_max_ulps = report.get("analysis_max_ulps", active_max_ulps)
    if not isinstance(active_max_ulps, int):
        raise AnalysisError("window analysis has no integer active_max_ulps")
    if not isinstance(analysis_max_ulps, int):
        raise AnalysisError("window analysis has no integer analysis_max_ulps")
    if target_max_ulps == active_max_ulps:
        raise AnalysisError("target window equals the active window")
    if not 0 <= target_max_ulps <= analysis_max_ulps:
        raise AnalysisError(
            f"target window {target_max_ulps} is outside analyzed range "
            f"0..{analysis_max_ulps}"
        )

    if target_max_ulps < active_max_ulps:
        payload = report.get("smaller_window_replay", {}).get(str(target_max_ulps))
        if not isinstance(payload, dict) or not isinstance(
            payload.get("question_ids"), list
        ):
            raise AnalysisError(
                f"window analysis has no smaller replay set for {target_max_ulps}"
            )
        question_ids = payload["question_ids"]
        selected = set(question_ids)
        if len(selected) != len(question_ids):
            raise AnalysisError(
                f"smaller replay set {target_max_ulps} contains duplicate IDs"
            )
    else:
        entries = report.get("larger_window_replay", {}).get(
            "first_possible_change", {}
        )
        if not isinstance(entries, dict):
            raise AnalysisError("window analysis has no larger-window entry buckets")
        selected: set[Any] = set()
        for distance in range(active_max_ulps + 1, target_max_ulps + 1):
            payload = entries.get(str(distance))
            if not isinstance(payload, dict) or not isinstance(
                payload.get("question_ids"), list
            ):
                raise AnalysisError(
                    f"window analysis has no larger replay bucket for {distance}"
                )
            question_ids = payload["question_ids"]
            bucket = set(question_ids)
            if len(bucket) != len(question_ids):
                raise AnalysisError(
                    f"larger replay bucket {distance} contains duplicate IDs"
                )
            overlap = selected.intersection(bucket)
            if overlap:
                raise AnalysisError(
                    f"larger replay bucket {distance} overlaps prior buckets: {overlap}"
                )
            selected.update(bucket)

    ordered = tuple(question_id for question_id in base_order if question_id in selected)
    if len(ordered) != len(selected):
        missing = selected.difference(base_order)
        raise AnalysisError(f"replay set contains IDs absent from base results: {missing}")
    return ordered


def row_could_be_correct_under_larger_window(
    row: dict[str, Any],
    group: inverse_window.RequestGroup,
    target_max_ulps: int,
    ascii_a_token_id: int,
) -> bool:
    """Conservatively bound correctness over retained-top-k observed-path winners."""

    answer = row.get("answer")
    if not isinstance(answer, str) or len(answer) != 1 or not "A" <= answer <= "J":
        raise AnalysisError(
            f"question {row.get('question_id')} has an invalid answer: {answer!r}"
        )
    unknown_divergent_continuation = False
    for step in group.steps:
        try:
            distances = inverse_window.step_candidate_ulp_distances(step)
        except inverse_window.AnalysisError as error:
            raise AnalysisError(str(error)) from error
        for token_id, distance in zip(step.topk_ids, distances):
            if distance > target_max_ulps or token_id == step.sampled_token:
                continue
            alternate_answer = inverse_window.answer_for_token(
                token_id, ascii_a_token_id
            )
            if alternate_answer is None:
                # A different non-answer token leaves the captured generation path.
                # Its continuation is not present in the base log, so it must retain
                # the possibility of eventually producing the ground-truth answer.
                unknown_divergent_continuation = True
            elif alternate_answer == answer:
                return True

        active_answer = inverse_window.answer_for_token(
            step.sampled_token, ascii_a_token_id
        )
        if active_answer is not None:
            return unknown_divergent_continuation or active_answer == answer

    return unknown_divergent_continuation or row.get("prediction") == answer


def conservative_topk_correctness_bound(
    report: dict[str, Any],
    service_log_path: Path,
    base: JsonlSnapshot,
    replay: JsonlSnapshot,
    base_by_id: dict[Any, dict[str, Any]],
    replay_by_id: dict[Any, dict[str, Any]],
    expected_ids: tuple[Any, ...],
    missing_ids: tuple[Any, ...],
    target_max_ulps: int,
) -> tuple[dict[str, Any], inverse_window.TextSnapshot]:
    active_max_ulps = report.get("active_max_ulps")
    if not isinstance(active_max_ulps, int):
        raise AnalysisError("window analysis has no integer active_max_ulps")
    if target_max_ulps <= active_max_ulps:
        raise AnalysisError(
            "a service-log correctness bound requires a larger target window"
        )
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise AnalysisError("window analysis has no input bindings")
    try:
        service_log = inverse_window.read_text_snapshot(service_log_path)
        groups, inflight_markers = inverse_window.parse_request_groups(
            service_log.text.splitlines()
        )
    except inverse_window.AnalysisError as error:
        raise AnalysisError(str(error)) from error
    if inputs.get("service_log_sha256") != service_log.sha256:
        raise AnalysisError("service log SHA-256 differs from window analysis")
    if inputs.get("service_log_bytes") != service_log.size_bytes:
        raise AnalysisError("service log byte count differs from window analysis")

    leading_request_groups = report.get("leading_request_groups")
    if (
        isinstance(leading_request_groups, bool)
        or not isinstance(leading_request_groups, int)
        or leading_request_groups < 0
    ):
        raise AnalysisError("window analysis has no valid leading-request count")
    if len(groups) != leading_request_groups + len(base.rows):
        raise AnalysisError(
            "service-log request-group alignment differs: "
            f"groups={len(groups)}, leading={leading_request_groups}, "
            f"base_rows={len(base.rows)}"
        )
    if report.get("completed_request_groups") != len(groups):
        raise AnalysisError("service-log group count differs from window analysis")
    if report.get("inflight_topk_markers") != inflight_markers:
        raise AnalysisError("service-log inflight-marker count differs from analysis")
    if inflight_markers != 0:
        raise AnalysisError("service log ends with an incomplete request group")

    ascii_a_token_id = report.get("ascii_a_token_id")
    if isinstance(ascii_a_token_id, bool) or not isinstance(ascii_a_token_id, int):
        raise AnalysisError("window analysis has no integer ASCII-A token ID")
    formal_groups = groups[leading_request_groups:]
    group_by_id: dict[Any, inverse_window.RequestGroup] = {}
    for row, group in zip(base.rows, formal_groups):
        question_id = row["question_id"]
        final_answer = inverse_window.answer_for_token(
            group.steps[-1].sampled_token, ascii_a_token_id
        )
        if final_answer != row.get("prediction"):
            raise AnalysisError(
                f"log/result prediction mismatch for question {question_id}: "
                f"token={group.steps[-1].sampled_token}, "
                f"result={row.get('prediction')}"
            )
        for step in group.steps:
            try:
                distances = inverse_window.step_candidate_ulp_distances(step)
            except inverse_window.AnalysisError as error:
                raise AnalysisError(str(error)) from error
            if distances[0] > active_max_ulps:
                raise AnalysisError(
                    f"question {question_id} selected a {distances[0]}-ULP "
                    f"candidate outside active window {active_max_ulps}"
                )
        group_by_id[question_id] = group

    could_be_correct_by_id = {
        question_id: row_could_be_correct_under_larger_window(
            base_by_id[question_id],
            group_by_id[question_id],
            target_max_ulps,
            ascii_a_token_id,
        )
        for question_id in expected_ids
    }
    could_be_correct_ids = tuple(
        question_id
        for question_id in expected_ids
        if could_be_correct_by_id[question_id]
    )
    provably_incorrect_ids = tuple(
        question_id
        for question_id in expected_ids
        if not could_be_correct_by_id[question_id]
    )
    observed_ids = tuple(row["question_id"] for row in replay.rows)
    impossible_observed_correct = tuple(
        question_id
        for question_id in observed_ids
        if replay_by_id[question_id]["correct"]
        and not could_be_correct_by_id[question_id]
    )
    if impossible_observed_correct:
        raise AnalysisError(
            "conservative top-k bound was contradicted by correct replay rows: "
            f"{impossible_observed_correct}"
        )
    missing_could_be_correct_ids = tuple(
        question_id
        for question_id in missing_ids
        if could_be_correct_by_id[question_id]
    )
    missing_provably_incorrect_ids = tuple(
        question_id
        for question_id in missing_ids
        if not could_be_correct_by_id[question_id]
    )
    observed_possible_ids = tuple(
        question_id
        for question_id in observed_ids
        if could_be_correct_by_id[question_id]
    )
    observed_impossible_ids = tuple(
        question_id
        for question_id in observed_ids
        if not could_be_correct_by_id[question_id]
    )
    return (
        {
            "method": "retained_topk_observed_path_overapproximation",
            "active_max_ulps": active_max_ulps,
            "target_max_ulps": target_max_ulps,
            "ascii_a_token_id": ascii_a_token_id,
            "leading_request_groups": leading_request_groups,
            "completed_request_groups": len(groups),
            "inflight_topk_markers": inflight_markers,
            "expected_rows": len(expected_ids),
            "could_be_correct_rows": len(could_be_correct_ids),
            "provably_incorrect_rows": len(provably_incorrect_ids),
            "could_be_correct_question_ids": list(could_be_correct_ids),
            "provably_incorrect_question_ids": list(provably_incorrect_ids),
            "missing_could_be_correct_rows": len(missing_could_be_correct_ids),
            "missing_provably_incorrect_rows": len(
                missing_provably_incorrect_ids
            ),
            "missing_could_be_correct_question_ids": list(
                missing_could_be_correct_ids
            ),
            "missing_provably_incorrect_question_ids": list(
                missing_provably_incorrect_ids
            ),
            "observed_validation": {
                "possible_rows": len(observed_possible_ids),
                "impossible_rows": len(observed_impossible_ids),
                "possible_correct_rows": sum(
                    int(replay_by_id[question_id]["correct"])
                    for question_id in observed_possible_ids
                ),
                "possible_incorrect_rows": sum(
                    int(not replay_by_id[question_id]["correct"])
                    for question_id in observed_possible_ids
                ),
                "impossible_correct_rows": 0,
                "impossible_incorrect_rows": len(observed_impossible_ids),
            },
            "interpretation": (
                "For each base observed-path generation step, every retained-top-k "
                "token inside the larger window is treated as a possible exact-F32 "
                "winner. A ground-truth answer token can be correct; a different "
                "non-answer token leaves the observed path and is conservatively "
                "allowed to become correct. A row is provably incorrect only when "
                "all represented terminal alternatives are wrong answers and no "
                "unknown continuation branch exists. This is an upper bound, not "
                "a substitute for a measured full-route score."
            ),
        },
        service_log,
    )


def analyze(
    analysis_report_path: Path,
    base_results_path: Path,
    replay_results_path: Path,
    reference_results_path: Path,
    target_max_ulps: int,
    expected_full_rows: int = 12032,
    allow_partial: bool = False,
    service_log_path: Path | None = None,
) -> dict[str, Any]:
    report_snapshot = read_json_object_snapshot(analysis_report_path)
    report = report_snapshot.value
    base = read_jsonl_snapshot(base_results_path)
    replay = read_jsonl_snapshot(replay_results_path)
    reference = read_jsonl_snapshot(reference_results_path)
    base_by_id, config_id = validate_rows(base.rows, "base")
    replay_by_id, _ = validate_rows(replay.rows, "replay", config_id)
    reference_by_id, _ = validate_rows(reference.rows, "reference", config_id)

    if len(base.rows) != expected_full_rows or len(reference.rows) != expected_full_rows:
        raise AnalysisError(
            "base/reference full-row contract failed: "
            f"base={len(base.rows)}, reference={len(reference.rows)}, "
            f"expected={expected_full_rows}"
        )
    if set(base_by_id) != set(reference_by_id):
        raise AnalysisError("base and reference question-ID sets differ")
    inputs = report.get("inputs")
    if not isinstance(inputs, dict):
        raise AnalysisError("window analysis has no input bindings")
    if inputs.get("candidate_results_sha256") != base.sha256:
        raise AnalysisError("base result SHA-256 differs from window analysis")
    if inputs.get("reference_results_sha256") != reference.sha256:
        raise AnalysisError("reference result SHA-256 differs from window analysis")
    if report.get("formal_rows") != expected_full_rows:
        raise AnalysisError("window analysis formal-row count is not full")

    base_order = tuple(row["question_id"] for row in base.rows)
    expected_ids = expected_replay_ids(report, target_max_ulps, base_order)
    expected_id_set = set(expected_ids)
    replay_id_set = set(replay_by_id)
    unexpected = replay_id_set.difference(expected_id_set)
    if unexpected:
        raise AnalysisError(f"replay contains unexpected question IDs: {unexpected}")
    missing_ids = tuple(
        question_id for question_id in expected_ids if question_id not in replay_id_set
    )
    replay_complete = not missing_ids and len(replay.rows) == len(expected_ids)
    if not allow_partial and not replay_complete:
        raise AnalysisError(
            f"replay is incomplete: observed={len(replay.rows)}, "
            f"expected={len(expected_ids)}"
        )

    observed_ids = tuple(row["question_id"] for row in replay.rows)
    expected_observed_order = tuple(
        question_id for question_id in expected_ids if question_id in replay_id_set
    )
    if observed_ids != expected_observed_order:
        raise AnalysisError("replay question IDs are not in base-result order")
    base_correct = sum(bool(row["correct"]) for row in base.rows)
    reference_correct = sum(bool(row["correct"]) for row in reference.rows)
    expected_base_correct = sum(
        bool(base_by_id[question_id]["correct"]) for question_id in expected_ids
    )
    observed_base_correct = sum(
        bool(base_by_id[question_id]["correct"]) for question_id in observed_ids
    )
    observed_replay_correct = sum(bool(row["correct"]) for row in replay.rows)
    observed_reference_correct = sum(
        bool(reference_by_id[question_id]["correct"]) for question_id in observed_ids
    )
    observed_gain = observed_replay_correct - observed_base_correct
    missing_count = len(missing_ids)
    minimum_final_correct = base_correct - expected_base_correct + observed_replay_correct
    unconstrained_maximum_final_correct = minimum_final_correct + missing_count
    maximum_final_correct = unconstrained_maximum_final_correct
    topk_bound = None
    service_log_snapshot = None
    if service_log_path is not None:
        topk_bound, service_log_snapshot = conservative_topk_correctness_bound(
            report,
            service_log_path,
            base,
            replay,
            base_by_id,
            replay_by_id,
            expected_ids,
            missing_ids,
            target_max_ulps,
        )
        maximum_final_correct = minimum_final_correct + topk_bound[
            "missing_could_be_correct_rows"
        ]
    projected_correct = minimum_final_correct if replay_complete else None
    score_gap = (
        projected_correct - reference_correct if projected_correct is not None else None
    )
    prediction_changes = sum(
        row.get("prediction") != base_by_id[row["question_id"]].get("prediction")
        for row in replay.rows
    )
    correctness_flips = sum(
        bool(row["correct"])
        != bool(base_by_id[row["question_id"]]["correct"])
        for row in replay.rows
    )

    active_max_ulps = report["active_max_ulps"]
    result = {
        "schema_version": 1,
        "record_type": "mmlu_pro_targeted_route_projection",
        "config_id": config_id,
        "route": {
            "active_max_ulps": active_max_ulps,
            "target_max_ulps": target_max_ulps,
            "direction": "smaller" if target_max_ulps < active_max_ulps else "larger",
            "expected_replay_rows": len(expected_ids),
            "observed_replay_rows": len(replay.rows),
            "missing_replay_rows": missing_count,
            "replay_complete": replay_complete,
        },
        "inputs": {
            "window_analysis": str(analysis_report_path.resolve()),
            "window_analysis_sha256": report_snapshot.sha256,
            "window_analysis_bytes": report_snapshot.size_bytes,
            "base_results": str(base.path.resolve()),
            "base_results_sha256": base.sha256,
            "base_results_bytes": base.size_bytes,
            "replay_results": str(replay.path.resolve()),
            "replay_results_sha256": replay.sha256,
            "replay_results_bytes": replay.size_bytes,
            "reference_results": str(reference.path.resolve()),
            "reference_results_sha256": reference.sha256,
            "reference_results_bytes": reference.size_bytes,
        },
        "observed": {
            "base_correct": observed_base_correct,
            "replay_correct": observed_replay_correct,
            "reference_correct": observed_reference_correct,
            "correct_gain_over_base": observed_gain,
            "prediction_changes": prediction_changes,
            "correctness_flips": correctness_flips,
        },
        "projection": {
            "base_full_correct": base_correct,
            "reference_full_correct": reference_correct,
            "hypothetical_correct_if_unobserved_unchanged": (
                base_correct + observed_gain
            ),
            "minimum_possible_correct": minimum_final_correct,
            "maximum_possible_correct": maximum_final_correct,
            "unconstrained_maximum_possible_correct": (
                unconstrained_maximum_final_correct
            ),
            "reference_score_still_possible": (
                minimum_final_correct <= reference_correct <= maximum_final_correct
            ),
            "hypothetical_full_correct": projected_correct,
            "hypothetical_score_numerator_delta": score_gap,
        },
        "missing_question_ids": list(missing_ids),
        "formal_acceptance": {
            "eligible": False,
            "reason": (
                "A targeted replay is route-selection evidence only. Even an equal "
                "projection requires a fresh 12,032-row run using only the selected "
                "global route."
            ),
        },
    }
    if service_log_snapshot is not None:
        result["inputs"].update(
            {
                "service_log": str(service_log_snapshot.path.resolve()),
                "service_log_sha256": service_log_snapshot.sha256,
                "service_log_bytes": service_log_snapshot.size_bytes,
            }
        )
        result["conservative_topk_correctness_bound"] = topk_bound
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-report", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--replay-results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--target-max-ulps", type=int, required=True)
    parser.add_argument("--expected-full-rows", type=int, default=12032)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--service-log",
        type=Path,
        help=(
            "frozen active-route service log used to tighten a larger-window "
            "partial-replay correctness upper bound"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = analyze(
            args.analysis_report,
            args.base_results,
            args.replay_results,
            args.reference_results,
            args.target_max_ulps,
            args.expected_full_rows,
            args.allow_partial,
            args.service_log,
        )
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            if args.output.exists() and not args.force:
                raise AnalysisError(f"refusing to overwrite output: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
        print(encoded, end="")
        return 0
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
