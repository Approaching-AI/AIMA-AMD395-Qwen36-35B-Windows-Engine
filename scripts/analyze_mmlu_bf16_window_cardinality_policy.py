#!/usr/bin/env python3
"""Combine base/min and max-ID paths for BF16-window cardinality policies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import analyze_mmlu_bf16_window_tie_break_followup as followup_analyzer
from analyze_mmlu_bf16_window_tie_break import select_window_token
from analyze_mmlu_inverse_window import (
    AnalysisError,
    RequestGroup,
    TopkStep,
    answer_for_token,
    parse_request_groups,
    read_jsonl,
    read_text_snapshot,
    step_candidate_ulp_distances,
)
from analyze_mmlu_route_replay import read_json_object_snapshot


ELIGIBLE_COUNTS = (2, 3, 4, 5)
COUNT3_ULP_SHAPES = (
    (0, 0, 0),
    (0, 0, 1),
    (0, 0, 2),
    (0, 1, 1),
    (0, 1, 2),
    (0, 2, 2),
)


@dataclass(frozen=True)
class PathProjection:
    exact: bool
    prediction: str | None
    reason: str


def select_cardinality_token(
    step: TopkStep,
    maximum_ulp_distance: int,
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int] = frozenset(),
    count3_ulp_shape_mask: int = 0,
) -> int:
    """Select an ID from cardinality and unique-maximum BF16 shape rules."""

    if not 0 <= maximum_ulp_distance <= 64:
        raise AnalysisError("maximum ULP distance must be between 0 and 64")
    if not maximum_eligible_counts.issubset(ELIGIBLE_COUNTS):
        raise AnalysisError("maximum eligible counts must be drawn from 2,3,4,5")
    if not unique_maximum_eligible_counts.issubset(ELIGIBLE_COUNTS):
        raise AnalysisError(
            "unique-maximum eligible counts must be drawn from 2,3,4,5"
        )
    if maximum_eligible_counts & unique_maximum_eligible_counts:
        raise AnalysisError("maximum and unique-maximum eligible counts overlap")
    if not 0 <= count3_ulp_shape_mask < (1 << len(COUNT3_ULP_SHAPES)):
        raise AnalysisError("count-3 ULP shape mask must use only bits 0 through 5")
    if count3_ulp_shape_mask and maximum_ulp_distance != 2:
        raise AnalysisError("count-3 ULP shape selection requires a 2-ULP window")
    distances = step_candidate_ulp_distances(step)
    eligible = tuple(
        (token_id, distance)
        for token_id, distance in zip(step.topk_ids, distances)
        if distance <= maximum_ulp_distance
    )
    if not eligible:
        raise AnalysisError("BF16 window has no retained-top-k candidate")
    unique_maximum = sum(distance == 0 for _, distance in eligible) == 1
    select_maximum = len(eligible) in maximum_eligible_counts or (
        unique_maximum
        and len(eligible) in unique_maximum_eligible_counts
    )
    if len(eligible) == 3:
        signature = tuple(sorted(distance for _, distance in eligible))
        try:
            signature_index = COUNT3_ULP_SHAPES.index(signature)
        except ValueError as error:
            raise AnalysisError(
                f"unsupported count-3 ULP signature: {signature}"
            ) from error
        select_maximum = select_maximum or (
            count3_ulp_shape_mask & (1 << signature_index)
        ) != 0
    token_ids = tuple(token_id for token_id, _ in eligible)
    return max(token_ids) if select_maximum else min(token_ids)


def project_path(
    group: RequestGroup,
    maximum_ulp_distance: int,
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int],
    count3_ulp_shape_mask: int,
    ascii_a_token_id: int,
    active_prediction: str | None,
) -> PathProjection:
    for step in group.steps:
        target_winner = select_cardinality_token(
            step,
            maximum_ulp_distance,
            maximum_eligible_counts,
            unique_maximum_eligible_counts,
            count3_ulp_shape_mask,
        )
        target_answer = answer_for_token(target_winner, ascii_a_token_id)
        if target_winner != step.sampled_token:
            if target_answer is not None:
                return PathProjection(True, target_answer, "diverged_to_answer_token")
            return PathProjection(False, None, "diverged_to_non_answer_token")
        if target_answer is not None:
            return PathProjection(True, target_answer, "matched_path_to_answer_token")
    return PathProjection(
        True, active_prediction, "matched_full_path_without_answer_token"
    )


def _formal_groups(
    log_path: Path,
    expected_rows: int,
    leading_groups: int,
    trailing_groups: int,
    label: str,
) -> tuple[list[RequestGroup], dict[str, Any]]:
    if leading_groups < 0 or trailing_groups < 0:
        raise AnalysisError(f"{label} request-group counts cannot be negative")
    snapshot = read_text_snapshot(log_path)
    groups, inflight = parse_request_groups(snapshot.text.splitlines())
    if leading_groups + trailing_groups > len(groups):
        raise AnalysisError(f"{label} excluded request groups exceed completed groups")
    end = len(groups) - trailing_groups
    formal = groups[leading_groups:end]
    if len(formal) != expected_rows:
        raise AnalysisError(
            f"{label} request-group alignment differs: formal_groups={len(formal)}, "
            f"rows={expected_rows}, completed={len(groups)}, leading={leading_groups}, "
            f"trailing={trailing_groups}"
        )
    return formal, {
        "path": str(log_path.resolve()),
        "sha256": snapshot.sha256,
        "bytes": snapshot.size_bytes,
        "completed_request_groups": len(groups),
        "leading_request_groups": leading_groups,
        "trailing_request_groups": trailing_groups,
        "inflight_topk_markers": inflight,
    }


def _validate_group_results(
    rows: list[dict[str, Any]],
    groups: list[RequestGroup],
    ascii_a_token_id: int,
    label: str,
) -> None:
    for row, group in zip(rows, groups):
        prediction = row.get("prediction")
        final_prediction = answer_for_token(
            group.steps[-1].sampled_token, ascii_a_token_id
        )
        if final_prediction != prediction:
            raise AnalysisError(
                f"{label} log/result prediction mismatch for question "
                f"{row.get('question_id')}: token_prediction={final_prediction}, "
                f"result={prediction}"
            )


def _policy_name(
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int],
    count3_ulp_shape_mask: int,
) -> str:
    if (
        not maximum_eligible_counts
        and not unique_maximum_eligible_counts
        and count3_ulp_shape_mask == 0
    ):
        return "minimum_token_id"
    parts: list[str] = []
    if maximum_eligible_counts:
        joined = "_".join(str(value) for value in sorted(maximum_eligible_counts))
        parts.append(f"maximum_counts_{joined}")
    if unique_maximum_eligible_counts:
        joined = "_".join(
            str(value) for value in sorted(unique_maximum_eligible_counts)
        )
        parts.append(f"unique_maximum_counts_{joined}")
    if count3_ulp_shape_mask:
        parts.append(f"count3_ulp_shape_mask_{count3_ulp_shape_mask}")
    return "maximum_token_id_for_" + "_and_".join(parts)


def _cap(
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int],
    count3_ulp_shape_mask: int,
) -> int | None:
    if unique_maximum_eligible_counts or count3_ulp_shape_mask:
        return None
    for cap in range(1, 6):
        if maximum_eligible_counts == frozenset(range(2, cap + 1)):
            return cap
    return None


def analyze(
    tie_break_analysis_path: Path,
    base_service_log_path: Path,
    replay_service_log_path: Path,
    replay_results_path: Path,
    base_results_path: Path,
    reference_results_path: Path,
    question_id_file_path: Path,
    question_id_manifest_path: Path,
    target_max_ulps: int,
    base_leading_request_groups: int = 0,
    base_trailing_request_groups: int = 0,
    replay_leading_request_groups: int = 0,
    replay_trailing_request_groups: int = 0,
    ascii_a_token_id: int = 32,
    expected_full_rows: int = 12032,
    allow_partial: bool = False,
) -> dict[str, Any]:
    followup = followup_analyzer.analyze(
        tie_break_analysis_path,
        replay_results_path,
        base_results_path,
        reference_results_path,
        question_id_file_path,
        question_id_manifest_path,
        target_max_ulps,
        "maximum_token_id",
        expected_full_rows,
        allow_partial,
    )
    analysis_snapshot = read_json_object_snapshot(tie_break_analysis_path)
    base_snapshot = read_text_snapshot(base_results_path)
    replay_snapshot = read_text_snapshot(replay_results_path)
    reference_snapshot = read_text_snapshot(reference_results_path)
    for label, snapshot, expected_sha256 in (
        ("base", base_snapshot, followup["inputs"]["base_results_sha256"]),
        ("replay", replay_snapshot, followup["inputs"]["replay_results_sha256"]),
        (
            "reference",
            reference_snapshot,
            followup["inputs"]["reference_results_sha256"],
        ),
    ):
        if snapshot.sha256 != expected_sha256:
            raise AnalysisError(f"{label} results changed while taking snapshots")
    base_rows = read_jsonl(base_snapshot)
    replay_rows = read_jsonl(replay_snapshot)
    reference_rows = read_jsonl(reference_snapshot)

    base_groups, base_log = _formal_groups(
        base_service_log_path,
        len(base_rows),
        base_leading_request_groups,
        base_trailing_request_groups,
        "base",
    )
    replay_groups, replay_log = _formal_groups(
        replay_service_log_path,
        len(replay_rows),
        replay_leading_request_groups,
        replay_trailing_request_groups,
        "replay",
    )
    analysis = analysis_snapshot.value
    analysis_inputs = analysis.get("inputs")
    if not isinstance(analysis_inputs, dict):
        raise AnalysisError("tie-break analysis has no input bindings")
    if analysis_inputs.get("service_log_sha256") != base_log["sha256"]:
        raise AnalysisError("base service-log SHA-256 differs from tie-break analysis")
    if analysis_snapshot.sha256 != followup["inputs"]["tie_break_analysis_sha256"]:
        raise AnalysisError("tie-break analysis changed while taking snapshots")

    _validate_group_results(base_rows, base_groups, ascii_a_token_id, "base")
    _validate_group_results(replay_rows, replay_groups, ascii_a_token_id, "replay")
    for row, group in zip(replay_rows, replay_groups):
        for step in group.steps:
            expected = select_window_token(
                step, target_max_ulps, "maximum_token_id"
            )
            if step.sampled_token != expected:
                raise AnalysisError(
                    "replay is not the maximum-token-ID route for question "
                    f"{row.get('question_id')}: sampled={step.sampled_token}, "
                    f"expected={expected}"
                )

    base_by_id = {
        row["question_id"]: (row, group)
        for row, group in zip(base_rows, base_groups)
    }
    replay_by_id = {
        row["question_id"]: (row, group)
        for row, group in zip(replay_rows, replay_groups)
    }
    reference_by_id = {row["question_id"]: row for row in reference_rows}
    reference_correct = sum(int(row["correct"]) for row in reference_rows)
    policies: list[dict[str, Any]] = []
    policy_configs: list[
        tuple[frozenset[int], frozenset[int], int]
    ] = []
    for modes in product(range(3), repeat=len(ELIGIBLE_COUNTS)):
        maximum_counts = frozenset(
            count
            for mode, count in zip(modes, ELIGIBLE_COUNTS)
            if mode == 2
        )
        unique_maximum_counts = frozenset(
            count
            for mode, count in zip(modes, ELIGIBLE_COUNTS)
            if mode == 1
        )
        policy_configs.append((maximum_counts, unique_maximum_counts, 0))
    for count3_ulp_shape_mask in range(1, 1 << len(COUNT3_ULP_SHAPES)):
        policy_configs.append(
            (frozenset({2}), frozenset(), count3_ulp_shape_mask)
        )
    for (
        maximum_counts,
        unique_maximum_counts,
        count3_ulp_shape_mask,
    ) in policy_configs:
        exact_correct = 0
        exact_reference_prediction_agreement = 0
        exact_prediction_changes = 0
        source_counts = {"base_path": 0, "replay_path": 0, "unresolved": 0}
        reason_counts: dict[str, int] = {}
        unresolved_ids: list[int] = []
        unresolved_base_correct = 0
        for question_id, (base_row, base_group) in base_by_id.items():
            outcome = project_path(
                base_group,
                target_max_ulps,
                maximum_counts,
                unique_maximum_counts,
                count3_ulp_shape_mask,
                ascii_a_token_id,
                base_row.get("prediction"),
            )
            source = "base_path"
            if not outcome.exact and question_id in replay_by_id:
                replay_row, replay_group = replay_by_id[question_id]
                outcome = project_path(
                    replay_group,
                    target_max_ulps,
                    maximum_counts,
                    unique_maximum_counts,
                    count3_ulp_shape_mask,
                    ascii_a_token_id,
                    replay_row.get("prediction"),
                )
                source = "replay_path"
            if outcome.exact:
                source_counts[source] += 1
                exact_correct += int(outcome.prediction == base_row["answer"])
                exact_reference_prediction_agreement += int(
                    outcome.prediction
                    == reference_by_id[question_id].get("prediction")
                )
                exact_prediction_changes += int(
                    outcome.prediction != base_row.get("prediction")
                )
                reason = f"{source}:{outcome.reason}"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            else:
                source_counts["unresolved"] += 1
                unresolved_ids.append(question_id)
                unresolved_base_correct += int(base_row["correct"])
        unknown_rows = len(unresolved_ids)
        exact_rows = expected_full_rows - unknown_rows
        policies.append(
            {
                "policy": _policy_name(
                    maximum_counts,
                    unique_maximum_counts,
                    count3_ulp_shape_mask,
                ),
                "maximum_eligible_counts": sorted(maximum_counts),
                "unique_maximum_eligible_counts": sorted(
                    unique_maximum_counts
                ),
                "count3_ulp_shape_mask": count3_ulp_shape_mask,
                "count3_ulp_shapes": [
                    list(shape)
                    for index, shape in enumerate(COUNT3_ULP_SHAPES)
                    if count3_ulp_shape_mask & (1 << index)
                ],
                "maximum_eligible_count_cap": _cap(
                    maximum_counts,
                    unique_maximum_counts,
                    count3_ulp_shape_mask,
                ),
                "projection_exact": unknown_rows == 0,
                "exact_rows": exact_rows,
                "unknown_rows": unknown_rows,
                "exact_correct": exact_correct,
                "minimum_correct": exact_correct,
                "maximum_correct": exact_correct + unknown_rows,
                "correct_if_unknown_base_unchanged": (
                    exact_correct + unresolved_base_correct
                ),
                "reference_correct": reference_correct,
                "reference_score_still_possible": (
                    exact_correct <= reference_correct
                    <= exact_correct + unknown_rows
                ),
                "projected_score_equal": (
                    unknown_rows == 0 and exact_correct == reference_correct
                ),
                "exact_reference_prediction_agreement": (
                    exact_reference_prediction_agreement
                ),
                "exact_prediction_changes": exact_prediction_changes,
                "source_counts": source_counts,
                "reason_counts": dict(sorted(reason_counts.items())),
                "unknown_question_ids": unresolved_ids,
            }
        )

    return {
        "schema_version": 1,
        "record_type": "mmlu_pro_bf16_window_cardinality_policy_analysis",
        "target_max_ulps": target_max_ulps,
        "ascii_a_token_id": ascii_a_token_id,
        "formal_rows": expected_full_rows,
        "reference_correct": reference_correct,
        "replay_complete": followup["complete"],
        "observed_replay_rows": followup["observed_replay_rows"],
        "expected_replay_rows": followup["expected_replay_rows"],
        "eligible_counts": list(ELIGIBLE_COUNTS),
        "count3_ulp_shape_bits": {
            str(index): list(shape)
            for index, shape in enumerate(COUNT3_ULP_SHAPES)
        },
        "policies": policies,
        "inputs": {
            "tie_break_analysis": {
                "path": str(tie_break_analysis_path.resolve()),
                "sha256": analysis_snapshot.sha256,
                "bytes": analysis_snapshot.size_bytes,
            },
            "base_service_log": base_log,
            "replay_service_log": replay_log,
            "base_results": {
                "path": str(base_results_path.resolve()),
                "sha256": base_snapshot.sha256,
                "bytes": base_snapshot.size_bytes,
            },
            "replay_results": {
                "path": str(replay_results_path.resolve()),
                "sha256": replay_snapshot.sha256,
                "bytes": replay_snapshot.size_bytes,
            },
            "reference_results": {
                "path": str(reference_results_path.resolve()),
                "sha256": reference_snapshot.sha256,
                "bytes": reference_snapshot.size_bytes,
            },
            "question_id_file_sha256": followup["inputs"][
                "question_id_file_sha256"
            ],
            "question_id_manifest_sha256": followup["inputs"][
                "question_id_manifest_sha256"
            ],
        },
        "formal_acceptance": {
            "eligible": False,
            "reason": (
                "This two-path analysis can select a global cardinality policy, "
                "but only a fresh 12,032-row run under that one policy can pass."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tie-break-analysis", type=Path, required=True)
    parser.add_argument("--base-service-log", type=Path, required=True)
    parser.add_argument("--replay-service-log", type=Path, required=True)
    parser.add_argument("--replay-results", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--question-id-file", type=Path, required=True)
    parser.add_argument("--question-id-manifest", type=Path, required=True)
    parser.add_argument("--target-max-ulps", type=int, required=True)
    parser.add_argument("--base-leading-request-groups", type=int, default=0)
    parser.add_argument("--base-trailing-request-groups", type=int, default=0)
    parser.add_argument("--replay-leading-request-groups", type=int, default=0)
    parser.add_argument("--replay-trailing-request-groups", type=int, default=0)
    parser.add_argument("--ascii-a-token-id", type=int, default=32)
    parser.add_argument("--expected-full-rows", type=int, default=12032)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = analyze(
            args.tie_break_analysis,
            args.base_service_log,
            args.replay_service_log,
            args.replay_results,
            args.base_results,
            args.reference_results,
            args.question_id_file,
            args.question_id_manifest,
            args.target_max_ulps,
            args.base_leading_request_groups,
            args.base_trailing_request_groups,
            args.replay_leading_request_groups,
            args.replay_trailing_request_groups,
            args.ascii_a_token_id,
            args.expected_full_rows,
            args.allow_partial,
        )
        encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        if args.output is None:
            sys.stdout.write(encoded)
        else:
            if args.output.exists() and not args.force:
                raise AnalysisError(f"refusing to overwrite output: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(encoded, encoding="utf-8")
    except (AnalysisError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
