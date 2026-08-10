#!/usr/bin/env python3
"""Audit a fresh full MMLU-Pro run of one selected BF16 shape policy.

Route-selection projections are deliberately not formal acceptance.  This tool
binds a completed fresh candidate result and its service log to the frozen
two-path projection, verifies every observed prediction, and then requires the
candidate score to equal the complete gb10 reference score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import analyze_mmlu_bf16_window_cardinality_policy as policy_analyzer
import analyze_mmlu_inverse_window as inverse_window
import analyze_mmlu_route_replay as replay_analyzer


class AcceptanceError(RuntimeError):
    """A formal-acceptance invariant was not satisfied."""


def snapshot_info(snapshot: Any) -> dict[str, Any]:
    return {
        "path": str(snapshot.path.resolve()),
        "sha256": snapshot.sha256,
        "bytes": snapshot.size_bytes,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def validate_bound_snapshot(
    inputs: dict[str, Any],
    name: str,
    snapshot: Any,
) -> None:
    binding = inputs.get(name)
    require(isinstance(binding, dict), f"policy analysis has no {name} binding")
    require(
        binding.get("sha256") == snapshot.sha256,
        f"{name} SHA-256 differs from the policy analysis",
    )
    require(
        binding.get("bytes") == snapshot.size_bytes,
        f"{name} byte length differs from the policy analysis",
    )


def validate_result_rows(
    rows: Iterable[dict[str, Any]],
    label: str,
    expected_rows: int | None,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    materialized = list(rows)
    if expected_rows is not None:
        require(
            len(materialized) == expected_rows,
            f"{label} row count is {len(materialized)}; expected {expected_rows}",
        )
    require(materialized, f"{label} has no rows")
    seen: set[Any] = set()
    config_id: str | None = None
    endpoint: dict[str, Any] | None = None
    for expected_sequence, row in enumerate(materialized):
        question_id = row.get("question_id")
        require(question_id is not None, f"{label} row {expected_sequence} has no question ID")
        require(
            question_id not in seen,
            f"duplicate {label} question ID: {question_id}",
        )
        seen.add(question_id)
        require(
            row.get("sequence") == expected_sequence,
            f"{label} sequence is not dense at {row.get('sequence')}; "
            f"expected {expected_sequence}",
        )
        require(row.get("ok") is True, f"{label} question {question_id} is not successful")
        require(
            isinstance(row.get("parsed"), bool),
            f"{label} question {question_id} has no Boolean parsed field",
        )
        require(
            isinstance(row.get("correct"), bool),
            f"{label} question {question_id} has no Boolean correctness",
        )
        answer = row.get("answer")
        require(
            isinstance(answer, str) and len(answer) == 1 and "A" <= answer <= "J",
            f"{label} question {question_id} has an invalid answer",
        )
        prediction = row.get("prediction")
        if row["parsed"]:
            require(
                isinstance(prediction, str)
                and len(prediction) == 1
                and "A" <= prediction <= "J",
                f"{label} question {question_id} has an invalid parsed prediction",
            )
        else:
            require(
                prediction is None,
                f"{label} question {question_id} is unparsed but has a prediction",
            )
        require(
            row["correct"] is (prediction == answer),
            f"{label} question {question_id} correctness is inconsistent",
        )
        row_config = row.get("config_id")
        require(
            isinstance(row_config, str) and row_config,
            f"{label} question {question_id} has no config ID",
        )
        if config_id is None:
            config_id = row_config
        require(
            row_config == config_id,
            f"{label} question {question_id} has a different config ID",
        )
        row_endpoint = row.get("endpoint")
        require(
            isinstance(row_endpoint, dict),
            f"{label} question {question_id} has no endpoint identity",
        )
        if endpoint is None:
            endpoint = row_endpoint
        require(
            row_endpoint == endpoint,
            f"{label} question {question_id} has a different endpoint identity",
        )
    assert config_id is not None
    assert endpoint is not None
    return materialized, config_id, endpoint


def formal_groups_from_binding(
    snapshot: inverse_window.TextSnapshot,
    binding: dict[str, Any],
    expected_rows: int,
    label: str,
) -> list[inverse_window.RequestGroup]:
    try:
        groups, inflight = inverse_window.parse_request_groups(snapshot.text.splitlines())
    except inverse_window.AnalysisError as error:
        raise AcceptanceError(f"could not parse {label} service log: {error}") from error
    require(
        len(groups) == binding.get("completed_request_groups"),
        f"{label} completed request-group count differs from the policy analysis",
    )
    require(
        inflight == binding.get("inflight_topk_markers"),
        f"{label} inflight top-k count differs from the policy analysis",
    )
    leading = binding.get("leading_request_groups")
    trailing = binding.get("trailing_request_groups")
    require(
        isinstance(leading, int) and leading >= 0,
        f"{label} has an invalid leading request-group binding",
    )
    require(
        isinstance(trailing, int) and trailing >= 0,
        f"{label} has an invalid trailing request-group binding",
    )
    require(
        leading + trailing <= len(groups),
        f"{label} excluded request groups exceed completed groups",
    )
    end = len(groups) - trailing
    formal = groups[leading:end]
    require(
        len(formal) == expected_rows,
        f"{label} formal request groups are {len(formal)}; expected {expected_rows}",
    )
    return formal


def fresh_candidate_groups(
    snapshot: inverse_window.TextSnapshot,
    expected_rows: int,
) -> list[inverse_window.RequestGroup]:
    try:
        groups, inflight = inverse_window.parse_request_groups(snapshot.text.splitlines())
    except inverse_window.AnalysisError as error:
        raise AcceptanceError(f"could not parse candidate service log: {error}") from error
    require(inflight == 0, f"candidate service log has {inflight} inflight top-k markers")
    require(
        len(groups) == expected_rows,
        f"candidate service log has {len(groups)} completed request groups; "
        f"expected {expected_rows}",
    )
    return groups


def validate_group_predictions(
    rows: list[dict[str, Any]],
    groups: list[inverse_window.RequestGroup],
    ascii_a_token_id: int,
    label: str,
) -> None:
    try:
        policy_analyzer._validate_group_results(  # pylint: disable=protected-access
            rows, groups, ascii_a_token_id, label
        )
    except inverse_window.AnalysisError as error:
        raise AcceptanceError(str(error)) from error


def selected_policy(
    report: dict[str, Any],
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int],
    count3_ulp_shape_mask: int,
) -> dict[str, Any]:
    matches = [
        policy
        for policy in report.get("policies", [])
        if policy.get("maximum_eligible_counts")
        == sorted(maximum_eligible_counts)
        and policy.get("unique_maximum_eligible_counts")
        == sorted(unique_maximum_eligible_counts)
        and policy.get("count3_ulp_shape_mask") == count3_ulp_shape_mask
    ]
    require(len(matches) == 1, "selected policy is absent or duplicated in the analysis")
    return matches[0]


def bit_mask(counts: frozenset[int]) -> int:
    return sum(1 << count for count in counts)


def validate_route_markers(
    service_log: inverse_window.TextSnapshot,
    groups: list[inverse_window.RequestGroup],
    target_max_ulps: int,
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int],
    count3_ulp_shape_mask: int,
) -> int:
    marker_name = "BATCH_MARK qwen36_exact_arbitrary_lm_head_bf16_window_high_id "
    markers = [line for line in service_log.text.splitlines() if marker_name in line]
    expected_steps = sum(len(group.steps) for group in groups)
    require(
        len(markers) == expected_steps,
        f"candidate route-marker count is {len(markers)}; expected {expected_steps}",
    )
    fields = (
        f"maximum_ulp_distance={target_max_ulps}",
        f"maximum_eligible_count_mask={bit_mask(maximum_eligible_counts)}",
        f"unique_maximum_eligible_count_mask={bit_mask(unique_maximum_eligible_counts)}",
        f"count3_ulp_shape_mask={count3_ulp_shape_mask}",
        "candidate_window=retained_topk",
        "tie_policy=maximum_token_id_by_bf16_shape",
    )
    for index, marker in enumerate(markers):
        missing = [field for field in fields if field not in marker]
        require(
            not missing,
            f"candidate route marker {index} has the wrong policy: missing {missing}",
        )
    return len(markers)


def basename(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def contains_string(value: Any, target: str) -> bool:
    if isinstance(value, str):
        return value.lower() == target.lower()
    if isinstance(value, dict):
        return any(contains_string(item, target) for item in value.values())
    if isinstance(value, list):
        return any(contains_string(item, target) for item in value)
    return False


def validate_manifest(
    manifest: dict[str, Any],
    candidate_results: Path,
    candidate_config: str,
    candidate_endpoint: dict[str, Any],
    expected_rows: int,
    required_runtime_sha256: tuple[str, ...],
) -> None:
    require(
        manifest.get("record_type") == "mmlu_pro_openai_manifest",
        "candidate manifest has the wrong record type",
    )
    config = manifest.get("config")
    require(
        isinstance(config, dict) and config.get("config_id") == candidate_config,
        "candidate manifest config differs from candidate rows",
    )
    require(
        manifest.get("selected_questions") == expected_rows,
        "candidate manifest is not a full-row selection",
    )
    selection = manifest.get("selection")
    require(isinstance(selection, dict), "candidate manifest has no selection object")
    require(selection.get("categories") is None, "candidate manifest filters categories")
    require(selection.get("question_ids") is None, "candidate manifest filters question IDs")
    require(selection.get("limit") == 0, "candidate manifest has a row limit")
    require(
        selection.get("limit_per_category") == 0,
        "candidate manifest has a per-category limit",
    )
    require(
        basename(manifest.get("results_path")) == candidate_results.name,
        "candidate manifest results path differs",
    )
    endpoints = manifest.get("endpoints")
    require(
        isinstance(endpoints, list)
        and len(endpoints) == 1
        and endpoints[0] == candidate_endpoint,
        "candidate manifest endpoint identity differs",
    )
    execution = manifest.get("execution")
    require(
        isinstance(execution, dict) and execution.get("workers") == 1,
        "candidate manifest is not the batch-one evaluation route",
    )
    runtime_evidence = manifest.get("runtime_evidence")
    require(
        isinstance(runtime_evidence, dict)
        and isinstance(runtime_evidence.get("candidate"), list)
        and runtime_evidence["candidate"],
        "candidate manifest has no candidate runtime evidence",
    )
    for digest in required_runtime_sha256:
        require(
            len(digest) == 64 and all(character in "0123456789abcdefABCDEF" for character in digest),
            f"invalid required runtime SHA-256: {digest}",
        )
        require(
            contains_string(runtime_evidence["candidate"], digest),
            f"candidate runtime evidence does not contain required SHA-256 {digest}",
        )


def endpoint_summary(
    summary: dict[str, Any],
    side: str,
    endpoint: dict[str, Any],
    expected_rows: int,
    expected_correct: int,
    expected_parsed: int,
) -> None:
    endpoints = summary.get("endpoints")
    require(isinstance(endpoints, dict), "summary has no endpoints object")
    value = endpoints.get(side)
    require(isinstance(value, dict), f"summary has no {side} endpoint")
    for key in ("side", "base_url", "model", "endpoint_id"):
        require(
            value.get(key) == endpoint.get(key),
            f"summary {side} endpoint {key} differs",
        )
    require(value.get("expected") == expected_rows, f"summary {side} expected count differs")
    require(value.get("completed") == expected_rows, f"summary {side} is incomplete")
    require(value.get("missing") == 0, f"summary {side} has missing rows")
    require(
        value.get("failed_without_success") == 0,
        f"summary {side} has failed rows without success",
    )
    require(value.get("parsed") == expected_parsed, f"summary {side} parsed count differs")
    require(
        value.get("unparsed") == expected_rows - expected_parsed,
        f"summary {side} unparsed count differs",
    )
    require(value.get("correct") == expected_correct, f"summary {side} correct count differs")


def validate_summary(
    summary: dict[str, Any],
    manifest: dict[str, Any],
    candidate_results: Path,
    reference_results: Path,
    candidate_config: str,
    candidate_endpoint: dict[str, Any],
    reference_endpoint: dict[str, Any],
    expected_rows: int,
    candidate_correct: int,
    reference_correct: int,
    candidate_parsed: int,
    reference_parsed: int,
) -> None:
    require(
        summary.get("record_type") == "mmlu_pro_openai_summary",
        "candidate summary has the wrong record type",
    )
    config = summary.get("config")
    require(
        isinstance(config, dict) and config.get("config_id") == candidate_config,
        "candidate summary config differs from candidate rows",
    )
    require(
        summary.get("expected_questions") == expected_rows,
        "candidate summary expected-question count differs",
    )
    endpoint_summary(
        summary,
        "candidate",
        candidate_endpoint,
        expected_rows,
        candidate_correct,
        candidate_parsed,
    )
    endpoint_summary(
        summary,
        "reference",
        reference_endpoint,
        expected_rows,
        reference_correct,
        reference_parsed,
    )
    parity = summary.get("parity")
    require(isinstance(parity, dict), "candidate summary has no parity object")
    require(parity.get("common_completed") == expected_rows, "summary parity is incomplete")
    require(parity.get("both_full") is True, "summary parity is not full")
    require(parity.get("score_equal") is True, "summary scores are not equal")
    require(parity.get("score_numerator_delta") == 0, "summary score delta is not zero")
    require(parity.get("passed") is True, "summary parity did not pass")
    require(
        summary.get("runtime_evidence") == manifest.get("runtime_evidence"),
        "summary runtime evidence differs from the manifest",
    )
    result_names = {basename(value) for value in summary.get("results_paths", [])}
    require(candidate_results.name in result_names, "summary does not bind candidate results")
    require(reference_results.name in result_names, "summary does not bind reference results")


def analyze(
    policy_analysis_path: Path,
    base_service_log_path: Path,
    replay_service_log_path: Path,
    base_results_path: Path,
    replay_results_path: Path,
    candidate_service_log_path: Path,
    candidate_results_path: Path,
    reference_results_path: Path,
    candidate_manifest_path: Path,
    candidate_summary_path: Path,
    target_max_ulps: int,
    maximum_eligible_counts: frozenset[int],
    unique_maximum_eligible_counts: frozenset[int],
    count3_ulp_shape_mask: int,
    ascii_a_token_id: int = 32,
    expected_full_rows: int = 12_032,
    required_runtime_sha256: tuple[str, ...] = (),
) -> dict[str, Any]:
    require(expected_full_rows > 0, "expected full-row count must be positive")
    require(
        not maximum_eligible_counts.intersection(unique_maximum_eligible_counts),
        "maximum and unique-maximum eligible counts overlap",
    )

    policy_snapshot = replay_analyzer.read_json_object_snapshot(policy_analysis_path)
    base_log = inverse_window.read_text_snapshot(base_service_log_path)
    replay_log = inverse_window.read_text_snapshot(replay_service_log_path)
    candidate_log = inverse_window.read_text_snapshot(candidate_service_log_path)
    base_results = replay_analyzer.read_jsonl_snapshot(base_results_path)
    replay_results = replay_analyzer.read_jsonl_snapshot(replay_results_path)
    candidate_results = replay_analyzer.read_jsonl_snapshot(candidate_results_path)
    reference_results = replay_analyzer.read_jsonl_snapshot(reference_results_path)
    manifest_snapshot = replay_analyzer.read_json_object_snapshot(candidate_manifest_path)
    summary_snapshot = replay_analyzer.read_json_object_snapshot(candidate_summary_path)

    report = policy_snapshot.value
    require(
        report.get("record_type") == "mmlu_pro_bf16_window_cardinality_policy_analysis",
        "policy analysis has the wrong record type",
    )
    require(report.get("formal_rows") == expected_full_rows, "policy analysis row count differs")
    require(report.get("target_max_ulps") == target_max_ulps, "policy analysis ULP window differs")
    require(report.get("ascii_a_token_id") == ascii_a_token_id, "policy ASCII-A token differs")
    inputs = report.get("inputs")
    require(isinstance(inputs, dict), "policy analysis has no input bindings")
    for name, snapshot in (
        ("base_service_log", base_log),
        ("replay_service_log", replay_log),
        ("base_results", base_results),
        ("replay_results", replay_results),
        ("reference_results", reference_results),
    ):
        validate_bound_snapshot(inputs, name, snapshot)

    base_rows, base_config, base_endpoint = validate_result_rows(
        base_results.rows, "base", expected_full_rows
    )
    replay_rows, replay_config, replay_endpoint = validate_result_rows(
        replay_results.rows, "replay", None
    )
    candidate_rows, candidate_config, candidate_endpoint = validate_result_rows(
        candidate_results.rows, "candidate", expected_full_rows
    )
    reference_rows, reference_config, reference_endpoint = validate_result_rows(
        reference_results.rows, "reference", expected_full_rows
    )
    require(
        len({base_config, replay_config, candidate_config, reference_config}) == 1,
        "evaluation config IDs differ",
    )
    require(base_endpoint.get("side") == "candidate", "base endpoint is not candidate")
    require(replay_endpoint.get("side") == "candidate", "replay endpoint is not candidate")
    require(candidate_endpoint.get("side") == "candidate", "fresh endpoint is not candidate")
    require(reference_endpoint.get("side") == "reference", "reference endpoint is not reference")
    base_ids = [row["question_id"] for row in base_rows]
    candidate_ids = [row["question_id"] for row in candidate_rows]
    reference_ids = [row["question_id"] for row in reference_rows]
    require(candidate_ids == base_ids, "candidate question order differs from the frozen base")
    require(reference_ids == base_ids, "reference question order differs from the frozen base")

    base_binding = inputs["base_service_log"]
    replay_binding = inputs["replay_service_log"]
    base_groups = formal_groups_from_binding(
        base_log, base_binding, len(base_rows), "base"
    )
    replay_groups = formal_groups_from_binding(
        replay_log, replay_binding, len(replay_rows), "replay"
    )
    candidate_groups = fresh_candidate_groups(candidate_log, expected_full_rows)
    validate_group_predictions(base_rows, base_groups, ascii_a_token_id, "base")
    validate_group_predictions(replay_rows, replay_groups, ascii_a_token_id, "replay")
    validate_group_predictions(candidate_rows, candidate_groups, ascii_a_token_id, "candidate")
    route_markers = validate_route_markers(
        candidate_log,
        candidate_groups,
        target_max_ulps,
        maximum_eligible_counts,
        unique_maximum_eligible_counts,
        count3_ulp_shape_mask,
    )

    policy = selected_policy(
        report,
        maximum_eligible_counts,
        unique_maximum_eligible_counts,
        count3_ulp_shape_mask,
    )
    reference_correct = sum(int(row["correct"]) for row in reference_rows)
    require(policy.get("projection_exact") is True, "selected policy projection is not exact")
    require(policy.get("exact_rows") == expected_full_rows, "selected policy exact-row count differs")
    require(policy.get("unknown_rows") == 0, "selected policy has unknown rows")
    require(policy.get("unknown_question_ids") == [], "selected policy has unknown IDs")
    require(policy.get("projected_score_equal") is True, "selected policy did not project score parity")
    require(policy.get("reference_correct") == reference_correct, "policy reference score differs")

    base_by_id = {
        row["question_id"]: (row, group)
        for row, group in zip(base_rows, base_groups)
    }
    replay_by_id = {
        row["question_id"]: (row, group)
        for row, group in zip(replay_rows, replay_groups)
    }
    reference_by_id = {row["question_id"]: row for row in reference_rows}
    source_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    projection_mismatches: list[Any] = []
    projected_correct = 0
    projected_reference_agreement = 0
    projected_changes = 0
    for candidate_row in candidate_rows:
        question_id = candidate_row["question_id"]
        base_row, base_group = base_by_id[question_id]
        try:
            outcome = policy_analyzer.project_path(
                base_group,
                target_max_ulps,
                maximum_eligible_counts,
                unique_maximum_eligible_counts,
                count3_ulp_shape_mask,
                ascii_a_token_id,
                base_row.get("prediction"),
            )
        except inverse_window.AnalysisError as error:
            raise AcceptanceError(f"could not project question {question_id}: {error}") from error
        source = "base_path"
        if not outcome.exact:
            require(
                question_id in replay_by_id,
                f"question {question_id} needs replay but is absent from replay results",
            )
            replay_row, replay_group = replay_by_id[question_id]
            try:
                outcome = policy_analyzer.project_path(
                    replay_group,
                    target_max_ulps,
                    maximum_eligible_counts,
                    unique_maximum_eligible_counts,
                    count3_ulp_shape_mask,
                    ascii_a_token_id,
                    replay_row.get("prediction"),
                )
            except inverse_window.AnalysisError as error:
                raise AcceptanceError(
                    f"could not project replay question {question_id}: {error}"
                ) from error
            source = "replay_path"
        require(outcome.exact, f"question {question_id} projection is unresolved")
        source_counts[source] += 1
        reason_counts[f"{source}:{outcome.reason}"] += 1
        projected_correct += int(outcome.prediction == candidate_row["answer"])
        projected_reference_agreement += int(
            outcome.prediction == reference_by_id[question_id].get("prediction")
        )
        projected_changes += int(outcome.prediction != base_row.get("prediction"))
        if outcome.prediction != candidate_row.get("prediction"):
            projection_mismatches.append(question_id)

    require(
        not projection_mismatches,
        f"fresh candidate differs from the frozen policy for "
        f"{len(projection_mismatches)} questions; first={projection_mismatches[:10]}",
    )
    require(projected_correct == policy.get("exact_correct"), "recomputed policy score differs")
    require(
        projected_reference_agreement == policy.get("exact_reference_prediction_agreement"),
        "recomputed reference-prediction agreement differs",
    )
    require(projected_changes == policy.get("exact_prediction_changes"), "recomputed changes differ")
    normalized_source_counts = {
        "base_path": source_counts.get("base_path", 0),
        "replay_path": source_counts.get("replay_path", 0),
        "unresolved": 0,
    }
    require(
        normalized_source_counts == policy.get("source_counts"),
        "recomputed source counts differ",
    )
    require(
        dict(sorted(reason_counts.items())) == policy.get("reason_counts"),
        "recomputed reason counts differ",
    )

    candidate_correct = sum(int(row["correct"]) for row in candidate_rows)
    candidate_parsed = sum(int(row["parsed"]) for row in candidate_rows)
    reference_parsed = sum(int(row["parsed"]) for row in reference_rows)
    require(candidate_correct == projected_correct, "candidate score differs from projection")
    require(candidate_correct == reference_correct, "candidate score differs from gb10 reference")
    require(
        report.get("reference_correct") == reference_correct,
        "policy-analysis reference numerator differs",
    )

    validate_manifest(
        manifest_snapshot.value,
        candidate_results_path,
        candidate_config,
        candidate_endpoint,
        expected_full_rows,
        required_runtime_sha256,
    )
    validate_summary(
        summary_snapshot.value,
        manifest_snapshot.value,
        candidate_results_path,
        reference_results_path,
        candidate_config,
        candidate_endpoint,
        reference_endpoint,
        expected_full_rows,
        candidate_correct,
        reference_correct,
        candidate_parsed,
        reference_parsed,
    )

    return {
        "schema_version": 1,
        "record_type": "mmlu_pro_bf16_window_cardinality_fresh_acceptance",
        "formal_acceptance": {
            "eligible": True,
            "reason": (
                "A fresh complete candidate run exactly follows the frozen global "
                "policy and equals the complete gb10 score numerator."
            ),
        },
        "policy": {
            "name": policy.get("policy"),
            "target_max_ulps": target_max_ulps,
            "maximum_eligible_counts": sorted(maximum_eligible_counts),
            "unique_maximum_eligible_counts": sorted(unique_maximum_eligible_counts),
            "count3_ulp_shape_mask": count3_ulp_shape_mask,
        },
        "score": {
            "rows": expected_full_rows,
            "candidate_correct": candidate_correct,
            "reference_correct": reference_correct,
            "score_numerator_delta": candidate_correct - reference_correct,
            "candidate_parsed": candidate_parsed,
            "reference_parsed": reference_parsed,
            "projection_mismatches": len(projection_mismatches),
            "projection_reference_prediction_agreement": projected_reference_agreement,
        },
        "service_log": {
            "completed_request_groups": len(candidate_groups),
            "topk_steps": sum(len(group.steps) for group in candidate_groups),
            "route_markers": route_markers,
            "inflight_topk_markers": 0,
        },
        "projection": {
            "source_counts": normalized_source_counts,
            "reason_counts": dict(sorted(reason_counts.items())),
            "prediction_changes_from_base": projected_changes,
        },
        "required_runtime_sha256": list(required_runtime_sha256),
        "inputs": {
            "policy_analysis": snapshot_info(policy_snapshot),
            "base_service_log": snapshot_info(base_log),
            "replay_service_log": snapshot_info(replay_log),
            "candidate_service_log": snapshot_info(candidate_log),
            "base_results": snapshot_info(base_results),
            "replay_results": snapshot_info(replay_results),
            "candidate_results": snapshot_info(candidate_results),
            "reference_results": snapshot_info(reference_results),
            "candidate_manifest": snapshot_info(manifest_snapshot),
            "candidate_summary": snapshot_info(summary_snapshot),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-analysis", type=Path, required=True)
    parser.add_argument("--base-service-log", type=Path, required=True)
    parser.add_argument("--replay-service-log", type=Path, required=True)
    parser.add_argument("--base-results", type=Path, required=True)
    parser.add_argument("--replay-results", type=Path, required=True)
    parser.add_argument("--candidate-service-log", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--target-max-ulps", type=int, required=True)
    parser.add_argument("--maximum-eligible-count", type=int, action="append", default=[])
    parser.add_argument(
        "--unique-maximum-eligible-count", type=int, action="append", default=[]
    )
    parser.add_argument("--count3-ulp-shape-mask", type=int, default=0)
    parser.add_argument("--ascii-a-token-id", type=int, default=32)
    parser.add_argument("--expected-full-rows", type=int, default=12_032)
    parser.add_argument("--required-runtime-sha256", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None and args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    try:
        report = analyze(
            args.policy_analysis,
            args.base_service_log,
            args.replay_service_log,
            args.base_results,
            args.replay_results,
            args.candidate_service_log,
            args.candidate_results,
            args.reference_results,
            args.candidate_manifest,
            args.candidate_summary,
            args.target_max_ulps,
            frozenset(args.maximum_eligible_count),
            frozenset(args.unique_maximum_eligible_count),
            args.count3_ulp_shape_mask,
            args.ascii_a_token_id,
            args.expected_full_rows,
            tuple(args.required_runtime_sha256),
        )
    except (
        AcceptanceError,
        inverse_window.AnalysisError,
        replay_analyzer.AnalysisError,
    ) as error:
        print(f"acceptance audit error: {error}", file=sys.stderr)
        return 2
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
