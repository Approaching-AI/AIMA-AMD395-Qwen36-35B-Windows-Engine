#!/usr/bin/env python3
"""Bound MMLU-Pro requests that can change under another global ULP window."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


TOPK_RE = re.compile(
    r"qwen36_exact_prefill_topk "
    r".*?sampled_token=(?P<sampled_token>\d+) "
    r"sampled_logit=(?P<sampled_logit>[^ ]+) "
    r".*?topk_ids=(?P<topk_ids>[^ ]+) "
    r"topk_logits=(?P<topk_logits>[^ ]+)"
)
DIRECT_END = "QRT_SERVER_MARK exact_first_token_prefill "
CONTINUATION_END = "QRT_SERVER_MARK exact_letter_classifier_continuation "


class AnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class TopkStep:
    sampled_token: int
    sampled_logit: float
    topk_ids: tuple[int, ...]
    topk_logits: tuple[float, ...]


@dataclass(frozen=True)
class RequestGroup:
    steps: tuple[TopkStep, ...]
    continuation: bool


@dataclass(frozen=True)
class TextSnapshot:
    path: Path
    text: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class SmallerWindowOutcome:
    correctness: bool | None
    prediction: str | None
    reason: str
    path_unchanged: bool


SCORE_FIELDS = (
    "n",
    "candidate_correct",
    "reference_correct",
    "prediction_mismatch",
    "correctness_flip",
    "candidate_only",
    "reference_only",
)


def read_text_snapshot(path: Path) -> TextSnapshot:
    payload = path.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise AnalysisError(f"input is not UTF-8: {path}: {error}") from error
    return TextSnapshot(
        path=path,
        text=text,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def read_jsonl(snapshot: TextSnapshot) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(snapshot.text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnalysisError(
                f"invalid JSON at {snapshot.path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise AnalysisError(
                f"row at {snapshot.path}:{line_number} is not an object"
            )
        rows.append(row)
    return rows


def parse_topk_step(line: str) -> TopkStep | None:
    match = TOPK_RE.search(line)
    if match is None:
        return None
    ids = tuple(int(value) for value in match.group("topk_ids").split(","))
    logits = tuple(float(value) for value in match.group("topk_logits").split(","))
    if not ids or len(ids) != len(logits):
        raise AnalysisError("top-k marker has inconsistent IDs and logits")
    return TopkStep(
        sampled_token=int(match.group("sampled_token")),
        sampled_logit=float(match.group("sampled_logit")),
        topk_ids=ids,
        topk_logits=logits,
    )


def parse_request_groups(lines: Iterable[str]) -> tuple[list[RequestGroup], int]:
    groups: list[RequestGroup] = []
    pending: list[TopkStep] = []
    for line in lines:
        step = parse_topk_step(line)
        if step is not None:
            pending.append(step)
        if DIRECT_END in line:
            if len(pending) != 1:
                raise AnalysisError(
                    f"direct request ended with {len(pending)} pending top-k markers"
                )
            groups.append(RequestGroup(tuple(pending), continuation=False))
            pending.clear()
        elif CONTINUATION_END in line:
            if len(pending) < 2:
                raise AnalysisError(
                    f"continuation ended with {len(pending)} pending top-k markers"
                )
            groups.append(RequestGroup(tuple(pending), continuation=True))
            pending.clear()
    return groups, len(pending)


def bf16_bits(value: float) -> int:
    if not math.isfinite(value):
        raise AnalysisError(f"logged BF16 value is not finite: {value}")
    try:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise AnalysisError(f"logged value is outside F32 range: {value}") from error
    if bits & 0xFFFF:
        raise AnalysisError(f"logged value is not exactly BF16-representable: {value}")
    return bits >> 16


def ordered_bf16(bits: int) -> int:
    magnitude = bits & 0x7FFF
    return 0x8000 - magnitude if bits & 0x8000 else 0x8000 + magnitude


def step_candidate_ulp_distances(step: TopkStep) -> tuple[int, ...]:
    if len(set(step.topk_ids)) != len(step.topk_ids):
        raise AnalysisError("top-k marker contains duplicate token IDs")
    if step.topk_ids[0] != step.sampled_token:
        raise AnalysisError("sampled token is not first in the arbitrated top-k")
    if step.topk_logits[0] != step.sampled_logit:
        raise AnalysisError("sampled logit differs from the arbitrated top-k logit")
    maximum = max(step.topk_logits)
    maximum_bits = bf16_bits(maximum)
    maximum_ordered = ordered_bf16(maximum_bits)
    distances: list[int] = []
    for logit in step.topk_logits:
        distance = maximum_ordered - ordered_bf16(bf16_bits(logit))
        if distance < 0:
            raise AnalysisError("top-k BF16 logit is greater than the captured maximum")
        distances.append(distance)
    return tuple(distances)


def step_ulp_distance(step: TopkStep) -> int:
    return step_candidate_ulp_distances(step)[0]


def answer_for_token(token_id: int, ascii_a_token_id: int) -> str | None:
    offset = token_id - ascii_a_token_id
    return chr(ord("A") + offset) if 0 <= offset < 10 else None


def project_smaller_window_outcome(
    row: dict[str, Any],
    group: RequestGroup,
    target_max_ulps: int,
    ascii_a_token_id: int,
) -> SmallerWindowOutcome:
    """Project score-exact terminal outcomes before requesting a smaller route."""

    answer = row.get("answer")
    if not isinstance(answer, str) or len(answer) != 1 or not "A" <= answer <= "J":
        raise AnalysisError(
            f"question {row.get('question_id')} has an invalid answer: {answer!r}"
        )
    if target_max_ulps < 0:
        raise AnalysisError("target max ULP distance cannot be negative")

    for step in group.steps:
        distances = step_candidate_ulp_distances(step)
        if distances[0] <= target_max_ulps:
            # The active minimum-exact-logit winner remains the minimum of a
            # smaller candidate subset that still contains it.
            continue
        eligible_answers = tuple(
            answer_for_token(token_id, ascii_a_token_id)
            for token_id, distance in zip(step.topk_ids, distances)
            if distance <= target_max_ulps
        )
        if not eligible_answers:
            raise AnalysisError(
                f"question {row.get('question_id')} has no smaller-window candidate"
            )
        if any(candidate is None for candidate in eligible_answers):
            return SmallerWindowOutcome(
                correctness=None,
                prediction=None,
                reason="eligible_non_answer_continuation",
                path_unchanged=False,
            )
        possible_correctness = {
            candidate == answer for candidate in eligible_answers
        }
        if len(possible_correctness) != 1:
            return SmallerWindowOutcome(
                correctness=None,
                prediction=None,
                reason="eligible_answers_mixed_correctness",
                path_unchanged=False,
            )
        prediction = eligible_answers[0] if len(eligible_answers) == 1 else None
        return SmallerWindowOutcome(
            correctness=next(iter(possible_correctness)),
            prediction=prediction,
            reason=(
                "singleton_answer"
                if prediction is not None
                else "eligible_answers_same_correctness"
            ),
            path_unchanged=False,
        )

    return SmallerWindowOutcome(
        correctness=bool(row.get("correct")),
        prediction=row.get("prediction") if isinstance(row.get("prediction"), str) else None,
        reason="active_path_unchanged",
        path_unchanged=True,
    )


def score_bucket(row: dict[str, Any], reference: dict[str, Any]) -> Counter[str]:
    candidate_correct = bool(row.get("correct"))
    reference_correct = bool(reference.get("correct"))
    return Counter(
        n=1,
        candidate_correct=int(candidate_correct),
        reference_correct=int(reference_correct),
        prediction_mismatch=int(row.get("prediction") != reference.get("prediction")),
        correctness_flip=int(candidate_correct != reference_correct),
        candidate_only=int(candidate_correct and not reference_correct),
        reference_only=int(reference_correct and not candidate_correct),
    )


def analyze(
    service_log: Path,
    candidate_results: Path,
    reference_results: Path,
    leading_request_groups: int,
    active_max_ulps: int,
    ascii_a_token_id: int,
    analysis_max_ulps: int = 64,
    trailing_request_groups: int = 0,
) -> dict[str, Any]:
    if leading_request_groups < 0:
        raise AnalysisError("leading request group count cannot be negative")
    if trailing_request_groups < 0:
        raise AnalysisError("trailing request group count cannot be negative")
    if not 0 <= active_max_ulps <= 64:
        raise AnalysisError("active max ULP distance must be between 0 and 64")
    if not active_max_ulps <= analysis_max_ulps <= 64:
        raise AnalysisError(
            "analysis max ULP distance must be between the active window and 64"
        )
    # Read each growing input once. The hashes below therefore bind exactly the
    # bytes analyzed even when this command is used during a live evaluation.
    candidate_snapshot = read_text_snapshot(candidate_results)
    reference_snapshot = read_text_snapshot(reference_results)
    service_log_snapshot = read_text_snapshot(service_log)
    candidates = read_jsonl(candidate_snapshot)
    references = read_jsonl(reference_snapshot)
    groups, inflight_markers = parse_request_groups(
        service_log_snapshot.text.splitlines()
    )
    excluded_request_groups = leading_request_groups + trailing_request_groups
    if excluded_request_groups > len(groups):
        raise AnalysisError(
            "excluded request groups exceed completed request groups: "
            f"groups={len(groups)}, leading={leading_request_groups}, "
            f"trailing={trailing_request_groups}"
        )
    formal_group_end = len(groups) - trailing_request_groups
    if formal_group_end - leading_request_groups != len(candidates):
        raise AnalysisError(
            "request-group alignment differs: "
            f"groups={len(groups)}, leading={leading_request_groups}, "
            f"trailing={trailing_request_groups}, "
            f"candidate_rows={len(candidates)}"
        )
    formal_groups = groups[leading_request_groups:formal_group_end]
    reference_by_id: dict[Any, dict[str, Any]] = {}
    for reference in references:
        question_id = reference.get("question_id")
        if question_id is None:
            raise AnalysisError("reference row has no question ID")
        if question_id in reference_by_id:
            raise AnalysisError(f"duplicate reference question ID: {question_id}")
        if reference.get("ok") is not True:
            raise AnalysisError(f"reference question {question_id} is not successful")
        if not isinstance(reference.get("correct"), bool):
            raise AnalysisError(
                f"reference question {question_id} has no Boolean correctness"
            )
        reference_by_id[question_id] = reference

    buckets: dict[int, Counter[str]] = {}
    threshold_replay: dict[int, list[Any]] = {
        threshold: [] for threshold in range(active_max_ulps)
    }
    larger_window_entry: dict[int, list[Any]] = {
        distance: []
        for distance in range(active_max_ulps + 1, analysis_max_ulps + 1)
    }
    observed_topk_max_candidate_ulp_distance = 0
    continuation_questions: list[Any] = []
    smaller_score_projection: dict[int, dict[str, Any]] = {
        threshold: {
            "known_correct": 0,
            "known_incorrect": 0,
            "unresolved_question_ids": [],
            "unresolved_active_correct": 0,
            "known_prediction_changes": 0,
            "reason_counts": Counter(),
        }
        for threshold in range(active_max_ulps)
    }
    candidate_question_ids: set[Any] = set()
    prior_sequence = -1
    for row, group in zip(candidates, formal_groups):
        sequence = row.get("sequence")
        if sequence != prior_sequence + 1:
            raise AnalysisError(
                f"candidate sequence is not dense at {sequence}; expected {prior_sequence + 1}"
            )
        prior_sequence = sequence
        question_id = row.get("question_id")
        if question_id is None:
            raise AnalysisError("candidate row has no question ID")
        if question_id in candidate_question_ids:
            raise AnalysisError(f"duplicate candidate question ID: {question_id}")
        candidate_question_ids.add(question_id)
        if row.get("ok") is not True:
            raise AnalysisError(f"candidate question {question_id} is not successful")
        if not isinstance(row.get("correct"), bool):
            raise AnalysisError(
                f"candidate question {question_id} has no Boolean correctness"
            )
        reference = reference_by_id.get(question_id)
        if reference is None:
            raise AnalysisError(f"candidate question {question_id} has no reference row")
        final_answer = answer_for_token(group.steps[-1].sampled_token, ascii_a_token_id)
        if final_answer != row.get("prediction"):
            raise AnalysisError(
                f"log/result prediction mismatch for question {question_id}: "
                f"token={group.steps[-1].sampled_token}, result={row.get('prediction')}"
            )
        candidate_distances = tuple(
            step_candidate_ulp_distances(step) for step in group.steps
        )
        distances = tuple(step_distances[0] for step_distances in candidate_distances)
        maximum_distance = max(distances)
        if maximum_distance > active_max_ulps:
            raise AnalysisError(
                f"question {question_id} selected a {maximum_distance}-ULP candidate "
                f"outside active window {active_max_ulps}"
            )
        buckets.setdefault(maximum_distance, Counter()).update(score_bucket(row, reference))
        if group.continuation:
            continuation_questions.append(question_id)
        for threshold, question_ids in threshold_replay.items():
            if maximum_distance > threshold:
                question_ids.append(question_id)
            outcome = project_smaller_window_outcome(
                row, group, threshold, ascii_a_token_id
            )
            projection = smaller_score_projection[threshold]
            projection["reason_counts"][outcome.reason] += 1
            if outcome.correctness is None:
                projection["unresolved_question_ids"].append(question_id)
                projection["unresolved_active_correct"] += int(row["correct"])
            elif outcome.correctness:
                projection["known_correct"] += 1
            else:
                projection["known_incorrect"] += 1
            if (
                outcome.prediction is not None
                and outcome.prediction != row.get("prediction")
            ):
                projection["known_prediction_changes"] += 1
        observed_topk_max_candidate_ulp_distance = max(
            observed_topk_max_candidate_ulp_distance,
            *(
                distance
                for step_distances in candidate_distances
                for distance in step_distances
            ),
        )
        nearest_excluded_distance = min(
            (
                distance
                for step_distances in candidate_distances
                for distance in step_distances
                if active_max_ulps < distance <= analysis_max_ulps
            ),
            default=None,
        )
        if nearest_excluded_distance is not None:
            larger_window_entry[nearest_excluded_distance].append(question_id)

    bucket_payload: dict[str, Any] = {}
    for distance in range(active_max_ulps + 1):
        counts = buckets.get(distance, Counter())
        candidate_correct = counts["candidate_correct"]
        reference_correct = counts["reference_correct"]
        bucket_payload[str(distance)] = {
            **{field: counts[field] for field in SCORE_FIELDS},
            "score_numerator_delta": candidate_correct - reference_correct,
        }

    candidate_correct = sum(bool(row.get("correct")) for row in candidates)
    common_reference_correct = sum(
        bool(reference_by_id[row.get("question_id")].get("correct"))
        for row in candidates
    )
    cumulative_larger_replay: dict[str, Any] = {}
    cumulative_rows = 0
    for threshold, question_ids in larger_window_entry.items():
        cumulative_rows += len(question_ids)
        cumulative_larger_replay[str(threshold)] = {
            "stable_rows": len(candidates) - cumulative_rows,
            "replay_rows": cumulative_rows,
        }

    smaller_score_payload: dict[str, Any] = {}
    for threshold, projection in smaller_score_projection.items():
        unresolved_question_ids = projection["unresolved_question_ids"]
        unresolved_rows = len(unresolved_question_ids)
        known_correct = projection["known_correct"]
        known_incorrect = projection["known_incorrect"]
        if known_correct + known_incorrect + unresolved_rows != len(candidates):
            raise AnalysisError(
                f"smaller-window score partition {threshold} is incomplete"
            )
        minimum_correct = known_correct
        maximum_correct = known_correct + unresolved_rows
        smaller_score_payload[str(threshold)] = {
            "known_correct": known_correct,
            "known_incorrect": known_incorrect,
            "unresolved_rows": unresolved_rows,
            "unresolved_question_ids": unresolved_question_ids,
            "unresolved_active_correct": projection["unresolved_active_correct"],
            "correct_if_unresolved_unchanged": (
                known_correct + projection["unresolved_active_correct"]
            ),
            "minimum_correct": minimum_correct,
            "maximum_correct": maximum_correct,
            "reference_correct": common_reference_correct,
            "reference_score_still_possible": (
                minimum_correct <= common_reference_correct <= maximum_correct
            ),
            "projection_exact": unresolved_rows == 0,
            "known_prediction_changes": projection["known_prediction_changes"],
            "reason_counts": dict(sorted(projection["reason_counts"].items())),
        }

    return {
        "schema_version": 3,
        "record_type": "mmlu_pro_inverse_window_analysis",
        "active_max_ulps": active_max_ulps,
        "analysis_max_ulps": analysis_max_ulps,
        "ascii_a_token_id": ascii_a_token_id,
        "inputs": {
            "service_log": str(service_log.resolve()),
            "service_log_sha256": service_log_snapshot.sha256,
            "service_log_bytes": service_log_snapshot.size_bytes,
            "candidate_results": str(candidate_results.resolve()),
            "candidate_results_sha256": candidate_snapshot.sha256,
            "candidate_results_bytes": candidate_snapshot.size_bytes,
            "reference_results": str(reference_results.resolve()),
            "reference_results_sha256": reference_snapshot.sha256,
            "reference_results_bytes": reference_snapshot.size_bytes,
        },
        "leading_request_groups": leading_request_groups,
        "trailing_request_groups": trailing_request_groups,
        "completed_request_groups": len(groups),
        "formal_rows": len(candidates),
        "inflight_topk_markers": inflight_markers,
        "candidate_correct": candidate_correct,
        "common_reference_correct": common_reference_correct,
        "score_numerator_delta": candidate_correct - common_reference_correct,
        "continuation_questions": continuation_questions,
        "observed_topk_max_candidate_ulp_distance": (
            observed_topk_max_candidate_ulp_distance
        ),
        "by_maximum_selected_ulp_distance": bucket_payload,
        "smaller_window_replay": {
            str(threshold): {
                "stable_rows": len(candidates) - len(question_ids),
                "replay_rows": len(question_ids),
                "question_ids": question_ids,
            }
            for threshold, question_ids in threshold_replay.items()
        },
        "smaller_window_score_projection": smaller_score_payload,
        "larger_window_replay": {
            "first_possible_change": {
                str(distance): {
                    "rows": len(question_ids),
                    "question_ids": question_ids,
                }
                for distance, question_ids in larger_window_entry.items()
            },
            "cumulative": cumulative_larger_replay,
        },
        "interpretation": (
            "A row whose every selected token is within a smaller window is "
            "provably unchanged because the active inverse-F32 arbitration chose "
            "the minimum exact logit from a superset that contains that token. "
            "When the active winner leaves a smaller window, a singleton A-J "
            "candidate or multiple A-J candidates with the same correctness "
            "also determine the score without replay; non-answer continuations "
            "and mixed-correctness answer sets remain unresolved. "
            "For a larger threshold, replay the union of first_possible_change "
            "question IDs from active_max_ulps + 1 through that threshold; rows "
            "outside that union have no newly eligible retained-top-k candidate "
            "on their observed generation path. Unresolved smaller-window rows "
            "and every listed larger-window row require measured replay; the "
            "analysis does not infer their alternate-window answers."
        ),
        "formal_acceptance": {
            "eligible": False,
            "reason": (
                "Window analysis is route-selection evidence only. Any selected "
                "global route requires a fresh 12,032-row acceptance run."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-log", type=Path, required=True)
    parser.add_argument("--candidate-results", type=Path, required=True)
    parser.add_argument("--reference-results", type=Path, required=True)
    parser.add_argument("--leading-request-groups", type=int, default=0)
    parser.add_argument("--trailing-request-groups", type=int, default=0)
    parser.add_argument("--active-max-ulps", type=int, required=True)
    parser.add_argument("--analysis-max-ulps", type=int, default=64)
    parser.add_argument("--ascii-a-token-id", type=int, default=32)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = analyze(
            args.service_log,
            args.candidate_results,
            args.reference_results,
            args.leading_request_groups,
            args.active_max_ulps,
            args.ascii_a_token_id,
            args.analysis_max_ulps,
            args.trailing_request_groups,
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
