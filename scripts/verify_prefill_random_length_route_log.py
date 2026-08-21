#!/usr/bin/env python3
"""Verify that every measured random length used the qualified terminal routes."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


class VerificationError(RuntimeError):
    pass


MARKER_PREFIXES = ("BATCH_MARK ", "QRT_SERVER_MARK ")
FIELD_PATTERN = re.compile(r"(?:^|\s)([A-Za-z0-9_]+)=([^\s]+)")
REQUEST_START_MARKER = "qwen36_resident_request_thread_plan_ready"
FULL_ATTENTION_MARKER = "full_attention_ck_compact_bf16"
FIXED_ATTENTION_MARKER = "full_attention_ck_q1_kv8192"
DYNAMIC_ATTENTION_MARKER = "full_attention_ck_q1_dynamic"
GDN_MARKER = "q8192_aiter_fused_gdn_provider"
SMOOTH_TAIL_MARKER = "q8192_triton_selected_moe_full_provider_v2"
EXACT_PREFILL_MARKER = "exact_first_token_prefill"
PREFIX_CACHE_MARKERS = ("prefix_cache_seed", "prefix_cache_hit")
WARMUP_CONTRACT = (
    "one_unmeasured_gb10_checked_q8192_with_server_prefix_cache_disabled"
)
TRITON_MARKERS = (
    SMOOTH_TAIL_MARKER,
    "layer39_q1_triton_0626_routed_backend",
    "q1_terminal_device_corridor_activate",
    "q1_terminal_device_corridor_triton_metadata_upload",
    "q1_terminal_device_corridor_output_activate",
    "q1_terminal_device_corridor_final_norm_activate",
)
FALLBACK_MARKERS = (
    "q1_terminal_device_corridor_fallback",
    "q1_terminal_device_corridor_output_fallback",
    "q1_terminal_device_corridor_final_norm_fallback",
)
EXPECTED_DYNAMIC_FULL_ATTENTION_LAYERS = tuple(range(3, 39, 4))
EXPECTED_DYNAMIC_GDN_LAYERS = tuple(
    layer for layer in range(39) if layer % 4 != 3
)
EXPECTED_DYNAMIC_MOE_LAYERS = tuple(range(39))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_marker_lines(text: str) -> list[dict[str, Any]]:
    markers = []
    for line_number, raw_line in enumerate(text.splitlines(), 1):
        match = next(
            (
                (prefix, raw_line.find(prefix))
                for prefix in MARKER_PREFIXES
                if raw_line.find(prefix) >= 0
            ),
            None,
        )
        if match is None:
            continue
        marker_prefix, marker_offset = match
        line = raw_line[marker_offset:]
        marker_and_fields = line[len(marker_prefix) :]
        marker = marker_and_fields.split(maxsplit=1)[0]
        fields = {
            match.group(1): match.group(2)
            for match in FIELD_PATTERN.finditer(marker_and_fields)
        }
        markers.append(
            {
                "marker": marker,
                "marker_prefix": marker_prefix.strip(),
                "fields": fields,
                "line_number": line_number,
                "line": line,
            }
        )
    return markers


def marker_prefill_tokens(marker: dict[str, Any]) -> int | None:
    fields = marker["fields"]
    value = fields.get("prefill_tokens")
    if value is None and marker["marker"] in (
        FULL_ATTENTION_MARKER,
        FIXED_ATTENTION_MARKER,
        DYNAMIC_ATTENTION_MARKER,
    ):
        value = fields.get("history_tokens")
    if value is None and marker["marker"] == SMOOTH_TAIL_MARKER:
        value = fields.get("selected_tokens")
    if value is None and marker["marker"] == GDN_MARKER:
        value = fields.get("tokens")
    if value is None and marker["marker"] == EXACT_PREFILL_MARKER:
        value = fields.get("input_tokens")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def audit_routes(
    report: dict[str, Any], markers: list[dict[str, Any]]
) -> dict[str, Any]:
    if report.get("record_type") != (
        "qrt_prefill_random_length_plateau_verification"
    ):
        raise VerificationError("input is not a random-length plateau report")
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise VerificationError("random-length report contains no records")
    plateau_report_passed = report.get("passed") is True
    try:
        expected_counts = collections.Counter(
            int(record["prompt_tokens"]) for record in records
        )
        expected_upper_anchors: dict[int, set[int]] = collections.defaultdict(set)
        for record in records:
            expected_upper_anchors[int(record["prompt_tokens"])].add(
                int(record["interval_upper_inclusive"])
            )
    except (KeyError, TypeError, ValueError) as error:
        raise VerificationError(
            "random-length report has an invalid prompt_tokens or interval "
            "upper-anchor field"
        ) from error
    if any(len(anchors) != 1 for anchors in expected_upper_anchors.values()):
        raise VerificationError(
            "a measured prompt length maps to multiple interval upper anchors"
        )

    warmup_request_count = 0
    warmup_record: dict[str, Any] | None = None
    request_policy = report.get("request_policy")
    warmup_contract = (
        request_policy.get("warmup_contract")
        if isinstance(request_policy, dict)
        else None
    )
    if warmup_contract is not None:
        warmup = report.get("warmup")
        expected_warmup = {
            "record_type": "qrt_prefill_random_length_warmup",
            "measured": False,
            "prompt_tokens": 8192,
            "gb10_match": True,
            "server_prefix_cache_enabled": False,
            "passed": True,
        }
        if warmup_contract != WARMUP_CONTRACT or not isinstance(warmup, dict):
            raise VerificationError("random-length warm-up contract differs")
        warmup_mismatches = {
            name: {"expected": value, "actual": warmup.get(name)}
            for name, value in expected_warmup.items()
            if warmup.get(name) != value
        }
        amd_tokens = warmup.get("amd_token_ids")
        gb10_tokens = warmup.get("gb10_token_ids")
        if (
            warmup_mismatches
            or not isinstance(amd_tokens, list)
            or len(amd_tokens) != 1
            or amd_tokens != gb10_tokens
        ):
            raise VerificationError(
                "random-length q8192 warm-up evidence differs: "
                + json.dumps(warmup_mismatches, sort_keys=True)
            )
        expected_counts[8192] += 1
        warmup_request_count = 1
        warmup_record = warmup

    indexed_records = list(enumerate(records))
    if report.get("schema_version") == 2:
        try:
            execution_positions = [
                int(record["execution_position"]) for record in records
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise VerificationError(
                "schema-v2 random-length records lack execution positions"
            ) from error
        if sorted(execution_positions) != list(range(len(records))):
            raise VerificationError(
                "schema-v2 random-length execution positions are not a permutation"
            )
        indexed_records.sort(key=lambda item: int(item[1]["execution_position"]))

    expected_requests: list[dict[str, Any]] = []
    if warmup_record is not None:
        expected_requests.append(
            {
                "kind": "warmup",
                "report_index": None,
                "execution_position": -1,
                "prompt_tokens": 8192,
                "output_token": warmup_record["amd_token_ids"][0],
            }
        )
    for report_index, record in indexed_records:
        amd_token_ids = record.get("amd_token_ids")
        output_token = None
        if (
            isinstance(amd_token_ids, list)
            and len(amd_token_ids) == 1
            and isinstance(amd_token_ids[0], int)
        ):
            output_token = amd_token_ids[0]
        expected_requests.append(
            {
                "kind": "measurement",
                "report_index": report_index,
                "execution_position": record.get(
                    "execution_position", report_index
                ),
                "prompt_tokens": int(record["prompt_tokens"]),
                "output_token": output_token,
            }
        )

    expected_exact_prefills = []
    for report_index, record in indexed_records:
        prompt_tokens = int(record["prompt_tokens"])
        if prompt_tokens >= 8192:
            continue
        amd_token_ids = record.get("amd_token_ids")
        expected_output_token = None
        if isinstance(amd_token_ids, list) and len(amd_token_ids) == 1:
            if isinstance(amd_token_ids[0], int):
                expected_output_token = amd_token_ids[0]
        expected_exact_prefills.append(
            {
                "report_index": report_index,
                "execution_position": record.get(
                    "execution_position", report_index
                ),
                "prompt_tokens": prompt_tokens,
                "output_token": expected_output_token,
            }
        )

    observed: dict[str, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    marker_rows: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for marker in markers:
        marker_rows[marker["marker"]].append(marker)
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens is not None:
            observed[marker["marker"]][prefill_tokens] += 1

    requirements = []
    missing = []
    for prompt_tokens, expected_count in sorted(expected_counts.items()):
        attention_marker = (
            FIXED_ATTENTION_MARKER
            if prompt_tokens == 8192
            else DYNAMIC_ATTENTION_MARKER
        )
        required_markers = [attention_marker, GDN_MARKER, *TRITON_MARKERS]
        if prompt_tokens < 8192:
            required_markers.append(EXACT_PREFILL_MARKER)
        for marker_name in required_markers:
            actual_count = observed[marker_name][prompt_tokens]
            passed = actual_count >= expected_count
            requirement = {
                "prompt_tokens": prompt_tokens,
                "marker": marker_name,
                "minimum_count": expected_count,
                "actual_count": actual_count,
                "passed": passed,
            }
            requirements.append(requirement)
            if not passed:
                missing.append(requirement)

    request_sequence_failures: list[dict[str, Any]] = []
    request_segments: list[dict[str, Any]] = []
    request_start_indices = [
        index
        for index, marker in enumerate(markers)
        if marker["marker"] == REQUEST_START_MARKER
    ]
    if len(request_start_indices) != len(expected_requests):
        request_sequence_failures.append(
            {
                "reason": "request-start marker count differs from the AMD request sequence",
                "expected_count": len(expected_requests),
                "actual_count": len(request_start_indices),
            }
        )

    scoped_route_markers = {
        FULL_ATTENTION_MARKER,
        FIXED_ATTENTION_MARKER,
        DYNAMIC_ATTENTION_MARKER,
        GDN_MARKER,
        *TRITON_MARKERS,
        EXACT_PREFILL_MARKER,
    }
    if request_start_indices:
        unscoped_before_first_request = [
            marker
            for marker in markers[: request_start_indices[0]]
            if marker["marker"] in scoped_route_markers
        ]
        if unscoped_before_first_request:
            request_sequence_failures.append(
                {
                    "reason": "route markers appeared before the first request boundary",
                    "line_numbers": [
                        marker["line_number"]
                        for marker in unscoped_before_first_request
                    ],
                }
            )

    layer_requirements = (
        (FULL_ATTENTION_MARKER, EXPECTED_DYNAMIC_FULL_ATTENTION_LAYERS),
        (GDN_MARKER, EXPECTED_DYNAMIC_GDN_LAYERS),
        (SMOOTH_TAIL_MARKER, EXPECTED_DYNAMIC_MOE_LAYERS),
    )
    single_terminal_markers = tuple(
        marker_name
        for marker_name in TRITON_MARKERS
        if marker_name != SMOOTH_TAIL_MARKER
    )
    for request_index, (expected, start_index) in enumerate(
        zip(expected_requests, request_start_indices)
    ):
        end_index = (
            request_start_indices[request_index + 1]
            if request_index + 1 < len(request_start_indices)
            else len(markers)
        )
        segment = markers[start_index:end_index]
        prompt_tokens = int(expected["prompt_tokens"])
        failures: list[dict[str, Any]] = []

        start_fields = segment[0]["fields"]
        if start_fields.get("pass") != "1":
            failures.append(
                {
                    "marker": REQUEST_START_MARKER,
                    "reason": "request thread-plan boundary did not pass",
                    "line_number": segment[0]["line_number"],
                }
            )

        expected_attention_marker = (
            FIXED_ATTENTION_MARKER
            if prompt_tokens == 8192
            else DYNAMIC_ATTENTION_MARKER
        )
        terminal_attention_rows = [
            marker
            for marker in segment
            if marker["marker"]
            in (FIXED_ATTENTION_MARKER, DYNAMIC_ATTENTION_MARKER)
        ]
        matching_attention_rows = [
            marker
            for marker in terminal_attention_rows
            if marker["marker"] == expected_attention_marker
            and marker_prefill_tokens(marker) == prompt_tokens
        ]
        if len(terminal_attention_rows) != 1 or len(matching_attention_rows) != 1:
            failures.append(
                {
                    "marker": expected_attention_marker,
                    "reason": "request does not contain exactly one matching terminal CK route",
                    "expected_count": 1,
                    "actual_rows": [
                        {
                            "marker": marker["marker"],
                            "prefill_tokens": marker_prefill_tokens(marker),
                            "line_number": marker["line_number"],
                        }
                        for marker in terminal_attention_rows
                    ],
                }
            )

        for marker_name, expected_layers in layer_requirements:
            rows = [
                marker
                for marker in segment
                if marker["marker"] == marker_name
            ]
            malformed_rows = []
            actual_layers = []
            for marker in rows:
                try:
                    layer = int(marker["fields"]["layer"])
                except (KeyError, TypeError, ValueError):
                    malformed_rows.append(marker["line_number"])
                    continue
                if marker_prefill_tokens(marker) != prompt_tokens:
                    malformed_rows.append(marker["line_number"])
                    continue
                actual_layers.append(layer)
            expected_layer_counts = collections.Counter(expected_layers)
            actual_layer_counts = collections.Counter(actual_layers)
            if malformed_rows or actual_layer_counts != expected_layer_counts:
                failures.append(
                    {
                        "marker": marker_name,
                        "reason": "request layer coverage differs from the exact dynamic route",
                        "expected_layers": list(expected_layers),
                        "actual_layers": sorted(actual_layers),
                        "malformed_or_wrong_length_lines": malformed_rows,
                    }
                )

        for marker_name in single_terminal_markers:
            rows = [
                marker
                for marker in segment
                if marker["marker"] == marker_name
            ]
            matching_rows = [
                marker
                for marker in rows
                if marker_prefill_tokens(marker) == prompt_tokens
            ]
            if len(rows) != 1 or len(matching_rows) != 1:
                failures.append(
                    {
                        "marker": marker_name,
                        "reason": "request does not contain exactly one matching terminal corridor marker",
                        "expected_count": 1,
                        "actual_count": len(rows),
                        "matching_length_count": len(matching_rows),
                    }
                )

        exact_prefill_rows = [
            marker
            for marker in segment
            if marker["marker"] == EXACT_PREFILL_MARKER
        ]
        expected_exact_prefill_count = 0 if prompt_tokens == 8192 else 1
        matching_exact_prefill_rows = [
            marker
            for marker in exact_prefill_rows
            if marker_prefill_tokens(marker) == prompt_tokens
        ]
        if (
            len(exact_prefill_rows) != expected_exact_prefill_count
            or len(matching_exact_prefill_rows) != expected_exact_prefill_count
        ):
            failures.append(
                {
                    "marker": EXACT_PREFILL_MARKER,
                    "reason": "request exact-prefill completion boundary differs",
                    "expected_count": expected_exact_prefill_count,
                    "actual_count": len(exact_prefill_rows),
                    "matching_length_count": len(matching_exact_prefill_rows),
                }
            )

        request_segment = {
            **expected,
            "request_index": request_index,
            "start_line_number": segment[0]["line_number"],
            "end_line_number": segment[-1]["line_number"],
            "failures": failures,
            "passed": not failures,
        }
        request_segments.append(request_segment)
        for failure in failures:
            request_sequence_failures.append(
                {
                    "request_index": request_index,
                    "execution_position": expected["execution_position"],
                    "prompt_tokens": prompt_tokens,
                    **failure,
                }
            )

    semantic_failures = []
    for marker in marker_rows[FULL_ATTENTION_MARKER]:
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens not in expected_counts:
            continue
        fields = marker["fields"]
        expected = {
            "target_tokens": str(prefill_tokens),
            "history_tokens": str(prefill_tokens),
            "direct_export": "1",
            "exact_q16384": "0",
            "exact_q32768": "0",
            "exact_q65536": "0",
            "exact_q131_context": "0",
            "exact_arbitrary_dynamic": "1",
            "legacy_prepare": "0",
            "q_dense_bf16": "1",
            "k_inplace_bf16": "1",
            "v_direct_bf16": "1",
            "gated_context_bf16": "1",
            "tail_fused": "1",
        }
        mismatches = {
            name: {"expected": value, "actual": fields.get(name)}
            for name, value in expected.items()
            if fields.get(name) != value
        }
        if mismatches:
            semantic_failures.append(
                {
                    "marker": marker["marker"],
                    "prefill_tokens": prefill_tokens,
                    "line_number": marker["line_number"],
                    "reason": "full attention did not execute the exact dynamic CK route",
                    "mismatches": mismatches,
                }
            )
    for marker_name in (FIXED_ATTENTION_MARKER, DYNAMIC_ATTENTION_MARKER):
        for marker in marker_rows[marker_name]:
            prefill_tokens = marker_prefill_tokens(marker)
            if prefill_tokens not in expected_counts:
                continue
            fields = marker["fields"]
            expected = {
                "target_tokens": "1",
                "history_tokens": str(prefill_tokens),
                "runtime_kv_length": str(prefill_tokens),
                "full_prefix_ck_launch": "0",
                "exact_terminal_correction": "1",
            }
            mismatches = {
                name: {"expected": value, "actual": fields.get(name)}
                for name, value in expected.items()
                if fields.get(name) != value
            }
            if mismatches:
                semantic_failures.append(
                    {
                        "marker": marker["marker"],
                        "prefill_tokens": prefill_tokens,
                        "line_number": marker["line_number"],
                        "reason": "terminal CK attention did not consume the exact request length",
                        "mismatches": mismatches,
                    }
                )
    for marker in marker_rows["q1_terminal_device_corridor_activate"]:
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens not in expected_counts:
            continue
        if marker["fields"].get("backend") != "triton_0626_raw_bf16":
            semantic_failures.append(
                {
                    "marker": marker["marker"],
                    "prefill_tokens": prefill_tokens,
                    "line_number": marker["line_number"],
                    "reason": "terminal corridor did not identify the Triton backend",
                }
            )
    for marker in marker_rows[
        "q1_terminal_device_corridor_triton_metadata_upload"
    ]:
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens not in expected_counts:
            continue
        if (
            marker["fields"].get("global_topk_ids") != "1"
            or marker["fields"].get("topk_weights") != "1"
        ):
            semantic_failures.append(
                {
                    "marker": marker["marker"],
                    "prefill_tokens": prefill_tokens,
                    "line_number": marker["line_number"],
                    "reason": "Triton metadata did not contain global IDs and weights",
                }
            )

    for marker in marker_rows[SMOOTH_TAIL_MARKER]:
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens not in expected_counts:
            continue
        fields = marker["fields"]
        expected = {
            "selected_routes": str(prefill_tokens * 8),
            "provider_tile_tokens": "8192",
            "provider_tile_count": "1",
            "provider_tail_tokens": (
                "0" if prefill_tokens == 8192 else str(prefill_tokens)
            ),
            "provider_tail_padded": "0",
            "smooth_tail_moe": "0",
            "smooth_tail_rounded_tokens": "0",
            "smooth_tail_padding_tokens": "0",
            "smooth_tail_transaction_count": "0",
            "smooth_tail_dense_ceil_provider": "0",
            "dynamic_logical_moe_provider": "1",
            "short_lossless_palette_requested": "0",
            "short_weight_int8_requested": "0",
        }
        mismatches = {
            name: {"expected": value, "actual": fields.get(name)}
            for name, value in expected.items()
            if fields.get(name) != value
        }
        if mismatches:
            semantic_failures.append(
                {
                    "marker": marker["marker"],
                    "prefill_tokens": prefill_tokens,
                    "line_number": marker["line_number"],
                    "reason": "selected-MoE did not execute the exact dynamic logical length",
                    "mismatches": mismatches,
                }
            )

    for marker in marker_rows[GDN_MARKER]:
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens not in expected_counts:
            continue
        fields = marker["fields"]
        expected = {
            "exact_arbitrary_dynamic": "1",
            "fla_chunk_gdn_arithmetic": "0",
            "secondary_fla_chunk_gdn_provider": "0",
            "gdn_input": "native_normalized_postconv",
            "full_sequence_recurrence": "1",
            "independent_q8192_state_merge": "0",
            "async": "1",
        }
        mismatches = {
            name: {"expected": value, "actual": fields.get(name)}
            for name, value in expected.items()
            if fields.get(name) != value
        }
        if mismatches:
            semantic_failures.append(
                {
                    "marker": marker["marker"],
                    "prefill_tokens": prefill_tokens,
                    "line_number": marker["line_number"],
                    "reason": "fused GDN did not execute the exact dynamic full-sequence route",
                    "mismatches": mismatches,
                }
            )

    for marker in marker_rows[EXACT_PREFILL_MARKER]:
        prefill_tokens = marker_prefill_tokens(marker)
        if prefill_tokens not in expected_counts or prefill_tokens == 8192:
            continue
        fields = marker["fields"]
        expected = {
            "input_tokens": str(prefill_tokens),
            "verifier_input_tokens": str(prefill_tokens),
            "requested_output_tokens": "1",
            "classifier": "0",
            "resident_prefix_mutated": "0",
        }
        mismatches = {
            name: {"expected": value, "actual": fields.get(name)}
            for name, value in expected.items()
            if fields.get(name) != value
        }
        if mismatches:
            semantic_failures.append(
                {
                    "marker": marker["marker"],
                    "prefill_tokens": prefill_tokens,
                    "line_number": marker["line_number"],
                    "reason": (
                        "server did not execute an uncached exact first-token "
                        "prefill for the complete request"
                    ),
                    "mismatches": mismatches,
                }
            )

    observed_exact_prefills = marker_rows[EXACT_PREFILL_MARKER]
    exact_prefill_sequence_failures = []
    if len(observed_exact_prefills) != len(expected_exact_prefills):
        exact_prefill_sequence_failures.append(
            {
                "reason": "exact-prefill marker count differs from request sequence",
                "expected_count": len(expected_exact_prefills),
                "actual_count": len(observed_exact_prefills),
            }
        )
    for sequence_index, (expected, observed) in enumerate(
        zip(expected_exact_prefills, observed_exact_prefills)
    ):
        fields = observed["fields"]
        expected_fields = {
            "input_tokens": str(expected["prompt_tokens"]),
        }
        if expected["output_token"] is not None:
            expected_fields["output_token"] = str(expected["output_token"])
        mismatches = {
            name: {"expected": value, "actual": fields.get(name)}
            for name, value in expected_fields.items()
            if fields.get(name) != value
        }
        if mismatches:
            exact_prefill_sequence_failures.append(
                {
                    "reason": "exact-prefill marker does not match request order",
                    "sequence_index": sequence_index,
                    "execution_position": expected["execution_position"],
                    "line_number": observed["line_number"],
                    "mismatches": mismatches,
                }
            )

    fallbacks = []
    for marker_name in FALLBACK_MARKERS:
        for marker in marker_rows[marker_name]:
            prefill_tokens = marker_prefill_tokens(marker)
            if prefill_tokens is None or prefill_tokens in expected_counts:
                fallbacks.append(
                    {
                        "marker": marker_name,
                        "prefill_tokens": prefill_tokens,
                        "line_number": marker["line_number"],
                        "reason": marker["fields"].get("reason"),
                    }
                )

    cold_cache_violations = [
        {
            "marker": marker_name,
            "line_number": marker["line_number"],
            "line": marker["line"],
        }
        for marker_name in PREFIX_CACHE_MARKERS
        for marker in marker_rows[marker_name]
    ]

    passed = (
        plateau_report_passed
        and not missing
        and not request_sequence_failures
        and not semantic_failures
        and not exact_prefill_sequence_failures
        and not fallbacks
        and not cold_cache_violations
    )
    return {
        "plateau_report_passed": plateau_report_passed,
        "expected_request_count": sum(expected_counts.values()),
        "expected_measurement_request_count": len(records),
        "expected_warmup_request_count": warmup_request_count,
        "expected_prompt_lengths": sorted(expected_counts),
        "request_start_marker": REQUEST_START_MARKER,
        "request_segment_count": len(request_segments),
        "request_segments": request_segments,
        "request_sequence_failures": request_sequence_failures,
        "requirements": requirements,
        "missing_requirements": missing,
        "semantic_failures": semantic_failures,
        "exact_prefill_sequence_failures": exact_prefill_sequence_failures,
        "fallbacks": fallbacks,
        "cold_cache_violations": cold_cache_violations,
        "passed": passed,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plateau-report", type=Path, required=True)
    parser.add_argument("--service-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.output.exists() and not args.force:
        raise VerificationError(f"refusing to overwrite {args.output}")
    try:
        report = json.loads(args.plateau_report.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"could not read plateau report: {error}") from error
    try:
        log_text = args.service_log.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise VerificationError(f"could not read service log: {error}") from error
    audit = audit_routes(report, parse_marker_lines(log_text))
    output = {
        "schema_version": 1,
        "record_type": "qrt_prefill_random_length_route_log_verification",
        "source_commit": report.get("source_commit", "unknown"),
        "plateau_report": str(args.plateau_report),
        "plateau_report_sha256": sha256_file(args.plateau_report),
        "service_log": str(args.service_log),
        "service_log_sha256": sha256_file(args.service_log),
        **audit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return 0 if output["passed"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
