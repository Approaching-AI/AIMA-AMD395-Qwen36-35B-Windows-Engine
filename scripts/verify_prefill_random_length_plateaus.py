#!/usr/bin/env python3
"""Compare random cold-prefill lengths with each interval's upper anchor."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import random
import re
import socket
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from verify_prefill_length_smoothness import (
    VerificationError,
    request_completion,
    utc_now,
)


DEFAULT_BOUNDARIES = (
    2048,
    2560,
    3072,
    3328,
    3584,
    4096,
    4608,
    5120,
    5632,
    6144,
    6656,
    7168,
    7680,
    8192,
)
QWEN36_VOCAB_SIZE = 248_320
RETAINED_PREFILL_TOKENS_PER_SECOND = 1506.407
RETAINED_Q8192_TTFT_MAX_MS = 4187.416
RETAINED_DECODE_TOKENS_PER_SECOND = 28.168
RETAINED_TPOT_MAX_MS = 35.502
RETAINED_RANDOM_TO_ANCHOR_RATIO_MIN = 0.97
RETAINED_RANDOM_TO_ANCHOR_RATIO_MAX = 1.03
RETAINED_ANCHOR_MAX_TO_MIN_RATIO = 1.03
MINIMUM_RANDOM_CASES_PER_INTERVAL = 6
INTEGER_SPIKE_ALIGNMENT = 64
EDGE_ALIGNMENT_PROBE_COUNT = 5
WARMUP_CONTRACT = (
    "one_unmeasured_gb10_checked_q8192_with_server_prefix_cache_disabled"
)
Q8192_MOE_EXPECTED_ARTIFACTS = (
    "qrt_triton_moe_q8192_provider.dll",
    "metadata.json",
    "q8192_selected_moe_route_count.hsaco",
    "q8192_selected_moe_route_prefix_by_program.hsaco",
    "q8192_selected_moe_route_padded_prefix.hsaco",
    "q8192_selected_moe_route_scatter.hsaco",
    "q8192_selected_moe_gate_up_silu.hsaco",
    "q8192_selected_moe_down.hsaco",
)
Q8192_MOE_NATIVE_SHARED_AUXILIARY_ARTIFACTS = (
    "q8192_triton_0626_row_major_sorted_conditional_exact_gate_rows256.hsaco",
    "q8192_triton_0626_zero_correction_gate_finalize.hsaco",
    "q8192_triton_0626_conditional_exact_down_rows4.hsaco",
)
Q8192_MOE_NATIVE_SHARED_REQUIRED_RUNTIME_ENVIRONMENT = (
    ("QRT_QWEN36_Q8192_ROUTER_HIPBLASLT_BF16", "1"),
    ("QRT_QWEN36_CUDA_VLLM_ROUTER_HAWKEYE_MIDPOINT_RADIUS", "173"),
    ("QRT_QWEN36_CUDA_VLLM_SHARED_HAWKEYE_MIDPOINT_RADIUS", "0"),
    ("QRT_QWEN36_Q8192_VLLM_BF16_RESIDUAL_CARRIER", "1"),
    ("QRT_QWEN36_EXACT_ARBITRARY_VLLM_SPLIT_VARIANCE", "1"),
    ("QRT_QWEN36_Q8192_VLLM_SORTED_BF16_ROUTE_SUM", "1"),
    ("QRT_QWEN36_CUDA_VLLM_MOE_HAWKEYE_MIDPOINT_RADIUS", "0"),
    ("QRT_QWEN36_CUDA_VLLM_MOE_UP_HAWKEYE_MIDPOINT_RADIUS", "0"),
    (
        "QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_CONTRIBUTION_"
        "HAWKEYE_MIDPOINT_RADIUS",
        "0",
    ),
    (
        "QRT_QWEN36_CUDA_VLLM_ROUTED_GATE_HAWKEYE_LOW_EXPONENT_THRESHOLD",
        "0",
    ),
    (
        "QRT_QWEN36_CUDA_VLLM_ROUTED_UP_HAWKEYE_LOW_EXPONENT_THRESHOLD",
        "0",
    ),
    (
        "QRT_QWEN36_CUDA_VLLM_ROUTED_DOWN_HAWKEYE_LOW_EXPONENT_THRESHOLD",
        "0",
    ),
)
GDN_EXPECTED_ARTIFACTS = (
    "qrt_aiter_fused_gdn_q8192_provider.dll",
    "q1024_seeded_aiter_fused_gdn.hsaco",
    "q8192_aiter_fused_gdn.hsaco",
    "q32768_seeded_aiter_fused_gdn_bf16.hsaco",
    "q262144_aiter_fused_gdn_bf16.hsaco",
)
MODEL_ENGINE_LOAD_MAX_MS = 30_000.0
GB10_Q8192_FIRST_TOKEN = 144
GB10_Q8192_RAW_LOGIT_TOLERANCE = 0.125
GB10_Q8192_CONTINUATION = (
    144,
    255,
    82,
    57,
    79,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
    220,
    220,
    196,
)
GB10_Q8192_CONTINUATION_SHA256 = (
    "97d1c7e3ae51a6aa3d55e41d38e410c3995c9dc017a133d8d1533ad3cc3ea9b4"
)
GB10_Q8192_CONTINUATION_FNV1A64 = "e60389077f247d2c"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GB10_Q8192_ORACLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "contracts"
    / "hprefill_q8192_gb10_oracle.json"
)
GB10_Q8192_CONTINUATION_CAPTURE_PATH = (
    Path(__file__).resolve().parent / "capture_gb10_q8192_continuation.py"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_gb10_q8192_oracle() -> tuple[dict[str, Any], str]:
    try:
        oracle = json.loads(GB10_Q8192_ORACLE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VerificationError(f"could not read q8192 GB10 oracle: {error}") from error
    try:
        identity = (
            oracle["schema_version"],
            oracle["record_type"],
            oracle["status"],
            oracle["correctness_authority"]["host"],
            oracle["correctness_authority"]["model"],
            oracle["prompt"]["token_count"],
            oracle["prompt"]["u32le_sha256"],
            oracle["prompt"]["u32le_fnv1a64"],
            oracle["expected"]["first_token_id"],
            oracle["expected"]["first_token_raw_logit"],
            oracle["expected"]["first_token_raw_logit_tolerance"],
            oracle["capture"]["output_token_ids"],
            oracle["capture"]["base_lm_head_capture"]["argmax_token_id"],
            oracle["capture"]["base_lm_head_capture"][
                "selected_token_raw_logit_bf16"
            ],
        )
    except (KeyError, TypeError) as error:
        raise VerificationError(f"q8192 GB10 oracle is incomplete: {error}") from error
    expected_identity = (
        1,
        "qrt_hprefill_q8192_gb10_oracle",
        "pass",
        "gb10-4t",
        "qwen3.6-35b-a3b",
        8192,
        "dda20edc609f935f34d3d41ca4a84ffefa66726676756fe26ff3bef4fbff0b96",
        "1584e34d56e5d78b",
        GB10_Q8192_FIRST_TOKEN,
        10.375,
        GB10_Q8192_RAW_LOGIT_TOLERANCE,
        [GB10_Q8192_FIRST_TOKEN],
        GB10_Q8192_FIRST_TOKEN,
        10.375,
    )
    if identity != expected_identity:
        raise VerificationError("q8192 GB10 oracle identity or raw logit differs")
    expected = oracle.get("expected", {})
    continuation_capture = oracle.get("continuation_capture", {})
    diagnostic_policy = oracle.get("diagnostic_policy", {})
    continuation_identity = (
        expected.get("continuation_comparison"),
        expected.get("continuation_token_count"),
        expected.get("decode_step_count"),
        expected.get("output_token_ids_u32le_sha256"),
        expected.get("output_token_ids_u32le_fnv1a64"),
        tuple(expected.get("output_token_ids", ())),
        continuation_capture.get("schema_version"),
        continuation_capture.get("record_type"),
        continuation_capture.get("authority", {}).get("host"),
        continuation_capture.get("authority", {}).get("model"),
        continuation_capture.get("authority", {}).get("model_reference"),
        continuation_capture.get("authority", {}).get("dtype"),
        continuation_capture.get("capture_hook_armed"),
        continuation_capture.get("request_policy", {}).get("max_tokens"),
        continuation_capture.get("request_policy", {}).get("temperature"),
        continuation_capture.get("request_policy", {}).get("top_p"),
        continuation_capture.get("request_policy", {}).get("ignore_eos"),
        continuation_capture.get("request_policy", {}).get("return_token_ids"),
        continuation_capture.get("http_status"),
        continuation_capture.get("usage", {}).get("prompt_tokens"),
        continuation_capture.get("usage", {}).get("completion_tokens"),
        continuation_capture.get("output_token_ids_u32le_sha256"),
        continuation_capture.get("output_token_ids_u32le_fnv1a64"),
        tuple(continuation_capture.get("output_token_ids", ())),
        continuation_capture.get("reproducibility", {}).get("request_count"),
        continuation_capture.get("reproducibility", {}).get(
            "token_for_token_repeat_pass"
        ),
        continuation_capture.get("reproducibility", {}).get(
            "repeat_output_token_ids_u32le_sha256"
        ),
        continuation_capture.get("reproducibility", {}).get(
            "repeat_output_token_ids_u32le_fnv1a64"
        ),
        diagnostic_policy.get("continuation_captured_in_this_contract"),
        diagnostic_policy.get(
            "decode_and_prefix_continuation_correctness_active"
        ),
    )
    expected_continuation_identity = (
        "token_for_token",
        32,
        31,
        GB10_Q8192_CONTINUATION_SHA256,
        GB10_Q8192_CONTINUATION_FNV1A64,
        GB10_Q8192_CONTINUATION,
        1,
        "gb10_q8192_continuation_capture",
        "gb10-4t",
        "qwen3.6-35b-a3b",
        "/models",
        "bfloat16",
        False,
        32,
        0,
        1,
        True,
        True,
        200,
        8192,
        32,
        GB10_Q8192_CONTINUATION_SHA256,
        GB10_Q8192_CONTINUATION_FNV1A64,
        GB10_Q8192_CONTINUATION,
        2,
        True,
        GB10_Q8192_CONTINUATION_SHA256,
        GB10_Q8192_CONTINUATION_FNV1A64,
        True,
        True,
    )
    if continuation_identity != expected_continuation_identity:
        raise VerificationError("q8192 GB10 continuation oracle differs")
    if (
        continuation_capture.get("command_file")
        != "scripts/capture_gb10_q8192_continuation.py"
        or continuation_capture.get("command_file_sha256")
        != sha256_file(GB10_Q8192_CONTINUATION_CAPTURE_PATH)
        or continuation_capture.get("reproducibility", {}).get(
            "repeat_request_sha256"
        )
        != continuation_capture.get("request_sha256")
    ):
        raise VerificationError("q8192 GB10 continuation capture provenance differs")
    for name in (
        "request_sha256",
        "response_sha256",
        "command_file_sha256",
    ):
        value = oracle["capture"].get(name)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise VerificationError(f"q8192 GB10 oracle has invalid hash: {name}")
    for name in ("request_sha256", "response_sha256", "command_file_sha256"):
        value = continuation_capture.get(name)
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise VerificationError(
                f"q8192 GB10 continuation capture has invalid hash: {name}"
            )
    repeat_response = continuation_capture.get("reproducibility", {}).get(
        "repeat_response_sha256"
    )
    if (
        not isinstance(repeat_response, str)
        or SHA256_PATTERN.fullmatch(repeat_response) is None
    ):
        raise VerificationError(
            "q8192 GB10 continuation capture has invalid repeat response hash"
        )
    return oracle, sha256_file(GB10_Q8192_ORACLE_PATH)


def validate_amd_service_record(
    record: dict[str, Any],
    *,
    amd_host: str,
    repo_commit: str,
    model_path: str,
    model_id: str,
) -> dict[str, Any]:
    gb10_oracle, gb10_oracle_sha256 = load_gb10_q8192_oracle()
    native_shared_selected = record.get(
        "dynamic_moe_native_shared_selected", False
    )
    if not isinstance(native_shared_selected, bool):
        raise VerificationError(
            "AMD service record has invalid native-shared selection"
        )
    required_contract = {
        "record_type": "qrt_random_length_terminal_service_start",
        "host": amd_host,
        "repo_commit": repo_commit,
        "model_path": model_path,
        "model_id": model_id,
        "gdn_route": "aiter",
        "engine_self_hashes_diagnostic_only": True,
        "q8192_correctness_pass": True,
        "q8192_performance_pass": True,
        "q8192_model_engine_load_pass": True,
        "q8192_route_pass": True,
        "q8192_moe_fixed_route_pass": True,
        "q8192_continuation_token_for_token_pass": True,
        "q8192_decode_route_pass": True,
        "q8192_prefix_continuation_pass": True,
        "q8192_prefix_continuation_correctness_pass": True,
        "q8192_prefix_continuation_performance_pass": True,
        "q8192_prefix_route_pass": True,
        "model_engine_load_pass": True,
        "dynamic_logical_moe_provider": True,
        "q8192_moe_dynamic_logical_component_pass": True,
        "q8192_moe_dynamic_logical_q8192_exact_pass": True,
        "q8192_moe_expected_full_provider_hash_diagnostic_only": True,
        "q8192_moe_dynamic_logical_timing_samples": 3,
        "q8192_moe_dynamic_logical_timing_stat": "median",
        "q8192_moe_dynamic_logical_nonfinite": 0,
        "q8192_moe_dynamic_logical_component_only": True,
        "q8192_moe_dynamic_logical_reset_included": False,
        "attention_dynamic_logical_case_count": 32,
        "attention_dynamic_logical_reset_included": False,
        "attention_dynamic_logical_component_only": True,
        "gdn_dynamic_logical_case_count": 32,
        "gdn_dynamic_logical_mode_count": 64,
        "gdn_dynamic_logical_reset_included": False,
        "gdn_dynamic_logical_component_only": True,
        "q8192_gb10_oracle_sha256": gb10_oracle_sha256,
        "q8192_gb10_oracle_capture_request_sha256": gb10_oracle["capture"][
            "request_sha256"
        ],
        "q8192_gb10_oracle_capture_response_sha256": gb10_oracle["capture"][
            "response_sha256"
        ],
        "q8192_gb10_oracle_capture_command_file_sha256": gb10_oracle["capture"][
            "command_file_sha256"
        ],
        "q8192_gb10_continuation_capture_request_sha256": gb10_oracle[
            "continuation_capture"
        ]["request_sha256"],
        "q8192_gb10_continuation_capture_response_sha256": gb10_oracle[
            "continuation_capture"
        ]["response_sha256"],
        "q8192_gb10_continuation_capture_command_file_sha256": gb10_oracle[
            "continuation_capture"
        ]["command_file_sha256"],
        "q8192_expected_output_token_ids_u32le_sha256": (
            GB10_Q8192_CONTINUATION_SHA256
        ),
        "q8192_expected_output_token_ids_u32le_fnv1a64": (
            GB10_Q8192_CONTINUATION_FNV1A64
        ),
        "q8192_continuation_token_count": 32,
        "q8192_decode_step_count": 31,
        "q8192_prefix_tokens": 8191,
        "q8192_prefix_suffix_tokens": 1,
        "q8192_prefix_hit_count": 2,
        "server_prefix_cache_enabled": False,
        "exact_first_token_prefill": True,
        "exact_letter_classifier": False,
        "cold_prefix_contract": (
            "server_prefix_cache_disabled_and_globally_unique_first_token_id"
        ),
        "dynamic_moe_native_shared_selected": native_shared_selected,
    }
    mismatches = {}
    for name, expected in required_contract.items():
        actual = (
            record.get(name, False)
            if name == "dynamic_moe_native_shared_selected"
            else record.get(name)
        )
        normalized_actual = (
            str(actual).casefold() if name == "host" else actual
        )
        normalized_expected = (
            str(expected).casefold() if name == "host" else expected
        )
        if normalized_actual != normalized_expected:
            mismatches[name] = {"expected": expected, "actual": actual}
    if mismatches:
        raise VerificationError(
            "AMD service record does not match the qualified run: "
            + json.dumps(mismatches, sort_keys=True)
        )

    def finite_number(name: str) -> float:
        value = record.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise VerificationError(
                f"AMD service record has invalid numeric field: {name}"
            )
        result = float(value)
        if not math.isfinite(result):
            raise VerificationError(
                f"AMD service record has non-finite numeric field: {name}"
            )
        return result

    service_load_ms = finite_number("model_engine_load_ms")
    service_load_max_ms = finite_number("model_engine_load_max_ms")
    q8192_load_ms = finite_number("q8192_model_engine_load_ms")
    q8192_speed = finite_number("q8192_prefill_tokens_per_second")
    q8192_metric_relative_error = finite_number(
        "q8192_prefill_metric_relative_error"
    )
    q8192_metric_relative_error_max = finite_number(
        "q8192_prefill_metric_relative_error_max"
    )
    q8192_speed_min = finite_number("q8192_prefill_min_tokens_per_second")
    q8192_ttft_ms = finite_number("q8192_ttft_ms")
    q8192_ttft_max_ms = finite_number("q8192_ttft_max_ms")
    q8192_decode_speed = finite_number("q8192_decode_tokens_per_second")
    q8192_decode_speed_min = finite_number(
        "q8192_decode_min_tokens_per_second"
    )
    q8192_tpot_ms = finite_number("q8192_tpot_ms")
    q8192_tpot_max_ms = finite_number("q8192_tpot_max_ms")
    q8192_prefix_decode_speed = finite_number(
        "q8192_prefix_decode_tokens_per_second"
    )
    q8192_prefix_tpot_ms = finite_number("q8192_prefix_tpot_ms")
    q8192_prefix_load_ms = finite_number(
        "q8192_prefix_model_engine_load_ms"
    )
    q8192_logit_difference = finite_number(
        "q8192_first_token_raw_logit_absolute_difference"
    )
    q8192_gb10_raw_logit = finite_number("q8192_first_token_raw_logit")
    q8192_logit_tolerance = finite_number(
        "q8192_first_token_raw_logit_tolerance"
    )
    numerical_failures = []
    if not 0.0 <= service_load_ms <= MODEL_ENGINE_LOAD_MAX_MS:
        numerical_failures.append("final service load exceeds 30000 ms")
    if not 0.0 < service_load_max_ms <= MODEL_ENGINE_LOAD_MAX_MS:
        numerical_failures.append("final service load limit was relaxed")
    if not 0.0 <= q8192_load_ms <= MODEL_ENGINE_LOAD_MAX_MS:
        numerical_failures.append("q8192 product load exceeds 30000 ms")
    if q8192_speed < RETAINED_PREFILL_TOKENS_PER_SECOND:
        numerical_failures.append("q8192 product throughput is below retained")
    derived_q8192_speed = (
        8192.0 * 1000.0 / q8192_ttft_ms
        if q8192_ttft_ms > 0.0
        else math.inf
    )
    if not math.isclose(q8192_speed, derived_q8192_speed, rel_tol=0.001):
        numerical_failures.append("q8192 throughput and TTFT are inconsistent")
    if not 0.0 <= q8192_metric_relative_error <= 0.001:
        numerical_failures.append("q8192 performance metric error is too large")
    if not 0.0 < q8192_metric_relative_error_max <= 0.001:
        numerical_failures.append("q8192 performance metric error limit was relaxed")
    if q8192_speed_min < RETAINED_PREFILL_TOKENS_PER_SECOND:
        numerical_failures.append("q8192 throughput limit was relaxed")
    if not 0.0 < q8192_ttft_ms <= RETAINED_Q8192_TTFT_MAX_MS:
        numerical_failures.append("q8192 product TTFT exceeds retained")
    if not 0.0 < q8192_ttft_max_ms <= RETAINED_Q8192_TTFT_MAX_MS:
        numerical_failures.append("q8192 TTFT limit was relaxed")
    if q8192_decode_speed < RETAINED_DECODE_TOKENS_PER_SECOND:
        numerical_failures.append("q8192 decode throughput is below retained")
    if q8192_decode_speed_min < RETAINED_DECODE_TOKENS_PER_SECOND:
        numerical_failures.append("q8192 decode throughput limit was relaxed")
    if not 0.0 < q8192_tpot_ms <= RETAINED_TPOT_MAX_MS:
        numerical_failures.append("q8192 TPOT exceeds retained")
    if not 0.0 < q8192_tpot_max_ms <= RETAINED_TPOT_MAX_MS:
        numerical_failures.append("q8192 TPOT limit was relaxed")
    if q8192_prefix_decode_speed < RETAINED_DECODE_TOKENS_PER_SECOND:
        numerical_failures.append("q8192 prefix decode throughput is below retained")
    if not 0.0 < q8192_prefix_tpot_ms <= RETAINED_TPOT_MAX_MS:
        numerical_failures.append("q8192 prefix TPOT exceeds retained")
    if not 0.0 <= q8192_prefix_load_ms <= MODEL_ENGINE_LOAD_MAX_MS:
        numerical_failures.append("q8192 prefix product load exceeds 30000 ms")
    if record.get("q8192_first_token") != gb10_oracle["expected"][
        "first_token_id"
    ]:
        numerical_failures.append("q8192 first token is not the gb10 oracle")
    if q8192_gb10_raw_logit != gb10_oracle["expected"][
        "first_token_raw_logit"
    ]:
        numerical_failures.append("q8192 raw logit is not the captured gb10 oracle")
    if not 0.0 <= q8192_logit_difference <= q8192_logit_tolerance:
        numerical_failures.append("q8192 raw-logit difference exceeds tolerance")
    if not 0.0 < q8192_logit_tolerance <= GB10_Q8192_RAW_LOGIT_TOLERANCE:
        numerical_failures.append("q8192 raw-logit tolerance was relaxed")
    if numerical_failures:
        raise VerificationError(
            "AMD service record fails retained numerical evidence: "
            + "; ".join(numerical_failures)
        )

    hash_fields = (
        "q8192_product_record_sha256",
        "qualified_environment_sha256",
        "service_environment_sha256",
        "q8192_moe_build_provenance_sha256",
        "q8192_moe_provider_sha256",
        "whole_provider_sha256",
        "attention_provider_sha256",
        "attention_build_provenance_sha256",
        "gdn_build_provenance_sha256",
        "gdn_provider_sha256",
        "q8192_gb10_oracle_sha256",
        "q8192_gb10_oracle_capture_request_sha256",
        "q8192_gb10_oracle_capture_response_sha256",
        "q8192_gb10_oracle_capture_command_file_sha256",
        "q8192_gb10_continuation_capture_request_sha256",
        "q8192_gb10_continuation_capture_response_sha256",
        "q8192_gb10_continuation_capture_command_file_sha256",
    )
    invalid_hashes = [
        name
        for name in hash_fields
        if not isinstance(record.get(name), str)
        or SHA256_PATTERN.fullmatch(record[name]) is None
    ]
    if invalid_hashes:
        raise VerificationError(
            "AMD service record has invalid SHA256 fields: "
            + ", ".join(invalid_hashes)
        )

    if record.get("q8192_moe_dynamic_logical_case_count") != 32:
        raise VerificationError(
            "AMD service record does not bind all dynamic logical component probes"
        )
    expected_runtime_environment = dict(
        Q8192_MOE_NATIVE_SHARED_REQUIRED_RUNTIME_ENVIRONMENT
        if native_shared_selected
        else ()
    )
    runtime_environment = record.get(
        "q8192_moe_required_runtime_environment", []
    )
    if (
        not isinstance(runtime_environment, list)
        or len(runtime_environment) != len(expected_runtime_environment)
    ):
        raise VerificationError(
            "AMD service record has incomplete selected-MoE runtime environment"
        )
    recorded_runtime_environment: dict[str, str] = {}
    for entry in runtime_environment:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("name"), str)
            or not isinstance(entry.get("value"), str)
            or entry["name"] in recorded_runtime_environment
        ):
            raise VerificationError(
                "AMD service record has invalid selected-MoE runtime environment"
            )
        recorded_runtime_environment[entry["name"]] = entry["value"]
    if recorded_runtime_environment != expected_runtime_environment:
        raise VerificationError(
            "AMD service record selected-MoE runtime environment differs"
        )
    if native_shared_selected:
        expected_base_aot = {
            "dynamic_moe_base_aot_mode": "generated_from_current_source",
            "dynamic_moe_base_aot_block_m": 64,
            "dynamic_moe_base_aot_group_m": 1,
            "dynamic_moe_base_aot_dynamic_logical_abi": True,
        }
        base_aot_mismatches = {
            name: {"expected": expected, "actual": record.get(name)}
            for name, expected in expected_base_aot.items()
            if record.get(name) != expected
        }
        if base_aot_mismatches:
            raise VerificationError(
                "AMD service record selected-MoE base AOT differs: "
                + json.dumps(base_aot_mismatches, sort_keys=True)
            )
    q8192_moe_environment = record.get("q8192_moe_environment", [])
    if native_shared_selected and (
        not isinstance(q8192_moe_environment, list)
        or not set(expected_runtime_environment).issubset(q8192_moe_environment)
    ):
        raise VerificationError(
            "AMD service record does not activate selected-MoE runtime environment"
        )
    attention_provenance = record.get("attention_build_provenance")
    if not isinstance(attention_provenance, str) or not attention_provenance:
        raise VerificationError(
            "AMD service record lacks CK attention build provenance"
        )
    for name in ("gdn_build_provenance", "gdn_provider", "gdn_kernel_dir"):
        value = record.get(name)
        if not isinstance(value, str) or not value:
            raise VerificationError(
                f"AMD service record lacks AITER GDN binding: {name}"
            )
    gdn_artifacts = record.get("gdn_artifacts")
    if not isinstance(gdn_artifacts, list) or len(gdn_artifacts) != 5:
        raise VerificationError(
            "AMD service record has incomplete AITER GDN artifact evidence"
        )
    for artifact in gdn_artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("path"), str)
            or not artifact["path"]
            or not isinstance(artifact.get("file"), str)
            or not artifact["file"]
            or not isinstance(artifact.get("bytes"), int)
            or artifact["bytes"] <= 0
            or SHA256_PATTERN.fullmatch(str(artifact.get("sha256", "")))
            is None
        ):
            raise VerificationError(
                "AMD service record has an invalid AITER GDN artifact binding"
            )
    gdn_artifact_names = [artifact["file"] for artifact in gdn_artifacts]
    if sorted(gdn_artifact_names) != sorted(GDN_EXPECTED_ARTIFACTS):
        raise VerificationError(
            "AMD service record has unexpected AITER GDN artifact names"
        )
    gdn_artifact_hashes = {
        artifact["file"]: artifact["sha256"] for artifact in gdn_artifacts
    }
    if gdn_artifact_hashes["qrt_aiter_fused_gdn_q8192_provider.dll"] != (
        record.get("gdn_provider_sha256")
    ):
        raise VerificationError(
            "AMD service record AITER GDN provider hash is not artifact-bound"
        )
    expected_moe_artifacts = Q8192_MOE_EXPECTED_ARTIFACTS + (
        Q8192_MOE_NATIVE_SHARED_AUXILIARY_ARTIFACTS
        if native_shared_selected
        else ()
    )
    artifacts = record.get("q8192_moe_artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(
        expected_moe_artifacts
    ):
        raise VerificationError(
            "AMD service record has incomplete q8192 MoE artifact evidence"
        )
    artifact_names = []
    artifact_hashes: dict[str, str] = {}
    for artifact in artifacts:
        if (
            not isinstance(artifact, dict)
            or not isinstance(artifact.get("file"), str)
            or not artifact["file"]
            or not isinstance(artifact.get("bytes"), int)
            or artifact["bytes"] <= 0
            or SHA256_PATTERN.fullmatch(str(artifact.get("sha256", "")))
            is None
        ):
            raise VerificationError(
                "AMD service record has an invalid q8192 MoE artifact binding"
            )
        artifact_names.append(artifact["file"])
        artifact_hashes[artifact["file"]] = artifact["sha256"]
    if sorted(artifact_names) != sorted(expected_moe_artifacts):
        raise VerificationError(
            "AMD service record q8192 MoE artifact inventory differs"
        )
    if artifact_hashes["qrt_triton_moe_q8192_provider.dll"] != record.get(
        "q8192_moe_provider_sha256"
    ):
        raise VerificationError(
            "AMD service record q8192 MoE provider hash is not artifact-bound"
        )
    command = record.get("command")
    if not isinstance(command, list) or not command:
        raise VerificationError("AMD service record has no exact command")
    command_text = " ".join(str(value) for value in command).casefold()
    if model_path.casefold() not in command_text or "qrt.exe" not in command_text:
        raise VerificationError(
            "AMD service record command does not identify runtime and model"
        )
    return required_contract


def choose_random_lengths(
    lower: int,
    upper: int,
    count: int,
    generator: random.Random,
) -> list[int]:
    if count < MINIMUM_RANDOM_CASES_PER_INTERVAL:
        raise VerificationError(
            f"at least {MINIMUM_RANDOM_CASES_PER_INTERVAL} lengths are "
            "required to cover boundary and integer-alignment probes"
        )
    candidates = list(range(lower + 1, upper))
    aligned_candidates = [
        value
        for value in candidates
        if value % INTEGER_SPIKE_ALIGNMENT == 0
        and lower + 2 <= value <= upper - 2
    ]
    if not aligned_candidates:
        raise VerificationError(
            f"interval ({lower}, {upper}] contains no interior "
            f"{INTEGER_SPIKE_ALIGNMENT}-token alignment probe"
        )
    if count > len(candidates):
        raise VerificationError(
            f"interval ({lower}, {upper}] has only {len(candidates)} "
            f"eligible random lengths, fewer than requested {count}"
        )
    aligned = generator.choice(aligned_candidates)
    selected = {
        lower + 1,
        aligned - 1,
        aligned,
        aligned + 1,
        upper - 1,
    }
    if len(selected) != EDGE_ALIGNMENT_PROBE_COUNT:
        raise VerificationError(
            f"interval ({lower}, {upper}] cannot provide distinct boundary "
            "and integer-alignment probes"
        )
    remaining = [value for value in candidates if value not in selected]
    generic_candidates = [
        value
        for value in remaining
        if value % INTEGER_SPIKE_ALIGNMENT not in (0, 1, 63)
    ]
    if not generic_candidates:
        raise VerificationError(
            f"interval ({lower}, {upper}] contains no ordinary unaligned "
            "random-length probe"
        )
    selected.add(generator.choice(generic_candidates))
    remaining = [value for value in remaining if value not in selected]
    selected.update(generator.sample(remaining, count - len(selected)))
    return sorted(selected)


def build_cases(
    boundaries: tuple[int, ...],
    random_per_interval: int,
    base_seed: int,
    first_token_base: int = 30_000,
) -> list[dict[str, int | str]]:
    generator = random.Random(base_seed)
    pairs: list[list[dict[str, int | str]]] = []
    ordinal = 0
    for interval_index, (lower, upper) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        content_seed = base_seed + interval_index * 1_000_003
        for pair_in_interval, prompt_tokens in enumerate(choose_random_lengths(
            lower, upper, random_per_interval, generator
        )):
            pair_id = interval_index * random_per_interval + pair_in_interval
            pair: list[dict[str, int | str]] = []
            for kind, case_tokens in (
                ("upper_anchor_before", upper),
                ("random", prompt_tokens),
                ("upper_anchor_after", upper),
            ):
                pair.append(
                    {
                        "ordinal": ordinal,
                        "pair_id": pair_id,
                        "interval_index": interval_index,
                        "interval_lower_exclusive": lower,
                        "interval_upper_inclusive": upper,
                        "kind": kind,
                        "prompt_tokens": case_tokens,
                        "first_token_id": first_token_base + ordinal,
                        "content_seed": content_seed,
                    }
                )
                ordinal += 1
            pairs.append(pair)
    random.Random(base_seed ^ 0xA11CE).shuffle(pairs)
    return [case for pair in pairs for case in pair]


def build_prompt(case: dict[str, int | str]) -> list[int]:
    prompt_tokens = int(case["prompt_tokens"])
    generator = random.Random(int(case["content_seed"]))
    return [int(case["first_token_id"])] + [
        32 + generator.randrange(256) for _ in range(prompt_tokens - 1)
    ]


def require_positive_ttft_ms(value: float | None, label: str) -> float:
    if value is None or not math.isfinite(value) or value <= 0.0:
        raise VerificationError(f"{label} has no finite positive qrt_metrics.ttft_ms")
    return value


def token_ids_u32le_sha256(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def summarize(
    records: list[dict[str, Any]],
    boundaries: tuple[int, ...],
    ratio_min: float,
    ratio_max: float,
    anchor_max_to_min_ratio: float,
    upper_anchor_min_tokens_per_second: float = 0.0,
    q8192_upper_anchor_ttft_max_ms: float = math.inf,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    paired: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["interval_index"]].append(record)
        paired[record["pair_id"]].append(record)

    intervals = []
    eligible_ratios = []
    random_count = 0
    eligible_pair_count = 0
    stable_pair_count = 0
    for interval_index, (lower, upper) in enumerate(
        zip(boundaries, boundaries[1:])
    ):
        interval_records = grouped[interval_index]
        interval_pair_ids = sorted(
            {int(record["pair_id"]) for record in interval_records}
        )
        if not interval_pair_ids:
            raise VerificationError(
                f"interval ({lower}, {upper}] has an invalid case layout"
            )
        interval_ratios = []
        interval_anchor_references = []
        interval_anchor_drifts = []
        interval_anchor_speeds = []
        interval_anchor_ttft_ms = []
        eligible_random_lengths = []
        interval_correctness = []
        interval_stability = []
        interval_anchor_absolute_performance = []
        interval_random_absolute_performance = []
        interval_q8192_ttft = []
        for pair_id in interval_pair_ids:
            pair_records = paired[pair_id]
            by_kind = {record["kind"]: record for record in pair_records}
            expected_kinds = {
                "upper_anchor_before",
                "random",
                "upper_anchor_after",
            }
            if len(pair_records) != 3 or set(by_kind) != expected_kinds:
                raise VerificationError(
                    f"pair {pair_id} in interval ({lower}, {upper}] "
                    "does not contain one bracketed random-length case"
                )
            anchor_before = by_kind["upper_anchor_before"]
            random_record = by_kind["random"]
            anchor_after = by_kind["upper_anchor_after"]
            random_count += 1
            anchor_speeds = (
                float(anchor_before["prefill_tokens_per_second"]),
                float(anchor_after["prefill_tokens_per_second"]),
            )
            anchor_reference = math.sqrt(anchor_speeds[0] * anchor_speeds[1])
            anchor_drift = max(anchor_speeds) / min(anchor_speeds)
            anchor_absolute_performance = all(
                speed >= upper_anchor_min_tokens_per_second
                for speed in anchor_speeds
            )
            random_absolute_performance = (
                float(random_record["prefill_tokens_per_second"])
                >= upper_anchor_min_tokens_per_second
            )
            q8192_ttft_passed = upper != 8192 or all(
                float(anchor["ttft_ms"]) <= q8192_upper_anchor_ttft_max_ms
                for anchor in (anchor_before, anchor_after)
            )
            ratio = (
                random_record["prefill_tokens_per_second"] / anchor_reference
            )
            correctness = bool(
                anchor_before["gb10_match"]
                and random_record["gb10_match"]
                and anchor_after["gb10_match"]
            )
            stable = anchor_drift <= anchor_max_to_min_ratio
            eligible = correctness and stable
            random_record["paired_upper_anchor_prefill_tokens_per_second"] = (
                anchor_reference
            )
            random_record["paired_upper_anchor_max_to_min_ratio"] = (
                anchor_drift
            )
            random_record["throughput_to_upper_anchor_ratio"] = ratio
            random_record["performance_eligible"] = eligible
            random_record["anchor_stability_passed"] = stable
            random_record["absolute_performance_passed"] = (
                random_absolute_performance
            )
            interval_anchor_references.append(anchor_reference)
            interval_anchor_drifts.append(anchor_drift)
            interval_anchor_speeds.extend(anchor_speeds)
            interval_anchor_ttft_ms.extend(
                float(anchor["ttft_ms"])
                for anchor in (anchor_before, anchor_after)
                if "ttft_ms" in anchor
            )
            interval_correctness.append(correctness)
            interval_stability.append(stable)
            interval_anchor_absolute_performance.append(
                anchor_absolute_performance
            )
            interval_random_absolute_performance.append(
                random_absolute_performance
            )
            interval_q8192_ttft.append(q8192_ttft_passed)
            if stable:
                stable_pair_count += 1
            if eligible:
                eligible_pair_count += 1
                eligible_ratios.append(ratio)
                interval_ratios.append(ratio)
                eligible_random_lengths.append(random_record["prompt_tokens"])
        interval_correctness_passed = all(interval_correctness)
        interval_measurement_stability_passed = all(interval_stability)
        interval_relative_performance_passed = bool(
            len(interval_ratios) == len(interval_pair_ids)
            and min(interval_ratios) >= ratio_min
            and max(interval_ratios) <= ratio_max
        )
        interval_anchor_absolute_performance_passed = all(
            interval_anchor_absolute_performance
        )
        interval_random_absolute_performance_passed = all(
            interval_random_absolute_performance
        )
        interval_q8192_ttft_passed = all(interval_q8192_ttft)
        interval_performance_passed = (
            interval_relative_performance_passed
            and interval_anchor_absolute_performance_passed
            and interval_random_absolute_performance_passed
            and interval_q8192_ttft_passed
        )
        intervals.append(
            {
                "interval_lower_exclusive": lower,
                "interval_upper_inclusive": upper,
                "paired_upper_anchor_prefill_tokens_per_second_median": median(
                    interval_anchor_references
                ),
                "maximum_paired_upper_anchor_max_to_min_ratio": max(
                    interval_anchor_drifts
                ),
                "minimum_upper_anchor_prefill_tokens_per_second": min(
                    interval_anchor_speeds
                ),
                "maximum_upper_anchor_prefill_tokens_per_second": max(
                    interval_anchor_speeds
                ),
                "maximum_upper_anchor_ttft_ms": (
                    max(interval_anchor_ttft_ms)
                    if interval_anchor_ttft_ms
                    else None
                ),
                "random_lengths": sorted(
                    int(by_kind["random"]["prompt_tokens"])
                    for pair_id in interval_pair_ids
                    for by_kind in [
                        {record["kind"]: record for record in paired[pair_id]}
                    ]
                ),
                "eligible_random_lengths": sorted(eligible_random_lengths),
                "minimum_eligible_random_to_upper_anchor_throughput_ratio": (
                    min(interval_ratios) if interval_ratios else None
                ),
                "maximum_eligible_random_to_upper_anchor_throughput_ratio": (
                    max(interval_ratios) if interval_ratios else None
                ),
                "correctness_passed": interval_correctness_passed,
                "measurement_stability_passed": (
                    interval_measurement_stability_passed
                ),
                "relative_performance_passed": (
                    interval_relative_performance_passed
                ),
                "upper_anchor_absolute_performance_passed": (
                    interval_anchor_absolute_performance_passed
                ),
                "random_absolute_performance_passed": (
                    interval_random_absolute_performance_passed
                ),
                "q8192_upper_anchor_ttft_passed": (
                    interval_q8192_ttft_passed
                ),
                "performance_passed": interval_performance_passed,
                "passed": (
                    interval_correctness_passed
                    and interval_measurement_stability_passed
                    and interval_performance_passed
                ),
            }
        )
    return {
        "random_pair_count": random_count,
        "stable_anchor_pair_count": stable_pair_count,
        "performance_eligible_pair_count": eligible_pair_count,
        "minimum_eligible_random_to_upper_anchor_throughput_ratio": (
            min(eligible_ratios) if eligible_ratios else None
        ),
        "maximum_eligible_random_to_upper_anchor_throughput_ratio": (
            max(eligible_ratios) if eligible_ratios else None
        ),
        "intervals": intervals,
        "correctness_passed": all(
            interval["correctness_passed"] for interval in intervals
        ),
        "measurement_stability_passed": all(
            interval["measurement_stability_passed"] for interval in intervals
        ),
        "performance_passed": all(
            interval["performance_passed"] for interval in intervals
        ),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amd-base-url", required=True)
    parser.add_argument("--gb10-base-url", required=True)
    parser.add_argument("--amd-host", default="baiying")
    parser.add_argument("--gb10-host", default="gb10-4t")
    parser.add_argument(
        "--amd-model-path", default=r"D:\models\Qwen3.6-35B-A3B"
    )
    parser.add_argument("--amd-service-record", type=Path)
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument(
        "--random-per-interval",
        type=int,
        default=MINIMUM_RANDOM_CASES_PER_INTERVAL,
    )
    parser.add_argument("--base-seed", type=int, default=39_536_119)
    parser.add_argument("--first-token-base", type=int, default=30_000)
    parser.add_argument(
        "--boundaries", type=int, nargs="+", default=DEFAULT_BOUNDARIES
    )
    parser.add_argument(
        "--ratio-min",
        type=float,
        default=RETAINED_RANDOM_TO_ANCHOR_RATIO_MIN,
    )
    parser.add_argument(
        "--ratio-max",
        type=float,
        default=RETAINED_RANDOM_TO_ANCHOR_RATIO_MAX,
    )
    parser.add_argument(
        "--anchor-max-to-min-ratio",
        type=float,
        default=RETAINED_ANCHOR_MAX_TO_MIN_RATIO,
    )
    parser.add_argument(
        "--upper-anchor-min-prefill-tokens-per-second",
        type=float,
        default=RETAINED_PREFILL_TOKENS_PER_SECOND,
    )
    parser.add_argument(
        "--q8192-upper-anchor-ttft-max-ms",
        type=float,
        default=RETAINED_Q8192_TTFT_MAX_MS,
    )
    parser.add_argument("--repo-commit", default="unknown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    effective_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(effective_argv)
    if args.output.exists() and not args.force:
        raise VerificationError(f"refusing to overwrite {args.output}")
    if args.random_per_interval < MINIMUM_RANDOM_CASES_PER_INTERVAL:
        raise VerificationError(
            "--random-per-interval cannot be below the retained minimum "
            f"of {MINIMUM_RANDOM_CASES_PER_INTERVAL}"
        )
    boundaries = tuple(args.boundaries)
    if (
        len(boundaries) < 2
        or boundaries != tuple(sorted(set(boundaries)))
        or boundaries[0] < 2
    ):
        raise VerificationError(
            "--boundaries must be at least two unique increasing integers"
        )
    if boundaries != DEFAULT_BOUNDARIES:
        raise VerificationError(
            "--boundaries must preserve the complete retained 2K-8K "
            "upper-anchor ladder"
        )
    if not 0.0 < args.ratio_min <= 1.0 <= args.ratio_max:
        raise VerificationError(
            "throughput ratio bounds must straddle one and be positive"
        )
    if args.ratio_min < RETAINED_RANDOM_TO_ANCHOR_RATIO_MIN:
        raise VerificationError(
            "--ratio-min cannot relax the retained random/anchor lower bound"
        )
    if args.ratio_max > RETAINED_RANDOM_TO_ANCHOR_RATIO_MAX:
        raise VerificationError(
            "--ratio-max cannot relax the retained random/anchor upper bound"
        )
    if args.anchor_max_to_min_ratio < 1.0:
        raise VerificationError(
            "--anchor-max-to-min-ratio must be at least one"
        )
    if args.anchor_max_to_min_ratio > RETAINED_ANCHOR_MAX_TO_MIN_RATIO:
        raise VerificationError(
            "--anchor-max-to-min-ratio cannot relax retained anchor stability"
        )
    if args.upper_anchor_min_prefill_tokens_per_second <= 0.0:
        raise VerificationError(
            "--upper-anchor-min-prefill-tokens-per-second must be positive"
        )
    if (
        args.upper_anchor_min_prefill_tokens_per_second
        < RETAINED_PREFILL_TOKENS_PER_SECOND
    ):
        raise VerificationError(
            "--upper-anchor-min-prefill-tokens-per-second cannot relax the "
            f"retained {RETAINED_PREFILL_TOKENS_PER_SECOND:.3f} tok/s target"
        )
    if args.q8192_upper_anchor_ttft_max_ms <= 0.0:
        raise VerificationError(
            "--q8192-upper-anchor-ttft-max-ms must be positive"
        )
    if args.q8192_upper_anchor_ttft_max_ms > RETAINED_Q8192_TTFT_MAX_MS:
        raise VerificationError(
            "--q8192-upper-anchor-ttft-max-ms cannot relax the retained "
            f"{RETAINED_Q8192_TTFT_MAX_MS:.3f} ms target"
        )
    if re.fullmatch(r"[0-9a-f]{40}", args.repo_commit) is None:
        raise VerificationError("--repo-commit must be a lowercase 40-hex SHA")
    if args.amd_service_record is None:
        raise VerificationError(
            "--amd-service-record is required for host/model/DLL provenance"
        )
    try:
        amd_service_record = json.loads(
            args.amd_service_record.read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(
            f"could not read AMD service record: {error}"
        ) from error
    required_service_contract = validate_amd_service_record(
        amd_service_record,
        amd_host=args.amd_host,
        repo_commit=args.repo_commit,
        model_path=args.amd_model_path,
        model_id=args.model,
    )

    case_count = 3 * args.random_per_interval * (len(boundaries) - 1)
    if (
        args.first_token_base < 0
        or args.first_token_base + case_count >= QWEN36_VOCAB_SIZE
    ):
        raise VerificationError(
            "--first-token-base must leave room for every globally unique "
            f"measurement and warm-up first token below the Qwen3.6 vocabulary size "
            f"({QWEN36_VOCAB_SIZE})"
        )

    started_at = utc_now()
    cases = build_cases(
        boundaries,
        args.random_per_interval,
        args.base_seed,
        args.first_token_base,
    )
    warmup_case = {
        "prompt_tokens": 8192,
        "first_token_id": args.first_token_base + case_count,
        "content_seed": args.base_seed ^ 0xC01D_8192,
    }
    warmup_prompt = build_prompt(warmup_case)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        amd_warmup_future = pool.submit(
            request_completion,
            args.amd_base_url,
            args.model,
            warmup_prompt,
            args.timeout_seconds,
        )
        gb10_warmup_future = pool.submit(
            request_completion,
            args.gb10_base_url,
            args.model,
            warmup_prompt,
            args.timeout_seconds,
        )
        (
            amd_warmup_tokens,
            amd_warmup_elapsed_ms,
            amd_warmup_ttft_ms,
        ) = amd_warmup_future.result()
        (
            gb10_warmup_tokens,
            gb10_warmup_elapsed_ms,
            _,
        ) = gb10_warmup_future.result()
    amd_warmup_ttft_ms = require_positive_ttft_ms(
        amd_warmup_ttft_ms, "AMD q8192 warm-up response"
    )
    warmup_passed = amd_warmup_tokens == gb10_warmup_tokens
    warmup = {
        "record_type": "qrt_prefill_random_length_warmup",
        "measured": False,
        "execution_position": -1,
        **warmup_case,
        "prompt_u32le_sha256": token_ids_u32le_sha256(warmup_prompt),
        "amd_token_ids": amd_warmup_tokens,
        "gb10_token_ids": gb10_warmup_tokens,
        "gb10_match": warmup_passed,
        "ttft_ms": amd_warmup_ttft_ms,
        "amd_elapsed_ms": amd_warmup_elapsed_ms,
        "gb10_elapsed_ms": gb10_warmup_elapsed_ms,
        "server_prefix_cache_enabled": False,
        "passed": warmup_passed,
    }
    print(
        "[warmup] q=8192 "
        f"ttft_ms={amd_warmup_ttft_ms:.3f} "
        f"gb10_match={warmup_passed} measured=False",
        flush=True,
    )
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
            ttft_ms = require_positive_ttft_ms(
                ttft_ms, f"AMD measurement response at position {position - 1}"
            )
            prompt_tokens = int(case["prompt_tokens"])
            record = {
                **case,
                "execution_position": position - 1,
                "amd_token_ids": amd_tokens,
                "gb10_token_ids": gb10_tokens,
                "gb10_match": amd_tokens == gb10_tokens,
                "ttft_ms": ttft_ms,
                "prefill_tokens_per_second": prompt_tokens * 1000.0 / ttft_ms,
                "amd_elapsed_ms": amd_elapsed_ms,
                "gb10_elapsed_ms": gb10_elapsed_ms,
            }
            records.append(record)
            print(
                f"[{position:02d}/{len(cases)}] "
                f"interval=({case['interval_lower_exclusive']},"
                f"{case['interval_upper_inclusive']}] kind={case['kind']} "
                f"q={prompt_tokens} ttft_ms={ttft_ms:.3f} "
                f"tok_s={record['prefill_tokens_per_second']:.3f} "
                f"gb10_match={record['gb10_match']}",
                flush=True,
            )

    records.sort(key=lambda record: record["ordinal"])
    summary = summarize(
        records,
        boundaries,
        args.ratio_min,
        args.ratio_max,
        args.anchor_max_to_min_ratio,
        args.upper_anchor_min_prefill_tokens_per_second,
        args.q8192_upper_anchor_ttft_max_ms,
    )
    report = {
        "schema_version": 2,
        "record_type": "qrt_prefill_random_length_plateau_verification",
        "started_at_utc": started_at,
        "completed_at_utc": utc_now(),
        "source_commit": args.repo_commit,
        "execution_host": socket.gethostname(),
        "command": [sys.executable, str(Path(__file__).resolve()), *effective_argv],
        "model": args.model,
        "amd_host": args.amd_host,
        "amd_model_path": args.amd_model_path,
        "amd_endpoint": args.amd_base_url,
        "amd_service_record": str(args.amd_service_record),
        "amd_service_record_sha256": sha256_file(args.amd_service_record),
        "amd_service_contract": required_service_contract,
        "amd_whole_provider_sha256": amd_service_record.get(
            "whole_provider_sha256"
        ),
        "amd_attention_provider_sha256": amd_service_record.get(
            "attention_provider_sha256"
        ),
        "amd_attention_build_provenance_sha256": amd_service_record.get(
            "attention_build_provenance_sha256"
        ),
        "amd_gdn_build_provenance_sha256": amd_service_record.get(
            "gdn_build_provenance_sha256"
        ),
        "amd_gdn_provider_sha256": amd_service_record.get(
            "gdn_provider_sha256"
        ),
        "amd_q8192_product_record_sha256": amd_service_record.get(
            "q8192_product_record_sha256"
        ),
        "amd_service_environment_sha256": amd_service_record.get(
            "service_environment_sha256"
        ),
        "amd_q8192_moe_build_provenance_sha256": amd_service_record.get(
            "q8192_moe_build_provenance_sha256"
        ),
        "amd_q8192_moe_provider_sha256": amd_service_record.get(
            "q8192_moe_provider_sha256"
        ),
        "amd_dynamic_moe_native_shared_selected": amd_service_record.get(
            "dynamic_moe_native_shared_selected", False
        ),
        "amd_q8192_moe_required_runtime_environment": amd_service_record.get(
            "q8192_moe_required_runtime_environment", []
        ),
        "amd_model_engine_load_ms": amd_service_record.get(
            "model_engine_load_ms"
        ),
        "gb10_host": args.gb10_host,
        "gb10_endpoint": args.gb10_base_url,
        "request_policy": {
            "boundaries": list(boundaries),
            "random_per_interval": args.random_per_interval,
            "measurement_order_contract": (
                "shuffled bracketed pairs: upper anchor, random, upper anchor"
            ),
            "base_seed": args.base_seed,
            "first_token_base": args.first_token_base,
            "stratified_length_contract": {
                "boundary_successor": "interval_lower_exclusive + 1",
                "boundary_predecessor": "interval_upper_inclusive - 1",
                "integer_alignment": INTEGER_SPIKE_ALIGNMENT,
                "integer_alignment_neighbors": [-1, 0, 1],
                "additional_lengths": "uniform_without_replacement",
            },
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "ignore_eos": True,
            "return_token_ids": True,
            "cold_prefix_contract": (
                "server_prefix_cache_disabled_and_globally_unique_first_token_id"
            ),
            "warmup_contract": WARMUP_CONTRACT,
            "warmup_request_count": 1,
            "warmup_measured": False,
            "server_prefix_cache_enabled": False,
            "exact_first_token_prefill": True,
            "exact_letter_classifier": False,
            "controlled_content_contract": (
                "same generated content stream within each interval"
            ),
        },
        "acceptance": {
            "random_to_upper_anchor_throughput_ratio_min": args.ratio_min,
            "random_to_upper_anchor_throughput_ratio_max": args.ratio_max,
            "paired_upper_anchor_max_to_min_ratio_max": (
                args.anchor_max_to_min_ratio
            ),
            "upper_anchor_min_prefill_tokens_per_second": (
                args.upper_anchor_min_prefill_tokens_per_second
            ),
            "random_min_prefill_tokens_per_second": (
                args.upper_anchor_min_prefill_tokens_per_second
            ),
            "q8192_upper_anchor_ttft_max_ms": (
                args.q8192_upper_anchor_ttft_max_ms
            ),
            "gb10_token_match_required_for_performance": True,
        },
        "warmup": warmup,
        **summary,
        "passed": (
            warmup_passed
            and summary["correctness_passed"]
            and summary["measurement_stability_passed"]
            and summary["performance_passed"]
        ),
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
