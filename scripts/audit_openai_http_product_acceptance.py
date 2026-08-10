#!/usr/bin/env python3
"""Audit the complete resident OpenAI HTTP and CLI lifecycle acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REQUIRED_CHECKS = frozenset(
    {
        "health",
        "ready",
        "models",
        "model_retrieve",
        "tokenizer_round_trip",
        "chat_template_tokenize",
        "service_state_identity",
        "cli_status",
        "build_provenance",
        "text_completion",
        "completion_sse",
        "chat_completion",
        "chat_sse",
        "structured_tool_call",
        "tool_result_continuation",
        "structured_tool_call_sse",
        "continuous_prompt_length_matrix",
        "context_limit_rejection",
        "http_prefix_reuse",
        "bounded_fifo_request_queue",
    }
)
EXPECTED_CAPABILITIES = {
    "batch_size": 1,
    "continuous_prompt_lengths": True,
    "streaming": True,
    "tool_calls": True,
    "prefix_cache": True,
    "bounded_fifo_queue": True,
}
FORBIDDEN_EVAL_COMMAND_FRAGMENTS = (
    "QRT_SERVER_EXACT_FIRST_TOKEN_PREFILL=1",
    "QRT_SERVER_EXACT_LETTER_CLASSIFIER=1",
    "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_BF16_INVERSE_F32=1",
    "QRT_QWEN36_Q1_MTP_TOP2_ARBITRATION=1",
)
REQUIRED_ORDINARY_COMMAND_FRAGMENTS = (
    "QRT_SERVER_EXACT_FIRST_TOKEN_PREFILL=0",
    "QRT_SERVER_EXACT_LETTER_CLASSIFIER=0",
    "QRT_QWEN36_EXACT_ARBITRARY_LM_HEAD_BF16_INVERSE_F32=0",
    "QRT_QWEN36_Q1_MTP_TOP2_ARBITRATION=0",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROVENANCE_IDENTITY_KEYS = (
    "schema_version",
    "host",
    "execution",
    "repo_commit",
    "dirty_tree",
    "command_file",
    "variant",
    "variant_defines",
    "executable",
    "executable_sha256",
    "provider_dll",
    "provider_sha256",
    "source_sha256",
)


class AuditError(RuntimeError):
    """An OpenAI product-acceptance invariant was not satisfied."""


@dataclass(frozen=True)
class JsonSnapshot:
    path: Path
    data: bytes
    sha256: str
    value: dict[str, Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json_snapshot(path: Path, label: str) -> JsonSnapshot:
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AuditError(f"{label} is not valid JSON: {path}: {error}") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return JsonSnapshot(path.resolve(), data, sha256_bytes(data), value)


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def require_same_service(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    for field in (
        "pid",
        "address",
        "model",
        "model_path",
        "provider_dll",
        "arbitrary_moe_provider_dll",
        "arbitrary_moe_kernel_dir",
        "max_model_len",
        "max_queue_depth",
        "queue_timeout_seconds",
        "repo_commit",
        "host",
        "started_unix_seconds",
        "ready_unix_seconds",
    ):
        require(left.get(field) == right.get(field), f"{label} field differs: {field}")


def validate_runtime_artifacts(
    ready: dict[str, Any],
    state: dict[str, Any],
    ready_executable_sha256: str,
    required_artifact_sha256: frozenset[str],
) -> list[dict[str, Any]]:
    artifacts = ready.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "ready runtime has no artifact snapshots")
    indexed: dict[str, dict[str, Any]] = {}
    observed_hashes = {ready_executable_sha256}
    for artifact in artifacts:
        require(isinstance(artifact, dict), "ready runtime artifact is not an object")
        path_text = artifact.get("path")
        require(isinstance(path_text, str) and path_text, "ready runtime artifact has no path")
        path = Path(path_text).resolve()
        key = str(path).casefold()
        require(key not in indexed, f"ready runtime artifact is duplicated: {path_text}")
        data = path.read_bytes()
        reported_bytes = artifact.get("bytes")
        require(isinstance(reported_bytes, int) and reported_bytes == len(data), f"ready runtime artifact size changed: {path_text}")
        reported_sha = require_sha256(artifact.get("sha256"), f"ready runtime artifact hash: {path_text}")
        require(sha256_bytes(data) == reported_sha, f"ready runtime artifact hash changed: {path_text}")
        observed_hashes.add(reported_sha)
        indexed[key] = artifact
    loaded_modules = ready.get("loaded_modules")
    require(isinstance(loaded_modules, list) and loaded_modules, "ready runtime has no loaded-module snapshots")
    loaded_by_path: dict[str, dict[str, Any]] = {}
    for module in loaded_modules:
        require(isinstance(module, dict), "ready loaded module is not an object")
        path_text = module.get("path")
        require(isinstance(path_text, str) and path_text, "ready loaded module has no path")
        key = str(Path(path_text).resolve()).casefold()
        require(key not in loaded_by_path, f"ready loaded module is duplicated: {path_text}")
        require_sha256(module.get("sha256"), f"ready loaded-module hash: {path_text}")
        loaded_by_path[key] = module
    for field in ("provider_dll", "arbitrary_moe_provider_dll"):
        path_text = state.get(field)
        require(isinstance(path_text, str) and path_text, f"service state lacks {field}")
        key = str(Path(path_text).resolve()).casefold()
        require(key in indexed, f"ready runtime does not snapshot service {field}")
        require(indexed[key]["sha256"] in required_artifact_sha256, f"service {field} is not an accepted build artifact")
        require(key in loaded_by_path, f"service {field} is not loaded in the ready process")
        require(loaded_by_path[key].get("sha256") == indexed[key]["sha256"], f"loaded service {field} hash differs")
        require(loaded_by_path[key].get("bytes") == indexed[key]["bytes"], f"loaded service {field} size differs")
    require(
        required_artifact_sha256.issubset(observed_hashes),
        "required build artifact is absent from the live runtime snapshot",
    )
    return artifacts


def index_checks(checks: Any) -> dict[str, dict[str, Any]]:
    require(isinstance(checks, list), "verification checks are not an array")
    indexed: dict[str, dict[str, Any]] = {}
    for check in checks:
        require(isinstance(check, dict), "verification check is not an object")
        name = check.get("name")
        require(isinstance(name, str) and name, "verification check has no name")
        require(name not in indexed, f"verification check is duplicated: {name}")
        require(check.get("passed") is True, f"verification check did not pass: {name}")
        require(isinstance(check.get("details"), dict), f"verification check has no details: {name}")
        indexed[name] = check
    missing = sorted(REQUIRED_CHECKS.difference(indexed))
    require(not missing, f"verification is missing required checks: {missing}")
    return indexed


def validate_endpoint(endpoint: Any) -> urllib.parse.SplitResult:
    require(isinstance(endpoint, str), "verification endpoint is not text")
    parsed = urllib.parse.urlsplit(endpoint)
    require(parsed.scheme == "http", "verification endpoint is not HTTP")
    require(parsed.hostname in {"127.0.0.1", "localhost", "::1"}, "verification endpoint is not loopback")
    require(parsed.port is not None, "verification endpoint has no explicit port")
    require(parsed.path in {"", "/"}, "verification endpoint has an unexpected path")
    return parsed


def validate_provenance(
    verification: dict[str, Any],
    required_provenance_sha256: frozenset[str],
    required_artifact_sha256: frozenset[str],
) -> list[dict[str, Any]]:
    provenance = verification.get("provenance")
    require(isinstance(provenance, list) and provenance, "verification has no build provenance")
    require(len(provenance) == len(required_provenance_sha256), "build provenance entry count differs")
    observed_snapshots: set[str] = set()
    observed_artifacts: set[str] = set()
    for item in provenance:
        require(isinstance(item, dict), "provenance entry is not an object")
        path_text = item.get("path")
        require(isinstance(path_text, str) and path_text, "provenance entry has no path")
        snapshot = read_json_snapshot(Path(path_text), "build provenance")
        reported_sha = require_sha256(item.get("sha256"), "reported provenance hash")
        require(snapshot.sha256 == reported_sha, f"build provenance hash changed: {path_text}")
        observed_snapshots.add(reported_sha)
        identity = item.get("identity")
        require(isinstance(identity, dict), "provenance entry has no identity")
        expected_identity = {
            key: snapshot.value[key]
            for key in PROVENANCE_IDENTITY_KEYS
            if key in snapshot.value
        }
        artifacts = snapshot.value.get("artifacts")
        if isinstance(artifacts, list):
            dll_artifacts = []
            for artifact in artifacts:
                if not isinstance(artifact, dict):
                    continue
                artifact_name = artifact.get("name") or artifact.get("path")
                if isinstance(artifact_name, str) and artifact_name.casefold().endswith(".dll"):
                    dll_artifacts.append(
                        {
                            "name": artifact_name,
                            "bytes": artifact.get("bytes"),
                            "sha256": artifact.get("sha256"),
                        }
                    )
            if dll_artifacts:
                expected_identity["dll_artifacts"] = dll_artifacts
        require(identity == expected_identity, f"reported provenance identity differs: {path_text}")
        if "dirty_tree" in identity:
            require(identity["dirty_tree"] is False, f"build provenance is dirty: {path_text}")
        executable_sha = identity.get("executable_sha256")
        if executable_sha is not None:
            observed_artifacts.add(require_sha256(executable_sha, "executable hash"))
        provider_sha = identity.get("provider_sha256")
        if provider_sha is not None:
            observed_artifacts.add(require_sha256(provider_sha, "provider hash"))
        dll_artifacts = identity.get("dll_artifacts", [])
        require(isinstance(dll_artifacts, list), "provenance DLL artifacts are not an array")
        for artifact in dll_artifacts:
            require(isinstance(artifact, dict), "DLL artifact is not an object")
            observed_artifacts.add(require_sha256(artifact.get("sha256"), "DLL artifact hash"))
    require(
        observed_snapshots == set(required_provenance_sha256),
        "build provenance snapshot set differs from the required set",
    )
    require(
        required_artifact_sha256.issubset(observed_artifacts),
        "required runtime artifact hash is absent from build provenance",
    )
    return provenance


def validate_matrix(
    check: dict[str, Any],
    expected_lengths: list[int],
    expected_reference_sha256: frozenset[str],
    expected_model: str,
) -> None:
    details = check["details"]
    require(details.get("prompt_token_id") == 32, "prompt-length matrix token ID differs")
    require(details.get("tested_lengths") == expected_lengths, "prompt-length matrix differs")
    cases = details.get("cases")
    require(isinstance(cases, list) and len(cases) == len(expected_lengths), "prompt-length matrix cases differ")
    for expected, case in zip(expected_lengths, cases):
        require(isinstance(case, dict), f"prompt-length case {expected} is not an object")
        require(case.get("prompt_tokens") == expected, f"prompt-length case order differs at {expected}")
        require(case.get("completion_tokens") == 1, f"prompt-length case did not emit one token: {expected}")
        require(case.get("gb10_text_match") is True, f"prompt-length case is not gb10-bound: {expected}")
        require_sha256(case.get("response_sha256"), f"prompt-length response hash {expected}")
        require_sha256(case.get("gb10_text_utf8_sha256"), f"gb10 text hash {expected}")
        require_sha256(case.get("gb10_response_sha256"), f"gb10 response hash {expected}")
        for metric in ("ttft_ms", "request_ms"):
            value = case.get(metric)
            require(isinstance(value, (int, float)) and value >= 0, f"prompt-length metric is invalid: {expected}/{metric}")
    references = details.get("gb10_references")
    require(isinstance(references, list) and references, "prompt-length matrix has no gb10 references")
    observed_hashes: set[str] = set()
    covered: list[int] = []
    for reference in references:
        require(isinstance(reference, dict), "gb10 reference binding is not an object")
        observed_hashes.add(require_sha256(reference.get("sha256"), "gb10 reference snapshot hash"))
        authority = reference.get("authority")
        require(isinstance(authority, dict), "gb10 reference has no authority")
        require(str(authority.get("host", "")).casefold() == "gb10-4t", "prompt reference authority is not gb10-4t")
        require(authority.get("model") == expected_model, "prompt reference model differs")
        lengths = reference.get("prompt_lengths")
        require(isinstance(lengths, list), "prompt reference lengths are not an array")
        covered.extend(lengths)
    require(observed_hashes == set(expected_reference_sha256), "gb10 prompt-reference hash set differs")
    require(sorted(covered) == sorted(expected_lengths), "gb10 prompt references do not exactly cover the matrix")


def validate_prefix(check: dict[str, Any]) -> None:
    details = check["details"]
    prefix_tokens = details.get("shared_prefix_tokens")
    require(isinstance(prefix_tokens, int) and prefix_tokens >= 256, "prefix reuse is below the product minimum")
    require(details.get("prompt_tokens") == prefix_tokens + 1, "prefix probe suffix is not exactly one token")
    responses = details.get("response_sha256")
    require(isinstance(responses, list) and len(responses) == 4, "prefix probe response set differs")
    for index, value in enumerate(responses):
        require_sha256(value, f"prefix response hash {index}")
    semantic = details.get("response_semantic_sha256")
    require(isinstance(semantic, list) and len(semantic) == 4, "prefix semantic response set differs")
    for index, value in enumerate(semantic):
        require_sha256(value, f"prefix semantic response hash {index}")
    require(semantic[0] == semantic[3], "prefix output changed after unrelated request")
    require(
        details.get("probe_order") == ["shared_a", "shared_b", "unrelated", "shared_a_repeat"],
        "prefix probe order differs",
    )
    unrelated_tokens = details.get("unrelated_prompt_tokens")
    require(
        isinstance(unrelated_tokens, int)
        and unrelated_tokens > 0
        and unrelated_tokens != details.get("prompt_tokens"),
        "unrelated prefix guard length differs",
    )
    require(details.get("unrelated_prefix_guard") is True, "unrelated prefix guard did not pass")
    require(details.get("repeat_output_match") is True, "prefix repeat output did not match")
    markers = details.get("markers")
    require(isinstance(markers, list), "prefix probe markers are not an array")
    seed = next((index for index, marker in enumerate(markers) if marker.get("kind") == "seed"), None)
    seeded = next(
        (index for index, marker in enumerate(markers) if marker.get("kind") == "hit" and marker.get("seed") is True),
        None,
    )
    reused = next(
        (index for index, marker in enumerate(markers) if marker.get("kind") == "hit" and marker.get("seed") is False),
        None,
    )
    require(seed is not None and seeded is not None and reused is not None, "prefix lifecycle markers are incomplete")
    require(seed < seeded < reused, "prefix lifecycle markers are out of order")


def validate_queue(
    check: dict[str, Any],
    health: dict[str, Any],
    state: dict[str, Any],
) -> None:
    queue = health.get("queue")
    require(isinstance(queue, dict), "health has no queue state")
    require(queue.get("accepting_requests") is True, "health queue is not accepting requests")
    require(queue.get("active_requests") == 0, "health queue was active at discovery")
    require(queue.get("waiting_requests") == 0, "health queue was not idle at discovery")
    capacity = queue.get("max_waiting_requests")
    timeout_seconds = queue.get("wait_timeout_seconds")
    require(isinstance(capacity, int) and capacity >= 2, "health queue capacity is too small")
    require(isinstance(timeout_seconds, int) and timeout_seconds > 0, "health queue timeout is invalid")
    require(state.get("max_queue_depth") == capacity, "state queue capacity differs")
    require(state.get("queue_timeout_seconds") == timeout_seconds, "state queue timeout differs")

    details = check["details"]
    request_count = details.get("request_count")
    require(isinstance(request_count, int) and request_count >= 3, "queue check request count is too small")
    require(details.get("prompt_tokens") == 17, "queue check prompt length differs")
    require(details.get("completion_tokens") == 1, "queue check completion length differs")
    require(
        details.get("queue_waiting_observed", 0) >= request_count - 1,
        "queue check did not observe every contending request",
    )
    requests = details.get("requests")
    require(isinstance(requests, list) and len(requests) == request_count, "queue request rows differ")
    request_ids: set[str] = set()
    waiting_rows = 0
    for row in requests:
        require(isinstance(row, dict), "queue request row is not an object")
        request_id = row.get("request_id")
        require(isinstance(request_id, str) and request_id and request_id not in request_ids, "queue request ID is absent or duplicated")
        request_ids.add(request_id)
        wait_ms = row.get("queue_wait_ms")
        require(isinstance(wait_ms, (int, float)) and wait_ms >= 0, "queue wait metric is invalid")
        waiting_rows += int(wait_ms > 1.0)
        elapsed_ms = row.get("elapsed_ms")
        require(isinstance(elapsed_ms, (int, float)) and elapsed_ms > 0, "queue request elapsed time is invalid")
        require_sha256(row.get("response_sha256"), "queue response hash")
    require(waiting_rows >= request_count - 1, "queue request rows do not prove contention")

    before = details.get("queue_before")
    after = details.get("queue_after")
    require(isinstance(before, dict) and isinstance(after, dict), "queue check lacks before/after snapshots")
    require(after.get("active_requests") == 0 and after.get("waiting_requests") == 0, "queue did not return to idle")
    require(after.get("started_total", 0) - before.get("started_total", 0) == request_count, "queue started counter differs")
    require(after.get("completed_total", 0) - before.get("completed_total", 0) == request_count, "queue completed counter differs")
    require(after.get("queued_total", 0) - before.get("queued_total", 0) >= request_count - 1, "queue contention counter differs")
    for counter in ("rejected_total", "timed_out_total"):
        require(after.get(counter) == before.get(counter), f"queue check changed {counter}")


def validate_generation_checks(checks: dict[str, dict[str, Any]]) -> None:
    for name in ("text_completion", "completion_sse", "chat_completion", "chat_sse", "tool_result_continuation"):
        completion_tokens = checks[name]["details"].get("completion_tokens")
        require(isinstance(completion_tokens, int) and completion_tokens >= 1, f"{name} emitted no tokens")
    for name in ("completion_sse", "chat_sse", "structured_tool_call_sse"):
        event_count = checks[name]["details"].get("event_count")
        require(isinstance(event_count, int) and event_count >= 2, f"{name} has too few SSE events")
    for name in ("structured_tool_call", "structured_tool_call_sse"):
        details = checks[name]["details"]
        require(details.get("tool_name") == "get_weather", f"{name} tool differs")
        require(details.get("argument_keys") == ["city"], f"{name} arguments differ")
    for nonstream_name, stream_name in (
        ("text_completion", "completion_sse"),
        ("chat_completion", "chat_sse"),
    ):
        nonstream = checks[nonstream_name]["details"]
        stream = checks[stream_name]["details"]
        require(stream.get("output_match_nonstream") is True, f"{stream_name} did not match non-stream output")
        nonstream_hash = require_sha256(nonstream.get("output_utf8_sha256"), f"{nonstream_name} output hash")
        stream_hash = require_sha256(stream.get("output_utf8_sha256"), f"{stream_name} output hash")
        require(stream_hash == nonstream_hash, f"{stream_name} output hash differs from non-stream output")


def audit_acceptance(
    verification_path: Path,
    ready_runtime_path: Path,
    shutdown_runtime_path: Path,
    expected_model: str,
    expected_model_path: str,
    expected_host: str,
    expected_max_model_len: int,
    expected_prompt_lengths: list[int],
    expected_reference_sha256: frozenset[str],
    required_provenance_sha256: frozenset[str],
    required_artifact_sha256: frozenset[str],
    max_load_ms: float,
) -> dict[str, Any]:
    verification_snapshot = read_json_snapshot(verification_path, "HTTP verification")
    ready_snapshot = read_json_snapshot(ready_runtime_path, "ready runtime evidence")
    shutdown_snapshot = read_json_snapshot(shutdown_runtime_path, "shutdown runtime evidence")
    verification = verification_snapshot.value
    ready = ready_snapshot.value
    shutdown = shutdown_snapshot.value

    require(verification.get("schema_version") == 1, "verification schema differs")
    require(verification.get("passed") is True, "HTTP verification did not pass")
    require(verification.get("generation_checks_skipped") is False, "generation checks were skipped")
    require(verification.get("api_key_recorded") is False, "verification recorded an API key")
    require(str(verification.get("host", "")).casefold() == expected_host.casefold(), "verification host differs")
    require(verification.get("model") == expected_model, "verification model differs")
    endpoint = validate_endpoint(verification.get("endpoint"))
    checks = index_checks(verification.get("checks"))

    health = verification.get("health")
    state = verification.get("service_state")
    cli_status = verification.get("cli_status")
    require(isinstance(health, dict) and isinstance(state, dict), "verification lacks health or state")
    require(isinstance(cli_status, dict), "verification lacks CLI status")
    require(health.get("status") == "ok" and health.get("ready") is True, "health is not ready")
    require(health.get("pid") == verification.get("pid") == state.get("pid"), "verification PID binding differs")
    require(health.get("model") == state.get("model") == expected_model, "service model binding differs")
    require(health.get("max_model_len") == state.get("max_model_len") == expected_max_model_len, "max model length differs")
    require(health.get("capabilities") == EXPECTED_CAPABILITIES, "health capabilities differ")
    validate_queue(checks["bounded_fifo_request_queue"], health, state)
    load = health.get("load")
    require(isinstance(load, dict), "health has no load metrics")
    total_load_ms = load.get("total_ms")
    require(isinstance(total_load_ms, (int, float)) and 0 <= total_load_ms <= max_load_ms, "model/engine load exceeds the acceptance bound")
    require(state.get("status") == "ready", "verification state is not ready")
    require(str(state.get("host", "")).casefold() == expected_host.casefold(), "state host differs")
    require(str(state.get("model_path", "")).casefold() == expected_model_path.casefold(), "state model path differs")
    require(isinstance(state.get("arbitrary_moe_provider_dll"), str) and state["arbitrary_moe_provider_dll"], "state lacks arbitrary MoE provider")
    require(isinstance(state.get("arbitrary_moe_kernel_dir"), str) and state["arbitrary_moe_kernel_dir"], "state lacks arbitrary MoE kernels")
    require(state.get("address") == endpoint.netloc, "state address differs from verification endpoint")
    require(cli_status.get("status") == "ready" and cli_status.get("reachable") is True, "CLI status is not ready and reachable")
    require(cli_status.get("endpoint") == verification.get("endpoint"), "CLI status endpoint differs")
    for field in (
        "pid",
        "model",
        "model_path",
        "provider_dll",
        "arbitrary_moe_provider_dll",
        "arbitrary_moe_kernel_dir",
        "max_model_len",
        "max_queue_depth",
        "queue_timeout_seconds",
        "repo_commit",
        "host",
        "started_unix_seconds",
        "ready_unix_seconds",
    ):
        require(cli_status.get(field) == state.get(field), f"CLI status field differs: {field}")
    cli_status_details = checks["cli_status"]["details"]
    cli_status_path = cli_status_details.get("path")
    require(isinstance(cli_status_path, str) and cli_status_path, "CLI status check has no snapshot path")
    cli_status_snapshot = read_json_snapshot(Path(cli_status_path), "CLI status snapshot")
    require(cli_status_snapshot.sha256 == cli_status_details.get("sha256"), "CLI status snapshot hash changed")
    require(cli_status_snapshot.value == cli_status, "CLI status snapshot payload differs")
    started = state.get("started_unix_seconds")
    ready_at = state.get("ready_unix_seconds")
    require(isinstance(started, int) and isinstance(ready_at, int) and ready_at >= started, "service startup timestamps differ")
    startup_seconds = ready_at - started
    require(startup_seconds * 1000 <= max_load_ms, "model/engine startup exceeds the acceptance bound")

    provenance = validate_provenance(verification, required_provenance_sha256, required_artifact_sha256)
    for item in provenance:
        identity = item["identity"]
        require(str(identity.get("host", "")).casefold() == expected_host.casefold(), "build provenance host differs")
        require(identity.get("execution") == "local_windows_process", "build provenance was not produced by local Windows")
    validate_generation_checks(checks)
    validate_matrix(checks["continuous_prompt_length_matrix"], expected_prompt_lengths, expected_reference_sha256, expected_model)
    context_limit = checks["context_limit_rejection"]["details"]
    require(context_limit.get("prompt_tokens") == expected_max_model_len, "context-limit prompt length differs")
    require(context_limit.get("completion_tokens") == 1, "context-limit completion length differs")
    require(context_limit.get("error_type") == "invalid_request_error", "context-limit error type differs")
    require(context_limit.get("error_code") == "context_length_exceeded", "context-limit error code differs")
    require_sha256(context_limit.get("response_sha256"), "context-limit response hash")
    validate_prefix(checks["http_prefix_reuse"])
    require(checks["model_retrieve"]["details"].get("model") == expected_model, "model retrieval check differs")

    require(ready.get("schema_version") == 1, "ready runtime schema differs")
    require(ready.get("record_type") == "qrt_live_service_runtime_evidence", "ready runtime record type differs")
    require(str(ready.get("host", "")).casefold() == expected_host.casefold(), "ready runtime host differs")
    require(ready.get("execution") == "local_windows_process", "ready runtime was not local Windows")
    ready_state = ready.get("service_state", {}).get("payload")
    ready_process = ready.get("service_process")
    require(isinstance(ready_state, dict) and isinstance(ready_process, dict), "ready runtime lacks state or process")
    require(ready_state.get("status") == "ready", "ready runtime state differs")
    require_same_service(state, ready_state, "verification/ready runtime")
    require(ready_process.get("pid") == state.get("pid"), "ready process PID differs")
    require(str(ready_process.get("name", "")).casefold() == "qrt.exe", "ready process is not qrt.exe")
    ready_executable_sha = require_sha256(ready_process.get("executable_sha256"), "ready executable hash")
    executable_path = ready_process.get("executable_path")
    require(isinstance(executable_path, str) and executable_path, "ready process has no executable path")
    require(sha256_bytes(Path(executable_path).read_bytes()) == ready_executable_sha, "ready executable hash changed")
    require(ready_executable_sha in required_artifact_sha256, "ready executable is not the accepted qrt build")
    executable_provenance = [
        item["identity"]
        for item in provenance
        if item["identity"].get("executable_sha256") == ready_executable_sha
    ]
    require(len(executable_provenance) == 1, "ready executable provenance is absent or duplicated")
    require(
        state.get("repo_commit") == executable_provenance[0].get("repo_commit"),
        "service commit differs from the ready executable provenance",
    )
    command_line = ready_process.get("command_line")
    require(isinstance(command_line, str) and command_line, "ready process has no command line")
    for fragment in FORBIDDEN_EVAL_COMMAND_FRAGMENTS:
        require(fragment.casefold() not in command_line.casefold(), f"ordinary service enabled evaluation-only route: {fragment}")
    for fragment in REQUIRED_ORDINARY_COMMAND_FRAGMENTS:
        require(fragment.casefold() in command_line.casefold(), f"ordinary service did not explicitly disable evaluation-only route: {fragment}")
    runtime_artifacts = validate_runtime_artifacts(
        ready,
        state,
        ready_executable_sha,
        required_artifact_sha256,
    )
    listeners = ready.get("listener")
    require(isinstance(listeners, list) and any(
        listener.get("owning_process") == state.get("pid")
        and listener.get("local_port") == endpoint.port
        for listener in listeners if isinstance(listener, dict)
    ), "ready runtime does not bind the service listener")
    ready_state_sha = require_sha256(ready.get("service_state", {}).get("sha256"), "ready state hash")
    require(checks["service_state_identity"]["details"].get("sha256") == ready_state_sha, "verification and ready-runtime state snapshots differ")

    require(shutdown.get("schema_version") == 1, "shutdown runtime schema differs")
    require(shutdown.get("record_type") == "qrt_stopped_service_runtime_evidence", "shutdown record type differs")
    require(str(shutdown.get("host", "")).casefold() == expected_host.casefold(), "shutdown host differs")
    require(shutdown.get("execution") == "local_windows_process", "shutdown audit was not local Windows")
    shutdown_ready = shutdown.get("ready_evidence")
    stopped_state = shutdown.get("stopped_state", {}).get("payload")
    assertions = shutdown.get("assertions")
    require(isinstance(shutdown_ready, dict) and isinstance(stopped_state, dict) and isinstance(assertions, dict), "shutdown evidence is incomplete")
    shutdown_ready_path = shutdown_ready.get("path")
    require(isinstance(shutdown_ready_path, str), "shutdown evidence has no ready-snapshot path")
    require(Path(shutdown_ready_path).resolve() == ready_snapshot.path, "shutdown audit names another ready snapshot")
    require(shutdown_ready.get("sha256") == ready_snapshot.sha256, "shutdown audit binds another ready snapshot")
    require(shutdown_ready.get("bytes") == len(ready_snapshot.data), "shutdown ready snapshot byte count differs")
    stopped_state_path = shutdown.get("stopped_state", {}).get("path")
    require(isinstance(stopped_state_path, str), "shutdown evidence has no stopped-state path")
    require(
        str(Path(stopped_state_path).resolve()).casefold()
        == str(Path(ready.get("service_state", {}).get("path", "")).resolve()).casefold(),
        "shutdown and ready evidence name different state files",
    )
    stopped_snapshot = read_json_snapshot(Path(stopped_state_path), "terminal service state")
    require(
        shutdown.get("stopped_state", {}).get("sha256") == stopped_snapshot.sha256,
        "terminal service-state hash changed",
    )
    require(stopped_state == stopped_snapshot.value, "terminal service-state payload differs from its file")
    require_same_service(ready_state, stopped_state, "ready/stopped runtime")
    require(stopped_state.get("status") == "stopped", "terminal service state is not stopped")
    require(isinstance(stopped_state.get("stopped_unix_seconds"), int), "terminal state has no stopped timestamp")
    require(stopped_state["stopped_unix_seconds"] >= stopped_state["ready_unix_seconds"], "stopped timestamp precedes readiness")
    for name in ("same_service_instance", "stopped_timestamp_present", "process_absent", "listener_absent"):
        require(assertions.get(name) is True, f"shutdown assertion did not pass: {name}")
    require(assertions.get("terminal_state") == "stopped", "shutdown terminal-state assertion differs")
    require(assertions.get("pid") == state.get("pid"), "shutdown PID assertion differs")
    require(assertions.get("address") == state.get("address"), "shutdown address assertion differs")
    require(assertions.get("model") == expected_model, "shutdown model assertion differs")
    resident_seconds = stopped_state["stopped_unix_seconds"] - stopped_state["started_unix_seconds"]
    require(resident_seconds >= startup_seconds, "resident lifetime is shorter than startup")

    return {
        "schema_version": 1,
        "record_type": "openai_http_product_acceptance",
        "passed": True,
        "host": expected_host,
        "model": expected_model,
        "model_path": expected_model_path,
        "endpoint": verification["endpoint"],
        "pid": state["pid"],
        "max_model_len": expected_max_model_len,
        "load_total_ms": total_load_ms,
        "startup_seconds": startup_seconds,
        "resident_seconds": resident_seconds,
        "required_checks": sorted(REQUIRED_CHECKS),
        "prompt_lengths": expected_prompt_lengths,
        "gb10_reference_sha256": sorted(expected_reference_sha256),
        "runtime_artifact_sha256": sorted(required_artifact_sha256),
        "runtime_artifact_count": len(runtime_artifacts),
        "provenance_count": len(provenance),
        "inputs": {
            "http_verification": {"path": str(verification_snapshot.path), "bytes": len(verification_snapshot.data), "sha256": verification_snapshot.sha256},
            "ready_runtime": {"path": str(ready_snapshot.path), "bytes": len(ready_snapshot.data), "sha256": ready_snapshot.sha256},
            "shutdown_runtime": {"path": str(shutdown_snapshot.path), "bytes": len(shutdown_snapshot.data), "sha256": shutdown_snapshot.sha256},
            "cli_status": {"path": str(cli_status_snapshot.path), "bytes": len(cli_status_snapshot.data), "sha256": cli_status_snapshot.sha256},
        },
        "requirements": {
            "standard_openai_json_and_sse": True,
            "model_listing_and_retrieval": True,
            "structured_tool_call_json_and_sse": True,
            "tool_result_continuation": True,
            "continuous_prompt_lengths_through_max_minus_one": True,
            "gb10_prompt_length_parity": True,
            "resident_prefix_reuse": True,
            "bounded_fifo_request_queue": True,
            "resident_model_load_within_30_seconds": True,
            "pid_bound_start_status_stop": True,
            "terminal_process_and_listener_absence": True,
        },
    }


def write_report(path: Path, report: dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise AuditError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_sha256_set(values: Iterable[str], label: str) -> frozenset[str]:
    parsed = frozenset(values)
    require(parsed, f"{label} set is empty")
    for value in parsed:
        require_sha256(value, label)
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--ready-runtime", type=Path, required=True)
    parser.add_argument("--shutdown-runtime", type=Path, required=True)
    parser.add_argument("--expected-model", default="qwen3.6-35b-a3b")
    parser.add_argument("--expected-model-path", default=r"D:\models\Qwen3.6-35B-A3B")
    parser.add_argument("--expected-host", default="baiying")
    parser.add_argument("--expected-max-model-len", type=int, default=262_144)
    parser.add_argument("--expected-prompt-length", type=int, action="append", required=True)
    parser.add_argument("--expected-reference-sha256", action="append", required=True)
    parser.add_argument("--required-provenance-sha256", action="append", required=True)
    parser.add_argument("--required-artifact-sha256", action="append", required=True)
    parser.add_argument("--max-load-ms", type=float, default=30_000.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require(args.expected_max_model_len > 1, "expected max model length is invalid")
    require(args.max_load_ms > 0, "maximum load time must be positive")
    require(len(set(args.expected_prompt_length)) == len(args.expected_prompt_length), "expected prompt lengths are duplicated")
    require(all(0 < length < args.expected_max_model_len for length in args.expected_prompt_length), "expected prompt length is outside the service context")
    report = audit_acceptance(
        args.verification,
        args.ready_runtime,
        args.shutdown_runtime,
        args.expected_model,
        args.expected_model_path,
        args.expected_host,
        args.expected_max_model_len,
        args.expected_prompt_length,
        parse_sha256_set(args.expected_reference_sha256, "reference SHA-256"),
        parse_sha256_set(args.required_provenance_sha256, "provenance SHA-256"),
        parse_sha256_set(args.required_artifact_sha256, "artifact SHA-256"),
        args.max_load_ms,
    )
    write_report(args.output, report, args.force)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
