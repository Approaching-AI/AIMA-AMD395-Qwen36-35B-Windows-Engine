#!/usr/bin/env python3
"""Verify the resident qrt OpenAI/CLI HTTP surface against a real service."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes

    def header(self, name: str) -> str | None:
        lower_name = name.lower()
        return next(
            (value for key, value in self.headers.items() if key.lower() == lower_name),
            None,
        )


class VerificationError(RuntimeError):
    pass


PREFIX_MARKER_RE = re.compile(
    r"QRT_SERVER_MARK prefix_cache_(?P<kind>seed|hit) "
    r"prefix_tokens=(?P<prefix_tokens>\d+) "
    r"suffix_tokens=(?P<suffix_tokens>\d+) "
    r"output_tokens=(?P<output_tokens>\d+)"
    r"(?: seed=(?P<seed>[01]))?"
)


def parse_prefix_markers(log_text: str) -> list[dict[str, Any]]:
    markers: list[dict[str, Any]] = []
    for match in PREFIX_MARKER_RE.finditer(log_text):
        seed = match.group("seed")
        markers.append(
            {
                "kind": match.group("kind"),
                "prefix_tokens": int(match.group("prefix_tokens")),
                "suffix_tokens": int(match.group("suffix_tokens")),
                "output_tokens": int(match.group("output_tokens")),
                "seed": None if seed is None else bool(int(seed)),
            }
        )
    return markers


def select_single_token_suffix_pair(
    tokenized_prompts: list[tuple[str, list[int]]],
    minimum_prefix_tokens: int,
    maximum_prompt_tokens: int,
) -> tuple[tuple[str, list[int]], tuple[str, list[int]], int]:
    groups: dict[tuple[int, ...], list[tuple[str, list[int]]]] = {}
    for prompt, tokens in tokenized_prompts:
        if not minimum_prefix_tokens < len(tokens) <= maximum_prompt_tokens:
            continue
        groups.setdefault(tuple(tokens[:-1]), []).append((prompt, tokens))
    for shared_tokens, entries in groups.items():
        distinct: dict[int, tuple[str, list[int]]] = {}
        for entry in entries:
            distinct.setdefault(entry[1][-1], entry)
        if len(distinct) >= 2:
            first, second = list(distinct.values())[:2]
            return first, second, len(shared_tokens)
    raise VerificationError(
        "could not construct two prefix-probe prompts differing in exactly one final token"
    )


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Verifier:
    def __init__(
        self,
        root_url: str,
        model: str,
        api_key: str | None,
        timeout_seconds: float,
        expected_max_model_len: int,
    ) -> None:
        self.root_url = root_url.rstrip("/")
        if self.root_url.endswith("/v1"):
            self.root_url = self.root_url[: -len("/v1")]
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.expected_max_model_len = expected_max_model_len
        self.checks: list[dict[str, Any]] = []
        self.snapshots: dict[str, Any] = {}
        self.provenance: list[dict[str, Any]] = []

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            raise VerificationError(message)

    def record(self, name: str, **details: Any) -> None:
        self.checks.append({"name": name, "passed": True, "details": details})

    def request(
        self,
        method: str,
        path: str,
        payload: Any | None = None,
    ) -> HttpResponse:
        headers = {"User-Agent": "qrt-openai-verifier/1"}
        body = None
        if payload is not None:
            body = canonical_json(payload)
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.root_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return HttpResponse(
                    status=response.status,
                    headers=dict(response.headers.items()),
                    body=response.read(),
                )
        except urllib.error.HTTPError as error:
            return HttpResponse(
                status=error.code,
                headers=dict(error.headers.items()),
                body=error.read(),
            )

    def request_json(self, method: str, path: str, payload: Any | None = None) -> tuple[HttpResponse, Any]:
        response = self.request(method, path, payload)
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(
                f"{method} {path} returned non-JSON status {response.status}: {error}"
            ) from error
        self.require(
            200 <= response.status < 300,
            f"{method} {path} returned status {response.status}: {value}",
        )
        return response, value

    def verify_discovery_and_tokenizer(self) -> None:
        _, health = self.request_json("GET", "/health")
        self.require(health.get("status") == "ok", "health status is not ok")
        self.require(health.get("ready") is True, "service is not ready")
        self.require(health.get("model") == self.model, "health model identity differs")
        self.require(isinstance(health.get("pid"), int) and health["pid"] > 0, "health PID is invalid")
        self.require(
            health.get("max_model_len") == self.expected_max_model_len,
            "health max_model_len differs from the expected service contract",
        )
        capabilities = health.get("capabilities")
        self.require(isinstance(capabilities, dict), "health capabilities are absent")
        for name, expected in (
            ("batch_size", 1),
            ("continuous_prompt_lengths", True),
            ("streaming", True),
            ("tool_calls", True),
            ("prefix_cache", True),
            ("bounded_fifo_queue", True),
        ):
            self.require(
                capabilities.get(name) == expected,
                f"health capability {name} differs",
            )
        queue = health.get("queue")
        self.require(isinstance(queue, dict), "health queue state is absent")
        self.require(
            queue.get("accepting_requests") is True,
            "inference queue is not accepting requests",
        )
        self.require(
            queue.get("active_requests") == 0
            and queue.get("waiting_requests") == 0,
            "inference queue is not idle before verification",
        )
        self.require(
            isinstance(queue.get("max_waiting_requests"), int)
            and queue["max_waiting_requests"] >= 0,
            "inference queue capacity is invalid",
        )
        self.require(
            isinstance(queue.get("wait_timeout_seconds"), int)
            and queue["wait_timeout_seconds"] > 0,
            "inference queue timeout is invalid",
        )
        for counter in (
            "started_total",
            "completed_total",
            "queued_total",
            "rejected_total",
            "timed_out_total",
            "shutdown_rejected_total",
        ):
            self.require(
                isinstance(queue.get(counter), int) and queue[counter] >= 0,
                f"inference queue counter is invalid: {counter}",
            )
        self.record(
            "health",
            pid=health["pid"],
            model=health["model"],
            max_model_len=health["max_model_len"],
            max_output_tokens=health.get("max_output_tokens"),
            queue=queue,
            load=health.get("load"),
        )
        self.snapshots["health"] = health

        _, ready = self.request_json("GET", "/ready")
        self.require(ready.get("status") == "ok", "ready status is not ok")
        self.require(ready.get("ready") is True, "ready endpoint is not ready")
        self.require(ready.get("pid") == health["pid"], "ready and health PIDs differ")
        self.require(
            ready.get("model") == health["model"],
            "ready and health model identities differ",
        )
        self.require(
            ready.get("max_model_len") == self.expected_max_model_len,
            "ready max_model_len differs from the expected service contract",
        )
        self.record("ready", pid=ready["pid"], model=ready["model"])

        _, models = self.request_json("GET", "/v1/models")
        self.require(models.get("object") == "list", "models response is not a list object")
        entries = models.get("data")
        self.require(isinstance(entries, list), "models.data is not an array")
        entry = next((item for item in entries if item.get("id") == self.model), None)
        self.require(entry is not None, "served model is absent from /v1/models")
        self.require(entry.get("object") == "model", "models entry has the wrong object type")
        self.require(
            entry.get("max_model_len") == self.expected_max_model_len,
            "models entry has the wrong max_model_len",
        )
        self.record("models", count=len(entries), served_model=self.model)

        encoded_model = urllib.parse.quote(self.model, safe="")
        _, retrieved_model = self.request_json(
            "GET", f"/v1/models/{encoded_model}"
        )
        self.require(
            retrieved_model.get("id") == self.model,
            "retrieved model identity differs",
        )
        self.require(
            retrieved_model.get("object") == "model",
            "retrieved model has the wrong object type",
        )
        self.require(
            retrieved_model.get("max_model_len") == self.expected_max_model_len,
            "retrieved model has the wrong max_model_len",
        )
        self.require(
            retrieved_model == entry,
            "retrieved model differs from its models-list entry",
        )
        self.record("model_retrieve", model=self.model)

        round_trip_text = "qrt HTTP UTF-8 round trip: 上海"
        _, tokenized = self.request_json(
            "POST",
            "/tokenize",
            {
                "model": self.model,
                "prompt": round_trip_text,
                "add_special_tokens": False,
                "return_token_strs": True,
            },
        )
        tokens = tokenized.get("tokens")
        self.require(isinstance(tokens, list) and tokens, "tokenize returned no tokens")
        self.require(tokenized.get("count") == len(tokens), "tokenize count differs from tokens")
        self.require(
            tokenized.get("max_model_len") == self.expected_max_model_len,
            "tokenize max_model_len differs",
        )
        token_strs = tokenized.get("token_strs")
        self.require(
            isinstance(token_strs, list) and len(token_strs) == len(tokens),
            "tokenize token_strs do not align with tokens",
        )
        _, detokenized = self.request_json(
            "POST",
            "/detokenize",
            {"model": self.model, "tokens": tokens, "skip_special_tokens": False},
        )
        self.require(detokenized.get("prompt") == round_trip_text, "tokenizer round trip differs")
        self.record("tokenizer_round_trip", token_count=len(tokens))

        _, chat_tokens = self.request_json(
            "POST",
            "/tokenize",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.require(
            isinstance(chat_tokens.get("tokens"), list) and chat_tokens["tokens"],
            "chat-template tokenization returned no tokens",
        )
        self.record("chat_template_tokenize", token_count=chat_tokens["count"])

    def verify_runtime_identity(
        self,
        state_file: Path | None,
        provenance_files: list[Path],
        require_clean_provenance: bool,
        status_file: Path | None = None,
    ) -> None:
        health = self.snapshots.get("health", {})
        if state_file is not None:
            state_bytes = state_file.read_bytes()
            try:
                state = json.loads(state_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(f"service state is not valid JSON: {state_file}") from error
            self.require(isinstance(state, dict), "service state is not a JSON object")
            self.require(state.get("status") == "ready", "service state is not ready")
            self.require(state.get("pid") == health.get("pid"), "state and health PIDs differ")
            self.require(state.get("model") == self.model, "state and requested models differ")
            self.require(
                state.get("max_model_len") == self.expected_max_model_len,
                "state max_model_len differs from the expected contract",
            )
            queue = health.get("queue", {})
            self.require(
                state.get("max_queue_depth") == queue.get("max_waiting_requests"),
                "state and health queue capacities differ",
            )
            self.require(
                state.get("queue_timeout_seconds")
                == queue.get("wait_timeout_seconds"),
                "state and health queue timeouts differ",
            )
            arbitrary_moe_provider = state.get("arbitrary_moe_provider_dll")
            arbitrary_moe_kernel_dir = state.get("arbitrary_moe_kernel_dir")
            self.require(
                isinstance(arbitrary_moe_provider, str)
                and arbitrary_moe_provider,
                "state has no arbitrary-prompt q1024 MoE provider",
            )
            self.require(
                isinstance(arbitrary_moe_kernel_dir, str)
                and arbitrary_moe_kernel_dir,
                "state has no arbitrary-prompt q1024 MoE kernel directory",
            )
            parsed_url = urllib.parse.urlsplit(self.root_url)
            expected_address = parsed_url.netloc
            self.require(
                state.get("address") == expected_address,
                f"state address {state.get('address')!r} differs from {expected_address!r}",
            )
            self.snapshots["service_state"] = state
            self.record(
                "service_state_identity",
                path=str(state_file.resolve()),
                sha256=sha256_bytes(state_bytes),
                host=state.get("host"),
                pid=state.get("pid"),
                repo_commit=state.get("repo_commit"),
                model_path=state.get("model_path"),
                provider_dll=state.get("provider_dll"),
                arbitrary_moe_provider_dll=arbitrary_moe_provider,
                arbitrary_moe_kernel_dir=arbitrary_moe_kernel_dir,
            )

        if status_file is not None:
            status_bytes = status_file.read_bytes()
            try:
                status = json.loads(status_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(f"CLI status is not valid JSON: {status_file}") from error
            self.require(isinstance(status, dict), "CLI status is not a JSON object")
            self.require(status.get("status") == "ready", "CLI status is not ready")
            self.require(status.get("reachable") is True, "CLI status is not reachable")
            self.require(status.get("endpoint") == self.root_url, "CLI status endpoint differs")
            self.require(status.get("pid") == health.get("pid"), "CLI status PID differs")
            self.require(status.get("model") == self.model, "CLI status model differs")
            self.require(
                status.get("max_model_len") == self.expected_max_model_len,
                "CLI status max model length differs",
            )
            state = self.snapshots.get("service_state")
            if state is not None:
                for field in (
                    "model_path",
                    "provider_dll",
                    "arbitrary_moe_provider_dll",
                    "arbitrary_moe_kernel_dir",
                    "max_queue_depth",
                    "queue_timeout_seconds",
                    "repo_commit",
                    "host",
                    "started_unix_seconds",
                    "ready_unix_seconds",
                ):
                    self.require(status.get(field) == state.get(field), f"CLI status field differs: {field}")
            self.snapshots["cli_status"] = status
            self.record(
                "cli_status",
                path=str(status_file.resolve()),
                sha256=sha256_bytes(status_bytes),
                pid=status["pid"],
                reachable=status["reachable"],
            )

        identity_keys = (
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
        for provenance_file in provenance_files:
            payload_bytes = provenance_file.read_bytes()
            try:
                payload = json.loads(payload_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(
                    f"build provenance is not valid JSON: {provenance_file}"
                ) from error
            self.require(isinstance(payload, dict), "build provenance is not a JSON object")
            if require_clean_provenance and "dirty_tree" in payload:
                self.require(
                    payload["dirty_tree"] is False,
                    f"build provenance is dirty: {provenance_file}",
                )
            identity = {key: payload[key] for key in identity_keys if key in payload}
            artifacts = payload.get("artifacts")
            if isinstance(artifacts, list):
                dll_artifacts = []
                for artifact in artifacts:
                    if not isinstance(artifact, dict):
                        continue
                    artifact_name = artifact.get("name") or artifact.get("path")
                    if isinstance(artifact_name, str) and artifact_name.lower().endswith(".dll"):
                        dll_artifacts.append(
                            {
                                "name": artifact_name,
                                "bytes": artifact.get("bytes"),
                                "sha256": artifact.get("sha256"),
                            }
                        )
                if dll_artifacts:
                    identity["dll_artifacts"] = dll_artifacts
            self.provenance.append(
                {
                    "path": str(provenance_file.resolve()),
                    "sha256": sha256_bytes(payload_bytes),
                    "identity": identity,
                }
            )
        if provenance_files:
            self.record(
                "build_provenance",
                files=len(provenance_files),
                clean_required=require_clean_provenance,
                commits=sorted(
                    {
                        item["identity"].get("repo_commit")
                        for item in self.provenance
                        if item["identity"].get("repo_commit")
                    }
                ),
            )

    def verify_usage(self, value: Any, endpoint: str) -> None:
        usage = value.get("usage") if isinstance(value, dict) else None
        self.require(isinstance(usage, dict), f"{endpoint} has no usage object")
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
        self.require(
            isinstance(prompt, int) and isinstance(completion, int) and isinstance(total, int),
            f"{endpoint} usage fields are not integers",
        )
        self.require(prompt + completion == total, f"{endpoint} usage total is inconsistent")

    def verify_request_id(self, response: HttpResponse, value: Any, endpoint: str) -> None:
        request_id = response.header("x-request-id")
        self.require(isinstance(value, dict) and isinstance(value.get("id"), str), f"{endpoint} has no id")
        self.require(request_id == value["id"], f"{endpoint} x-request-id does not match body id")

    def verify_text_completion(self, max_tokens: int) -> None:
        response, value = self.request_json(
            "POST",
            "/v1/completions",
            {
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "n": 1,
            },
        )
        self.require(value.get("object") == "text_completion", "completion object differs")
        choices = value.get("choices")
        self.require(isinstance(choices, list) and len(choices) == 1, "completion choices differ")
        self.require(isinstance(choices[0].get("text"), str), "completion text is absent")
        self.verify_usage(value, "completions")
        self.verify_request_id(response, value, "completions")
        output = {
            "text": choices[0]["text"],
            "finish_reason": choices[0].get("finish_reason"),
            "completion_tokens": value["usage"]["completion_tokens"],
        }
        self.snapshots["text_completion_output"] = output
        self.record(
            "text_completion",
            finish_reason=output["finish_reason"],
            completion_tokens=output["completion_tokens"],
            output_utf8_sha256=sha256_bytes(output["text"].encode("utf-8")),
            response_sha256=sha256_bytes(response.body),
        )

    def verify_chat_completion(self, max_tokens: int) -> None:
        response, value = self.request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_completion_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.require(value.get("object") == "chat.completion", "chat completion object differs")
        choices = value.get("choices")
        self.require(isinstance(choices, list) and len(choices) == 1, "chat completion choices differ")
        choice = choices[0]
        message = choice.get("message")
        self.require(isinstance(message, dict), "chat completion message is absent")
        content = message.get("content")
        self.require(isinstance(content, str), "chat completion content is absent")
        self.require(not message.get("tool_calls"), "ordinary chat completion emitted a tool call")
        self.verify_usage(value, "chat completion")
        self.verify_request_id(response, value, "chat completion")
        output = {
            "text": content,
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": value["usage"]["completion_tokens"],
        }
        self.snapshots["chat_completion_output"] = output
        self.record(
            "chat_completion",
            finish_reason=output["finish_reason"],
            completion_tokens=output["completion_tokens"],
            output_utf8_sha256=sha256_bytes(content.encode("utf-8")),
            response_sha256=sha256_bytes(response.body),
        )

    def verify_bounded_request_queue(
        self,
        request_count: int,
        prompt_token_id: int,
    ) -> None:
        self.require(request_count >= 2, "queue verification needs at least two requests")
        payload = {
            "model": self.model,
            "prompt": [prompt_token_id] * 17,
            "max_tokens": 1,
            "temperature": 0,
            "top_p": 1,
            "n": 1,
            "ignore_eos": True,
        }
        baseline_response, baseline = self.request_json(
            "POST", "/v1/completions", payload
        )
        self.verify_request_id(baseline_response, baseline, "queue baseline")
        baseline_choice = baseline.get("choices", [{}])[0]
        baseline_semantics = {
            "text": baseline_choice.get("text"),
            "finish_reason": baseline_choice.get("finish_reason"),
            "token_ids": baseline_choice.get("token_ids"),
        }

        _, before_health = self.request_json("GET", "/health")
        before = before_health.get("queue")
        self.require(isinstance(before, dict), "pre-queue health has no queue state")
        self.require(
            before.get("active_requests") == 0
            and before.get("waiting_requests") == 0,
            "inference queue is not idle before concurrent verification",
        )
        self.require(
            isinstance(before.get("max_waiting_requests"), int)
            and before["max_waiting_requests"] >= request_count - 1,
            "configured queue is too small for the requested concurrency check",
        )

        barrier = threading.Barrier(request_count)

        def issue(index: int) -> dict[str, Any]:
            barrier.wait(timeout=min(self.timeout_seconds, 30.0))
            started = time.monotonic()
            response = self.request("POST", "/v1/completions", payload)
            elapsed_ms = (time.monotonic() - started) * 1000.0
            try:
                value = json.loads(response.body)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(
                    f"queued request {index} returned non-JSON: {error}"
                ) from error
            return {
                "index": index,
                "response": response,
                "value": value,
                "elapsed_ms": elapsed_ms,
            }

        concurrent_started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=request_count,
            thread_name_prefix="qrt-queue-verifier",
        ) as executor:
            futures = [executor.submit(issue, index) for index in range(request_count)]
            rows = [future.result(timeout=self.timeout_seconds) for future in futures]
        concurrent_elapsed_ms = (time.monotonic() - concurrent_started) * 1000.0

        request_ids: set[str] = set()
        queue_waits: list[float] = []
        reported_rows: list[dict[str, Any]] = []
        for row in rows:
            response = row["response"]
            value = row["value"]
            self.require(
                response.status == 200,
                f"queued request {row['index']} returned status {response.status}: {value}",
            )
            self.verify_request_id(
                response, value, f"queued request {row['index']}"
            )
            self.verify_usage(value, f"queued request {row['index']}")
            self.require(
                value["usage"].get("prompt_tokens") == 17
                and value["usage"].get("completion_tokens") == 1,
                f"queued request {row['index']} usage differs",
            )
            choice = value.get("choices", [{}])[0]
            semantics = {
                "text": choice.get("text"),
                "finish_reason": choice.get("finish_reason"),
                "token_ids": choice.get("token_ids"),
            }
            self.require(
                semantics == baseline_semantics,
                f"queued request {row['index']} output differs from the baseline",
            )
            request_id = response.header("x-request-id")
            self.require(
                isinstance(request_id, str) and request_id not in request_ids,
                f"queued request {row['index']} has a duplicate request ID",
            )
            request_ids.add(request_id)
            wait_header = response.header("x-qrt-queue-wait-ms")
            try:
                queue_wait_ms = float(wait_header) if wait_header is not None else -1.0
            except ValueError as error:
                raise VerificationError(
                    f"queued request {row['index']} has invalid queue wait metadata"
                ) from error
            metrics = value.get("qrt_metrics")
            self.require(
                isinstance(metrics, dict)
                and isinstance(metrics.get("queue_wait_ms"), (int, float)),
                f"queued request {row['index']} has no queue wait metric",
            )
            self.require(
                queue_wait_ms >= 0
                and abs(queue_wait_ms - float(metrics["queue_wait_ms"])) <= 1.0,
                f"queued request {row['index']} queue wait header and body differ",
            )
            queue_waits.append(queue_wait_ms)
            reported_rows.append(
                {
                    "index": row["index"],
                    "elapsed_ms": round(row["elapsed_ms"], 6),
                    "queue_wait_ms": round(queue_wait_ms, 6),
                    "request_id": request_id,
                    "response_sha256": sha256_bytes(response.body),
                }
            )

        self.require(
            sum(wait > 1.0 for wait in queue_waits) >= request_count - 1,
            "concurrent requests did not exercise the waiting queue",
        )
        _, after_health = self.request_json("GET", "/health")
        after = after_health.get("queue")
        self.require(isinstance(after, dict), "post-queue health has no queue state")
        self.require(
            after.get("active_requests") == 0
            and after.get("waiting_requests") == 0,
            "inference queue did not return to idle",
        )
        self.require(
            after.get("started_total", 0) - before.get("started_total", 0)
            == request_count,
            "queue started-request counter differs",
        )
        self.require(
            after.get("completed_total", 0) - before.get("completed_total", 0)
            == request_count,
            "queue completed-request counter differs",
        )
        self.require(
            after.get("queued_total", 0) - before.get("queued_total", 0)
            >= request_count - 1,
            "queue did not observe every contending request",
        )
        for counter in ("rejected_total", "timed_out_total"):
            self.require(
                after.get(counter) == before.get(counter),
                f"concurrent queue verification changed {counter}",
            )
        self.record(
            "bounded_fifo_request_queue",
            request_count=request_count,
            prompt_tokens=17,
            completion_tokens=1,
            concurrent_elapsed_ms=round(concurrent_elapsed_ms, 6),
            queue_waiting_observed=sum(wait > 1.0 for wait in queue_waits),
            queue_before=before,
            queue_after=after,
            requests=sorted(reported_rows, key=lambda row: row["index"]),
        )

    def verify_prompt_length_matrix(
        self,
        prompt_lengths: list[int],
        prompt_token_id: int,
        reference_paths: list[Path] | None = None,
    ) -> None:
        reference_cases: dict[int, dict[str, Any]] = {}
        reference_bindings: list[dict[str, Any]] = []
        for reference_path in reference_paths or []:
            reference_bytes = reference_path.read_bytes()
            try:
                reference = json.loads(reference_bytes.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise VerificationError(
                    f"prompt-length reference is not valid JSON: {reference_path}"
                ) from error
            self.require(
                isinstance(reference, dict)
                and reference.get("record_type")
                == "openai_prompt_length_reference",
                "prompt-length reference has the wrong record type",
            )
            authority = reference.get("authority")
            self.require(
                isinstance(authority, dict)
                and str(authority.get("host", "")).lower() == "gb10-4t",
                "prompt-length reference is not bound to gb10-4t",
            )
            self.require(
                authority.get("model") == self.model,
                "prompt-length reference model differs",
            )
            request = reference.get("request")
            self.require(
                isinstance(request, dict)
                and request.get("prompt_token_id") == prompt_token_id,
                "prompt-length reference token ID differs",
            )
            self.require(
                isinstance(request.get("prompt_lengths"), list)
                and request["prompt_lengths"]
                and set(request["prompt_lengths"]).issubset(prompt_lengths),
                "prompt-length reference matrix is not a requested subset",
            )
            self.require(
                request.get("max_tokens") == 1
                and request.get("temperature") == 0
                and request.get("top_p") == 1
                and request.get("n") == 1
                and request.get("ignore_eos") is True,
                "prompt-length reference request policy differs",
            )
            cases = reference.get("cases")
            self.require(
                isinstance(cases, list)
                and len(cases) == len(request["prompt_lengths"]),
                "prompt-length reference cases do not align",
            )
            for case in cases:
                self.require(
                    isinstance(case, dict)
                    and isinstance(case.get("prompt_tokens"), int),
                    "prompt-length reference contains an invalid case",
                )
                length = case["prompt_tokens"]
                self.require(
                    length not in reference_cases,
                    "prompt-length reference duplicates a length",
                )
                text = case.get("text")
                self.require(
                    isinstance(text, str)
                    and case.get("text_utf8_sha256")
                    == sha256_bytes(text.encode("utf-8")),
                    f"prompt-length reference text binding differs at {length}",
                )
                self.require(
                    case.get("completion_tokens") == 1,
                    f"prompt-length reference completion count differs at {length}",
                )
                reference_cases[length] = case
            self.require(
                set(request["prompt_lengths"]).issubset(reference_cases),
                "prompt-length reference is missing one of its declared lengths",
            )
            reference_bindings.append(
                {
                    "path": str(reference_path.resolve()),
                    "sha256": sha256_bytes(reference_bytes),
                    "authority": authority,
                    "prompt_lengths": request["prompt_lengths"],
                }
            )
        if reference_paths:
            self.require(
                set(reference_cases) == set(prompt_lengths),
                "prompt-length references do not cover the requested matrix",
            )
        cases: list[dict[str, Any]] = []
        for prompt_length in prompt_lengths:
            response, value = self.request_json(
                "POST",
                "/v1/completions",
                {
                    "model": self.model,
                    "prompt": [prompt_token_id] * prompt_length,
                    "max_tokens": 1,
                    "temperature": 0,
                    "top_p": 1,
                    "n": 1,
                    "ignore_eos": True,
                },
            )
            self.require(
                value.get("object") == "text_completion",
                f"length-{prompt_length} completion object differs",
            )
            choices = value.get("choices")
            self.require(
                isinstance(choices, list) and len(choices) == 1,
                f"length-{prompt_length} completion choices differ",
            )
            self.require(
                isinstance(choices[0].get("text"), str),
                f"length-{prompt_length} completion text is absent",
            )
            label = f"prompt length {prompt_length}"
            self.verify_usage(value, label)
            self.verify_request_id(response, value, label)
            self.require(
                value["usage"]["prompt_tokens"] == prompt_length,
                f"{label} usage does not preserve the exact token-array length",
            )
            self.require(
                value["usage"]["completion_tokens"] == 1,
                f"{label} did not emit exactly one token",
            )
            choice_text = choices[0]["text"]
            reference_case = reference_cases.get(prompt_length)
            if reference_case is not None:
                self.require(
                    choice_text == reference_case["text"],
                    f"{label} text differs from gb10-4t",
                )
                self.require(
                    choices[0].get("finish_reason")
                    == reference_case.get("finish_reason"),
                    f"{label} finish reason differs from gb10-4t",
                )
            metrics = value.get("qrt_metrics")
            self.require(isinstance(metrics, dict), f"{label} has no qrt_metrics")
            cases.append(
                {
                    "prompt_tokens": prompt_length,
                    "completion_tokens": 1,
                    "finish_reason": choices[0].get("finish_reason"),
                    "ttft_ms": metrics.get("ttft_ms"),
                    "request_ms": metrics.get("request_ms"),
                    "response_sha256": sha256_bytes(response.body),
                    "gb10_text_match": reference_case is not None,
                    "gb10_text_utf8_sha256": (
                        reference_case.get("text_utf8_sha256")
                        if reference_case is not None
                        else None
                    ),
                    "gb10_response_sha256": (
                        reference_case.get("response_sha256")
                        if reference_case is not None
                        else None
                    ),
                }
            )
        self.record(
            "continuous_prompt_length_matrix",
            prompt_token_id=prompt_token_id,
            tested_lengths=prompt_lengths,
            gb10_references=reference_bindings,
            cases=cases,
        )

    def verify_context_limit_rejection(self, prompt_token_id: int) -> None:
        response = self.request(
            "POST",
            "/v1/completions",
            {
                "model": self.model,
                "prompt": [prompt_token_id] * self.expected_max_model_len,
                "max_tokens": 1,
                "temperature": 0,
                "top_p": 1,
                "n": 1,
                "ignore_eos": True,
            },
        )
        self.require(response.status == 400, f"context-limit status is {response.status}")
        try:
            value = json.loads(response.body)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise VerificationError(f"context-limit response is not JSON: {error}") from error
        error = value.get("error") if isinstance(value, dict) else None
        self.require(isinstance(error, dict), "context-limit response has no error object")
        self.require(error.get("type") == "invalid_request_error", "context-limit error type differs")
        self.require(error.get("code") == "context_length_exceeded", "context-limit error code differs")
        self.require(error.get("param") == "messages", "context-limit error parameter differs")
        self.record(
            "context_limit_rejection",
            prompt_tokens=self.expected_max_model_len,
            completion_tokens=1,
            error_type=error["type"],
            error_code=error["code"],
            response_sha256=sha256_bytes(response.body),
        )

    def tokenize_prompt(self, prompt: str) -> list[int]:
        _, value = self.request_json(
            "POST",
            "/tokenize",
            {
                "model": self.model,
                "prompt": prompt,
                "add_special_tokens": False,
            },
        )
        tokens = value.get("tokens")
        self.require(
            isinstance(tokens, list) and tokens,
            "prefix probe returned no tokens",
        )
        self.require(
            all(
                isinstance(token, int)
                and not isinstance(token, bool)
                and token >= 0
                for token in tokens
            ),
            "prefix probe returned invalid token IDs",
        )
        self.require(value.get("count") == len(tokens), "prefix probe token count differs")
        return tokens

    def verify_prefix_reuse(
        self,
        service_log: Path,
        minimum_prefix_tokens: int,
        log_wait_seconds: float,
    ) -> None:
        self.require(service_log.is_file(), f"service log does not exist: {service_log}")
        log_offset = service_log.stat().st_size
        nonce = time.time_ns()
        common_text = (
            f"QRT resident HTTP prefix verification {nonce}. "
            + "This deterministic segment must remain shared. " * 384
            + "\nProbe:"
        )
        suffixes = [
            " A",
            " B",
            " C",
            " D",
            " E",
            " F",
            " G",
            " H",
            " I",
            " J",
            " 0",
            " 1",
            " 2",
            " 3",
            "\nA",
            "\nB",
        ]
        tokenized_prompts = [
            (prompt, self.tokenize_prompt(prompt))
            for prompt in (common_text + suffix for suffix in suffixes)
        ]
        first, second, shared_token_count = select_single_token_suffix_pair(
            tokenized_prompts,
            minimum_prefix_tokens,
            min(self.expected_max_model_len - 1, 8191),
        )
        unrelated_text = (
            f"QRT unrelated prefix contamination guard {nonce}. "
            + "This text must not share the resident verification prefix. " * 96
            + "\nUnrelated: Z"
        )
        unrelated_tokens = self.tokenize_prompt(unrelated_text)
        for _ in range(8):
            if len(unrelated_tokens) != len(first[1]):
                break
            unrelated_text += " extra"
            unrelated_tokens = self.tokenize_prompt(unrelated_text)
        else:
            raise VerificationError(
                "could not construct an unrelated prefix with a distinct prompt length"
            )
        self.require(
            len(unrelated_tokens) < self.expected_max_model_len,
            "unrelated prefix guard leaves no output capacity",
        )

        response_hashes: list[str] = []
        semantic_hashes: list[str] = []
        probe_order = (first, second, (unrelated_text, unrelated_tokens), first)
        for probe_index, (prompt, tokens) in enumerate(probe_order):
            response, value = self.request_json(
                "POST",
                "/v1/completions",
                {
                    "model": self.model,
                    "prompt": prompt,
                    "max_tokens": 1,
                    "temperature": 0,
                    "top_p": 1,
                    "n": 1,
                    "ignore_eos": True,
                },
            )
            self.verify_request_id(response, value, "prefix probe")
            self.verify_usage(value, "prefix probe")
            self.require(
                value["usage"]["prompt_tokens"] == len(tokens),
                "prefix probe usage does not match tokenizer output",
            )
            self.require(
                value["usage"]["completion_tokens"] == 1,
                "prefix probe did not emit exactly one token",
            )
            choices = value.get("choices")
            self.require(
                isinstance(choices, list) and len(choices) == 1,
                "prefix probe completion choices differ",
            )
            choice = choices[0]
            self.require(isinstance(choice.get("text"), str), "prefix probe text is absent")
            response_hashes.append(sha256_bytes(response.body))
            semantic_hashes.append(
                sha256_bytes(
                    canonical_json(
                        {
                            "text": choice["text"],
                            "finish_reason": choice.get("finish_reason"),
                            "completion_tokens": value["usage"]["completion_tokens"],
                        }
                    )
                )
            )
            if probe_index == 3:
                self.require(
                    semantic_hashes[3] == semantic_hashes[0],
                    "original prefix output changed after the unrelated request",
                )

        deadline = time.monotonic() + log_wait_seconds
        relevant: list[dict[str, Any]] = []
        delta = b""
        while time.monotonic() <= deadline:
            size = service_log.stat().st_size
            self.require(size >= log_offset, "service log was truncated during prefix probe")
            with service_log.open("rb") as handle:
                handle.seek(log_offset)
                delta = handle.read()
            try:
                log_text = delta.decode("utf-8")
            except UnicodeDecodeError as error:
                raise VerificationError(f"service log is not UTF-8: {error}") from error
            relevant = [
                marker
                for marker in parse_prefix_markers(log_text)
                if marker["prefix_tokens"] == shared_token_count
                and marker["suffix_tokens"] == 1
                and marker["output_tokens"] == 1
            ]
            seed_index = next(
                (index for index, marker in enumerate(relevant) if marker["kind"] == "seed"),
                None,
            )
            seeded_hit_index = next(
                (
                    index
                    for index, marker in enumerate(relevant)
                    if marker["kind"] == "hit" and marker["seed"] is True
                ),
                None,
            )
            reused_hit_index = next(
                (
                    index
                    for index, marker in enumerate(relevant)
                    if marker["kind"] == "hit" and marker["seed"] is False
                ),
                None,
            )
            if (
                seed_index is not None
                and seeded_hit_index is not None
                and reused_hit_index is not None
                and seed_index < seeded_hit_index < reused_hit_index
            ):
                break
            time.sleep(0.1)
        else:
            raise VerificationError(
                "service log did not prove an ordered prefix seed, seeded hit, and resident hit"
            )

        self.record(
            "http_prefix_reuse",
            service_log=str(service_log.resolve()),
            log_offset=log_offset,
            log_delta_sha256=sha256_bytes(delta),
            shared_prefix_tokens=shared_token_count,
            prompt_tokens=len(first[1]),
            prompt_sha256=[
                sha256_bytes(prompt.encode("utf-8")) for prompt, _ in (first, second)
            ],
            prompt_token_ids_sha256=[
                sha256_bytes(canonical_json(tokens)) for _, tokens in (first, second)
            ],
            response_sha256=response_hashes,
            response_semantic_sha256=semantic_hashes,
            probe_order=["shared_a", "shared_b", "unrelated", "shared_a_repeat"],
            unrelated_prompt_tokens=len(unrelated_tokens),
            unrelated_prefix_guard=True,
            repeat_output_match=True,
            markers=relevant,
        )

    @staticmethod
    def parse_sse(body: bytes) -> list[str]:
        try:
            text = body.decode("utf-8")
        except UnicodeDecodeError as error:
            raise VerificationError(f"SSE body is not UTF-8: {error}") from error
        events: list[str] = []
        for block in text.replace("\r\n", "\n").split("\n\n"):
            data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
            if data_lines:
                events.append("\n".join(data_lines))
        return events

    def verify_chat_stream(self, max_tokens: int) -> None:
        response = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "Reply with exactly OK."}],
                "max_completion_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.require(response.status == 200, f"chat stream status is {response.status}")
        content_type = response.header("content-type") or ""
        self.require(content_type.startswith("text/event-stream"), "chat stream content type differs")
        events = self.parse_sse(response.body)
        self.require(events and events[-1] == "[DONE]", "chat stream does not end with [DONE]")
        chunks = [json.loads(event) for event in events[:-1]]
        self.require(chunks, "chat stream has no JSON chunks")
        first_choices = chunks[0].get("choices")
        self.require(
            isinstance(first_choices, list)
            and first_choices
            and first_choices[0].get("delta", {}).get("role") == "assistant",
            "chat stream does not begin with an assistant role chunk",
        )
        usage_chunks = [chunk for chunk in chunks if chunk.get("choices") == []]
        self.require(len(usage_chunks) == 1, "chat stream must contain one usage-only chunk")
        self.verify_usage(usage_chunks[0], "chat stream")
        finish_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("choices")
            and chunk["choices"][0].get("finish_reason") is not None
        ]
        self.require(len(finish_chunks) == 1, "chat stream must contain one finish chunk")
        request_id = response.header("x-request-id")
        self.require(request_id == chunks[0].get("id"), "chat stream request id differs")
        content_parts: list[str] = []
        for chunk in chunks:
            choices = chunk.get("choices")
            if not choices:
                continue
            content = choices[0].get("delta", {}).get("content")
            if content is not None:
                self.require(isinstance(content, str), "chat stream content delta is not text")
                content_parts.append(content)
        stream_text = "".join(content_parts)
        expected = self.snapshots.get("chat_completion_output")
        self.require(isinstance(expected, dict), "chat non-stream output was not captured")
        self.require(stream_text == expected.get("text"), "chat stream output differs from non-stream output")
        finish_reason = finish_chunks[0]["choices"][0]["finish_reason"]
        self.require(finish_reason == expected.get("finish_reason"), "chat stream finish reason differs from non-stream output")
        completion_tokens = usage_chunks[0]["usage"]["completion_tokens"]
        self.require(completion_tokens == expected.get("completion_tokens"), "chat stream token count differs from non-stream output")
        self.record(
            "chat_sse",
            event_count=len(events),
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            output_match_nonstream=True,
            output_utf8_sha256=sha256_bytes(stream_text.encode("utf-8")),
            response_sha256=sha256_bytes(response.body),
        )

    def verify_completion_stream(self, max_tokens: int) -> None:
        response = self.request(
            "POST",
            "/v1/completions",
            {
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "n": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
        )
        self.require(
            response.status == 200,
            f"completion stream status is {response.status}",
        )
        content_type = response.header("content-type") or ""
        self.require(
            content_type.startswith("text/event-stream"),
            "completion stream content type differs",
        )
        events = self.parse_sse(response.body)
        self.require(
            events and events[-1] == "[DONE]",
            "completion stream does not end with [DONE]",
        )
        try:
            chunks = [json.loads(event) for event in events[:-1]]
        except json.JSONDecodeError as error:
            raise VerificationError(
                f"completion stream chunk is not JSON: {error}"
            ) from error
        self.require(chunks, "completion stream has no JSON chunks")
        self.require(
            all(chunk.get("object") == "text_completion" for chunk in chunks),
            "completion stream object type differs",
        )
        request_id = response.header("x-request-id")
        self.require(
            isinstance(request_id, str)
            and request_id
            and all(chunk.get("id") == request_id for chunk in chunks),
            "completion stream request IDs differ",
        )
        usage_chunks = [chunk for chunk in chunks if chunk.get("choices") == []]
        self.require(
            len(usage_chunks) == 1,
            "completion stream must contain one usage-only chunk",
        )
        self.verify_usage(usage_chunks[0], "completion stream")
        completion_tokens = usage_chunks[0]["usage"]["completion_tokens"]
        self.require(
            1 <= completion_tokens <= max_tokens,
            "completion stream emitted an invalid completion-token count",
        )
        finish_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("choices")
            and chunk["choices"][0].get("finish_reason") is not None
        ]
        self.require(
            len(finish_chunks) == 1,
            "completion stream must contain one finish chunk",
        )
        text_parts: list[str] = []
        for chunk in chunks:
            choices = chunk.get("choices")
            if not choices:
                continue
            text = choices[0].get("text")
            self.require(isinstance(text, str), "completion stream text delta is absent")
            text_parts.append(text)
        stream_text = "".join(text_parts)
        expected = self.snapshots.get("text_completion_output")
        self.require(isinstance(expected, dict), "completion non-stream output was not captured")
        self.require(stream_text == expected.get("text"), "completion stream output differs from non-stream output")
        finish_reason = finish_chunks[0]["choices"][0]["finish_reason"]
        self.require(finish_reason == expected.get("finish_reason"), "completion stream finish reason differs from non-stream output")
        self.require(completion_tokens == expected.get("completion_tokens"), "completion stream token count differs from non-stream output")
        self.record(
            "completion_sse",
            event_count=len(events),
            finish_reason=finish_reason,
            completion_tokens=completion_tokens,
            output_match_nonstream=True,
            output_utf8_sha256=sha256_bytes(stream_text.encode("utf-8")),
            response_sha256=sha256_bytes(response.body),
        )

    def verify_tool_call_and_continuation(self, max_tokens: int) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for one city.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": "Call get_weather for Shanghai now. Do not answer in prose.",
            }
        ]
        response, value = self.request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "tools": [tool],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "parallel_tool_calls": False,
                "max_completion_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.verify_request_id(response, value, "tool call")
        self.verify_usage(value, "tool call")
        choices = value.get("choices")
        self.require(isinstance(choices, list) and len(choices) == 1, "tool-call choices differ")
        choice = choices[0]
        self.require(choice.get("finish_reason") == "tool_calls", "tool call finish reason differs")
        assistant = choice.get("message")
        self.require(isinstance(assistant, dict), "tool-call assistant message is absent")
        calls = assistant.get("tool_calls")
        self.require(isinstance(calls, list) and len(calls) == 1, "expected one tool call")
        call = calls[0]
        self.require(call.get("type") == "function", "tool call type differs")
        function = call.get("function")
        self.require(isinstance(function, dict), "tool call function is absent")
        self.require(function.get("name") == "get_weather", "tool call name differs")
        try:
            arguments = json.loads(function.get("arguments", ""))
        except json.JSONDecodeError as error:
            raise VerificationError(f"tool arguments are not JSON: {error}") from error
        self.require(isinstance(arguments, dict), "tool arguments are not an object")
        self.require(
            isinstance(arguments.get("city"), str) and arguments["city"].strip(),
            "tool arguments have no city",
        )
        self.record(
            "structured_tool_call",
            tool_name=function["name"],
            argument_keys=sorted(arguments),
            response_sha256=sha256_bytes(response.body),
        )

        messages.extend(
            [
                assistant,
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(
                        {"city": "Shanghai", "temperature_c": 25, "condition": "clear"},
                        separators=(",", ":"),
                    ),
                },
            ]
        )
        continuation_response, continuation = self.request_json(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": messages,
                "tools": [tool],
                "tool_choice": "none",
                "max_completion_tokens": min(max_tokens, 64),
                "temperature": 0,
                "top_p": 1,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.verify_request_id(continuation_response, continuation, "tool continuation")
        self.verify_usage(continuation, "tool continuation")
        continuation_choice = continuation.get("choices", [{}])[0]
        continuation_message = continuation_choice.get("message", {})
        self.require(
            isinstance(continuation_message.get("content"), str)
            and continuation_message["content"].strip(),
            "tool continuation returned no assistant content",
        )
        self.require(
            not continuation_message.get("tool_calls"),
            "tool continuation emitted another tool call despite tool_choice=none",
        )
        self.record(
            "tool_result_continuation",
            finish_reason=continuation_choice.get("finish_reason"),
            completion_tokens=continuation["usage"]["completion_tokens"],
            response_sha256=sha256_bytes(continuation_response.body),
        )

    def verify_tool_call_stream(self, max_tokens: int) -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for one city.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
        response = self.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Call get_weather for Shanghai now. "
                            "Do not answer in prose."
                        ),
                    }
                ],
                "tools": [tool],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
                "parallel_tool_calls": False,
                "max_completion_tokens": max_tokens,
                "temperature": 0,
                "top_p": 1,
                "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        self.require(response.status == 200, f"tool-call stream status is {response.status}")
        content_type = response.header("content-type") or ""
        self.require(
            content_type.startswith("text/event-stream"),
            "tool-call stream content type differs",
        )
        events = self.parse_sse(response.body)
        self.require(
            events and events[-1] == "[DONE]",
            "tool-call stream does not end with [DONE]",
        )
        try:
            chunks = [json.loads(event) for event in events[:-1]]
        except json.JSONDecodeError as error:
            raise VerificationError(
                f"tool-call stream chunk is not JSON: {error}"
            ) from error
        self.require(chunks, "tool-call stream has no JSON chunks")
        self.require(
            all(chunk.get("object") == "chat.completion.chunk" for chunk in chunks),
            "tool-call stream object type differs",
        )
        request_id = response.header("x-request-id")
        self.require(
            isinstance(request_id, str)
            and request_id
            and all(chunk.get("id") == request_id for chunk in chunks),
            "tool-call stream request IDs differ",
        )
        first_choices = chunks[0].get("choices")
        self.require(
            isinstance(first_choices, list)
            and first_choices
            and first_choices[0].get("delta", {}).get("role") == "assistant",
            "tool-call stream does not begin with an assistant role chunk",
        )
        usage_chunks = [chunk for chunk in chunks if chunk.get("choices") == []]
        self.require(
            len(usage_chunks) == 1,
            "tool-call stream must contain one usage-only chunk",
        )
        self.verify_usage(usage_chunks[0], "tool-call stream")
        completion_tokens = usage_chunks[0]["usage"]["completion_tokens"]
        self.require(
            1 <= completion_tokens <= max_tokens,
            "tool-call stream emitted an invalid completion-token count",
        )
        finish_chunks = [
            chunk
            for chunk in chunks
            if chunk.get("choices")
            and chunk["choices"][0].get("finish_reason") is not None
        ]
        self.require(
            len(finish_chunks) == 1
            and finish_chunks[0]["choices"][0]["finish_reason"] == "tool_calls",
            "tool-call stream finish reason differs",
        )

        calls: dict[int, dict[str, Any]] = {}
        for chunk in chunks:
            choices = chunk.get("choices")
            if not choices:
                continue
            delta_calls = choices[0].get("delta", {}).get("tool_calls", [])
            self.require(
                isinstance(delta_calls, list),
                "tool-call stream delta tool_calls is not an array",
            )
            for delta_call in delta_calls:
                self.require(
                    isinstance(delta_call, dict)
                    and isinstance(delta_call.get("index"), int)
                    and delta_call["index"] >= 0,
                    "tool-call stream has an invalid call index",
                )
                call = calls.setdefault(
                    delta_call["index"],
                    {"id": None, "type": None, "name": "", "arguments": ""},
                )
                for key in ("id", "type"):
                    value = delta_call.get(key)
                    if value is not None:
                        self.require(
                            isinstance(value, str)
                            and (call[key] is None or call[key] == value),
                            f"tool-call stream has an inconsistent {key}",
                        )
                        call[key] = value
                function = delta_call.get("function")
                if function is not None:
                    self.require(
                        isinstance(function, dict),
                        "tool-call stream function delta is not an object",
                    )
                    for key in ("name", "arguments"):
                        value = function.get(key)
                        if value is not None:
                            self.require(
                                isinstance(value, str),
                                f"tool-call stream function {key} is not text",
                            )
                            call[key] += value
        self.require(set(calls) == {0}, "tool-call stream did not emit exactly one call")
        call = calls[0]
        self.require(
            isinstance(call["id"], str) and call["id"],
            "tool-call stream has no call ID",
        )
        self.require(call["type"] == "function", "tool-call stream call type differs")
        self.require(call["name"] == "get_weather", "tool-call stream name differs")
        try:
            arguments = json.loads(call["arguments"])
        except json.JSONDecodeError as error:
            raise VerificationError(
                f"tool-call stream arguments are not JSON: {error}"
            ) from error
        self.require(isinstance(arguments, dict), "tool-call stream arguments are not an object")
        self.require(
            isinstance(arguments.get("city"), str) and arguments["city"].strip(),
            "tool-call stream arguments have no city",
        )
        self.record(
            "structured_tool_call_sse",
            tool_name=call["name"],
            argument_keys=sorted(arguments),
            event_count=len(events),
            completion_tokens=completion_tokens,
            response_sha256=sha256_bytes(response.body),
        )

    def report(self, skipped_generation: bool, started: float) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "passed": True,
            "host": self.snapshots.get("service_state", {}).get("host") or platform.node(),
            "pid": self.snapshots.get("health", {}).get("pid"),
            "endpoint": self.root_url,
            "model": self.model,
            "api_key_recorded": False,
            "generation_checks_skipped": skipped_generation,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "checks": self.checks,
            "health": self.snapshots.get("health"),
            "service_state": self.snapshots.get("service_state"),
            "cli_status": self.snapshots.get("cli_status"),
            "provenance": self.provenance,
        }


def write_report(path: Path | None, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="qwen3.6-35b-a3b")
    parser.add_argument("--api-key")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--expected-max-model-len", type=int, default=262_144)
    parser.add_argument("--max-completion-tokens", type=int, default=128)
    parser.add_argument("--queue-concurrency", type=int, default=3)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--prompt-length", type=int, action="append", default=[])
    parser.add_argument("--prompt-token-id", type=int, default=32)
    parser.add_argument(
        "--prompt-length-reference", type=Path, action="append", default=[]
    )
    parser.add_argument("--verify-prefix-reuse", action="store_true")
    parser.add_argument("--service-log", type=Path)
    parser.add_argument("--prefix-min-tokens", type=int, default=256)
    parser.add_argument("--prefix-log-wait-seconds", type=float, default=5.0)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--provenance-file", type=Path, action="append", default=[])
    parser.add_argument("--require-clean-provenance", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be positive")
    if not 1 <= args.max_completion_tokens <= 512:
        raise SystemExit("--max-completion-tokens must be between 1 and 512")
    if not 2 <= args.queue_concurrency <= 16:
        raise SystemExit("--queue-concurrency must be between 2 and 16")
    if args.prompt_length and args.skip_generation:
        raise SystemExit("--prompt-length cannot be combined with --skip-generation")
    if args.prompt_length_reference and not args.prompt_length:
        raise SystemExit("--prompt-length-reference requires --prompt-length")
    if len(set(args.prompt_length)) != len(args.prompt_length):
        raise SystemExit("--prompt-length values must be unique")
    if any(
        length < 1 or length >= args.expected_max_model_len
        for length in args.prompt_length
    ):
        raise SystemExit(
            "--prompt-length must leave room for the one-token completion"
        )
    if not 0 <= args.prompt_token_id <= 0xFFFF_FFFF:
        raise SystemExit("--prompt-token-id must be an unsigned 32-bit integer")
    if args.verify_prefix_reuse and args.skip_generation:
        raise SystemExit("--verify-prefix-reuse cannot be combined with --skip-generation")
    if args.verify_prefix_reuse and args.service_log is None:
        raise SystemExit("--verify-prefix-reuse requires --service-log")
    if args.prefix_min_tokens < 1:
        raise SystemExit("--prefix-min-tokens must be positive")
    if args.prefix_log_wait_seconds <= 0:
        raise SystemExit("--prefix-log-wait-seconds must be positive")
    if args.output is not None and args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite output: {args.output}")
    started = time.monotonic()
    verifier = Verifier(
        root_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        expected_max_model_len=args.expected_max_model_len,
    )
    try:
        verifier.verify_discovery_and_tokenizer()
        verifier.verify_runtime_identity(
            args.state_file,
            args.provenance_file,
            args.require_clean_provenance,
            args.status_file,
        )
        if not args.skip_generation:
            verifier.verify_text_completion(min(args.max_completion_tokens, 32))
            verifier.verify_completion_stream(min(args.max_completion_tokens, 32))
            verifier.verify_chat_completion(min(args.max_completion_tokens, 32))
            verifier.verify_chat_stream(min(args.max_completion_tokens, 32))
            verifier.verify_tool_call_and_continuation(args.max_completion_tokens)
            verifier.verify_tool_call_stream(args.max_completion_tokens)
            verifier.verify_bounded_request_queue(
                args.queue_concurrency,
                args.prompt_token_id,
            )
            if args.prompt_length:
                verifier.verify_prompt_length_matrix(
                    args.prompt_length,
                    args.prompt_token_id,
                    args.prompt_length_reference,
                )
            verifier.verify_context_limit_rejection(args.prompt_token_id)
            if args.verify_prefix_reuse:
                verifier.verify_prefix_reuse(
                    args.service_log,
                    args.prefix_min_tokens,
                    args.prefix_log_wait_seconds,
                )
        report = verifier.report(args.skip_generation, started)
    except Exception as error:
        report = {
            "schema_version": 1,
            "passed": False,
            "endpoint": verifier.root_url,
            "model": verifier.model,
            "api_key_recorded": False,
            "generation_checks_skipped": args.skip_generation,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "checks": verifier.checks,
            "error": f"{type(error).__name__}: {error}",
        }
        write_report(args.output, report)
        return 1
    write_report(args.output, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
