#!/usr/bin/env python3

import hashlib
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import verify_qrt_openai_server as subject


class VerifyQrtOpenAiServerTest(unittest.TestCase):
    def test_runtime_identity_hashes_raw_bom_provenance_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            state_path = root / "service.json"
            state_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "pid": 123,
                        "address": "127.0.0.1:8000",
                        "model": "test-model",
                        "model_path": r"D:\models\test",
                        "provider_dll": r"D:\runtime\whole.dll",
                        "arbitrary_moe_provider_dll": r"D:\runtime\q1024.dll",
                        "arbitrary_moe_kernel_dir": r"D:\runtime\kernels",
                        "max_model_len": 262_144,
                        "max_queue_depth": 64,
                        "queue_timeout_seconds": 600,
                        "repo_commit": "a" * 40,
                        "host": "BAIYING",
                    }
                ),
                encoding="utf-8",
            )
            provenance_path = root / "build-provenance.json"
            provenance_payload = {
                "schema_version": 1,
                "host": "baiying",
                "repo_commit": "a" * 40,
                "dirty_tree": False,
                "executable_sha256": "1" * 64,
            }
            provenance_bytes = b"\xef\xbb\xbf" + json.dumps(
                provenance_payload
            ).encode("utf-8")
            provenance_path.write_bytes(provenance_bytes)
            state = json.loads(state_path.read_text(encoding="utf-8"))
            status_path = root / "status.json"
            status_path.write_text(
                json.dumps(
                    {
                        "status": "ready",
                        "reachable": True,
                        "endpoint": "http://127.0.0.1:8000",
                        **{
                            field: state[field]
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
                            )
                        },
                        "started_unix_seconds": None,
                        "ready_unix_seconds": None,
                    }
                ),
                encoding="utf-8",
            )

            verifier = subject.Verifier(
                "http://127.0.0.1:8000",
                "test-model",
                None,
                1.0,
                262_144,
            )
            verifier.snapshots["health"] = {
                "pid": 123,
                "queue": {
                    "max_waiting_requests": 64,
                    "wait_timeout_seconds": 600,
                },
            }
            verifier.verify_runtime_identity(
                state_path,
                [provenance_path],
                True,
                status_path,
            )

            self.assertEqual(
                verifier.provenance[0]["sha256"],
                hashlib.sha256(provenance_bytes).hexdigest(),
            )
            self.assertEqual(
                verifier.provenance[0]["identity"]["repo_commit"],
                "a" * 40,
            )
            self.assertEqual(verifier.snapshots["cli_status"]["pid"], 123)

    def test_discovery_retrieves_the_served_model(self):
        model = {
            "id": "test-model",
            "object": "model",
            "created": 1,
            "owned_by": "qrt",
            "max_model_len": 262_144,
        }
        requested_paths = []

        class FakeVerifier(subject.Verifier):
            def request_json(self, method, path, payload=None):
                requested_paths.append((method, path))
                if method == "GET" and path in ("/health", "/ready"):
                    return None, {
                        "status": "ok",
                        "ready": True,
                        "model": "test-model",
                        "pid": 123,
                        "max_model_len": 262_144,
                        "max_output_tokens": 1024,
                        "capabilities": {
                            "batch_size": 1,
                            "continuous_prompt_lengths": True,
                            "streaming": True,
                            "tool_calls": True,
                            "prefix_cache": True,
                            "bounded_fifo_queue": True,
                        },
                        "queue": {
                            "accepting_requests": True,
                            "active_requests": 0,
                            "waiting_requests": 0,
                            "max_waiting_requests": 64,
                            "wait_timeout_seconds": 600,
                            "started_total": 0,
                            "completed_total": 0,
                            "queued_total": 0,
                            "rejected_total": 0,
                            "timed_out_total": 0,
                            "shutdown_rejected_total": 0,
                            "wait_mean_ms": 0.0,
                            "wait_max_ms": 0.0,
                        },
                    }
                if method == "GET" and path == "/v1/models":
                    return None, {"object": "list", "data": [model]}
                if method == "GET" and path == "/v1/models/test-model":
                    return None, model
                if method == "POST" and path == "/tokenize" and "prompt" in payload:
                    return None, {
                        "tokens": [1, 2],
                        "count": 2,
                        "max_model_len": 262_144,
                        "token_strs": ["one", "two"],
                    }
                if method == "POST" and path == "/detokenize":
                    return None, {"prompt": "qrt HTTP UTF-8 round trip: 上海"}
                if method == "POST" and path == "/tokenize" and "messages" in payload:
                    return None, {"tokens": [3], "count": 1}
                raise AssertionError(f"unexpected fake request: {method} {path}")

        verifier = FakeVerifier(
            "http://127.0.0.1:8000",
            "test-model",
            None,
            1.0,
            262_144,
        )
        verifier.verify_discovery_and_tokenizer()
        self.assertIn(("GET", "/v1/models/test-model"), requested_paths)
        self.assertIn("model_retrieve", [check["name"] for check in verifier.checks])

    def test_prompt_length_matrix_runs_exact_token_arrays(self):
        requested_lengths = [17, 8193]

        class FakeVerifier(subject.Verifier):
            def request_json(self, method, path, payload=None):
                if method != "POST" or path != "/v1/completions":
                    raise AssertionError("unexpected fake request")
                prompt = payload.get("prompt")
                if not isinstance(prompt, list) or set(prompt) != {32}:
                    raise AssertionError("prompt matrix must use the selected token ID")
                if len(prompt) not in requested_lengths:
                    raise AssertionError("unexpected prompt length")
                request_id = f"cmpl-length-{len(prompt)}"
                value = {
                    "id": request_id,
                    "object": "text_completion",
                    "choices": [
                        {"index": 0, "text": "X", "finish_reason": "length"}
                    ],
                    "usage": {
                        "prompt_tokens": len(prompt),
                        "completion_tokens": 1,
                        "total_tokens": len(prompt) + 1,
                    },
                    "qrt_metrics": {
                        "ttft_ms": 1.25,
                        "request_ms": 1.5,
                    },
                }
                response = subject.HttpResponse(
                    status=200,
                    headers={"x-request-id": request_id},
                    body=json.dumps(value).encode("utf-8"),
                )
                return response, value

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_paths = []
            for length in requested_lengths:
                reference_path = Path(temp_dir) / f"gb10-reference-{length}.json"
                reference = {
                    "schema_version": 1,
                    "record_type": "openai_prompt_length_reference",
                    "authority": {
                        "host": "gb10-4t",
                        "base_url": "http://gb10.test:8000",
                        "model": "test-model",
                    },
                    "request": {
                        "prompt_token_id": 32,
                        "prompt_lengths": [length],
                        "max_tokens": 1,
                        "temperature": 0,
                        "top_p": 1,
                        "n": 1,
                        "ignore_eos": True,
                    },
                    "cases": [
                        {
                            "prompt_tokens": length,
                            "completion_tokens": 1,
                            "text": "X",
                            "text_utf8_sha256": subject.sha256_bytes(b"X"),
                            "finish_reason": "length",
                            "response_sha256": f"reference-{length}",
                        }
                    ],
                }
                reference_path.write_text(json.dumps(reference), encoding="utf-8")
                reference_paths.append(reference_path)

            verifier = FakeVerifier(
                "http://127.0.0.1:8000",
                "test-model",
                None,
                1.0,
                262_144,
            )
            verifier.verify_prompt_length_matrix(
                requested_lengths,
                32,
                reference_paths,
            )
            check = verifier.checks[-1]
            self.assertEqual(check["name"], "continuous_prompt_length_matrix")
            self.assertEqual(check["details"]["tested_lengths"], requested_lengths)
            self.assertEqual(
                [case["prompt_tokens"] for case in check["details"]["cases"]],
                requested_lengths,
            )
            self.assertEqual(
                check["details"]["gb10_references"][0]["authority"]["host"],
                "gb10-4t",
            )
            self.assertEqual(len(check["details"]["gb10_references"]), 2)
            self.assertTrue(
                all(case["gb10_text_match"] for case in check["details"]["cases"])
            )

    def test_bounded_queue_verifier_requires_contention_and_stable_outputs(self):
        class FakeVerifier(subject.Verifier):
            def __init__(self):
                super().__init__(
                    "http://127.0.0.1:8000",
                    "test-model",
                    None,
                    5.0,
                    262_144,
                )
                self.lock = threading.Lock()
                self.generation_calls = 0
                self.started = 0
                self.completed = 0
                self.queued = 0

            def request(self, method, path, payload=None):
                if method == "GET" and path == "/health":
                    with self.lock:
                        queue = {
                            "accepting_requests": True,
                            "active_requests": 0,
                            "waiting_requests": 0,
                            "max_waiting_requests": 8,
                            "wait_timeout_seconds": 60,
                            "started_total": self.started,
                            "completed_total": self.completed,
                            "queued_total": self.queued,
                            "rejected_total": 0,
                            "timed_out_total": 0,
                            "shutdown_rejected_total": 0,
                            "wait_mean_ms": 0.0,
                            "wait_max_ms": 20.0,
                        }
                    body = json.dumps({"status": "ok", "queue": queue}).encode()
                    return subject.HttpResponse(200, {}, body)
                if method != "POST" or path != "/v1/completions":
                    raise AssertionError("unexpected fake request")
                with self.lock:
                    call = self.generation_calls
                    self.generation_calls += 1
                    self.started += 1
                    self.completed += 1
                    queue_wait_ms = 0.0 if call <= 1 else 10.0 * (call - 1)
                    if call > 1:
                        self.queued += 1
                request_id = f"cmpl-queue-{call}"
                value = {
                    "id": request_id,
                    "object": "text_completion",
                    "choices": [
                        {
                            "index": 0,
                            "text": "X",
                            "finish_reason": "length",
                            "token_ids": [42],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 17,
                        "completion_tokens": 1,
                        "total_tokens": 18,
                    },
                    "qrt_metrics": {"queue_wait_ms": queue_wait_ms},
                }
                return subject.HttpResponse(
                    200,
                    {
                        "x-request-id": request_id,
                        "x-qrt-queue-wait-ms": str(queue_wait_ms),
                    },
                    json.dumps(value).encode(),
                )

        verifier = FakeVerifier()
        verifier.verify_bounded_request_queue(3, 32)
        check = verifier.checks[-1]
        self.assertEqual(check["name"], "bounded_fifo_request_queue")
        self.assertEqual(check["details"]["request_count"], 3)
        self.assertEqual(check["details"]["queue_waiting_observed"], 2)

    def test_context_limit_rejection_uses_exact_maximum_boundary(self):
        class FakeVerifier(subject.Verifier):
            def request(self, method, path, payload=None):
                if method != "POST" or path != "/v1/completions":
                    raise AssertionError("unexpected fake request")
                if payload.get("prompt") != [32] * 8 or payload.get("max_tokens") != 1:
                    raise AssertionError("context boundary request differs")
                body = json.dumps(
                    {
                        "error": {
                            "message": "maximum context exceeded",
                            "type": "invalid_request_error",
                            "param": "messages",
                            "code": "context_length_exceeded",
                        }
                    }
                ).encode("utf-8")
                return subject.HttpResponse(
                    status=400,
                    headers={"content-type": "application/json"},
                    body=body,
                )

        verifier = FakeVerifier(
            "http://127.0.0.1:8000",
            "test-model",
            None,
            1.0,
            8,
        )
        verifier.verify_context_limit_rejection(32)
        check = verifier.checks[-1]
        self.assertEqual(check["name"], "context_limit_rejection")
        self.assertEqual(check["details"]["prompt_tokens"], 8)
        self.assertEqual(check["details"]["error_code"], "context_length_exceeded")

    def test_tool_call_stream_reassembles_fragmented_arguments(self):
        request_id = "chatcmpl-tool-test"
        chunks = [
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-test-0",
                                    "type": "function",
                                    "function": {
                                        "name": "get_",
                                        "arguments": '{"ci',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "name": "weather",
                                        "arguments": 'ty":"Shanghai"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                ],
                "usage": None,
            },
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 32,
                    "completion_tokens": 12,
                    "total_tokens": 44,
                },
            },
        ]
        body = (
            "".join(
                "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        ).encode("utf-8")

        class FakeVerifier(subject.Verifier):
            def request(self, method, path, payload=None):
                if method != "POST" or path != "/v1/chat/completions":
                    raise AssertionError("unexpected fake request")
                if payload.get("stream") is not True:
                    raise AssertionError("tool call must stream")
                if payload.get("stream_options") != {"include_usage": True}:
                    raise AssertionError("tool-call stream must request usage")
                if payload.get("parallel_tool_calls") is not False:
                    raise AssertionError("tool-call stream must request one call")
                return subject.HttpResponse(
                    status=200,
                    headers={
                        "content-type": "text/event-stream; charset=utf-8",
                        "x-request-id": request_id,
                    },
                    body=body,
                )

        verifier = FakeVerifier(
            "http://127.0.0.1:8000",
            "test-model",
            None,
            1.0,
            262_144,
        )
        verifier.verify_tool_call_stream(32)
        check = verifier.checks[-1]
        self.assertEqual(check["name"], "structured_tool_call_sse")
        self.assertEqual(check["details"]["tool_name"], "get_weather")
        self.assertEqual(check["details"]["argument_keys"], ["city"])
        self.assertEqual(check["details"]["completion_tokens"], 12)

    def test_completion_stream_requires_usage_finish_and_done(self):
        request_id = "cmpl-test"
        chunks = [
            {
                "id": request_id,
                "object": "text_completion",
                "choices": [
                    {"index": 0, "text": "Paris", "finish_reason": None}
                ],
                "usage": None,
            },
            {
                "id": request_id,
                "object": "text_completion",
                "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": request_id,
                "object": "text_completion",
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            },
        ]
        body = (
            "".join(
                "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        ).encode("utf-8")

        class FakeVerifier(subject.Verifier):
            def request(self, method, path, payload=None):
                if method != "POST" or path != "/v1/completions":
                    raise AssertionError("unexpected fake request")
                if payload.get("stream_options") != {"include_usage": True}:
                    raise AssertionError("completion stream must request usage")
                return subject.HttpResponse(
                    status=200,
                    headers={
                        "content-type": "text/event-stream; charset=utf-8",
                        "x-request-id": request_id,
                    },
                    body=body,
                )

        verifier = FakeVerifier(
            "http://127.0.0.1:8000",
            "test-model",
            None,
            1.0,
            262_144,
        )
        verifier.snapshots["text_completion_output"] = {
            "text": "Paris",
            "finish_reason": "stop",
            "completion_tokens": 1,
        }
        verifier.verify_completion_stream(8)
        check = verifier.checks[-1]
        self.assertEqual(check["name"], "completion_sse")
        self.assertEqual(check["details"]["completion_tokens"], 1)
        self.assertEqual(check["details"]["finish_reason"], "stop")
        self.assertTrue(check["details"]["output_match_nonstream"])

    def test_chat_stream_matches_nonstream_output(self):
        nonstream_id = "chatcmpl-nonstream"
        stream_id = "chatcmpl-stream"
        nonstream = {
            "id": nonstream_id,
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "OK"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
        chunks = [
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            },
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [
                    {"index": 0, "delta": {"content": "OK"}, "finish_reason": None}
                ],
                "usage": None,
            },
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": None,
            },
            {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            },
        ]
        stream_body = (
            "".join(
                "data: " + json.dumps(chunk, separators=(",", ":")) + "\n\n"
                for chunk in chunks
            )
            + "data: [DONE]\n\n"
        ).encode("utf-8")

        class FakeVerifier(subject.Verifier):
            def request(self, method, path, payload=None):
                if method != "POST" or path != "/v1/chat/completions":
                    raise AssertionError("unexpected fake request")
                if payload.get("stream") is True:
                    return subject.HttpResponse(
                        status=200,
                        headers={
                            "content-type": "text/event-stream; charset=utf-8",
                            "x-request-id": stream_id,
                        },
                        body=stream_body,
                    )
                return subject.HttpResponse(
                    status=200,
                    headers={
                        "content-type": "application/json",
                        "x-request-id": nonstream_id,
                    },
                    body=json.dumps(nonstream).encode("utf-8"),
                )

        verifier = FakeVerifier(
            "http://127.0.0.1:8000",
            "test-model",
            None,
            1.0,
            262_144,
        )
        verifier.verify_chat_completion(8)
        verifier.verify_chat_stream(8)
        self.assertEqual(verifier.checks[-2]["name"], "chat_completion")
        self.assertEqual(verifier.checks[-1]["name"], "chat_sse")
        self.assertTrue(verifier.checks[-1]["details"]["output_match_nonstream"])
        self.assertEqual(
            verifier.checks[-2]["details"]["output_utf8_sha256"],
            verifier.checks[-1]["details"]["output_utf8_sha256"],
        )

    def test_parse_prefix_markers_preserves_seed_state(self):
        markers = subject.parse_prefix_markers(
            "noise\n"
            "QRT_SERVER_MARK prefix_cache_seed prefix_tokens=300 "
            "suffix_tokens=1 output_tokens=1 elapsed_ms=1.25\n"
            "QRT_SERVER_MARK prefix_cache_hit prefix_tokens=300 "
            "suffix_tokens=1 output_tokens=1 seed=1 ttft_ms=2 tpot_ms=0\n"
            "QRT_SERVER_MARK prefix_cache_hit prefix_tokens=300 "
            "suffix_tokens=1 output_tokens=1 seed=0 ttft_ms=1 tpot_ms=0\n"
        )
        self.assertEqual(
            markers,
            [
                {
                    "kind": "seed",
                    "prefix_tokens": 300,
                    "suffix_tokens": 1,
                    "output_tokens": 1,
                    "seed": None,
                },
                {
                    "kind": "hit",
                    "prefix_tokens": 300,
                    "suffix_tokens": 1,
                    "output_tokens": 1,
                    "seed": True,
                },
                {
                    "kind": "hit",
                    "prefix_tokens": 300,
                    "suffix_tokens": 1,
                    "output_tokens": 1,
                    "seed": False,
                },
            ],
        )

    def test_select_suffix_pair_requires_one_distinct_final_token(self):
        shared = list(range(256))
        first, second, shared_count = subject.select_single_token_suffix_pair(
            [
                ("first", shared + [1000]),
                ("duplicate", shared + [1000]),
                ("second", shared + [1001]),
                ("short", [1, 2]),
            ],
            minimum_prefix_tokens=256,
            maximum_prompt_tokens=8191,
        )
        self.assertEqual(first[0], "first")
        self.assertEqual(second[0], "second")
        self.assertEqual(shared_count, 256)

    def test_select_suffix_pair_rejects_nonmatching_or_oversized_prompts(self):
        with self.assertRaisesRegex(subject.VerificationError, "differing in exactly one"):
            subject.select_single_token_suffix_pair(
                [
                    ("one", [1] * 256 + [2]),
                    ("two", [1] * 255 + [3, 4]),
                    ("large-a", [5] * 8191 + [6]),
                    ("large-b", [5] * 8191 + [7]),
                ],
                minimum_prefix_tokens=256,
                maximum_prompt_tokens=8191,
            )

    def test_prefix_probe_binds_ordered_new_log_markers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "service.log"
            log_path.write_text("old log data\n", encoding="utf-8")

            class FakeVerifier(subject.Verifier):
                def __init__(self):
                    super().__init__(
                        "http://127.0.0.1:8000",
                        "test-model",
                        None,
                        1.0,
                        262_144,
                    )
                    self.completion_count = 0

                def tokenize_prompt(self, prompt):
                    if "unrelated prefix contamination guard" in prompt:
                        return list(range(300)) + [ord(prompt[-1])]
                    return list(range(256)) + [ord(prompt[-1])]

                def request_json(self, method, path, payload=None):
                    self.assert_completion_request(method, path, payload)
                    self.completion_count += 1
                    request_id = f"cmpl-prefix-{self.completion_count}"
                    if self.completion_count == 1:
                        markers = (
                            "QRT_SERVER_MARK prefix_cache_seed prefix_tokens=256 "
                            "suffix_tokens=1 output_tokens=1 elapsed_ms=1\n"
                            "QRT_SERVER_MARK prefix_cache_hit prefix_tokens=256 "
                            "suffix_tokens=1 output_tokens=1 seed=1 ttft_ms=1 tpot_ms=0\n"
                        )
                    elif self.completion_count == 3:
                        markers = (
                            "QRT_SERVER_MARK prefix_cache_seed prefix_tokens=300 "
                            "suffix_tokens=1 output_tokens=1 elapsed_ms=1\n"
                            "QRT_SERVER_MARK prefix_cache_hit prefix_tokens=300 "
                            "suffix_tokens=1 output_tokens=1 seed=1 ttft_ms=1 tpot_ms=0\n"
                        )
                    else:
                        markers = (
                            "QRT_SERVER_MARK prefix_cache_hit prefix_tokens=256 "
                            "suffix_tokens=1 output_tokens=1 seed=0 ttft_ms=1 tpot_ms=0\n"
                        )
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(markers)
                    response = subject.HttpResponse(
                        status=200,
                        headers={"x-request-id": request_id},
                        body=request_id.encode("utf-8"),
                    )
                    value = {
                        "id": request_id,
                        "choices": [
                            {
                                "index": 0,
                                "text": payload["prompt"][-1],
                                "finish_reason": "length",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": (
                                301
                                if "unrelated prefix contamination guard" in payload["prompt"]
                                else 257
                            ),
                            "completion_tokens": 1,
                            "total_tokens": (
                                302
                                if "unrelated prefix contamination guard" in payload["prompt"]
                                else 258
                            ),
                        },
                    }
                    return response, value

                def assert_completion_request(self, method, path, payload):
                    if method != "POST" or path != "/v1/completions":
                        raise AssertionError("unexpected fake request")
                    if payload.get("ignore_eos") is not True:
                        raise AssertionError("prefix probe must ignore EOS")

            verifier = FakeVerifier()
            verifier.verify_prefix_reuse(log_path, 256, 0.1)
            self.assertEqual(verifier.completion_count, 4)
            check = verifier.checks[-1]
            self.assertEqual(check["name"], "http_prefix_reuse")
            self.assertEqual(check["details"]["shared_prefix_tokens"], 256)
            self.assertEqual(
                [marker["seed"] for marker in check["details"]["markers"]],
                [None, True, False, False],
            )
            self.assertTrue(check["details"]["unrelated_prefix_guard"])
            self.assertTrue(check["details"]["repeat_output_match"])
            self.assertEqual(
                check["details"]["response_semantic_sha256"][0],
                check["details"]["response_semantic_sha256"][3],
            )


if __name__ == "__main__":
    unittest.main()
