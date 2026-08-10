#!/usr/bin/env python3

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_openai_http_product_acceptance as subject


def write_json(path: Path, value: object, *, utf8_bom: bool = False) -> str:
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if utf8_bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def provenance_identity(payload: dict) -> dict:
    identity = {
        key: payload[key]
        for key in subject.PROVENANCE_IDENTITY_KEYS
        if key in payload
    }
    dlls = []
    for artifact in payload.get("artifacts", []):
        name = artifact.get("name") or artifact.get("path")
        if isinstance(name, str) and name.casefold().endswith(".dll"):
            dlls.append(
                {
                    "name": name,
                    "bytes": artifact.get("bytes"),
                    "sha256": artifact.get("sha256"),
                }
            )
    if dlls:
        identity["dll_artifacts"] = dlls
    return identity


class OpenAiHttpProductAcceptanceAuditTest(unittest.TestCase):
    def build_fixture(self, root: Path) -> dict:
        model = "qwen3.6-35b-a3b"
        model_path = r"D:\models\Qwen3.6-35B-A3B"
        executable = root / "qrt.exe"
        executable.write_bytes(b"clean qrt executable")
        qrt_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
        whole_provider = root / "whole-provider.dll"
        whole_provider.write_bytes(b"clean whole provider")
        whole_sha = hashlib.sha256(whole_provider.read_bytes()).hexdigest()
        q1024_provider = root / "q1024-provider.dll"
        q1024_provider.write_bytes(b"clean q1024 provider")
        q1024_sha = hashlib.sha256(q1024_provider.read_bytes()).hexdigest()

        provenance_payloads = [
            {
                "schema_version": 1,
                "host": "baiying",
                "execution": "local_windows_process",
                "repo_commit": "a" * 40,
                "dirty_tree": False,
                "command_file": "build-qrt.ps1",
                "executable": str(executable),
                "executable_sha256": qrt_sha,
            },
            {
                "schema_version": 1,
                "host": "baiying",
                "execution": "local_windows_process",
                "repo_commit": "b" * 40,
                "dirty_tree": False,
                "command_file": "build-whole.ps1",
                "source_sha256": "3" * 64,
                "artifacts": [
                    {
                        "name": "qrt_qwen36_whole_provider.dll",
                        "bytes": 10,
                        "sha256": whole_sha,
                    }
                ],
            },
            {
                "schema_version": 1,
                "host": "baiying",
                "execution": "local_windows_process",
                "repo_commit": "c" * 40,
                "dirty_tree": False,
                "command_file": "build-q1024.ps1",
                "variant": "exact",
                "artifacts": [
                    {
                        "path": "qrt_triton_moe_q1024_exact_provider_slots64.dll",
                        "bytes": 11,
                        "sha256": q1024_sha,
                    }
                ],
            },
        ]
        provenance = []
        provenance_hashes = set()
        for index, payload in enumerate(provenance_payloads):
            path = root / f"provenance-{index}.json"
            digest = write_json(path, payload, utf8_bom=index == 0)
            provenance_hashes.add(digest)
            provenance.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "identity": provenance_identity(payload),
                }
            )

        state_path = root / "service.json"
        ready_state = {
            "schema_version": 1,
            "status": "ready",
            "pid": 4242,
            "address": "127.0.0.1:18000",
            "model": model,
            "model_path": model_path,
            "provider_dll": str(whole_provider),
            "arbitrary_moe_provider_dll": str(q1024_provider),
            "arbitrary_moe_kernel_dir": "q1024-kernels",
            "max_model_len": 262_144,
            "max_queue_depth": 64,
            "queue_timeout_seconds": 600,
            "repo_commit": "a" * 40,
            "host": "BAIYING",
            "started_unix_seconds": 100,
            "ready_unix_seconds": 120,
            "stopped_unix_seconds": None,
            "message": "ready",
        }
        ready_state_data = (json.dumps(ready_state, indent=2) + "\n").encode()
        ready_state_sha = hashlib.sha256(ready_state_data).hexdigest()

        ready_runtime = {
            "schema_version": 1,
            "record_type": "qrt_live_service_runtime_evidence",
            "host": "BAIYING",
            "execution": "local_windows_process",
            "service_state": {
                "path": str(state_path),
                "sha256": ready_state_sha,
                "payload": ready_state,
            },
            "service_process": {
                "pid": 4242,
                "name": "qrt.exe",
                "executable_path": str(executable),
                "executable_sha256": qrt_sha,
                "command_line": "qrt.exe serve " + " ".join(
                    f"--set-env {fragment}"
                    for fragment in subject.REQUIRED_ORDINARY_COMMAND_FRAGMENTS
                ),
                "api_key_redacted": False,
            },
            "listener": [
                {
                    "local_address": "127.0.0.1",
                    "local_port": 18000,
                    "owning_process": 4242,
                }
            ],
            "artifacts": [
                {
                    "path": str(whole_provider),
                    "bytes": whole_provider.stat().st_size,
                    "sha256": whole_sha,
                },
                {
                    "path": str(q1024_provider),
                    "bytes": q1024_provider.stat().st_size,
                    "sha256": q1024_sha,
                },
            ],
            "loaded_modules": [
                {
                    "name": whole_provider.name,
                    "path": str(whole_provider),
                    "bytes": whole_provider.stat().st_size,
                    "sha256": whole_sha,
                },
                {
                    "name": q1024_provider.name,
                    "path": str(q1024_provider),
                    "bytes": q1024_provider.stat().st_size,
                    "sha256": q1024_sha,
                },
            ],
        }
        ready_path = root / "ready-runtime.json"
        ready_sha = write_json(ready_path, ready_runtime)

        cli_status = {
            "status": "ready",
            "reachable": True,
            "endpoint": "http://127.0.0.1:18000",
            **{
                field: ready_state[field]
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
                    "stopped_unix_seconds",
                    "message",
                )
            },
        }
        cli_status_path = root / "status.json"
        cli_status_sha = write_json(cli_status_path, cli_status)

        stopped_state = copy.deepcopy(ready_state)
        stopped_state.update(
            {
                "status": "stopped",
                "stopped_unix_seconds": 900,
                "message": None,
            }
        )
        write_json(state_path, stopped_state)
        shutdown_runtime = {
            "schema_version": 1,
            "record_type": "qrt_stopped_service_runtime_evidence",
            "host": "BAIYING",
            "execution": "local_windows_process",
            "ready_evidence": {
                "path": str(ready_path),
                "bytes": ready_path.stat().st_size,
                "sha256": ready_sha,
                "service_process": ready_runtime["service_process"],
            },
            "stopped_state": {
                "path": str(state_path),
                "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
                "payload": stopped_state,
            },
            "assertions": {
                "same_service_instance": True,
                "terminal_state": "stopped",
                "stopped_timestamp_present": True,
                "process_absent": True,
                "listener_absent": True,
                "pid": 4242,
                "address": "127.0.0.1:18000",
                "model": model,
            },
        }
        shutdown_path = root / "shutdown-runtime.json"
        write_json(shutdown_path, shutdown_runtime)

        reference_hashes = {"4" * 64, "5" * 64}
        prompt_lengths = [17, 262_143]

        def check(name, **details):
            return {"name": name, "passed": True, "details": details}

        common_generation = {
            "completion_tokens": 1,
            "response_sha256": "6" * 64,
        }
        checks = [
            check("health", pid=4242),
            check("ready", pid=4242),
            check("models", count=1, served_model=model),
            check("model_retrieve", model=model),
            check("tokenizer_round_trip", token_count=5),
            check("chat_template_tokenize", token_count=9),
            check(
                "service_state_identity",
                path=str(state_path),
                sha256=ready_state_sha,
            ),
            check(
                "cli_status",
                path=str(cli_status_path),
                sha256=cli_status_sha,
                pid=4242,
                reachable=True,
            ),
            check("build_provenance", files=3, clean_required=True),
            check(
                "text_completion",
                output_utf8_sha256="d" * 64,
                **common_generation,
            ),
            check(
                "completion_sse",
                event_count=3,
                output_match_nonstream=True,
                output_utf8_sha256="d" * 64,
                **common_generation,
            ),
            check(
                "chat_completion",
                output_utf8_sha256="e" * 64,
                **common_generation,
            ),
            check(
                "chat_sse",
                event_count=3,
                output_match_nonstream=True,
                output_utf8_sha256="e" * 64,
                **common_generation,
            ),
            check(
                "structured_tool_call",
                tool_name="get_weather",
                argument_keys=["city"],
                **common_generation,
            ),
            check("tool_result_continuation", **common_generation),
            check(
                "structured_tool_call_sse",
                tool_name="get_weather",
                argument_keys=["city"],
                event_count=3,
                **common_generation,
            ),
            check(
                "continuous_prompt_length_matrix",
                prompt_token_id=32,
                tested_lengths=prompt_lengths,
                gb10_references=[
                    {
                        "path": "short.json",
                        "sha256": "4" * 64,
                        "authority": {"host": "gb10-4t", "model": model},
                        "prompt_lengths": [17],
                    },
                    {
                        "path": "max.json",
                        "sha256": "5" * 64,
                        "authority": {"host": "gb10-4t", "model": model},
                        "prompt_lengths": [262_143],
                    },
                ],
                cases=[
                    {
                        "prompt_tokens": length,
                        "completion_tokens": 1,
                        "gb10_text_match": True,
                        "response_sha256": "7" * 64,
                        "gb10_text_utf8_sha256": "8" * 64,
                        "gb10_response_sha256": "9" * 64,
                        "ttft_ms": 1.0,
                        "request_ms": 2.0,
                    }
                    for length in prompt_lengths
                ],
            ),
            check(
                "context_limit_rejection",
                prompt_tokens=262_144,
                completion_tokens=1,
                error_type="invalid_request_error",
                error_code="context_length_exceeded",
                response_sha256="c" * 64,
            ),
            check(
                "http_prefix_reuse",
                shared_prefix_tokens=300,
                prompt_tokens=301,
                response_sha256=["a" * 64, "b" * 64, "c" * 64, "d" * 64],
                response_semantic_sha256=["e" * 64, "f" * 64, "0" * 64, "e" * 64],
                probe_order=["shared_a", "shared_b", "unrelated", "shared_a_repeat"],
                unrelated_prompt_tokens=512,
                unrelated_prefix_guard=True,
                repeat_output_match=True,
                markers=[
                    {"kind": "seed", "seed": None},
                    {"kind": "hit", "seed": True},
                    {"kind": "hit", "seed": False},
                ],
            ),
            check(
                "bounded_fifo_request_queue",
                request_count=3,
                prompt_tokens=17,
                completion_tokens=1,
                concurrent_elapsed_ms=30.0,
                queue_waiting_observed=2,
                queue_before={
                    "active_requests": 0,
                    "waiting_requests": 0,
                    "started_total": 10,
                    "completed_total": 10,
                    "queued_total": 0,
                    "rejected_total": 0,
                    "timed_out_total": 0,
                },
                queue_after={
                    "active_requests": 0,
                    "waiting_requests": 0,
                    "started_total": 13,
                    "completed_total": 13,
                    "queued_total": 2,
                    "rejected_total": 0,
                    "timed_out_total": 0,
                },
                requests=[
                    {
                        "index": index,
                        "elapsed_ms": 10.0 * (index + 1),
                        "queue_wait_ms": 10.0 * index,
                        "request_id": f"cmpl-queue-{index}",
                        "response_sha256": str(index + 1) * 64,
                    }
                    for index in range(3)
                ],
            ),
        ]
        verification = {
            "schema_version": 1,
            "passed": True,
            "host": "BAIYING",
            "pid": 4242,
            "endpoint": "http://127.0.0.1:18000",
            "model": model,
            "api_key_recorded": False,
            "generation_checks_skipped": False,
            "checks": checks,
            "health": {
                "status": "ok",
                "ready": True,
                "pid": 4242,
                "model": model,
                "max_model_len": 262_144,
                "capabilities": subject.EXPECTED_CAPABILITIES,
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
                "load": {"total_ms": 19_900.0},
            },
            "service_state": ready_state,
            "cli_status": cli_status,
            "provenance": provenance,
        }
        verification_path = root / "verification.json"
        write_json(verification_path, verification)
        return {
            "verification": verification,
            "verification_path": verification_path,
            "ready_path": ready_path,
            "shutdown_path": shutdown_path,
            "model": model,
            "model_path": model_path,
            "prompt_lengths": prompt_lengths,
            "reference_hashes": frozenset(reference_hashes),
            "provenance_hashes": frozenset(provenance_hashes),
            "artifact_hashes": frozenset({qrt_sha, whole_sha, q1024_sha}),
        }

    def invoke(self, fixture: dict):
        return subject.audit_acceptance(
            fixture["verification_path"],
            fixture["ready_path"],
            fixture["shutdown_path"],
            fixture["model"],
            fixture["model_path"],
            "baiying",
            262_144,
            fixture["prompt_lengths"],
            fixture["reference_hashes"],
            fixture["provenance_hashes"],
            fixture["artifact_hashes"],
            30_000.0,
        )

    def test_complete_http_and_lifecycle_evidence_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            report = self.invoke(fixture)
            self.assertTrue(report["passed"])
            self.assertTrue(report["requirements"]["structured_tool_call_json_and_sse"])
            self.assertEqual(report["prompt_lengths"], [17, 262_143])

    def test_unbound_max_length_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            matrix = next(
                check
                for check in fixture["verification"]["checks"]
                if check["name"] == "continuous_prompt_length_matrix"
            )
            matrix["details"]["cases"][-1]["gb10_text_match"] = False
            write_json(fixture["verification_path"], fixture["verification"])
            with self.assertRaisesRegex(subject.AuditError, "not gb10-bound"):
                self.invoke(fixture)

    def test_shutdown_pid_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            shutdown = json.loads(fixture["shutdown_path"].read_text())
            shutdown["assertions"]["pid"] += 1
            write_json(fixture["shutdown_path"], shutdown)
            with self.assertRaisesRegex(subject.AuditError, "shutdown PID"):
                self.invoke(fixture)

    def test_service_commit_must_match_live_executable_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            entry = fixture["verification"]["provenance"][0]
            provenance_path = Path(entry["path"])
            payload = json.loads(provenance_path.read_bytes().decode("utf-8-sig"))
            payload["repo_commit"] = "d" * 40
            old_digest = entry["sha256"]
            new_digest = write_json(provenance_path, payload, utf8_bom=True)
            entry["sha256"] = new_digest
            entry["identity"] = provenance_identity(payload)
            fixture["provenance_hashes"] = frozenset(
                (set(fixture["provenance_hashes"]) - {old_digest}) | {new_digest}
            )
            write_json(fixture["verification_path"], fixture["verification"])
            with self.assertRaisesRegex(subject.AuditError, "service commit"):
                self.invoke(fixture)

    def test_unsnapshotted_live_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            ready = json.loads(fixture["ready_path"].read_text())
            ready["artifacts"] = ready["artifacts"][:1]
            ready_sha = write_json(fixture["ready_path"], ready)
            shutdown = json.loads(fixture["shutdown_path"].read_text())
            shutdown["ready_evidence"]["bytes"] = fixture["ready_path"].stat().st_size
            shutdown["ready_evidence"]["sha256"] = ready_sha
            write_json(fixture["shutdown_path"], shutdown)
            with self.assertRaisesRegex(subject.AuditError, "arbitrary_moe_provider_dll"):
                self.invoke(fixture)

    def test_unloaded_live_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self.build_fixture(Path(temporary))
            ready = json.loads(fixture["ready_path"].read_text())
            ready["loaded_modules"] = ready["loaded_modules"][:1]
            ready_sha = write_json(fixture["ready_path"], ready)
            shutdown = json.loads(fixture["shutdown_path"].read_text())
            shutdown["ready_evidence"]["bytes"] = fixture["ready_path"].stat().st_size
            shutdown["ready_evidence"]["sha256"] = ready_sha
            write_json(fixture["shutdown_path"], shutdown)
            with self.assertRaisesRegex(subject.AuditError, "not loaded"):
                self.invoke(fixture)


if __name__ == "__main__":
    unittest.main()
