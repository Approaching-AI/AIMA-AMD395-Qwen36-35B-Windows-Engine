#!/usr/bin/env python3
"""Two-request real-token HTTP regression, inside baiying's Windows Job guard.

This is a token/transport/stability check, not the raw-logit product gate.
Without --execute only fingerprints and the saved oracle are checked.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    # Every invocation owns a new output directory; do not replace old evidence.
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=True, indent=2)
        output.write("\n")


def fingerprint(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def token_digest(tokens):
    require(isinstance(tokens, list), "tokens must be an array")
    require(all(type(t) is int and 0 <= t < 2**32 for t in tokens), "invalid token ID")
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def token_fnv(tokens, offset_basis):
    token_digest(tokens)  # Apply the same explicit u32 validation.
    value = offset_basis
    for byte in struct.pack(f"<{len(tokens)}I", *tokens):
        value = ((value ^ byte) * 1099511628211) & (2**64 - 1)
    return f"{value:016x}"


def validate_fixture(prompt, oracle):
    require(oracle["status"] == "pass", "reference contract did not pass")
    require(oracle["correctness_authority"]["host"] == "gb10-4t", "wrong authority")
    expected = oracle["expected"]["output_token_ids"]
    require(len(prompt) == oracle["prompt"]["token_count"] == 8192, "requires q8192")
    require(len(expected) == oracle["expected"]["continuation_token_count"] == 32,
            "requires the complete 32-token reference")
    require(token_digest(prompt) == oracle["prompt"]["u32le_sha256"], "prompt digest differs")
    require(token_digest(expected) == oracle["expected"]["output_token_ids_u32le_sha256"],
            "reference continuation digest differs")
    require(expected[0] == oracle["expected"]["first_token_id"], "reference first token differs")
    return expected


def verify_inventory(manifest_path):
    root = Path(manifest_path).resolve().parent
    artifacts = read_json(manifest_path)["artifacts"]
    require(bool(artifacts), "empty runtime inventory")
    seen = set()
    for entry in artifacts:
        path = (root / entry["path"]).resolve()
        require(path.is_relative_to(root) and path != root, "artifact escapes runtime root")
        require(path not in seen, "duplicate runtime artifact")
        seen.add(path)
        require(path.stat().st_size == entry["bytes"], f"artifact size differs: {entry['path']}")
        require(fingerprint(path) == entry["sha256"], f"artifact hash differs: {entry['path']}")
    return len(artifacts)


def preflight(config):
    files = config["files"]
    for name in ("server", "provider", "env", "prompt", "oracle", "runtime_manifest"):
        require(fingerprint(files[name]["path"]) == files[name]["sha256"],
                f"input fingerprint differs: {name}")
    prompt = read_json(files["prompt"]["path"])
    expected = validate_fixture(prompt, read_json(files["oracle"]["path"]))
    count = verify_inventory(files["runtime_manifest"]["path"])
    require(type(config["port"]) is int and 1024 <= config["port"] <= 65535, "invalid port")
    require(Path(config["model"]).is_dir(), "model directory missing")
    source = subprocess.run(["git", "-C", config["working_directory"], "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    require(source == config["server_commit"], "server checkout commit differs")
    dirty = subprocess.run(["git", "-C", config["working_directory"], "status", "--porcelain"],
                           capture_output=True, text=True, check=True, timeout=5).stdout.strip()
    require(not dirty, "server source checkout is dirty")
    return prompt, expected, {"pass": True, "runtime_artifacts_verified": count,
                              "files": files, "server_checkout_commit": source}


def require_windows_job():
    require(os.name == "nt" and platform.node().split(".")[0].lower() == "baiying",
            "execution requires native Windows on baiying")
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE,
                                     ctypes.POINTER(wintypes.BOOL)]
    kernel.IsProcessInJob.restype = wintypes.BOOL
    member = wintypes.BOOL()
    require(kernel.IsProcessInJob(kernel.GetCurrentProcess(), None, ctypes.byref(member)),
            "cannot check Windows Job ownership")
    require(member.value, "launch this controller through baiying_guarded_inference.ps1")


class Client:
    def __init__(self, port):
        self.root = f"http://127.0.0.1:{port}"
        # Machine-to-machine loopback requests must never use a system proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open(self, path, payload=None, timeout=15):
        request = urllib.request.Request(
            self.root + path,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            return self.opener.open(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            return error

    def json(self, path, payload=None, status=200, timeout=15):
        with self.open(path, payload, timeout) as response:
            data = response.read(1024 * 1024 + 1)
            require(len(data) <= 1024 * 1024, "HTTP response size limit")
            value = json.loads(data)
            require(response.status == status, f"{path}: HTTP {response.status}: {value}")
            return value


def check_usage(value):
    require(value == {"prompt_tokens": 8192, "completion_tokens": 32, "total_tokens": 8224},
            "HTTP usage differs from the oracle shape")


def check_completion(value, expected, expected_text):
    require(value["object"] == "text_completion", "wrong completion type")
    require(len(value["choices"]) == 1, "batch-one completion required")
    choice = value["choices"][0]
    require(choice["token_ids"] == expected, "non-stream token sequence differs from GB10")
    require(choice["text"] == expected_text, "completion text differs from reference detokenization")
    require(choice["finish_reason"] == "length", "unexpected completion finish reason")
    check_usage(value["usage"])


def check_stream(events, expected_text):
    require(events and events[-1]["data"] == "[DONE]", "SSE has no terminal DONE")
    chunks = [json.loads(event["data"]) for event in events[:-1]]
    text = ""
    finishes = []
    usage = []
    for chunk in chunks:
        require("error" not in chunk, f"SSE error: {chunk}")
        for choice in chunk["choices"]:
            require(not finishes, "SSE choice after finish")
            text += choice["text"]
            if choice["finish_reason"] is not None:
                finishes.append(choice["finish_reason"])
        if chunk.get("usage") is not None:
            require(bool(finishes) and chunk["choices"] == [], "usage before finish")
            usage.append(chunk)
    require(text == expected_text, "SSE text differs from oracle detokenization")
    require(finishes == ["length"] and len(usage) == 1, "SSE finish/usage count differs")
    check_usage(usage[0]["usage"])
    return usage[0]["qrt_metrics"]


def check_first_token_observations(log_text, oracle, prompt):
    validate_fixture(prompt, oracle)
    # Observation v1 uses the standard offset; the frozen QRT oracle uses a
    # historical shorter offset. Bind both to the *same SHA-verified tokens*,
    # without changing either frozen contract or the native observation.
    observation_fnv = token_fnv(prompt, 14695981039346656037)
    require(token_fnv(prompt, 1469598103934665603) == oracle["prompt"]["u32le_fnv1a64"],
            "legacy oracle digest does not match the SHA-verified tokens")
    rows = [json.loads(line) for line in log_text.splitlines()
            if line.startswith('{"type":"qrt_server_first_token_observation"')]
    require(len(rows) == 2, "requires two same-run first-token observations")
    expected = oracle["expected"]
    for row in rows:
        require(row["contract_version"] == 1 and row["available"] is True
                and row["prefix_route_used"] is False, "unavailable or stale first-token report")
        require(row["source"] == "qrt_engine_report.baseline_output_head_topk_logits[0]",
                "unexpected raw-logit source")
        require(row["input_tokens"] == 8192 and row["output_tokens"] == 32, "observation shape differs")
        require(row["prompt_token_ids_fnv1a64"] == observation_fnv,
                "observation prompt differs")
        require(row["output_token_id"] == expected["first_token_id"], "observed first token differs")
        logit = row["first_token_raw_logit"]
        require(type(logit) in (float, int) and math.isfinite(logit)
                and abs(logit - expected["first_token_raw_logit"]) <= expected["first_token_raw_logit_tolerance"],
                "first-token raw logit fails GB10 tolerance")
    return rows


def check_timing_contract(metrics):
    require(metrics.get("timing_contract_version") == 2, "requires corrected timing contract")
    require(metrics["tpot_samples"] == 31, "wrong decode sample count")
    total, mean = metrics["decode_total_ms"], metrics["tpot_ms"]
    require(type(total) in (int, float) and type(mean) in (int, float)
            and math.isfinite(total) and math.isfinite(mean) and total > 0
            and math.isclose(mean * 31, total, rel_tol=1e-12), "TPOT is not the per-token mean")


def replay_saved_run(run_dir, guard_path, prompt_path, oracle_path):
    """Validate saved transport after a parser repair without more GPU work.

    The original controller status is immutable and stays separate. A replay
    cannot supply missing raw logits, post-request queue samples or token IDs
    which the streaming protocol did not expose.
    """
    paths = {"result": run_dir / "result.json", "completion": run_dir / "completion.json",
             "stream": run_dir / "stream.json", "state": run_dir / "service.json",
             "guard": guard_path, "prompt": prompt_path, "oracle": oracle_path}
    fingerprints = {name: fingerprint(path) for name, path in paths.items()}
    original = read_json(paths["result"])
    completion = read_json(paths["completion"])
    stream = read_json(paths["stream"])
    state = read_json(paths["state"])
    guard = read_json(paths["guard"])
    prompt, oracle = read_json(prompt_path), read_json(oracle_path)
    expected = validate_fixture(prompt, oracle)
    require(original["preflight"]["pass"] is True, "original preflight failed")
    for name in ("prompt", "oracle"):
        require(fingerprints[name] == original["preflight"]["files"][name]["sha256"],
                f"saved {name} does not match executed input")
    require(original.get("nonstream_exact_32_tokens") is True,
            "original run did not verify reference detokenization")
    check_completion(completion, expected, completion["choices"][0]["text"])
    metrics = check_stream(stream, completion["choices"][0]["text"])
    require(original["server_exit_code"] == 0 and not original.get("forced_owned_process_cleanup"),
            "original server did not exit gracefully")
    require(state["status"] == "stopped" and state["pid"] == original["pid"],
            "stopped service identity differs")
    require(state["repo_commit"] == original["server_commit"], "service commit differs")
    require(guard["host"].lower() == state["host"].lower() == "baiying", "guard host differs")
    require(guard["spec"]["model"] == state["model_path"] == original["model"], "guard model differs")
    require(guard["spec"]["repo_commit"] == original["server_commit"], "guard source differs")
    require(guard["reason"] == "completed" and guard["host_checks_pass"] is True,
            "supervisor timeout or unhealthy cleanup")
    require(guard["host_checks"] and all(guard["host_checks"].values()), "host check failed")
    during = original["health_during_sse"]
    require(during["pid"] == original["pid"] and during["ready"] is True,
            "live health identity differs")
    require(during["queue"]["started_total"] == 2 and during["queue"]["completed_total"] == 1
            and during["queue"]["active_requests"] == 1, "no active-generation health observation")
    result = {"status": "offline_http_result_validation_pass", "gpu_executed": False,
            "source_fingerprints": fingerprints, "original_controller_status": original["status"],
            "original_controller_error": original.get("error"), "host": state["host"],
            "model": state["model_path"], "server_commit": state["repo_commit"],
            "provider_dll": state["provider_dll"], "nonstream_exact_32_tokens": True,
            "stream_text_and_usage_match": True, "live_health_during_generation": True,
            "graceful_shutdown": True, "host_checks_pass": True,
            "post_request_queue_sample_available": "health_after" in original,
            "stream_raw_token_ids_exposed": False, "first_token_raw_logit_observed": False,
            "full_product_gate_pass": False, "retained_performance_claimed": False,
            "ready_wall_ms": original["ready_wall_ms"], "load": original["health_before"]["load"],
            "nonstream_metrics": completion["qrt_metrics"], "stream_metrics": metrics}
    stderr_path = run_dir / "server.stderr.log"
    if stderr_path.is_file():
        require(stderr_path.stat().st_size <= 32 * 1024 * 1024, "saved stderr size limit")
        log_text = stderr_path.read_text(encoding="utf-8")
        result["source_fingerprints"]["stderr"] = fingerprint(stderr_path)
        if '"type":"qrt_server_first_token_observation"' in log_text:
            rows = check_first_token_observations(log_text, oracle, prompt)
            check_timing_contract(completion["qrt_metrics"])
            check_timing_contract(metrics)
            result["first_token_logit_observations"] = rows
            result["first_token_raw_logit_observed"] = True
            result["q8192_nonstream_first_token_logit_and_32_tokens_pass"] = True
            result["observation_fnv_offset_basis"] = 14695981039346656037
            result["oracle_fnv_offset_basis"] = 1469598103934665603
    return result


def server_command(config, output):
    path = lambda name: config["files"][name]["path"]
    runtime = Path(path("runtime_manifest")).parent
    command = [path("server"), "serve", "--model", config["model"],
            "--provider", path("provider"), "--env-file", path("env"),
            "--arbitrary-moe-provider", str(runtime / "q1024-moe" / "qrt_triton_moe_q1024_exact_provider_slots64.dll"),
            "--arbitrary-moe-kernel-dir", str(runtime / "q1024-moe" / "moe-kernels"),
            "--smooth-tail-moe-root", str(runtime / "smooth-tail"),
            "--set-env", "QRT_QWEN36_HAWKEYE_CORRECTION_MAXIMUM_BLOCKS_PER_LAUNCH=8",
            "--model-id", config["model_id"], "--host", "127.0.0.1", "--port", str(config["port"]),
            "--max-model-len", "262144", "--max-queue-depth", "1", "--queue-timeout-seconds", "15",
            "--state-file", str(output / "service.json")]
    if config.get("require_first_token_logit", False):
        command += ["--set-env", "QRT_SERVER_FIRST_TOKEN_LOGIT_DIAGNOSTIC=1"]
    return command


def execute(config, output, prompt, expected, report):
    require_windows_job()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", config["port"]))
    command = server_command(config, output)
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("QRT_")}
    report.update(command=command, inherited_qrt_environment_removed=True,
                  host=platform.node(), model=config["model"], server_commit=config["server_commit"])
    save_json(output / "launch.json", report.copy())
    stop = threading.Event()
    deadline_failure = []
    log_paths = [output / "server.stdout.log", output / "server.stderr.log"]
    client = Client(config["port"])
    with log_paths[0].open("xb") as stdout, log_paths[1].open("xb") as stderr:
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=config["working_directory"], env=environment,
                                   stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL)
        report["pid"] = process.pid
        report["model_process_launched"] = True
        # Startup can load GPU modules before readiness, so failure before
        # /health is not evidence that no GPU API was touched.
        report["gpu_executed"] = None

        def watchdog():
            while not stop.wait(0.25):
                reason = None
                if time.monotonic() - started > 75:
                    reason = "75-second controller wall limit"
                elif sum(p.stat().st_size for p in log_paths) > 32 * 1024 * 1024:
                    reason = "32-MiB server log limit"
                if reason:
                    deadline_failure.append(reason)
                    if process.poll() is None:
                        process.kill()
                    return

        watch = threading.Thread(target=watchdog, daemon=True)
        watch.start()
        identified = False
        try:
            while True:
                require(process.poll() is None, "server exited before readiness")
                require(time.monotonic() - started < 35, "35-second readiness limit")
                try:
                    health = client.json("/health", timeout=0.5)
                except (OSError, ValueError):
                    time.sleep(0.25)
                    continue
                require(health["pid"] == process.pid and health["model"] == config["model_id"],
                        "health identity does not match owned server")
                require(health["ready"] is True, "service not ready")
                identified = True
                break
            report["ready_wall_ms"] = (time.monotonic() - started) * 1000
            report["gpu_executed"] = True
            report["health_before"] = health
            require(client.json("/ready")["pid"] == process.pid, "readiness identity differs")
            models = client.json("/v1/models")
            require(config["model_id"] in [v["id"] for v in models["data"]], "model discovery differs")
            model = config["model_id"]
            expected_text = client.json("/detokenize", {"model": model, "tokens": expected,
                                                        "skip_special_tokens": True})["prompt"]
            base_chat = {"model": model, "messages": [{"role": "user", "content": "x"}], "max_tokens": 16}
            for thinking in ({"type": "enabled", "budget_tokens": 0},
                             {"type": "enabled", "budget_tokens": 17}):
                client.json("/v1/chat/completions", {**base_chat, "thinking": thinking}, status=400)
            client.json("/v1/chat/completions", {**base_chat, "thinking": {"type": "enabled"},
                        "chat_template_kwargs": {"enable_thinking": False}}, status=400)
            after_negative = client.json("/health")
            require(after_negative["queue"]["started_total"] == health["queue"]["started_total"] == 0,
                    "negative HTTP checks invoked inference")
            report["negative_checks_without_inference"] = 3
            request = {"model": model, "prompt": prompt, "max_tokens": 32,
                       "temperature": 0, "top_p": 1, "ignore_eos": True}
            completion = client.json("/v1/completions", {**request, "stream": False})
            save_json(output / "completion.json", completion)
            check_completion(completion, expected, expected_text)
            report["nonstream_exact_32_tokens"] = True
            report["nonstream_metrics"] = completion["qrt_metrics"]
            stream_started = time.monotonic()
            events = []
            with client.open("/v1/completions", {**request, "stream": True,
                             "stream_options": {"include_usage": True}}) as response:
                require(response.status == 200, f"SSE HTTP {response.status}")
                require("text/event-stream" in response.headers.get("Content-Type", ""), "SSE type differs")
                during = client.json("/health", timeout=2)
                require(during["pid"] == process.pid and during["ready"] is True,
                        "health not responsive during SSE")
                report["health_during_sse"] = during
                while True:
                    line = response.readline(65537)
                    require(len(line) <= 65536, "SSE line size limit")
                    if not line:
                        break
                    if line.startswith(b"data:"):
                        data = line[5:].strip().decode("utf-8")
                        events.append({"ms": (time.monotonic() - stream_started) * 1000, "data": data})
                        require(len(events) <= 128, "SSE event count limit")
                        if data == "[DONE]":
                            break
            save_json(output / "stream.json", events)
            report["stream_metrics"] = check_stream(events, expected_text)
            report["stream_text_and_usage_match"] = True
            report["stream_raw_token_ids_exposed"] = False
            report["same_prompt_second_request_is_not_a_cold_measurement"] = True
            after = client.json("/health")
            require(after["queue"]["started_total"] == after["queue"]["completed_total"] == 2,
                    "expected exactly two serial inference requests")
            require(after["queue"]["active_requests"] == after["queue"]["waiting_requests"] == 0,
                    "queue did not drain")
            report["health_after"] = after
            require(not deadline_failure, str(deadline_failure))
        finally:
            if process.poll() is None and identified:
                try:
                    report["shutdown_response"] = client.json("/admin/shutdown", {}, timeout=2)
                    process.wait(timeout=5)
                except (OSError, ValueError, subprocess.TimeoutExpired) as error:
                    report["shutdown_error"] = str(error)
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
                report["forced_owned_process_cleanup"] = True
            report["server_exit_code"] = process.returncode
            stop.set()
            watch.join(timeout=1)
            report["wall_ms"] = (time.monotonic() - started) * 1000
        require(process.returncode == 0 and not report.get("forced_owned_process_cleanup"),
                "server did not exit gracefully")
        require(not deadline_failure, str(deadline_failure))
        state = read_json(output / "service.json")
        require(state["status"] == "stopped" and state["pid"] == process.pid, "stopped state differs")
        require(state["repo_commit"] == config["server_commit"], "binary source commit differs")
        report["service_state"] = state
        if config.get("require_first_token_logit", False):
            oracle = read_json(config["files"]["oracle"]["path"])
            rows = check_first_token_observations(log_paths[1].read_text(encoding="utf-8"), oracle, prompt)
            check_timing_contract(report["nonstream_metrics"])
            check_timing_contract(report["stream_metrics"])
            report["first_token_logit_observations"] = rows
            report["first_token_raw_logit_observed"] = True
            report["q8192_nonstream_first_token_logit_and_32_tokens_pass"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay-run", type=Path)
    parser.add_argument("--guard-record", type=Path)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--oracle", type=Path)
    args = parser.parse_args()
    if args.replay_run:
        if not all((args.guard_record, args.prompt, args.oracle)) or args.config:
            parser.error("--replay-run requires --guard-record, --prompt, --oracle and no --config")
    elif not args.config:
        parser.error("--config is required for preflight or execution")
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "failed", "gpu_executed": False,
              "command_file": str(Path(__file__).resolve()), "command_file_sha256": fingerprint(__file__),
              "config_sha256": fingerprint(args.config) if args.config else None,
              "model_process_launched": False,
              "first_token_raw_logit_observed": False,
              "full_product_gate_pass": False, "retained_performance_claimed": False}
    try:
        if args.replay_run:
            report.update(replay_saved_run(args.replay_run, args.guard_record, args.prompt, args.oracle))
        else:
            config = read_json(args.config)
            prompt, expected, report["preflight"] = preflight(config)
            if args.execute:
                # Job membership is checked before creating the model process.
                require_windows_job()
                execute(config, args.out, prompt, expected, report)
                report["status"] = "http_token_transport_pass"
            else:
                report["status"] = "cpu_preflight_pass"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        save_json(args.out / "result.json", report)
    print(json.dumps({k: report.get(k) for k in ("status", "error", "gpu_executed", "full_product_gate_pass")}))
    return 0 if report["status"] != "failed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
