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
        if "usage" in chunk:
            require(bool(finishes) and chunk["choices"] == [], "usage before finish")
            usage.append(chunk)
    require(text == expected_text, "SSE text differs from oracle detokenization")
    require(finishes == ["length"] and len(usage) == 1, "SSE finish/usage count differs")
    check_usage(usage[0]["usage"])
    return usage[0]["qrt_metrics"]


def server_command(config, output):
    path = lambda name: config["files"][name]["path"]
    runtime = Path(path("runtime_manifest")).parent
    return [path("server"), "serve", "--model", config["model"],
            "--provider", path("provider"), "--env-file", path("env"),
            "--arbitrary-moe-provider", str(runtime / "q1024-moe" / "qrt_triton_moe_q1024_exact_provider_slots64.dll"),
            "--arbitrary-moe-kernel-dir", str(runtime / "q1024-moe" / "moe-kernels"),
            "--smooth-tail-moe-root", str(runtime / "smooth-tail"),
            "--set-env", "QRT_QWEN36_HAWKEYE_CORRECTION_MAXIMUM_BLOCKS_PER_LAUNCH=8",
            "--model-id", config["model_id"], "--host", "127.0.0.1", "--port", str(config["port"]),
            "--max-model-len", "262144", "--max-queue-depth", "1", "--queue-timeout-seconds", "15",
            "--state-file", str(output / "service.json")]


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "status": "failed", "gpu_executed": False,
              "command_file": str(Path(__file__).resolve()), "command_file_sha256": fingerprint(__file__),
              "config_sha256": fingerprint(args.config), "model_process_launched": False,
              "first_token_raw_logit_observed": False,
              "full_product_gate_pass": False, "retained_performance_claimed": False}
    try:
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
