#!/usr/bin/env python3
"""Check a completed standalone core run against the unchanged GB10 boundary."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import re
import struct

ROOT = Path(__file__).resolve().parents[1]
ORACLE_SHA256 = "7a6feb488f4dd136e49c4d7227c1fc325e3216af3dae3e9b60aa0b3025f3d1b1"
UPSTREAM_COMMIT = "ec9934446911fdf376da8eebcd83e7b137efbb7c"
MODEL = "d:/models/qwen3.6-35b-a3b"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def token_hash(ids):
    require(isinstance(ids, list) and all(type(x) is int and 0 <= x < 248320 for x in ids),
            "Invalid token ID array")
    return sha(struct.pack("<" + "I" * len(ids), *ids))


def path_key(path):
    return str(path).replace("\\", "/").lower()


def check_tokens(events, oracle, commit):
    require(re.fullmatch(r"[0-9a-f]{40}", commit), "Missing compiled source commit")
    require(len(events) == 515, "Expected start, ready, 512 callbacks and result")
    start, ready, result = events[0], events[1], events[-1]
    require([start.get("event"), ready.get("event"), result.get("event")] == ["start", "ready", "result"],
            "Event framing differs")
    for event in (start, result):
        require(event.get("host", "").lower() == "baiying", "Unexpected execution host")
        require(path_key(event.get("model", "")) == MODEL, "Unexpected model path")
        require(event.get("source_commit") == commit and event.get("upstream_commit") == UPSTREAM_COMMIT,
                "Compiled source identity differs")
        require(event.get("input_u32_sha256") == oracle["prompt"]["u32le_sha256"],
                "Prompt token identity differs")
    require(start.get("prompt_tokens") == 8192 and start.get("requested_outputs") == 512,
            "Unexpected requested shape")
    require(result.get("complete") is True and result.get("callback_count") == 512 and
            result.get("oracle_tensor_reads") == 0, "Incomplete or oracle-seeded execution")
    require(result.get("prompt_execution") == "cold-aot", "Request did not use cold AOT prefill")
    require(result.get("prefix_cache_lookup") in ("disabled", "miss"), "Cold run restored a prefix")
    output = result.get("output_token_ids")
    require(isinstance(output, list) and len(output) == 512, "Output extent differs")
    require(token_hash(output) == oracle["expected"]["output_token_ids_u32le_sha256"],
            "Output continuation differs from GB10")
    require(output == oracle["expected"]["output_token_ids"], "Output token IDs differ from GB10")
    times = []
    for index, callback in enumerate(events[2:-1]):
        require(callback.get("event") == "token" and type(callback.get("index")) is int and
                callback["index"] == index and type(callback.get("token")) is int and
                callback["token"] == output[index], "Streaming callback order or token differs")
        value = callback.get("elapsed_ms")
        require(type(value) in (int, float) and math.isfinite(value) and value > 0 and
                (not times or value >= times[-1]), "Callback clock is invalid")
        times.append(value)
    raw = result.get("first_token_raw_logit")
    require(type(raw) in (int, float) and math.isfinite(raw), "First logit is invalid")
    error = abs(raw - oracle["expected"]["first_token_raw_logit"])
    require(oracle["expected"]["first_token_raw_logit_tolerance"] == 0.125 and error <= 0.125,
            "First logit exceeds original GB10 tolerance")
    require(result.get("ttft_ms") == times[0], "TTFT is not the first callback clock")
    require(math.isclose(result.get("tpot_ms", -1), (times[-1] - times[0]) / 511,
                         rel_tol=1e-12, abs_tol=1e-9), "TPOT is not the callback interval")
    require(type(ready.get("command_to_ready_ms")) in (int, float) and
            math.isfinite(ready["command_to_ready_ms"]) and ready["command_to_ready_ms"] > 0 and
            ready["command_to_ready_ms"] == result.get("command_to_ready_ms"), "Load clock is invalid")
    return dict(gb10_boundary_pass=True, prompt_tokens=8192, output_tokens=512, stream_callbacks=512,
                first_token_id=output[0], first_token_raw_logit=raw, first_logit_absolute_error=error,
                first_logit_tolerance=0.125, ttft_ms=times[0], tpot_ms=result["tpot_ms"],
                command_to_ready_ms=ready["command_to_ready_ms"],
                ttft_below_10000_ms=times[0] < 10000,
                retained_4187_415605_ms_ttft_met=times[0] <= 4187.415605,
                load_at_most_30000_ms=ready["command_to_ready_ms"] <= 30000,
                performance_acceptance=False, release_qualified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--build-metadata", type=Path, required=True)
    parser.add_argument("--build-run", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    oracle_path = ROOT / "contracts/gb10_cold_token_matrix_20260911_oracle.json"
    require(sha(oracle_path.read_bytes()) == ORACLE_SHA256, "Original oracle changed")
    oracle = next(x for x in read(oracle_path)["cases"] if x["name"] == "q8192-out512")
    plan = read(args.plan)
    build, build_run, run = read(args.build_metadata), read(args.build_run), read(args.run / "run-record.json")
    commit = plan["source_commit"]
    for record in (build_run, run):
        require(record["host"].lower() == "baiying" and record["reason"] == "completed" and
                record["exit_code"] == 0 and record["host_checks_pass"] is True and
                record["after_processes"] == [] and
                all(record["host_checks"].get(k) is True for k in (
                    "same_boot", "no_engine_process", "host_memory_reserve", "commit_reserve", "amd_gpu_ok")),
                "Guarded run did not complete cleanly")
        require(record["preflight"]["pass"] is True and
                all(value is True for value in record["preflight"]["checks"].values()),
                "Guarded preflight did not pass")
        require(record["spec"]["repo_commit"] == commit, "Guarded source commit differs")
    require(build["completed"] is True and build["dirty_tree"] is False and build["repo_commit"] == commit and
            build["upstream_commit"] == UPSTREAM_COMMIT, "Build provenance is incomplete")
    require(build["source_inputs"] == plan["build_source_inputs"] and
            build_run["spec"]["source_inputs"] == plan["build_source_inputs"], "Compiled source inputs differ")
    require(run["spec"]["build_metadata_sha256"] == sha(args.build_metadata.read_bytes()) and
            run["spec"]["build_run_sha256"] == sha(args.build_run.read_bytes()) and
            run["spec"]["source_manifest_sha256"] == sha(args.plan.read_bytes()), "Run/build/plan binding differs")
    for key in ("source_inputs", "arguments", "executable", "model", "command_file_sha256"):
        require(run["spec"][key] == plan["run_spec"][key], "Frozen run spec differs: " + key)
    require(run["spec"]["executable_sha256"] == next(x["sha256"] for x in build["artifacts"]
            if path_key(x["path"]) == path_key(run["spec"]["executable"])), "Runtime artifact differs")
    require(run["preflight"]["executable_sha256"] == run["spec"]["executable_sha256"],
            "Actually launched executable differs from the qualified build")
    events = [json.loads(line) for line in (args.run / "product.stdout.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    require(events[0]["ck_provider_sha256"] == plan["ck_provider"]["sha256"], "CK provider differs")
    require(events[0]["vision_image_sha256"] == plan["vision_image"]["sha256"], "Vision image differs")
    report = check_tokens(events, oracle, commit)
    report.update(host=run["host"], model=run["spec"]["model"], command_file=run["spec"]["command_file"],
                  source_commit=commit, upstream_commit=UPSTREAM_COMMIT, oracle_sha256=ORACLE_SHA256,
                  run_record_sha256=sha((args.run / "run-record.json").read_bytes()),
                  stdout_sha256=sha((args.run / "product.stdout.jsonl").read_bytes()),
                  build_metadata_sha256=sha(args.build_metadata.read_bytes()),
                  source_manifest_sha256=sha(args.plan.read_bytes()), output_token_ids=events[-1]["output_token_ids"])
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print(json.dumps({k: v for k, v in report.items() if k != "output_token_ids"}))


if __name__ == "__main__":
    main()
