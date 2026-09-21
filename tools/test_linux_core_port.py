#!/usr/bin/env python3
"""Local portability/observer controls. This never loads a model or runs HIP."""
from pathlib import Path
import argparse
import copy
import json
import struct
import subprocess
import sys

from check_linux_core_q8192 import ORACLE_SHA256, UPSTREAM_COMMIT, check_tokens, sha
from linux_core_coff import verify_coff
from prepare_linux_core_windows import make_overlays, verify_import

ROOT = Path(__file__).resolve().parents[1]


def rejects(operation):
    try:
        operation()
    except (ValueError, TypeError, KeyError) as error:
        return str(error)
    raise AssertionError("Negative control was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    reports = []
    def run(label, command, timeout=90):
        command = [str(x) for x in command]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        report = dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        (out / (label + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        result.check_returncode()
        reports.append(label)
    inventory = verify_import()
    overlays = make_overlays()
    prepared = out / "prepared"
    run("prepare", [sys.executable, ROOT / "tools/prepare_linux_core_windows.py", "--out", prepared])
    run("coff-compile", [args.cxx, "--target=x86_64-pc-windows-msvc", "-c", "-x", "assembler",
                         prepared / "aot_images.S", "-o", out / "aot_images.obj"])
    data = (out / "aot_images.obj").read_bytes()
    images = json.loads((prepared / "prepare.json").read_text())["images"]
    coff = verify_coff(data, images)
    count, = struct.unpack_from("<H", data, 2)
    section = next(20 + i * 40 for i in range(count)
                   if data[20 + i * 40:28 + i * 40].rstrip(b"\0") == b".rdata")
    start, = struct.unpack_from("<I", data, section + 20)
    flags, = struct.unpack_from("<I", data, section + 36)
    corrupt = bytearray(data); corrupt[start + 8] ^= 1
    writable = bytearray(data); struct.pack_into("<I", writable, section + 36, flags | 0x80000000)
    executable = bytearray(data); struct.pack_into("<I", executable, section + 36, flags | 0x20000000)
    coff["negative_controls"] = {key: rejects(lambda payload=payload: verify_coff(payload, images))
        for key, payload in dict(image_byte_changed=corrupt, writable_section=writable,
                                  executable_section=executable, truncated_symbols=data[:-70]).items()}
    (out / "coff-contract.json").write_text(json.dumps(coff, indent=2) + "\n")
    run("host-build", [args.cxx, "-std=c++17", "-O1", "-g", "-Wall", "-Wextra", "-Werror",
                       "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                       ROOT / "native/linux_core_port/host_contract_test.cpp", "-o", out / "host-test"])
    run("host-contract", [out / "host-test", out / "fixtures"], 30)
    oracle_path = ROOT / "contracts/gb10_cold_token_matrix_20260911_oracle.json"
    assert sha(oracle_path.read_bytes()) == ORACLE_SHA256
    oracle = next(x for x in json.loads(oracle_path.read_text())["cases"] if x["name"] == "q8192-out512")
    commit = "a" * 40
    common = dict(host="BAIYING", model="D:/models/Qwen3.6-35B-A3B", source_commit=commit,
                  upstream_commit=UPSTREAM_COMMIT, input_u32_sha256=oracle["prompt"]["u32le_sha256"])
    ids = oracle["expected"]["output_token_ids"]
    # These are synthetic observer inputs, never runtime or inference evidence.
    events = [dict(common, event="start", prompt_tokens=8192, requested_outputs=512),
              dict(event="ready", command_to_ready_ms=25000)]
    events.extend(dict(event="token", index=i, token=x, elapsed_ms=7000 + i * 40) for i, x in enumerate(ids))
    events.append(dict(common, event="result", complete=True, callback_count=512, oracle_tensor_reads=0,
                       prompt_execution="cold-aot", prefix_cache_lookup="disabled", output_token_ids=ids,
                       first_token_raw_logit=10.375, ttft_ms=7000, tpot_ms=40, command_to_ready_ms=25000))
    result = check_tokens(events, oracle, commit)
    assert result["gb10_boundary_pass"] and not result["performance_acceptance"]
    mutations = {
        "wrong_host": (0, "host", "other"), "wrong_model": (-1, "model", "D:/models/other"),
        "wrong_prompt": (0, "input_u32_sha256", "0" * 64),
        "wrong_commit": (-1, "source_commit", "b" * 40),
        "wrong_upstream": (0, "upstream_commit", "b" * 40),
        "wrong_shape": (0, "prompt_tokens", 7169),
        "wrong_count": (-1, "callback_count", 511),
        "incomplete": (-1, "complete", False), "oracle_seed": (-1, "oracle_tensor_reads", 1),
        "cached_request": (-1, "prefix_cache_lookup", "exact"),
        "decode_fallback": (-1, "prompt_execution", "cold-decode"),
        "missing_output": (-1, "output_token_ids", ids[:-1]),
        "wrong_output": (-1, "output_token_ids", ids[:-1] + [(ids[-1] + 1) % 248320]),
        "boolean_token": (-1, "output_token_ids", [True] + ids[1:]),
        "reordered_callback": (3, "index", 2), "wrong_callback": (5, "token", 0),
        "clock_reversed": (3, "elapsed_ms", 6999), "clock_nan": (3, "elapsed_ms", float("nan")),
        "bad_first_logit": (-1, "first_token_raw_logit", 10.375 + 0.126),
        "nonfinite_first_logit": (-1, "first_token_raw_logit", float("inf")),
        "boolean_first_logit": (-1, "first_token_raw_logit", True),
        "wrong_ttft": (-1, "ttft_ms", 6999), "wrong_tpot": (-1, "tpot_ms", 20),
        "wrong_load_time": (-1, "command_to_ready_ms", 10000),
    }
    failures = {}
    for name, (index, key, value) in mutations.items():
        changed = copy.deepcopy(events); changed[index][key] = value
        failures[name] = rejects(lambda: check_tokens(changed, oracle, commit))
    failures["missing_event"] = rejects(lambda: check_tokens(events[:-1], oracle, commit))
    for sign in (-1, 1):
        edge = copy.deepcopy(events); edge[-1]["first_token_raw_logit"] += sign * 0.125
        assert check_tokens(edge, oracle, commit)["gb10_boundary_pass"]
    summary = dict(upstream_files=len(inventory["files"]), overlays=len(overlays),
        coff=coff, observer_negative_controls=failures, observer_tolerance_edge_controls=2,
        compiler_stages=reports, model_loaded=False, gpu_executed=False,
        windows_build_qualified=False, inference_acceptance=False, performance_acceptance=False)
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("coff", "observer_negative_controls")}
                     | dict(coff_images=coff["images"], observer_negative_controls=len(failures))))


if __name__ == "__main__":
    main()
