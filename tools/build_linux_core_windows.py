#!/usr/bin/env python3
"""Bounded standalone Windows build; run inside baiying_guarded_inference.ps1."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time

from linux_core_coff import verify_coff
from prepare_linux_core_windows import verify_import

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rocm", type=Path, default=Path("C:/Program Files/AMD/ROCm/7.1"))
    parser.add_argument("--timeout-seconds", type=int, default=1500)
    parser.add_argument("--windows-rectangular-ck", action="store_true",
                        help="Opt in to the Windows suffix ABI mapping; not model-qualified")
    parser.add_argument("--current-text-decode", action="store_true",
                        help="Use current upstream decode arithmetic for text; not model-qualified")
    parser.add_argument("--gb10-convolution", action="store_true",
                        help="Use the GB10 convolution rounding and SiLU boundary")
    parser.add_argument("--gb10-gdn", action="store_true",
                        help="Use the Windows FLA and GB10 Q2 arithmetic experiment")
    parser.add_argument("--gb10-projections", action="store_true",
                        help="Use SM121 decode projection and embedding normalization arithmetic")
    parser.add_argument("--gb10-prefill-projections", action="store_true",
                        help="Use FP32 prefill GEMMs with staged SM121 exact replay")
    parser.add_argument("--gb10-normalization", action="store_true",
                        help="Use GB10 prefill gated norm and unrounded residual variance")
    parser.add_argument("--gb10-moe", action="store_true",
                        help="Use the Windows prefill MoE provider and live FP32 carriers")
    args = parser.parse_args()
    host = socket.gethostname()
    if platform.system() != "Windows" or host.split(".")[0].lower() != "baiying":
        raise SystemExit("Native compilation requires the local baiying Windows environment")
    if not 60 <= args.timeout_seconds <= 1680:
        raise SystemExit("Build timeout must be 60..1680 seconds, inside the 1800-second owner bound")
    inventory = verify_import()
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(ROOT), *arguments], timeout=15, text=True).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise SystemExit("Build requires a clean committed source tree")
    out = args.out.resolve()
    if out.exists():
        raise SystemExit("Build output exists; choose a fresh evidence directory")
    out.mkdir(parents=True)
    deadline = time.monotonic() + args.timeout_seconds
    commands = []
    record = dict(schema=1, host=host, repo_commit=commit, dirty_tree=False,
                  upstream_commit=inventory["revision"], commands=commands,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  completed=False, model_loaded=False, inference_acceptance=False,
                  performance_acceptance=False)
    def publish():
        (out / "build-provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    def run(label, command, seconds=180):
        remaining = int(deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Total compilation deadline reached")
        timeout = min(seconds, remaining)
        command = [str(x) for x in command]
        start = time.monotonic()
        stdout_path, stderr_path = out / (label + ".stdout.txt"), out / (label + ".stderr.txt")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
            timed_out = False
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, timeout=15, check=False)
                code = process.wait(timeout=10)
        commands.append(dict(label=label, argv=command, timeout_seconds=timeout,
                             exit_code=code, timed_out=timed_out,
                             wall_ms=(time.monotonic() - start) * 1000,
                             stdout_sha256=sha(stdout_path), stderr_sha256=sha(stderr_path)))
        publish()
        print(json.dumps(dict(stage=label, exit_code=code, timed_out=timed_out)), flush=True)
        if code or timed_out:
            raise RuntimeError(f"{label} failed; inspect preserved compiler output")
    try:
        hipcc, clang = args.rocm / "bin/hipcc.exe", args.rocm / "bin/clang++.exe"
        import_lib = args.rocm / "lib/libhipblaslt.dll.a"
        for p in (hipcc, clang, import_lib):
            if not p.is_file():
                raise FileNotFoundError(p)
        os.environ["PATH"] = str(args.rocm / "bin") + os.pathsep + os.environ.get("PATH", "")
        record["compiler_inputs"] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                                     for p in (hipcc, clang, import_lib)]
        source_paths = [ROOT / "third_party/aima_linux" / e["path"] for e in inventory["files"]]
        source_paths += list((ROOT / "native/linux_core_port").glob("*"))
        source_paths += [ROOT / "tools" / name for name in (
            "prepare_linux_core_windows.py", "build_linux_core_windows.py", "linux_core_coff.py")]
        if args.gb10_convolution:
            source_paths += [ROOT / "native/providers/gdn" / name for name in (
                "sm121_silu_table.h", "sm121_exp2_table.h")]
        if args.gb10_gdn:
            source_paths += [ROOT / "native/providers/gdn" / name for name in (
                "fla_checkpoint.h", "sm121_q1_gdn.h", "sm121_q1_math.h",
                "sm121_rsqrt_table.h", "sm121_sqrt_table.h", "sm121_attention_rcp.h")]
        if args.gb10_projections:
            source_paths += [ROOT / "native/providers/moe_accumulator" / name for name in (
                "sm121_wave16.h", "q1_moe_hawkeye_bf16_accumulator.h", "sm121_group16_modulo.h",
                "sm121_lane_reduce.h", "sm121_canonical_normalize.h")]
        if args.gb10_prefill_projections:
            source_paths += [ROOT / "native/providers/moe_accumulator" / name for name in (
                "bf16_midpoint_selector.h", "sm121_dot_certificate.h", "sm121_float_alignment.h",
                "sm121_float_subgroup.h", "sm121_paired_products.h", "sm121_prepared_integer_pairs.h",
                "sm121_scalar_projection.h", "sm121_scaled_half_products.h", "sm121_scaled_half_projection.h",
                "sm121_staged_half_projection.h", "sm121_subgroup.h")]
        if args.gb10_normalization:
            source_paths += [ROOT / "native/providers/gdn" / name for name in (
                "sm121_q2_gated_math.h", "sm121_mtp_residual.h",
                "sm121_mtp_residual_math.h", "sm121_mtp_math.h",
                "sm121_mtp_kv_math.h", "sm121_bf16_fma.h", "sm121_exp2_interpolated.h")]
            source_paths += [ROOT / "native/providers/ck_fmha" / name for name in (
                "blackwell_attention.h", "float_pv_replay.h", "inplace_probability_storage.h",
                "long_attention_layout.h", "packed_probability_storage.h")]
            source_paths += [ROOT / "native/providers/moe_accumulator" / name for name in (
                "sm121_integer_core.h", "sm121_integer_parts.h", "sm121_mantissa_parts.h",
                "sm121_native_product.h", "sm121_prepared_bf16.h", "sm121_pv_error_bound.h",
                "sm121_pv_final_bound.h", "sm121_strided_pair.h")]
            source_paths += [ROOT / "native/providers/sm121_attention_capacity.h",
                             ROOT / "native/src/qrt_context_limits.h"]
        record["source_inputs"] = [dict(path=p.relative_to(ROOT).as_posix(), bytes=p.stat().st_size,
                                        sha256=sha(p)) for p in sorted(source_paths) if p.is_file()]
        prepared = out / "prepared"
        preparation = [sys.executable, ROOT / "tools/prepare_linux_core_windows.py", "--out", prepared]
        if args.windows_rectangular_ck:
            preparation.append("--windows-rectangular-ck")
        if args.current_text_decode:
            preparation.append("--current-text-decode")
        if args.gb10_convolution:
            preparation.append("--gb10-convolution")
        if args.gb10_gdn:
            preparation.append("--gb10-gdn")
        if args.gb10_projections:
            preparation.append("--gb10-projections")
        if args.gb10_prefill_projections:
            preparation.append("--gb10-prefill-projections")
        if args.gb10_normalization:
            preparation.append("--gb10-normalization")
        if args.gb10_moe:
            preparation.append("--gb10-moe")
        run("prepare", preparation, 120)
        plan = json.loads((prepared / "prepare.json").read_text())
        if "optional_adaptations" in plan:
            record["optional_adaptations"] = plan["optional_adaptations"]
        record["prepare_sha256"] = sha(prepared / "prepare.json")
        identity = '#pragma once\n#define AIMA_PORT_SOURCE_COMMIT ' + json.dumps(commit) + \
            '\n#define AIMA_PORT_UPSTREAM_COMMIT ' + json.dumps(inventory["revision"]) + '\n'
        (prepared / "aima_port_build_identity.h").write_text(identity, encoding="utf-8", newline="\n")
        record["build_identity_sha256"] = sha(prepared / "aima_port_build_identity.h")
        run("host-contract-build", [clang, "-std=c++17", "-O1", "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN",
            ROOT / "native/linux_core_port/host_contract_test.cpp", "-o", out / "host-contract.exe"])
        run("host-contract-run", [out / "host-contract.exe", out / "host-contract-fixtures"], 30)
        obj = out / "aot_images.obj"
        run("embed-images", [clang, "--target=x86_64-pc-windows-msvc", "-c", "-x", "assembler",
                              prepared / "aot_images.S", "-o", obj], 60)
        record["coff_verification"] = verify_coff(obj.read_bytes(), plan["images"])
        shutil.copyfile(import_lib, out / "hipblaslt.lib")
        upstream = ROOT / "third_party/aima_linux"
        includes = [prepared, prepared / "overlay/native/include", ROOT / "native/linux_core_port",
                    upstream / "native/include", upstream / "native/generated"]
        flags = ["-std=c++17", "-O3", "-DNDEBUG", "--offload-arch=gfx1151",
                 "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1", "-fno-gpu-rdc", "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN"]
        if args.gb10_convolution:
            flags.append("-DAIMA_PORT_GB10_CONVOLUTION=1")
        if args.gb10_gdn:
            flags.append("-DAIMA_PORT_GB10_GDN=1")
        if args.gb10_projections:
            flags.append("-DAIMA_PORT_GB10_PROJECTIONS=1")
        if args.gb10_prefill_projections:
            flags.append("-DAIMA_PORT_GB10_PREFILL_PROJECTIONS=1")
        if args.gb10_normalization:
            flags.append("-DAIMA_PORT_GB10_NORMALIZATION=1")
        if args.gb10_moe:
            flags.append("-DAIMA_PORT_GB10_MOE=1")
        flags += [x for p in includes for x in ("-I", p)]
        record["compile_flags"] = [str(x) for x in flags]
        objects = [obj]
        for index, source in enumerate(plan["sources"]):
            target = out / f"unit-{index:02d}.obj"
            arithmetic_flags = (["-fno-fast-math", "-fno-reciprocal-math", "-ffp-contract=off"]
                                if Path(source).name == "gb10_decode_attention.hip.cpp" else [])
            run(f"compile-{index:02d}", [hipcc, *flags, *arithmetic_flags, "-c", source, "-o", target])
            objects.append(target)
        # The existing process owner recognizes qrt* engine processes.
        executable = out / "qrt-linux-core-q8192-probe.exe"
        run("link", [hipcc, "--offload-arch=gfx1151", "-fno-gpu-rdc", *objects,
                     "-L", out, "-lhipblaslt", "-Xlinker", "/STACK:268435456", "-o", executable], 180)
        record["artifacts"] = [dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
                               for p in (executable, obj, out / "host-contract.exe", out / "hipblaslt.lib")]
        # Recheck every source and generated input after compilation.
        for entry in record["source_inputs"]:
            if sha(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("Source changed while compiling")
        for entry in plan["generated"]:
            if sha(prepared / entry["path"]) != entry["sha256"]:
                raise ValueError("Generated source changed while compiling")
        if git("rev-parse", "HEAD") != commit or git("status", "--porcelain", "--untracked-files=normal"):
            raise ValueError("Source tree changed while compiling")
        record["completed"] = True
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        publish()


if __name__ == "__main__":
    main()
