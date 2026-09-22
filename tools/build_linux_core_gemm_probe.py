#!/usr/bin/env python3
"""Build the bounded full-shape hipBLASLt diagnostic on native baiying."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rocm", type=Path, default=Path("C:/Program Files/AMD/ROCm/7.1"))
    args = parser.parse_args()
    if platform.system() != "Windows" or socket.gethostname().split(".")[0].lower() != "baiying":
        raise SystemExit("Requires native baiying")
    git = lambda *a: subprocess.check_output(["git", "-C", str(ROOT), *a], text=True, timeout=15).strip()
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        raise SystemExit("Requires clean committed source")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sha = lambda p: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    source = ROOT / "native/diagnostics/linux_core_gemm_algorithms.hip.cpp"
    compiler = args.rocm / "bin/hipcc.exe"
    library = args.rocm / "lib/libhipblaslt.dll.a"
    inputs = [source, Path(__file__), compiler, args.rocm / "bin/clang++.exe", library,
              args.rocm / "bin/libhipblaslt.dll", args.rocm / "include/hipblaslt/hipblaslt.h"]
    record = dict(host=socket.gethostname(), source_commit=commit, dirty_tree=False,
                  source_inputs=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in inputs],
                  completed=False, model_loaded=False, inference_acceptance=False, performance_acceptance=False)
    identity = out / "gemm_probe_identity.h"
    identity.write_text("#pragma once\n#define GEMM_PROBE_SOURCE_COMMIT " + json.dumps(commit) + "\n", encoding="utf-8")
    shutil.copyfile(library, out / "hipblaslt.lib")
    executable = out / "qrt-linux-core-gemm-algorithms.exe"
    command = [str(x) for x in [compiler, "-std=c++17", "-O3", "--offload-arch=gfx1151",
        "-fno-gpu-rdc", "-DNOMINMAX", "-DWIN32_LEAN_AND_MEAN", "-I", out,
        source, "-L", out, "-lhipblaslt", "-o", executable]]
    record.update(command=command, identity_sha256=sha(identity), timeout_seconds=180)
    def publish():
        (out / "build-provenance.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    publish()
    env = os.environ.copy()
    env["PATH"] = str(args.rocm / "bin") + os.pathsep + env.get("PATH", "")
    started = time.monotonic()
    with (out / "compile.stdout.txt").open("xb") as stdout, (out / "compile.stderr.txt").open("xb") as stderr:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        try:
            record.update(exit_code=process.wait(timeout=180), timed_out=False)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill.exe", "/PID", str(process.pid), "/T", "/F"], timeout=15, capture_output=True)
            record.update(exit_code=process.wait(timeout=10), timed_out=True)
    record.update(wall_ms=(time.monotonic() - started) * 1000,
                  stdout_sha256=sha(out / "compile.stdout.txt"), stderr_sha256=sha(out / "compile.stderr.txt"))
    if not record["exit_code"] and not record["timed_out"]:
        if any(sha(Path(x["path"])) != x["sha256"] for x in record["source_inputs"]):
            raise RuntimeError("Build input changed")
        record.update(completed=True, artifact=dict(path=str(executable), bytes=executable.stat().st_size, sha256=sha(executable)))
    publish()
    if not record["completed"]:
        raise SystemExit("Build failed; compiler output preserved")
    print(json.dumps(dict(type="summary", status="pass", native_build=True, model_loaded=False,
                         inference_acceptance=False), separators=(",", ":")))


if __name__ == "__main__":
    main()
