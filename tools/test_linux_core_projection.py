#!/usr/bin/env python3
"""Exercise projection bindings, live embedding selection and artifact failures."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from test_linux_core_gdn import STUB

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stub = out / "stub/hip/hip_runtime.h"
    stub.parent.mkdir(parents=True)
    stub.write_text(STUB + r'''
#define __forceinline__ inline
#define __host__
inline unsigned __clz(unsigned v) { return __builtin_clz(v); }
template<class T> T __shfl_xor(T v, unsigned, unsigned) { return v; }
''', encoding="utf-8")
    source = ROOT / "native/linux_core_port/gb10_projection_host_contract_test.cpp"
    executable = out / "host-contract"
    command = ["clang++", "-std=c++17", "-O1", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-I", str(out / "stub"),
        "-I", str(ROOT / "third_party/aima_linux/native/include"), str(source),
        str(ROOT / "third_party/aima_linux/native/src/sha256.cpp"), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable), str(out / "inverse-fixture.bin")],
        capture_output=True, text=True, timeout=30)
    (out / "run.stderr").write_text(run.stderr)
    run.check_returncode()
    inputs = [source, ROOT / "native/linux_core_port/gb10_projection.hip.cpp",
              ROOT / "native/linux_core_port/gb10_projection.h", Path(__file__),
              ROOT / "tools/test_linux_core_gdn.py"]
    report = dict(result=json.loads(run.stdout), build_command=command,
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        stub_sha256=hashlib.sha256(stub.read_bytes()).hexdigest(),
        host_only=True, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
