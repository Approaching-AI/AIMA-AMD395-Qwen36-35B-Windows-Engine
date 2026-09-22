#!/usr/bin/env python3
"""Check prefill operand transport, compaction bounds and dispatch ownership."""
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
inline dim3 gridDim;
inline unsigned __clz(unsigned v) { return __builtin_clz(v); }
template<class T> T __shfl_xor(T v, unsigned, unsigned) { return v; }
inline unsigned atomicAdd(unsigned* p, unsigned v) { unsigned old=*p; *p+=v; return old; }
inline unsigned atomicOr(unsigned* p, unsigned v) { unsigned old=*p; *p|=v; return old; }
inline int hipMemsetAsync(void* p, int v, size_t bytes, void*) {
 memset(p,v,bytes);fake_events.push_back("memset");return 0;
}
''', encoding="utf-8")
    source = ROOT / "native/linux_core_port/gb10_prefill_projection_host_contract_test.cpp"
    executable = out / "host-contract"
    command = ["clang++", "-std=c++17", "-O1", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-I", str(out / "stub"), str(source),
        "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    (out / "run.stderr").write_text(run.stderr)
    run.check_returncode()
    inputs = [source, ROOT / "native/linux_core_port/gb10_prefill_projection.hip.cpp",
              ROOT / "native/linux_core_port/gb10_prefill_projection.h", Path(__file__),
              ROOT / "tools/test_linux_core_gdn.py"]
    report = dict(result=json.loads(run.stdout), build_command=command,
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs],
        stub_sha256=hashlib.sha256(stub.read_bytes()).hexdigest(),
        host_only=True, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
