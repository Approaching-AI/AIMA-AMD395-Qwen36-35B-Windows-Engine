#!/usr/bin/env python3
"""Check normalization table ownership, dispatch bindings and artifact failures."""
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
    parser.add_argument("--silu-table", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stub = out / "stub/hip/hip_runtime.h"
    stub.parent.mkdir(parents=True)
    stub.write_text(STUB[:STUB.index("template<class K,class...A>")] + r'''
#include <cstdint>
#include <type_traits>
#define __forceinline__ inline
#define __host__
constexpr int hipErrorInvalidValue = 2;
template<class T> T __shfl(T v, unsigned, unsigned) { return v; }
template<class T> T __shfl_down(T v, unsigned, unsigned) { return v; }
template<class T> T __shfl_xor(T v, unsigned, unsigned) { return v; }
struct Launch { std::string name; dim3 grid, block; void* stream; std::vector<std::uintptr_t> args; };
inline std::vector<Launch> launches;
template<class T> std::uintptr_t value(T v) {
 if constexpr (std::is_pointer_v<T>) return reinterpret_cast<std::uintptr_t>(v);
 else return static_cast<std::uintptr_t>(v);
}
template<class K,class...A>void fake_launch(const char*name,dim3 grid,dim3 block,void*stream,K kernel,A...args) {
 if(false)kernel(args...);
 launches.push_back({name,grid,block,stream,{value(args)...}});
}
#define hipLaunchKernelGGL(k,g,b,z,s,...) fake_launch(#k,g,b,s,k,__VA_ARGS__)
''', encoding="utf-8")
    source = ROOT / "native/linux_core_port/gb10_normalization_host_contract_test.cpp"
    executable = out / "host-contract"
    command = ["clang++", "-std=c++17", "-O1", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-I", str(out / "stub"),
        "-I", str(ROOT / "third_party/aima_linux/native/include"), str(source),
        str(ROOT / "third_party/aima_linux/native/src/sha256.cpp"), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable), str(args.silu_table.resolve()), str(out / "bad-table.bin")],
        capture_output=True, text=True, timeout=30)
    (out / "run.stderr").write_text(run.stderr)
    run.check_returncode()
    inputs = [source, ROOT / "native/linux_core_port/gb10_normalization.hip.cpp",
              ROOT / "native/linux_core_port/gb10_normalization.h", Path(__file__),
              ROOT / "native/linux_core_port/gb10_gdn.h",
              ROOT / "tools/test_linux_core_gdn.py"]
    inputs += [ROOT / "native/providers/gdn" / name for name in (
        "sm121_q2_gated_math.h", "sm121_mtp_residual.h",
        "sm121_mtp_residual_math.h", "sm121_mtp_math.h")]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(result=json.loads(run.stdout), build_command=command,
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in inputs],
        stub_sha256=sha(stub), silu_table_sha256=sha(args.silu_table),
        host_only=True, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
