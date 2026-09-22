#!/usr/bin/env python3
"""Check MoE ABI, carrier lifetimes and byte conversions with ASan/UBSan."""
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
    stub.write_text(STUB[:STUB.index("template<class K,class...A>")] + r'''
#define __forceinline__ inline
#define __host__
#include <type_traits>
constexpr int hipMemcpyDeviceToHost=2;
template<class T> T __shfl_xor(T v,unsigned,unsigned) { return v; }
template<class T> T __shfl_down(T v,unsigned,unsigned) { return v; }
inline unsigned atomicOr(unsigned* p,unsigned v) { const unsigned old=*p; *p|=v; return old; }
inline int hipMemsetAsync(void* p,int value,size_t size,void* stream) {
 assert(!stream && !value && size==4); memset(p,value,size); return 0;
}
struct RecordedMoeLaunch { unsigned blocks; std::vector<std::uintptr_t> args; };
inline std::vector<RecordedMoeLaunch> moe_launches;
template<class T>std::uintptr_t moe_address(T x) {
 if constexpr(std::is_pointer_v<T>)return reinterpret_cast<std::uintptr_t>(x);
 else return static_cast<std::uintptr_t>(x);
}
template<class K,class...A> void fake_launch(const char* name,dim3 grid,dim3 block,void* stream,K kernel,A...args) {
 if(false)kernel(args...);
 assert(!stream && (block.x==256 || block.x==64 || block.x==32) && grid.x>0 && grid.x<=131072);
 moe_launches.push_back({grid.x,{moe_address(args)...}});
 fake_events.push_back(name);
}
#define hipLaunchKernelGGL(k,g,b,z,s,...) fake_launch(#k,g,b,s,k,__VA_ARGS__)
''')
    source = ROOT / "native/linux_core_port/gb10_moe_host_contract_test.cpp"
    executable = out / "host-contract"
    command = ["clang++", "-std=c++17", "-O1", "-fno-fast-math", "-ffp-contract=off", "-fsanitize=address,undefined",
               "-I", str(out / "stub"), "-I", str(ROOT / "third_party/aima_linux/native/include"),
               str(source), str(ROOT / "third_party/aima_linux/native/src/sha256.cpp"), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable), str(out / "sha-fixture.bin")], capture_output=True, text=True, timeout=30)
    (out / "run.stderr").write_text(run.stderr)
    run.check_returncode()
    inputs = [source, Path(__file__), ROOT / "tools/test_linux_core_gdn.py"]
    inputs += [ROOT / "native/linux_core_port" / n for n in
               ("gb10_moe.hip.cpp", "gb10_moe.h", "gb10_moe_math.h", "gb10_moe_assets.inc", "gb10_gdn.h", "gb10_decode_moe.h")]
    inputs += [ROOT / "native/providers/gdn" / n for n in
               ("sm121_mtp_residual_math.h", "sm121_mtp_math.h", "sm121_q1_math.h", "sm121_mtp_moe_math.h")]
    inputs += [ROOT / "native/providers/moe_accumulator" / n for n in ("sm121_router_exp.h", "sm121_shared_gate.h")]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(result=json.loads(run.stdout), build_command=command,
                  inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in inputs],
                  stub_sha256=sha(stub), host_only=True, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
