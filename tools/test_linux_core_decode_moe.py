#!/usr/bin/env python3
"""Check complete decode MoE wiring and terminal lifetime under ASan/UBSan.

Includes the actual existing kernel headers with a recording HIP substitute;
kernel arithmetic itself is qualified separately on the native host.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
from test_linux_core_decode_attention import STUB as ATTENTION_STUB

ROOT = Path(__file__).resolve().parents[1]
STUB = ATTENTION_STUB.replace("#define __device__", "#define __host__\n#define __forceinline__ inline\n#define __device__")
STUB = STUB.replace("#define HIP_KERNEL_NAME", "#define __shared__ static\n#define __syncthreads() ((void)0)\n#define HIP_KERNEL_NAME")
STUB = STUB.replace("constexpr int hipSuccess=0, hipMemcpyHostToDevice=1;",
    "constexpr int hipSuccess=0, hipMemcpyHostToDevice=1, hipMemcpyDeviceToHost=2, hipMemcpyDeviceToDevice=3, hipErrorInvalidValue=2;")
STUB = STUB.replace("inline bool fail_copy=false;", "inline bool fail_copy=false;\ninline int device_reads=0, resets=0, snapshots=0;\ninline unsigned inject_flag=0;")
STUB = STUB.replace("assert(kind==hipMemcpyHostToDevice);if(fail_copy)return 1;",
    "assert(kind==hipMemcpyHostToDevice || kind==hipMemcpyDeviceToHost);if(fail_copy)return 1;\n if(kind==hipMemcpyDeviceToHost)++device_reads;")
STUB = STUB[:STUB.index("template<class T> std::uintptr_t address")] + r'''
#include <cmath>
inline unsigned __clz(unsigned v){return v ? __builtin_clz(v) : 32u;}
template<class T>T __shfl(T v,unsigned,unsigned){return v;}
template<class T>T __shfl_down(T v,unsigned,unsigned){return v;}
template<class T>T __shfl_xor(T v,unsigned,unsigned){return v;}
inline unsigned atomicOr(unsigned* p,unsigned x){unsigned old=*p;*p|=x;return old;}
inline int hipMemsetAsync(void* p,int x,size_t n,void* stream){assert(!stream && x==0 && n==4);memset(p,x,n);++resets;return 0;}
inline int hipMemcpyAsync(void* d,const void* s,size_t n,int kind,void* stream){
 assert(!stream && kind==hipMemcpyDeviceToDevice && n==4096);if(fail_copy)return 1;
 memcpy(d,s,n);++snapshots;return 0;
}
template<class T>std::uintptr_t address(T v){return reinterpret_cast<std::uintptr_t>(v);}
template<class T>void flatten(std::vector<std::uintptr_t>& out,T v){
 if constexpr(std::is_pointer_v<T>)out.push_back(address(v));
 else if constexpr(std::is_arithmetic_v<T>)out.push_back(static_cast<std::uintptr_t>(v));
 else {
  for(const void* p:{static_cast<void*>(v.router),static_cast<void*>(v.shared_gate),static_cast<void*>(v.shared_gate_up),
      static_cast<void*>(v.shared_activated),static_cast<void*>(v.shared_down),static_cast<void*>(v.shared),
      static_cast<void*>(v.routed_gate_up),static_cast<void*>(v.routed_activated),static_cast<void*>(v.routed_weighted),
      static_cast<void*>(v.routed),static_cast<void*>(v.output),static_cast<void*>(v.topk_ids),
      static_cast<void*>(v.topk_weights),static_cast<void*>(v.invalid)})out.push_back(address(p));
  if(inject_flag)*v.invalid|=inject_flag;
 }
}
template<class K,class...A>void record(const char* name,dim3 grid,dim3 block,void* stream,K kernel,A...args){
 if(false)kernel(args...);
 Launch launch{name,grid,block,stream,{}};(flatten(launch.args,args),...);launches.push_back(launch);
}
#define hipLaunchKernelGGL(k,g,b,z,s,...) record(#k,g,b,s,k,__VA_ARGS__)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--silu-table", type=Path, required=True)
    parser.add_argument("--router-table", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stub = out / "stub/hip/hip_runtime.h"
    stub.parent.mkdir(parents=True)
    stub.write_text(STUB)
    source = ROOT / "native/linux_core_port/gb10_decode_moe_host_contract_test.cpp"
    executable = out / "host-contract"
    command = ["clang++", "-std=c++17", "-O1", "-fno-fast-math", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-I", str(out / "stub"),
        "-I", str(ROOT / "third_party/aima_linux/native/include"), str(source),
        str(ROOT / "third_party/aima_linux/native/src/sha256.cpp"), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable), str(args.silu_table.resolve()), str(args.router_table.resolve()), str(out / "bad-table.bin")],
        capture_output=True, text=True, timeout=60)
    (out / "run.stderr").write_text(run.stderr)
    run.check_returncode()
    sha = lambda p: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    paths = [source, Path(__file__), ROOT / "tools/test_linux_core_decode_attention.py",
        ROOT / "native/linux_core_port/gb10_decode_moe.hip.cpp", ROOT / "native/linux_core_port/gb10_decode_moe.h",
        ROOT / "native/linux_core_port/gb10_gdn.h", ROOT / "native/linux_core_port/gb10_gdn.hip.cpp"]
    paths += [ROOT / "native/providers/gdn" / n for n in ("sm121_mtp_moe.h", "sm121_mtp_moe_math.h", "sm121_mtp_moe_layout.h", "sm121_mtp_projection.h")]
    report = dict(result=json.loads(run.stdout), build_command=command,
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in paths],
        stub_sha256=sha(stub), silu_table_sha256=sha(args.silu_table), router_table_sha256=sha(args.router_table),
        host_only=True, kernel_arithmetic_executed=False, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
