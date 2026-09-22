#!/usr/bin/env python3
"""Sanitize attention ownership, failure cleanup and native launch bindings.

The recording HIP/kernel substitutes do not execute attention arithmetic.
Actual numerical qualification requires the native model run.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUB = r'''
#pragma once
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>
#define __global__
#define __device__
#define HIP_KERNEL_NAME(...) __VA_ARGS__
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
inline dim3 blockIdx, blockDim, threadIdx;
using hipStream_t=void*; using hipError_t=int;
constexpr int hipSuccess=0, hipMemcpyHostToDevice=1;
inline int allocations=0, allocation_calls=0, fail_allocation=-1, copies=0, drains=0;
inline bool fail_copy=false;
inline int hipMalloc(void** p,size_t n) {
 if(allocation_calls++==fail_allocation)return 1;
 *p=malloc(n);if(!*p)return 1;++allocations;return 0;
}
inline int hipFree(void* p){if(p){free(p);--allocations;}return 0;}
inline int hipMemcpy(void* d,const void* s,size_t n,int kind) {
 assert(kind==hipMemcpyHostToDevice);if(fail_copy)return 1;
 memcpy(d,s,n);++copies;return 0;
}
inline int hipDeviceSynchronize(){++drains;return 0;}
inline const char* hipGetErrorString(int){return "injected HIP failure";}
struct Launch { std::string name;dim3 grid,block;void* stream;std::vector<std::uintptr_t> args; };
inline std::vector<Launch> launches;
inline size_t fail_launch=0;
inline int hipGetLastError(){return fail_launch && launches.size()==fail_launch ? 1 : 0;}
template<class T> std::uintptr_t address(T v) {
 if constexpr(std::is_pointer_v<T>)return reinterpret_cast<std::uintptr_t>(v);
 else if constexpr(std::is_arithmetic_v<T>)return static_cast<std::uintptr_t>(v);
 else return reinterpret_cast<std::uintptr_t>(v.plane);
}
template<class K,class...A>void record(const char* name,dim3 grid,dim3 block,void* stream,K kernel,A...args){
 if(false)kernel(args...);
 launches.push_back({name,grid,block,stream,{address(args)...}});
}
#define hipLaunchKernelGGL(k,g,b,z,s,...) record(#k,g,b,s,k,__VA_ARGS__)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reciprocal-table", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stub = out / "stub/hip/hip_runtime.h"
    stub.parent.mkdir(parents=True)
    stub.write_text(STUB)
    source = ROOT / "native/linux_core_port/gb10_decode_attention_host_contract_test.cpp"
    executable = out / "host-contract"
    command = ["clang++", "-std=c++17", "-O1", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-I", str(out / "stub"),
        "-I", str(ROOT / "third_party/aima_linux/native/include"), str(source),
        str(ROOT / "third_party/aima_linux/native/src/sha256.cpp"), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr)
    build.check_returncode()
    run = subprocess.run([str(executable), str(args.reciprocal_table.resolve()), str(out / "bad-table.bin")],
                         capture_output=True, text=True, timeout=30)
    (out / "run.stderr").write_text(run.stderr)
    run.check_returncode()
    sha = lambda p: hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
    paths = [source, ROOT / "native/linux_core_port/gb10_decode_attention.hip.cpp",
        ROOT / "native/linux_core_port/gb10_decode_attention.h", ROOT / "native/linux_core_port/gb10_gdn.h",
        ROOT / "native/linux_core_port/gb10_gdn.hip.cpp", Path(__file__)]
    report = dict(result=json.loads(run.stdout), build_command=command,
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in paths],
        stub_sha256=sha(stub), reciprocal_table_sha256=sha(args.reciprocal_table),
        host_only=True, kernel_arithmetic_executed=False, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
