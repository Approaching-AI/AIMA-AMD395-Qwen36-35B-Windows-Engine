#!/usr/bin/env python3
"""Check GDN conversions, provider ABI and failure handling without HIP or a model."""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUB = r'''
#pragma once
#include <cstdlib>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>
#include <cassert>
#include <cstdint>
#define __global__
#define __device__
#define __shared__ static
#define __syncthreads() ((void)0)
struct dim3{unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
inline dim3 blockIdx,blockDim,threadIdx;
inline float __shfl_xor(float value,unsigned,unsigned){return value;}
using hipStream_t=void*;using hipError_t=int;
using hipModule_t=void*;using hipFunction_t=void*;using hipDeviceptr_t=std::uintptr_t;
constexpr int hipSuccess=0,hipMemcpyHostToDevice=1;
inline int hipSetDevice(int){return 0;}
inline int hipMalloc(void**p,size_t n){*p=malloc(n);return *p?0:1;}
inline int hipFree(void*p){free(p);return 0;}
inline int hipMemcpy(void*d,const void*s,size_t n,int){memcpy(d,s,n);return 0;}
inline int hipDeviceSynchronize(){return 0;}
inline int hipGetLastError(){return 0;}
inline const char* hipGetErrorString(int){return "stub";}
inline std::vector<std::string> fake_events;
inline std::vector<std::uintptr_t> fake_module_pointers;
inline int hipModuleLoadData(void** module,const void* bytes){
 assert(std::memcmp(bytes,"\177ELF",4)==0);*module=reinterpret_cast<void*>(0x71);return 0;
}
inline int hipModuleGetFunction(void** function,void* module,const char* name){
 assert(module==reinterpret_cast<void*>(0x71)&&std::string(name)=="recompute_w_u_fwd_kernel");
 *function=reinterpret_cast<void*>(0x72);return 0;
}
inline int hipModuleUnload(void* module){assert(module==reinterpret_cast<void*>(0x71));return 0;}
inline int hipModuleLaunchKernel(void* function,unsigned x,unsigned y,unsigned z,
 unsigned bx,unsigned by,unsigned bz,unsigned shared,void* stream,void** args,void* extra){
 assert(function==reinterpret_cast<void*>(0x72)&&x==128&&y==32&&z==1&&bx==64&&by==1&&bz==1);
 assert(shared==8192&&stream==nullptr&&extra==nullptr);
 fake_module_pointers.clear();
 for(unsigned i=0;i<7;++i){std::uintptr_t p;std::memcpy(&p,args[i],sizeof(p));fake_module_pointers.push_back(p);}
 assert(*static_cast<std::int32_t*>(args[7])==8192);
 assert(*static_cast<hipDeviceptr_t*>(args[8])==0&&*static_cast<hipDeviceptr_t*>(args[9])==0);
 fake_events.push_back("native_wu_module");return 0;
}
inline void* fake_stream=nullptr;
inline bool fake_q2_flags_verified=false;
template<class K,class...A>void fake_launch(const char*name,dim3 grid,dim3 block,void*stream,K kernel,A...args){
 if(false)kernel(args...);
 assert(stream==fake_stream);assert(block.x==256||block.x==128);
 fake_events.push_back(name);
 if constexpr(sizeof...(A)==16){auto t=std::make_tuple(args...);assert(std::get<4>(t)==false&&std::get<13>(t)==false);assert(grid.x==32&&block.x==128);fake_q2_flags_verified=true;}
}
#define hipLaunchKernelGGL(k,g,b,z,s,...) fake_launch(#k,g,b,s,k,__VA_ARGS__)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cxx", default="clang++")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    stub = out / "stub/hip/hip_runtime.h"
    stub.parent.mkdir(parents=True)
    stub.write_text(STUB, encoding="utf-8")
    (stub.parent / "hip_runtime_api.h").write_text('#include "hip_runtime.h"\n', encoding="utf-8")
    executable = out / "host-contract"
    source = ROOT / "native/linux_core_port/gb10_gdn_host_contract_test.cpp"
    command = [args.cxx, "-std=c++17", "-O1", "-ffp-contract=off",
        "-fsanitize=address,undefined", "-I", str(out / "stub"),
        "-I", str(ROOT / "third_party/aima_linux/native/include"), str(source),
        str(ROOT / "third_party/aima_linux/native/src/sha256.cpp"),
        str(ROOT / "third_party/aima_linux/native/src/aot_kernel.hip.cpp"), "-o", str(executable)]
    build = subprocess.run(command, capture_output=True, text=True, timeout=90)
    (out / "build.stderr").write_text(build.stderr, encoding="utf-8")
    build.check_returncode()
    run = subprocess.run([str(executable), str(out / "sha-fixture.bin")],
        capture_output=True, text=True, timeout=30)
    (out / "run.stderr").write_text(run.stderr, encoding="utf-8")
    run.check_returncode()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inputs = [source, ROOT / "native/linux_core_port/gb10_gdn.hip.cpp",
              ROOT / "native/linux_core_port/gb10_gdn.h", ROOT / "native/linux_core_port/gb10_gdn_assets.inc",
              ROOT / "native/linux_core_port/gb10_gdn_wu_image.inc",
              ROOT / "third_party/aima_linux/native/src/aot_kernel.hip.cpp"]
    report = dict(result=json.loads(run.stdout), build_command=command,
        inputs=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p)) for p in inputs],
        test_source_sha256=sha(Path(__file__)), stub_sha256=sha(stub),
        host_only=True, model_inference_acceptance=False)
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["result"]))


if __name__ == "__main__":
    main()
