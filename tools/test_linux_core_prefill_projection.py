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
#define __launch_bounds__(...)
inline dim3 gridDim;
inline unsigned __clz(unsigned v) { return __builtin_clz(v); }
inline unsigned __popc(unsigned v) { return __builtin_popcount(v); }
inline unsigned __ballot(bool v) { return unsigned(v); }
template<class T> T __shfl_xor(T v, unsigned, unsigned) { return v; }
template<class A,class B,class C> C recording_wmma(A,B,C value) { return value; }
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32 recording_wmma
inline unsigned atomicAdd(unsigned* p, unsigned v) { unsigned old=*p; *p+=v; return old; }
inline unsigned atomicOr(unsigned* p, unsigned v) { unsigned old=*p; *p|=v; return old; }
inline int hipMemsetAsync(void* p, int v, size_t bytes, void*) {
 memset(p,v,bytes);fake_events.push_back("memset");return 0;
}
struct FakeProfileEvent { unsigned serial = 0; };
using hipEvent_t = FakeProfileEvent*;
inline unsigned fake_live_profile_events = 0, fake_profile_serial = 0;
inline unsigned fake_profile_sync = 0, fake_count_copies = 0;
inline bool fake_negative_elapsed = false;
inline size_t fake_count_host_bytes = 0;
constexpr int hipMemcpyDeviceToDevice = 2, hipMemcpyDeviceToHost = 3;
inline int hipEventCreate(hipEvent_t* p) {
 *p=new FakeProfileEvent;++fake_live_profile_events;return 0;
}
inline int hipEventDestroy(hipEvent_t p) {delete p;--fake_live_profile_events;return 0;}
inline int hipEventRecord(hipEvent_t p, void* stream) {
 assert(!stream);p->serial=++fake_profile_serial;fake_events.push_back("event");return 0;
}
inline int hipEventSynchronize(hipEvent_t p) {
 assert(p->serial);fake_profile_sync=p->serial;return 0;
}
inline int hipEventElapsedTime(float* ms,hipEvent_t first,hipEvent_t last) {
 assert(first->serial&&last->serial>=first->serial&&last->serial<=fake_profile_sync);
 *ms=float(last->serial-first->serial)*0.25f;
 if(fake_negative_elapsed)*ms=-*ms;return 0;
}
inline int hipMemcpyAsync(void* d,const void* s,size_t n,int kind,void* stream) {
 assert(!stream&&kind==hipMemcpyDeviceToDevice&&n&&n%sizeof(unsigned)==0&&n<=97*sizeof(unsigned));
 memcpy(d,s,n);++fake_count_copies;fake_events.push_back("count-copy");return 0;
}
inline int profile_memcpy(void* d,const void* s,size_t n,int kind) {
 assert(kind==hipMemcpyDeviceToHost&&fake_profile_sync==fake_profile_serial);
 fake_count_host_bytes=n;memcpy(d,s,n);return 0;
}
#define hipMemcpy profile_memcpy
struct FakeProjectionLaunch { std::string name; dim3 grid; };
inline std::vector<FakeProjectionLaunch> fake_projection_launches;
template<class K,class... A> void projection_launch(const char* name,dim3 grid,
    dim3 block,void* stream,K kernel,A... args) {
 fake_projection_launches.push_back({name,grid});
 fake_launch(name,grid,block,stream,kernel,args...);
}
#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(k,g,b,z,s,...) projection_launch(#k,g,b,s,k,__VA_ARGS__)
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
