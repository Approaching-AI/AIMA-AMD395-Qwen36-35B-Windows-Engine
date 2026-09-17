"""Exercise actual native EXP storage publication, cleanup and launch validation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeExp2WorkspaceTests(unittest.TestCase):
    def test_failure_cleanup_reuse_source_identity_and_launch_ranges(self):
        body = (ROOT / "native/providers/ck_fmha/native_exp2_workspace.h").read_text()
        body = "\n".join(line for line in body.splitlines()
                         if not line.startswith(("#pragma", "#include")))
        harness = r'''
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <initializer_list>
using hipStream_t=void*;
enum hipError_t{hipSuccess,hipErrorUnknown,hipErrorInvalidValue};
constexpr int hipMemcpyDeviceToHost=1;
struct dim3{unsigned x,y,z;explicit dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
namespace qrt_sm121_exp2_native_delta{
constexpr size_t packed_bytes=82182144u;
void build(){} void verify(){}
}
namespace qrt_native_delta_probability{void probabilities(){}}
namespace qrt_blackwell_attention{constexpr unsigned kQueryHeads=16u;}
unsigned calls=0,fail_call=0,errors=0,submissions=0,completions=0;
bool pending=false;
const char* last_kernel=nullptr;
std::set<void*> live;
hipError_t step(){return ++calls==fail_call?hipErrorUnknown:hipSuccess;}
hipError_t hipMalloc(void** p,size_t bytes){
    if(bytes!=4u&&bytes!=qrt_sm121_exp2_native_delta::packed_bytes)std::abort();
    auto status=step();if(status!=hipSuccess)return status;
    *p=new unsigned char[16];live.insert(*p);return hipSuccess;
}
hipError_t hipFree(void* p){
    if(!p)return hipSuccess;
    if(pending||live.erase(p)!=1u)std::abort();
    delete[] static_cast<unsigned char*>(p);return hipSuccess;
}
hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t){
    if(!live.count(p)||value||bytes!=4u)std::abort();return step();
}
hipError_t hipMemcpyAsync(void* out,const void* input,size_t bytes,int kind,hipStream_t){
    if(!live.count(const_cast<void*>(input))||bytes!=4u||kind!=hipMemcpyDeviceToHost)std::abort();
    auto status=step();if(status==hipSuccess)*static_cast<unsigned*>(out)=errors;return status;
}
hipError_t hipStreamSynchronize(hipStream_t){pending=false;++completions;return step();}
hipError_t hipGetLastError(){return step();}
template<class... Args>void submit(const char* name,void(*)(),dim3,dim3,unsigned,hipStream_t,Args...){
    last_kernel=name;pending=true;++submissions;
}
#define hipLaunchKernelGGL(kernel,...) submit(#kernel,kernel,__VA_ARGS__)
''' + body + r'''
void reset(){if(!live.empty()||pending)std::abort();calls=fail_call=errors=submissions=completions=0;}
int main(){
    using namespace qrt_native_exp2_workspace;
    unsigned char source=0,other=0;Workspace w;
    if(prepare(w,nullptr,nullptr)!=hipErrorInvalidValue||calls)return 1;
    for(unsigned failure=1;failure<=7;++failure){
        reset();fail_call=failure;
        if(prepare(w,&source,nullptr)!=hipErrorUnknown||w.packed||w.original||!live.empty()||completions!=1u)return 2;
    }
    reset();errors=1u;
    if(prepare(w,&source,nullptr)!=hipErrorInvalidValue||w.packed||w.original||!live.empty())return 3;
    reset();
    if(prepare(w,&source,nullptr)!=hipSuccess||!w.packed||w.original!=&source||live.size()!=1u||
        calls!=7u||submissions!=2u||std::strcmp(last_kernel,"delta::verify"))return 4;
    auto* original_owner=w.packed;
    if(prepare(w,&source,nullptr)!=hipSuccess||calls!=7u||w.packed!=original_owner)return 5;
    if(prepare(w,&other,nullptr)!=hipErrorInvalidValue||calls!=7u||w.packed!=original_owner)return 6;
    float score=0,scale=0;uint16_t probability=0;
    auto call=[&](unsigned start,unsigned count,unsigned stride,const unsigned char* src=nullptr){
        return launch(&w,&score,&probability,&scale,start,count,stride,src?src:&source,true,nullptr);
    };
    if(call(0,0,0)!=hipErrorInvalidValue||call(0,129,129)!=hipErrorInvalidValue||
        call(UINT32_MAX,2,1)!=hipErrorInvalidValue||call(8192,1,8193)!=hipErrorInvalidValue||
        call(8191,2,8193)!=hipErrorInvalidValue||call(0,128,127)!=hipErrorInvalidValue||
        call(0,128,128,&other)!=hipErrorInvalidValue||calls!=7u)return 7;
    for(unsigned count:{1u,17u,128u}){
        if(call(8192u-count,count,8192u)!=hipSuccess||!pending||
            std::strcmp(last_kernel,"qrt_native_delta_probability::probabilities"))return 8;
        if(hipStreamSynchronize(nullptr)!=hipSuccess)return 9;
    }
    fail_call=calls+1u;
    if(call(0,128,128)!=hipErrorUnknown)return 10;
    (void)hipStreamSynchronize(nullptr);
    release(w);release(w);
    if(w.packed||w.original||!live.empty())return 11;
    if(call(0,128,128)!=hipErrorInvalidValue)return 12;
    reset();
    if(prepare(w,&other,nullptr)!=hipSuccess||w.original!=&other)return 13;
    release(w);
    std::puts("native_exp2_workspace_pass failures=7 mismatch_rejected=1 reuse_and_release=1 native_kernel_executed=0");
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-native-exp-owner-") as directory:
            source = Path(directory) / "check.cpp"
            source.write_text(harness)
            executable = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                            "-fno-sanitize-recover=all", str(source), "-o", executable],
                           check=True, timeout=30)
            subprocess.run([executable], check=True, timeout=20)


if __name__ == "__main__":
    unittest.main()
