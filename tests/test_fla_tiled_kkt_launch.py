"""Execute the actual KKT dispatcher admission and failure propagation."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaTiledKktLaunchTests(unittest.TestCase):
    def test_modes_ranges_and_failed_launch(self):
        header = (ROOT / "native/providers/gdn/blackwell_kkt.h").read_text()
        actual = "inline int tiled_kkt_mode()" + header.split(
            "inline int tiled_kkt_mode()", 1
        )[1].split("__global__ void gate_kernel", 1)[0]
        source = r'''
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c) {} };
constexpr unsigned kChunk=64,kThreads=256,kGroup=16;
unsigned launches=0,error_queries=0,first_seen=0,kind=0;
dim3 grid_seen,threads_seen;
hipStream_t stream_seen=nullptr;
bool fail=false;
void dot_kernel() {}
template<bool FloatProducts> void tiled_dot_kernel() {}
void record(void(*kernel)(),dim3 grid,dim3 threads,unsigned,hipStream_t stream,
    const uint16_t*,const uint16_t*,float*,unsigned first) {
    ++launches;first_seen=first;grid_seen=grid;threads_seen=threads;stream_seen=stream;
    kind=kernel==dot_kernel?0:kernel==tiled_dot_kernel<false>?1:2;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError() { ++error_queries;return fail?hipErrorUnknown:hipSuccess; }
''' + actual + r'''
int main() {
    unsetenv("QRT_FLA_GDN_TILED_KKT");assert(tiled_kkt_mode()==0);
    for (const char* s:{"","0","1","2","3","-1","01","1 ","true"}) {
        setenv("QRT_FLA_GDN_TILED_KKT",s,1);
        const int expected=!s[0]||!std::strcmp(s,"0")?0:!std::strcmp(s,"1")?1:!std::strcmp(s,"2")?2:-1;
        assert(tiled_kkt_mode()==expected);
    }
    uint16_t k[2]{},beta[2]{};float a[2]{};
    auto rejected=[&](const uint16_t* kp,const uint16_t* bp,float* ap,unsigned first,unsigned chunks,unsigned mode) {
        const unsigned before=launches,queries=error_queries;
        assert(launch_dot_chunks(kp,bp,ap,first,chunks,mode,nullptr)==hipErrorInvalidValue);
        assert(launches==before&&error_queries==queries);
    };
    rejected(nullptr,beta,a,0,1,0);rejected(k,nullptr,a,0,1,1);rejected(k,beta,nullptr,0,1,2);
    for (unsigned mode:{3u,UINT32_MAX}) rejected(k,beta,a,0,1,mode);
    for (unsigned chunks:{0u,17u,UINT32_MAX}) rejected(k,beta,a,0,chunks,2);
    for (unsigned first:{16u,17u,UINT32_MAX}) rejected(k,beta,a,first,1,1);
    rejected(k,beta,a,15,2,0);rejected(k,beta,a,1,16,2);
    const auto stream=reinterpret_cast<void*>(uintptr_t(0x1234));
    for (unsigned mode:{0u,1u,2u}) for (unsigned first:{0u,1u,15u}) for (unsigned chunks:{1u,16u-first}) {
        const unsigned before=launches,queries=error_queries;
        assert(launch_dot_chunks(k,beta,a,first,chunks,mode,stream)==hipSuccess);
        assert(launches==before+1&&error_queries==queries+1&&first_seen==first&&kind==mode);
        assert(grid_seen.x==(mode?2u:256u)&&grid_seen.y==32u&&grid_seen.z==chunks*(mode?8u:1u));
        assert(threads_seen.x==256u&&threads_seen.y==1u&&threads_seen.z==1u&&stream_seen==stream);
        fail=true;
        assert(launch_dot_chunks(k,beta,a,first,chunks,mode,stream)==hipErrorUnknown);
        assert(launches==before+2&&error_queries==queries+2);fail=false;
    }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.cpp").write_text(source)
            subprocess.run([
                os.environ.get("CXX", "clang++"), "-std=c++17", "-O1", "-g",
                "-fsanitize=address,undefined", str(path / "main.cpp"),
                "-o", str(path / "check"),
            ], check=True, capture_output=True, text=True, timeout=30)
            subprocess.run([str(path / "check")], check=True, capture_output=True,
                           text=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
