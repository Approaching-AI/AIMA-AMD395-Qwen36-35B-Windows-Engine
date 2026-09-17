"""Execute the actual scalar matrix option and both host launch wrappers."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaScalarMatrixLaunchTests(unittest.TestCase):
    def test_option_launch_shapes_and_error_propagation(self):
        header = (ROOT / "native/providers/gdn/blackwell_scalar_matrices.h").read_text()
        source = (ROOT / "native/providers/gdn/blackwell_cooperative.cpp").read_text()
        mode = "inline int mode()" + header.split("inline int mode()", 1)[1].split("__device__", 1)[0]
        wu = "hipError_t wu(" + source.split("hipError_t wu(", 1)[1].split("hipError_t scores(", 1)[0]
        output = "hipError_t output(" + source.split("hipError_t output(", 1)[1].split("hipError_t state(", 1)[0]
        code = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <tuple>
#include <type_traits>
#include "native/providers/gdn/coarse_interval_policy.h"
#include "native/providers/gdn/paired_score_policy.h"
namespace qrt_fla_lifetime { void output_kernel(){} template<unsigned C>void state_kernel(){} }
namespace qrt_fla_interval { template<bool C,bool A>void wu_kernel(){} template<bool C,bool A>void state_kernel(){} template<bool C,bool A>void output_kernel(){} }
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t=void*;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c) {} };
namespace qrt_fla_blackwell_scalar {
void wu_kernel() {} void output_kernel() {}
''' + mode + r'''
}
constexpr unsigned threads=256,tile_columns=8;
void wu_kernel() {} void output_kernel() {}
unsigned launches=0,queries=0,count_seen=0;
void(*kernel_seen)()=nullptr;hipStream_t stream_seen=nullptr;
dim3 grid_seen,threads_seen;bool fail=false;
template<class... Args> void record(void(*kernel)(),dim3 grid,dim3 block,unsigned,hipStream_t stream,Args... args) {
    ++launches;kernel_seen=kernel;grid_seen=grid;threads_seen=block;stream_seen=stream;
    const auto values=std::make_tuple(args...);
    if constexpr(sizeof...(Args)==8u)count_seen=std::get<6>(values);
    else if constexpr(sizeof...(Args)==10u)count_seen=std::get<7>(values);
    else if constexpr(std::is_integral_v<std::tuple_element_t<6,decltype(values)>>)count_seen=std::get<6>(values);
    else count_seen=std::get<7>(values);
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError() { ++queries;return fail?hipErrorUnknown:hipSuccess; }
''' + wu + output + r'''
int main() {
    unsetenv("QRT_FLA_GDN_PAIRED_SCORE_ARENAS");
    unsetenv("QRT_FLA_GDN_COARSE_INTERVAL");
    unsetenv("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES");assert(qrt_fla_blackwell_scalar::mode()==0);
    for (const char* v:{"","0","1","2","-1","01","1 ","true"}) {
        setenv("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES",v,1);
        const int expected=!v[0]||!std::strcmp(v,"0")?0:!std::strcmp(v,"1")?1:-1;
        assert(qrt_fla_blackwell_scalar::mode()==expected);
    }
    uint16_t words[16]{};float values[16]{};unsigned char table[1]{};
    const auto stream=reinterpret_cast<void*>(uintptr_t(0x1234));
    for (const char* setting:{"0","1","invalid"}) for (unsigned count:{1u,63u,64u,65u,1024u}) {
        setenv("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES",setting,1);
        const int selected=qrt_fla_blackwell_scalar::mode();
        for (bool output_call:{false,true}) for (bool error:{false,true}) {
            fail=error;const unsigned before=launches,q=queries;
            const auto status=output_call?output(words,words+1,words+2,values,words+3,values+1,count,table,stream)
                :wu(words,words+1,words+2,words+3,values,words+4,words+1,count,table,stream);
            if (selected<0) {assert(status==hipErrorInvalidValue&&launches==before&&queries==q);continue;}
            assert(status==(error?hipErrorUnknown:hipSuccess)&&launches==before+1&&queries==q+1);
            const auto kernel=output_call?(selected?qrt_fla_blackwell_scalar::output_kernel:output_kernel)
                :(selected?qrt_fla_blackwell_scalar::wu_kernel:wu_kernel);
            assert(kernel_seen==kernel&&count_seen==count&&stream_seen==stream);
            assert(grid_seen.x==16u&&grid_seen.y==32u&&grid_seen.z==(count+63u)/64u);
            assert(threads_seen.x==256u&&threads_seen.y==1u&&threads_seen.z==1u);
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.cpp").write_text(code)
            subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-O1",
                            "-fsanitize=address,undefined", "-I", str(ROOT), str(path / "main.cpp"),
                            "-o", str(path / "check")], check=True, capture_output=True,
                           text=True, timeout=30)
            subprocess.run([str(path / "check")], check=True, capture_output=True,
                           text=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
