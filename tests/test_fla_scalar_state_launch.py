"""Execute the actual scalar state option and bounded segment launcher."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FlaScalarStateLaunchTests(unittest.TestCase):
    def test_modes_layout_and_failed_launch(self):
        header = (ROOT / "native/providers/gdn/blackwell_scalar_state.h").read_text()
        source = (ROOT / "native/providers/gdn/blackwell_cooperative.cpp").read_text()
        mode = "inline int state_columns()" + header.split("inline int state_columns()", 1)[1].split("// Each", 1)[0].split("// Retain", 1)[0]
        launch = "hipError_t state(" + source.split("hipError_t state(", 1)[1].split("hipError_t state_checkpoints(", 1)[0]
        code = r'''
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <tuple>
#include "native/providers/gdn/coarse_interval_policy.h"
namespace qrt_fla_interval { template<bool C,bool A>void wu_kernel(){} template<bool C,bool A>void state_kernel(){} template<bool C,bool A>void output_kernel(){} }
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t=void*;
struct dim3 { unsigned x,y,z; dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c) {} };
namespace qrt_fla_checkpoint { struct Segment {}; }
namespace qrt_fla_blackwell_scalar {
template<unsigned Columns> void state_kernel() {}
''' + mode + r'''
}
constexpr unsigned threads=256,state_columns=4;
template<bool Capture> void state_kernel() {}
unsigned launches=0,queries=0,count_seen=0;
void(*kernel_seen)()=nullptr;hipStream_t stream_seen=nullptr;
dim3 grid_seen,threads_seen;bool fail=false;
template<class... Args> void record(void(*kernel)(),dim3 grid,dim3 block,unsigned,hipStream_t stream,Args... args) {
    ++launches;kernel_seen=kernel;grid_seen=grid;threads_seen=block;stream_seen=stream;
    count_seen=std::get<7>(std::make_tuple(args...));
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError() { ++queries;return fail?hipErrorUnknown:hipSuccess; }
''' + launch + r'''
int main() {
    unsetenv("QRT_FLA_GDN_COARSE_INTERVAL");
    unsetenv("QRT_FLA_GDN_SCALAR_FLOAT_STATE");assert(qrt_fla_blackwell_scalar::state_columns()==0);
    for (const char* value:{"","0","4","8","1","-1","04","8 ","true"}) {
        setenv("QRT_FLA_GDN_SCALAR_FLOAT_STATE",value,1);
        const int expected=!value[0]||!std::strcmp(value,"0")?0:!std::strcmp(value,"4")?4:!std::strcmp(value,"8")?8:-1;
        assert(qrt_fla_blackwell_scalar::state_columns()==expected);
    }
    uint16_t words[16]{};float values[16]{};unsigned char table[1]{};
    const auto stream=reinterpret_cast<void*>(uintptr_t(0x1234));
    for (const char* setting:{"0","4","8","invalid"}) for (unsigned count:{1u,63u,64u,65u,1024u}) {
        setenv("QRT_FLA_GDN_SCALAR_FLOAT_STATE",setting,1);
        const int selected=qrt_fla_blackwell_scalar::state_columns();
        for (bool error:{false,true}) {
            fail=error;const unsigned before=launches,q=queries;
            const auto status=state(words,words+1,words+2,values,words+3,words+4,values+1,count,table,stream);
            if (selected<0) {assert(status==hipErrorInvalidValue&&launches==before&&queries==q);continue;}
            assert(status==(error?hipErrorUnknown:hipSuccess)&&launches==before+1&&queries==q+1);
            const auto kernel=selected==4?qrt_fla_blackwell_scalar::state_kernel<4u>:
                selected==8?qrt_fla_blackwell_scalar::state_kernel<8u>:state_kernel<false>;
            assert(kernel_seen==kernel&&count_seen==count&&stream_seen==stream);
            assert(grid_seen.x==(selected==8?16u:32u)&&grid_seen.y==32u&&grid_seen.z==1u);
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
