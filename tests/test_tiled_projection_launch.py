"""Execute the bitmap and tiled replay launch guards and failure paths."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TiledProjectionLaunchTests(unittest.TestCase):
    def test_capacity_empty_selection_and_submission_errors(self):
        header = (ROOT / "native/providers/moe_accumulator/sm121_tiled_projection.h").read_text()
        launch = 'inline hipError_t mark(' + header.split('inline hipError_t mark(', 1)[1].split(
            '} // namespace qrt_sm121_tiled_projection', 1)[0]
        code = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c) {}};
constexpr unsigned threads=256u;
void mark_kernel() {}
template<unsigned R,unsigned T,unsigned C> void replay_kernel() {}
unsigned launches=0,queries=0,clears=0;size_t cleared=0;bool fail_clear=false,fail_launch=false;
dim3 grid_seen,block_seen;void(*kernel_seen)()=nullptr;hipStream_t stream_seen=nullptr;
hipError_t hipMemsetAsync(void*,int value,size_t bytes,hipStream_t stream) {
    assert(!value);++clears;cleared=bytes;stream_seen=stream;return fail_clear ? hipErrorUnknown : hipSuccess;
}
hipError_t hipGetLastError() {++queries;return fail_launch ? hipErrorUnknown : hipSuccess;}
template<class... Args> void record(void(*kernel)(),dim3 grid,dim3 block,unsigned,hipStream_t stream,Args...) {
    ++launches;kernel_seen=kernel;grid_seen=grid;block_seen=block;stream_seen=stream;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
''' + launch + r'''
int main() {
    uint16_t words[1]{};unsigned flags[1]{};float out[1]{};
    const auto stream=reinterpret_cast<void*>(uintptr_t(0x1234));
    for(unsigned cells:{0u,134217729u,UINT32_MAX})
        assert(mark(flags,0u,cells,flags,SIZE_MAX,stream)==hipErrorInvalidValue);
    assert(mark(flags,10u,9u,flags,1u,stream)==hipErrorInvalidValue);
    assert(mark(flags,10u,65u,flags,2u,stream)==hipErrorInvalidValue);
    assert(!launches&&!queries&&!clears);
    for(unsigned count:{0u,1u,257u}) for(bool error:{false,true}) {
        fail_clear=error;fail_launch=false;launches=queries=clears=0;
        const auto status=mark(flags,count,257u,flags,9u,stream);
        assert(status==(error?hipErrorUnknown:hipSuccess)&&clears==1u&&cleared==36u&&stream_seen==stream);
        assert(launches==unsigned(!error&&count)&&queries==launches);
    }
    fail_clear=false;fail_launch=true;launches=queries=clears=0;
    assert(mark(flags,257u,257u,flags,9u,stream)==hipErrorUnknown&&launches==1u&&queries==1u);
    assert(grid_seen.x==2u&&block_seen.x==256u&&kernel_seen==mark_kernel);
    fail_launch=false;launches=queries=clears=0;
    for(unsigned width:{0u,15u,17u,4112u})
        assert(launch(words,words,flags,flags,flags,263u,out,129u,65u,width,64u,stream)==hipErrorInvalidValue);
    assert(launch(words,words,flags,flags,flags,262u,out,129u,65u,272u,64u,stream)==hipErrorInvalidValue);
    assert(launch(words,words,flags,flags,flags,SIZE_MAX,out,16385u,65u,272u,64u,stream)==hipErrorInvalidValue);
    assert(launch(words,words,flags,flags,flags,SIZE_MAX,out,129u,8193u,272u,64u,stream)==hipErrorInvalidValue);
    assert(launch(words,words,flags,flags,flags,SIZE_MAX,out,129u,65u,272u,32u,stream)==hipErrorInvalidValue);
    assert(!launches&&!queries&&!clears);
    for(unsigned tile:{64u,128u}) for(bool error:{false,true}) {
        fail_launch=error;launches=queries=0;
        assert(launch(words,words,flags,flags,flags,263u,out,129u,65u,272u,tile,stream)==(error?hipErrorUnknown:hipSuccess));
        assert(launches==1u&&queries==1u&&stream_seen==stream&&block_seen.x==256u);
        assert(grid_seen.x==(tile==64u?3u:2u)&&grid_seen.y==3u&&grid_seen.z==1u);
        assert((kernel_seen==(tile==64u?replay_kernel<64u,32u,256u>:replay_kernel<128u,32u,512u>)));
    }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "main.cpp").write_text(code)
            subprocess.run([os.environ.get("CXX", "clang++"), "-std=c++17", "-O1",
                            "-fsanitize=address,undefined", str(path / "main.cpp"),
                            "-o", str(path / "check")], check=True, capture_output=True,
                           text=True, timeout=30)
            subprocess.run([str(path / "check")], check=True, capture_output=True,
                           text=True, timeout=10)


if __name__ == "__main__":
    unittest.main()
