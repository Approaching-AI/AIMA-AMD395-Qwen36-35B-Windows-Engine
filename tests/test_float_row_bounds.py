"""Check reusable row bounds against the original wide signed K16 sum."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FloatRowBoundsTests(unittest.TestCase):
    def test_metadata_capacity_and_launch_errors(self):
        header = (ROOT / "native/providers/moe_accumulator/sm121_bounded_projection.h").read_text()
        functions = "inline hipError_t prepare(" + header.split("inline hipError_t prepare(", 1)[1].split(
            "} // namespace qrt_sm121_bounded_projection", 1)[0]
        code = r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
enum hipError_t {hipSuccess,hipErrorInvalidValue,hipErrorUnknown};
using hipStream_t=void*;
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c) {}};
constexpr unsigned threads=256u,lanes=4u;
void prepare_kernel() {}
template<bool Audit> void replay_kernel() {}
unsigned calls=0,queries=0;bool fail=false;dim3 grid,block;hipStream_t seen_stream=nullptr;void(*seen_kernel)()=nullptr;
hipError_t hipGetLastError() {++queries;return fail?hipErrorUnknown:hipSuccess;}
template<class... A> void record(void(*kernel)(),dim3 g,dim3 b,unsigned shared,hipStream_t stream,A...) {
    assert(!shared);++calls;grid=g;block=b;seen_stream=stream;seen_kernel=kernel;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
''' + functions + r'''
int main() {
    uint16_t values[1]{};uint32_t metadata[1]{};float output[1]{};
    auto stream=reinterpret_cast<void*>(uintptr_t(0x1234));
    for(unsigned width:{0u,15u,17u,4112u}) assert(prepare(values,metadata,SIZE_MAX,65u,width,stream)==hipErrorInvalidValue);
    for(unsigned rows:{0u,16385u,UINT32_MAX}) assert(prepare(values,metadata,SIZE_MAX,rows,272u,stream)==hipErrorInvalidValue);
    assert(prepare(values,metadata,1104u,65u,272u,stream)==hipErrorInvalidValue);
    assert(prepare(nullptr,metadata,SIZE_MAX,65u,272u,stream)==hipErrorInvalidValue);
    assert(prepare(values,nullptr,SIZE_MAX,65u,272u,stream)==hipErrorInvalidValue);
    assert(!calls&&!queries);
    for(bool error:{false,true}) {
        fail=error;calls=queries=0;
        assert(prepare(values,metadata,1105u,65u,272u,stream)==(error?hipErrorUnknown:hipSuccess));
        assert(calls==1u&&queries==1u&&grid.x==5u&&block.x==256u&&seen_stream==stream&&seen_kernel==prepare_kernel);
    }
    calls=queries=0;fail=false;
    const auto invoke=[&](unsigned count,unsigned rows,unsigned tokens,unsigned width,size_t wc,size_t ic) {
        return launch(values,values,metadata,wc,metadata,ic,metadata,count,output,rows,tokens,width,stream);
    };
    assert(invoke(0u,65u,33u,272u,1104u,561u)==hipErrorInvalidValue);
    assert(invoke(0u,65u,33u,272u,1105u,560u)==hipErrorInvalidValue);
    assert(invoke(2146u,65u,33u,272u,1105u,561u)==hipErrorInvalidValue);
    for(unsigned width:{0u,15u,17u,4112u}) assert(invoke(0u,65u,33u,width,SIZE_MAX,SIZE_MAX)==hipErrorInvalidValue);
    for(unsigned rows:{0u,16385u,UINT32_MAX}) assert(invoke(0u,rows,33u,272u,SIZE_MAX,SIZE_MAX)==hipErrorInvalidValue);
    for(unsigned tokens:{0u,8193u,UINT32_MAX}) assert(invoke(0u,65u,tokens,272u,SIZE_MAX,SIZE_MAX)==hipErrorInvalidValue);
    assert(invoke(0u,65u,33u,272u,1105u,561u)==hipSuccess&&!calls&&!queries);
    for(bool error:{false,true}) {
        fail=error;calls=queries=0;
        assert(invoke(129u,65u,33u,272u,1105u,561u)==(error?hipErrorUnknown:hipSuccess));
        assert(calls==1u&&queries==1u&&grid.x==3u&&block.x==256u&&seen_stream==stream&&seen_kernel==replay_kernel<false>);
    }
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "main.cpp"
            source.write_text(code)
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O1",
                            "-fsanitize=address,undefined", str(source), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=10)

    def test_ordered_carries_and_all_bf16_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            exe = str(Path(directory) / "check")
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2",
                            "-fsanitize=undefined,float-cast-overflow", "-fno-sanitize-recover=all",
                            str(ROOT / "tests/native/float_row_bounds_selftest.cpp"), "-o", exe],
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
