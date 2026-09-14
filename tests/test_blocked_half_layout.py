"""Check the component weight permutation and its actual preparation launcher."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class BlockedHalfLayoutTests(unittest.TestCase):
    def test_permutation_padding_bounds_and_submission(self):
        header = (ROOT / 'native/providers/moe_accumulator/sm121_blocked_half_projection.h').read_text()
        actual = '\n'.join(function(header, signature) for signature in (
            '__global__ void prepare_weights(', 'inline hipError_t prepare('))
        harness = r'''
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>
#include "''' + str(ROOT / 'native/providers/moe_accumulator/sm121_blocked_half_layout.h') + r'''"
#define __global__
using hipStream_t = void*;
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
struct dim3 { unsigned x,y,z; explicit dim3(unsigned a=0,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
dim3 blockIdx,blockDim,threadIdx;
unsigned launches=0;
dim3 last_grid,last_threads;
bool execute=true,fail=false;
struct Row { uint32_t words[9]; };
namespace staged { namespace half {
Row prepare(const uint16_t* values) {
    Row row{};
    std::memcpy(row.words,values,32u);
    row.words[8]=0x12340000u|values[0];
    return row;
}
}}
template<class Kernel> void launch(Kernel kernel,dim3 grid,dim3 threads,unsigned shared,hipStream_t stream,
    const uint16_t* input,Row* output,unsigned rows,unsigned width) {
    ++launches;last_grid=grid;last_threads=threads;
    if(shared || stream) std::abort();
    if(!execute || fail) return;
    blockDim=threads;
    for(blockIdx.x=0;blockIdx.x<grid.x;++blockIdx.x)
        for(threadIdx.x=0;threadIdx.x<threads.x;++threadIdx.x) kernel(input,output,rows,width);
}
#define hipLaunchKernelGGL(kernel,...) launch(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){return fail?hipErrorUnknown:hipSuccess;}
namespace qrt_sm121_blocked_half_projection {
namespace layout=qrt_sm121_blocked_half_layout;
''' + actual + r'''
}
int main() {
    namespace layout=qrt_sm121_blocked_half_layout;
    using qrt_sm121_blocked_half_projection::prepare;
    static_assert(layout::records(32,32)==64u);
    static_assert(layout::records(33,48)==256u);
    static_assert(layout::records(524288,8192)==268435456u);
    static_assert(layout::records(0,32)==0u && layout::records(32,0)==0u);
    static_assert(layout::records(524289,32)==0u && layout::records(32,8208)==0u);
    static_assert(layout::records(32,17)==0u);
    static_assert(layout::offset(32,32,32,0)==layout::invalid_offset);
    static_assert(layout::offset(32,32,0,2)==layout::invalid_offset);
    static_assert(layout::offset(524288,8192,524287,511)==268435455u);
    constexpr unsigned guard=65u;
    Row sentinel;std::memset(&sentinel,0xa5,sizeof(sentinel));
    for(unsigned rows:{1u,31u,32u,33u,65u,257u})
    for(unsigned width:{16u,32u,48u,272u,2048u,4096u,4112u,8192u}) {
        const size_t count=layout::records(rows,width);
        const unsigned groups=width/16u;
        std::vector<uint16_t> input(size_t(rows)*width+2u*guard,0xa5a5u);
        for(size_t i=0;i<size_t(rows)*width;++i)input[guard+i]=uint16_t(i*1973u+rows*31u+width);
        const auto original=input;
        std::vector<Row> output(count+2u*guard,sentinel);
        const unsigned previous=launches;
        if(prepare(input.data()+guard,output.data()+guard,count,rows,width,nullptr)!=hipSuccess ||
           launches!=previous+1u || last_grid.x!=(size_t(rows)*groups+255u)/256u ||
           last_threads.x!=256u) return 1;
        std::vector<bool> visited(count,false);
        for(unsigned row=0;row<rows;++row) {
            size_t block=layout::offset(rows,width,row,0);
            for(unsigned group=0;group<groups;++group) {
                const size_t index=layout::offset(rows,width,row,group);
                if(index>=count || visited[index] || index!=block+group%2u) return 2;
                visited[index]=true;
                const uint16_t* source=input.data()+guard+size_t(row)*width+group*16u;
                if(std::memcmp(output[guard+index].words,source,32u) ||
                   output[guard+index].words[8]!=(0x12340000u|source[0])) return 3;
                if(group%2u && group+1u<groups)block+=64u;
            }
        }
        if(input!=original) return 4;
        for(size_t i=0;i<output.size();++i) {
            const bool live=i>=guard && i<guard+count && visited[i-guard];
            if(!live && std::memcmp(&output[i],&sentinel,sizeof(Row))) return 5;
        }
        if(prepare(input.data()+guard,output.data()+guard,count-1u,rows,width,nullptr)!=hipErrorInvalidValue ||
           launches!=previous+1u) return 6;
    }
    uint16_t input=0;Row output{};unsigned previous=launches;
    for(unsigned rows:{0u,524289u,0xffffffffu})
        if(prepare(&input,&output,SIZE_MAX,rows,32,nullptr)!=hipErrorInvalidValue) return 7;
    for(unsigned width:{0u,1u,17u,8208u,0xffffffffu})
        if(prepare(&input,&output,SIZE_MAX,32,width,nullptr)!=hipErrorInvalidValue) return 8;
    if(prepare(nullptr,&output,SIZE_MAX,32,32,nullptr)!=hipErrorInvalidValue ||
       prepare(&input,nullptr,SIZE_MAX,32,32,nullptr)!=hipErrorInvalidValue || launches!=previous) return 9;
    execute=false;fail=true;
    if(prepare(&input,&output,64,32,32,nullptr)!=hipErrorUnknown || launches!=previous+1u) return 10;
    fail=false;
    if(prepare(&input,&output,268435456u,524288,8192,nullptr)!=hipSuccess ||
       last_grid.x!=1048576u || launches!=previous+2u) return 11;
    return 0;
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-blocked-half-') as tmp:
            exe = str(Path(tmp) / 'layout')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-fno-omit-frame-pointer', '-x', 'c++', '-', '-o', exe],
                           input=harness, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
