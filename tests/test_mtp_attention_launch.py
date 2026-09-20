"""Exercise the real MTP attention launch bounds and partial-submit ordering."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

from test_attention_workspace import function

ROOT = Path(__file__).resolve().parents[1]


class MtpAttentionLaunchTests(unittest.TestCase):
    def test_original_cache_view_and_failed_stage_stop(self):
        header = (ROOT / 'native/providers/gdn/sm121_mtp_attention.h').read_text()
        actual = '\n'.join(function(header, name) for name in (
            'inline size_t attention_workspace_bytes(', 'inline hipError_t launch_attention('))
        source = r'''
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <cassert>
#include <initializer_list>
enum hipError_t { hipSuccess, hipErrorInvalidValue, hipErrorUnknown };
using hipStream_t = void*;
struct dim3 { unsigned x,y,z; explicit dim3(unsigned a,unsigned b=1,unsigned c=1):x(a),y(b),z(c){} };
void attention_scores() {}
void publish_attention_context() {}
namespace qrt_blackwell_attention {
template<bool A,bool B=false,bool C=false,bool D=false,bool E=false,bool F=false,bool G=false,bool H=false>
void blackwell_exact_attention_kernel() {}
}
unsigned launches=0,fail_stage=0,expected_first=0,expected_rows=0,expected_stride=0;
uint16_t query[8192]{},cache[1024]{},output[8192]{};
unsigned char table[1]{};
float scores[1]{},context[1]{};
template<class... Args> void record(void(*kernel)(),dim3 grid,dim3 block,size_t shared,hipStream_t stream,Args... args) {
    assert(block.x==256u && !shared && !stream);auto values=std::make_tuple(args...);
    if constexpr(sizeof...(Args)==6) {
        assert(kernel==attention_scores && launches==0u);
        assert(grid.x==expected_rows*expected_stride && grid.y==1u);
        assert(std::get<0>(values)==query && std::get<1>(values)==cache && std::get<2>(values)==scores);
        assert(std::get<3>(values)==expected_first && std::get<4>(values)==expected_rows && std::get<5>(values)==expected_stride);
    } else if constexpr(sizeof...(Args)==15) {
        assert((kernel==qrt_blackwell_attention::blackwell_exact_attention_kernel<true,true,false,false,false,false,false,true>));
        assert(launches==1u && grid.x==16u && grid.y==expected_rows);
        assert(!std::get<0>(values) && !std::get<1>(values) && std::get<2>(values)==cache+512u);
        assert(std::get<3>(values)==context && std::get<4>(values)==expected_first && !std::get<5>(values));
        assert(std::get<6>(values)==table && !std::get<7>(values) && !std::get<8>(values));
        assert(std::get<9>(values) && std::get<10>(values)==table && std::get<11>(values)==scores);
        assert(std::get<12>(values)==expected_stride && !std::get<13>(values) && !std::get<14>(values));
    } else {
        static_assert(sizeof...(Args)==3);
        assert(kernel==publish_attention_context && launches==2u && grid.x==expected_rows*16u);
        assert(std::get<0>(values)==context && std::get<1>(values)==output && std::get<2>(values)==expected_rows*4096u);
    }
    ++launches;
}
#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define hipLaunchKernelGGL(kernel,...) record(kernel,__VA_ARGS__)
hipError_t hipGetLastError(){return fail_stage && launches==fail_stage?hipErrorUnknown:hipSuccess;}
''' + actual + r'''
int main() {
    assert(attention_workspace_bytes(2,262144)==33587200u);
    for(unsigned rows:{0u,3u,~0u})assert(!attention_workspace_bytes(rows,32));
    for(unsigned stride:{0u,1u,33u,262145u,~0u})assert(!attention_workspace_bytes(1,stride));
    const auto launch=[&](unsigned tokens,unsigned first,unsigned rows,unsigned stride) {
        return launch_attention(query,cache,tokens,first,rows,table,table,scores,stride,context,output);
    };
    for(unsigned rows:{0u,3u,~0u})assert(launch(262144,0,rows,262144)==hipErrorInvalidValue);
    for(unsigned tokens:{0u,262145u,~0u})assert(launch(tokens,0,1,262144)==hipErrorInvalidValue);
    for(unsigned first:{262144u,~0u})assert(launch(262144,first,1,262144)==hipErrorInvalidValue);
    assert(launch(262144,262143,2,262144)==hipErrorInvalidValue);
    assert(launch(8192,8191,1,8160)==hipErrorInvalidValue);
    assert(launch_attention(nullptr,cache,8192,0,1,table,table,scores,8192,context,output)==hipErrorInvalidValue);
    assert(launch_attention(query,cache,8192,0,1,table,table,scores,8192,scores,output)==hipErrorInvalidValue);
    assert(launch_attention(query,cache,8192,0,1,table,table,scores,8192,context,query)==hipErrorInvalidValue);
    assert(launch_attention(query,cache,8192,0,1,table,table,scores,8192,context,cache)==hipErrorInvalidValue);
    assert(!launches);
    for(unsigned first:{0u,8191u,262142u})for(unsigned rows:{1u,2u}) {
        expected_first=first;expected_rows=rows;expected_stride=(first+rows+31u)&~31u;
        for(unsigned fail=0;fail<=3u;++fail) {
            launches=0;fail_stage=fail;
            assert(launch(262144,first,rows,expected_stride)==(fail?hipErrorUnknown:hipSuccess));
            assert(launches==(fail?fail:3u));
        }
    }
}
'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'mtp_attention.cpp'
            exe = Path(temporary) / 'mtp_attention'
            path.write_text(source)
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra',
                            '-Werror', '-fsanitize=address,undefined', '-fno-sanitize-recover=all',
                            str(path), '-o', str(exe)], check=True, timeout=60)
            subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
