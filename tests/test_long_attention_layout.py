"""Long attention allocation extents and opt-in scope at context boundaries."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LongAttentionLayoutTests(unittest.TestCase):
    def test_extents_and_inactive_short_decode_scope(self):
        source = r'''
#include <cassert>
#include <climits>
#include <initializer_list>
#include "native/providers/ck_fmha/long_attention_layout.h"
int main() {
    using namespace qrt_long_attention_layout;
    unsigned cases=0;
    for(unsigned stride=1;stride<=maximum_tokens;++stride) {
        for(unsigned queries:{1u,17u,32u,64u,127u,128u}) {
            const auto x=layout(queries,stride);
            if(queries>stride){assert(!x.elements);continue;}
            const size_t rows=size_t(queries)*16;
            // Independent byte extents, including the unused causal tails.
            const size_t score_bytes=rows*stride*sizeof(float);
            const size_t probability_bytes=rows*stride*sizeof(unsigned short);
            const size_t scale_bytes=rows*((stride+31)/32+1)*sizeof(float);
            const size_t error_bytes=size_t(queries)*4096*sizeof(float);
            assert(x.probability*4==score_bytes);
            assert(x.scales*4==score_bytes+probability_bytes);
            assert(x.errors*4==score_bytes+probability_bytes+scale_bytes);
            assert(x.indices*4==x.errors*4+error_bytes);
            assert(x.count*4==x.indices*4+error_bytes);
            assert(x.elements*4==x.count*4+sizeof(unsigned));
            assert(x.elements<UINT_MAX && x.scales%2==0);
            ++cases;
        }
    }
    assert(cases>1500000);
    assert(!layout(0,32).elements&&!layout(129,129).elements);
    assert(!layout(1,maximum_tokens+1).elements&&!layout(UINT_MAX,UINT_MAX).elements);
    for(const char* option:std::initializer_list<const char*>{nullptr,"","0","1","2","10","-1"}) {
        const bool valid=!option||!*option||!std::strcmp(option,"0")||!std::strcmp(option,"1");
        for(unsigned start:{0u,8191u,8192u,16384u,maximum_tokens-2u}) {
            for(unsigned count:{1u,2u,1024u,8192u}) {
                for(unsigned flags=0;flags<128u;++flags) {
                    bool selected=true;
                    const bool good=select(option,start,count,flags&1,flags&2,flags&4,
                        flags&8?1u:3u,flags&16,flags&32,flags&64,selected);
                    if(!valid){assert(!good&&!selected);continue;}
                    if(!option||std::strcmp(option,"1")){assert(good&&!selected);continue;}
                    if(count>maximum_tokens-start){assert(!good&&!selected);continue;}
                    if(count==1||start+count<=8192){assert(good&&!selected);continue;}
                    assert(good==(flags==15u));assert(selected==good);
                }
            }
        }
    }
    bool enabled=true;
    for(const char* option:std::initializer_list<const char*>{nullptr,"","0","1","2","01","1 ","true"}) {
        for(bool pipeline:{false,true}) {
            enabled=true;
            const bool valid=!option||!*option||!std::strcmp(option,"0")||!std::strcmp(option,"1");
            assert(select_final_bound(option,pipeline,enabled)==valid);
            assert(enabled==(valid&&pipeline&&option&&!std::strcmp(option,"1")));
        }
    }
    assert(!select("1",UINT_MAX,2,true,true,true,1,false,false,false,enabled)&&!enabled);
    assert(!select("1",8192,8193,true,true,true,1,false,false,false,enabled)&&!enabled);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-long-attention-layout-') as tmp:
            cpp = Path(tmp) / 'layout.cpp'
            cpp.write_text(source)
            exe = Path(tmp) / 'layout'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O1',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=address,undefined',
                            '-fno-omit-frame-pointer', '-I', str(ROOT), str(cpp), '-o', str(exe)],
                           check=True, timeout=30)
            subprocess.run([str(exe)], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
