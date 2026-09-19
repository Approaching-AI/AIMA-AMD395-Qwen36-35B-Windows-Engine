"""Check every supported extent against an independent byte inventory."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LongProbabilityStoragePolicyTests(unittest.TestCase):
    def test_modes_extents_and_unchanged_short_selection(self):
        code = r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <limits>
#include "native/providers/ck_fmha/long_probability_storage_policy.h"
int main(){
    namespace policy=qrt_long_probability_storage;
    unsigned choices=0;unsigned long long layouts=0;
    const char* options[]={nullptr,"","0","1","2","3","01","-1","+1","1 "," 2","true","12"};
    for(const char* option:options)for(bool active:{false,true}){
        unsigned mode=19u;
        const bool valid=!option||!*option||!std::strcmp(option,"0")||!std::strcmp(option,"1")||!std::strcmp(option,"2");
        assert(policy::select(option,active,mode)==valid);
        const unsigned expected=valid&&active&&option&&*option?unsigned(option[0]-'0'):0u;
        assert(mode==expected);++choices;
    }
    for(unsigned stride=1u;stride<=qrt_long_attention_layout::maximum_tokens;++stride)
        for(unsigned queries:{1u,17u,32u,64u,127u,128u})for(unsigned mode:{0u,1u,2u}){
            const auto x=policy::layout(mode,queries,stride);
            if(queries>stride){assert(!x.elements);continue;}
            const size_t rows=size_t(queries)*16u,scores=rows*stride*4u;
            const size_t probabilities=mode?0u:rows*stride*2u;
            const size_t scales=rows*((stride+31u)/32u+1u)*4u,errors=rows*256u*4u;
            assert(x.probability*4u==(mode?0u:scores));
            assert(x.scales*4u==scores+probabilities);
            assert(x.errors*4u==scores+probabilities+scales);
            assert(x.indices*4u==scores+probabilities+scales+errors);
            assert(x.count*4u==scores+probabilities+scales+errors*2u);
            assert(x.elements*4u==scores+probabilities+scales+errors*2u+4u);++layouts;
        }
    for(unsigned mode:{0u,1u,2u,3u,std::numeric_limits<unsigned>::max()}){
        assert(!policy::layout(mode,0u,32u).elements);
        assert(!policy::layout(mode,129u,8192u).elements);
        assert(!policy::layout(mode,32u,31u).elements);
        assert(!policy::layout(mode,1u,264737u).elements);
        if(mode>2u)assert(!policy::layout(mode,128u,264736u).elements);
    }
    std::printf("{\"storage_policy_choices\":%u,\"storage_policy_layouts\":%llu,"
        "\"independent_byte_inventory\":true,\"non_long_selection_unchanged\":true,\"gpu_execution\":false}\n",choices,layouts);
}
'''
        with tempfile.TemporaryDirectory(prefix="qrt-long-probability-policy-") as tmp:
            cpp = Path(tmp) / "policy.cpp"
            cpp.write_text(code)
            exe = Path(tmp) / "policy"
            subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                            "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-I", str(ROOT),
                            str(cpp), "-o", str(exe)], check=True, timeout=40)
            subprocess.run([str(exe)], check=True, timeout=30)


if __name__ == "__main__":
    unittest.main()
