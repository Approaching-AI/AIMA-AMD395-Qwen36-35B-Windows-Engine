"""Validate four matrix partials and per-product truncation compensation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Sm121MantissaPartsTests(unittest.TestCase):
    def test_exact_partial_decomposition_and_rejected_exponent_ranges(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include "sm121_mantissa_parts.h"
using namespace qrt_q1_moe_hawkeye;
uint32_t seed=0x8191395u;
uint32_t random_word(){seed^=seed<<13u;seed^=seed>>17u;seed^=seed<<5u;return seed;}
float bf16(uint16_t value){return qrt_sm121_native_product::from_bits(uint32_t(value)<<16u);}
int main(){
    unsigned accepted=0,rejected=0,compensated=0;
    for(unsigned group=0;group<200000u;++group){
        uint32_t pairs[16],old[16];float partials[4]{};
        unsigned first=80u+random_word()%65u, second=80u+random_word()%65u;
        unsigned spread=group%3u==0?20u:6u;
        if(group%67u==0){first=1u+group%8u;second=240u;}
        for(unsigned i=0;i<16;++i){
            uint16_t a=uint16_t((random_word()&0x807fu)|((first+random_word()%spread)<<7u));
            uint16_t b=uint16_t((random_word()&0x807fu)|((second+random_word()%spread)<<7u));
            if(group%41u==0)a=0;
            if(group%47u==0)b=0x8000;
            if(group%53u==0&&i==1)a=1;
            if(group%59u==0&&i==2)b=0x7f80;
            if(group%61u==0&&i==3)b=0x7fc0;
            pairs[i]=uint32_t(a)|(uint32_t(b)<<16u);
            old[i]=qrt_sm121_group16::pack_product(multiply_bf16(a,b,-133));
            uint16_t av[2]={qrt_sm121_mantissa_parts::high(a),qrt_sm121_mantissa_parts::low(a)};
            uint16_t bv[2]={qrt_sm121_mantissa_parts::high(b),qrt_sm121_mantissa_parts::low(b)};
            for(unsigned h=0;h<2;++h)for(unsigned k=0;k<2;++k){
                volatile float product=bf16(av[h])*bf16(bv[k]);
                volatile float rounded=partials[h*2+k]+product;partials[h*2+k]=rounded;
            }
        }
        // Include carries below, within, and far above the product group.
        const int exponent=int(first+second)-254+int(random_word()%28u)-2;
        Value carry{(random_word()&0x7fffffu)|0x800000u,int16_t(exponent),bool(random_word()&1u)};
        if(group%7u==0)carry={0u,-133,false};
        auto expected=qrt_sm121_group16::sum_packed(carry,old);
        qrt_sm121_group16::AlignedSum actual{{123u,true},777};
        bool ok=qrt_sm121_mantissa_parts::sum(carry,pairs,partials,&actual);
        if(ok){
            ++accepted;
            assert(actual.max_exponent==expected.max_exponent);
            assert(actual.value.magnitude==expected.value.magnitude);
            assert(actual.value.negative==expected.value.negative);
            if(carry.exponent>int(first+second)-254+11)++compensated;
        }else{
            ++rejected;assert(actual.value.magnitude==123u&&actual.value.negative&&actual.max_exponent==777);
        }
    }
    assert(accepted>50000u&&rejected>1000u&&compensated>1000u);
    std::printf("groups=200000 accepted=%u rejected=%u large_carry=%u exact=1\n",accepted,rejected,compensated);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-mantissa-parts-') as tmp:
            exe = str(Path(tmp) / 'parts-test')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined,float-cast-overflow', '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
