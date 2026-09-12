"""Check lossless BF16 staging against the original independent product code."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class PreparedBf16Tests(unittest.TestCase):
    def test_all_encodings_and_signed_products(self):
        source=r'''
#include <cassert>
#include <cstdio>
#include "sm121_prepared_bf16.h"
#include "sm121_group16_modulo.h"
namespace p=qrt_sm121_prepared_bf16;
uint32_t seed=395u;
uint32_t rnd() { seed^=seed<<13u;seed^=seed>>17u;seed^=seed<<5u;return seed; }
void verify(uint16_t a,uint16_t b) {
    assert(p::eligible(a)&&p::eligible(b));
    const auto expected=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133));
    assert(p::multiply(p::encode(a),p::encode(b))==expected);
}
void verify_wide(uint16_t a,uint16_t b) {
    const auto expected=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133));
    assert(p::multiply_wide(p::encode_wide(a),p::encode_wide(b))==expected);
}
int main() {
    const uint16_t controls[]={0u,0x8000u,0x2000u,0xa000u,0x207fu,0xa07fu,0x3f80u,0xbf80u,0x5f80u,0xdf80u,0x5fffu,0xdfffu};
    const uint16_t wide_controls[]={0u,0x8000u,1u,0x8001u,127u,0x807fu,128u,0x8080u,
        0x3f80u,0xbf80u,0x7f7fu,0xff7fu,0x7f80u,0xff80u,0x7fc1u,0xffc1u};
    unsigned eligible=0u,products=0u,wide_products=0u;
    for(unsigned i=0u;i<65536u;++i) {
        for(uint16_t b:wide_controls) { verify_wide(uint16_t(i),b);verify_wide(b,uint16_t(i));wide_products+=2u; }
        const unsigned e=(i>>7u)&255u;
        assert(p::eligible(uint16_t(i))==((i&0x7fffu)==0u||(e>=64u&&e<=191u)));
        if(!p::eligible(uint16_t(i))) continue;
        ++eligible;
        const uint16_t encoded=p::encode(uint16_t(i));
        const uint16_t restored=(encoded&255u)==0u ? encoded&0x8000u :
            uint16_t((encoded&0x8000u)|((((encoded>>8u)&127u)+64u)<<7u)|(encoded&127u));
        assert(restored==i);
        for(uint16_t b:controls) { verify(uint16_t(i),b);++products; }
    }
    for(unsigned i=0u;i<1000000u;++i) {
        const uint16_t a=uint16_t((rnd()&0x807fu)|((64u+rnd()%128u)<<7u));
        const uint16_t b=uint16_t((rnd()&0x807fu)|((64u+rnd()%128u)<<7u));
        verify(a,b);++products;
        const uint16_t wa=uint16_t(rnd()),wb=uint16_t(rnd());
        verify_wide(wa,wb);++wide_products;
    }
    assert(eligible==32770u);
    std::printf("bf16_encodings=65536 eligible=%u packed_products=%u mismatches=0\n",eligible,products);
    std::printf("wide_bf16_encodings=65536 signed_zero_subnormal_special_products=%u mismatches=0\n",wide_products);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-prepared-bf16-') as tmp:
            exe=str(Path(tmp)/'prepared')
            subprocess.run([os.environ.get('CXX','c++'),'-std=c++17','-O2','-Wall','-Wextra','-Werror',
                            '-fsanitize=undefined','-I',str(ROOT/'native/providers/moe_accumulator'),
                            '-x','c++','-','-o',exe],input=source,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=15)


if __name__=='__main__': unittest.main()
