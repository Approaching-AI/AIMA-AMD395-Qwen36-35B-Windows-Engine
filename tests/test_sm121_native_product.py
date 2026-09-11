"""Compare the native-product encoding/alignment with the original integer K16."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class Sm121NativeProductTests(unittest.TestCase):
    def test_all_operand_encodings_extreme_groups_and_random_carries(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include "sm121_native_product.h"
using namespace qrt_q1_moe_hawkeye;
uint32_t seed=0x3958192u;
uint32_t random_word() { seed^=seed<<13u; seed^=seed>>17u; seed^=seed<<5u; return seed; }
int main() {
    uint32_t normal_groups=0, fallback_groups=0;
    for (uint32_t group=0; group<200000u; ++group) {
        uint32_t old[16], current[16], tags=0;
        for (uint32_t i=0;i<16;++i) {
            uint16_t a,b;
            if (group<65536u) {
                a=static_cast<uint16_t>(group);
                constexpr uint16_t controls[16]={0,0x8000,1,0x7f,0x80,0x3f00,0x3f80,0x3fff,
                    0xbf80,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc0,0x4000,0x8001};
                b=controls[i];
            } else if (group<100000u) {
                a=static_cast<uint16_t>(random_word()); b=static_cast<uint16_t>(random_word());
            } else {
                a=static_cast<uint16_t>((random_word()&0x807fu)|((110u+random_word()%32u)<<7u));
                b=static_cast<uint16_t>((random_word()&0x807fu)|((110u+random_word()%32u)<<7u));
            }
            old[i]=qrt_sm121_group16::pack_product(multiply_bf16(a,b,-133));
            current[i]=qrt_sm121_native_product::pack(a,b);
            assert(qrt_sm121_native_product::original_pack(current[i])==old[i]);
            tags|=current[i];
        }
        uint32_t carry_bits=random_word();
        if (group>=100000u) carry_bits=(carry_bits&0x807fffffu)|((100u+random_word()%60u)<<23u);
        if (((carry_bits>>23u)&0xffu)==255u) carry_bits&=0x807fffffu;
        float carry; std::memcpy(&carry,&carry_bits,4);
        auto accumulator=value_from_float(carry,-133);
        auto expected=qrt_sm121_group16::sum_packed(accumulator,old);
        auto actual=qrt_sm121_native_product::sum(accumulator,current);
        assert(actual.max_exponent==expected.max_exponent);
        assert(actual.value.negative==expected.value.negative);
        assert(actual.value.magnitude==expected.value.magnitude);
        if(tags&2u)++fallback_groups;else ++normal_groups;
    }
    assert(normal_groups>=100000u&&fallback_groups>0);
    std::printf("groups=200000 products=3200000 normal_groups=%u fallback_groups=%u exact=1\n",normal_groups,fallback_groups);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-native-product-') as tmp:
            exe = str(Path(tmp) / 'native-product-test')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
