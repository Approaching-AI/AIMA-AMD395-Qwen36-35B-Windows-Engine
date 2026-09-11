"""Exercise packed halfword arithmetic against the original BF16 product."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PairedProductTests(unittest.TestCase):
    def test_all_encodings_and_independent_halfwords(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include "sm121_paired_products.h"
#include "sm121_group16_modulo.h"
uint32_t reference(uint16_t a,uint16_t b){
 return qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133));
}
int main(){
 unsigned seed=0x3952026u;auto random=[&](){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;};
 const uint16_t controls[]={0,0x8000,1,0x7f,0x80,0x807f,0x3f80,0xbf80,0x3fff,0xbfff,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0xffff};
 for(unsigned a=0;a<65536u;++a)for(uint16_t b:controls){
  const uint32_t left=a|(random()&0xffff0000u),right=uint32_t(b)|(random()&0xffff0000u);
  const auto out=qrt_sm121_paired_products::multiply(left,right);
  assert(out.low==reference(uint16_t(left),uint16_t(right)));
  assert(out.high==reference(uint16_t(left>>16u),uint16_t(right>>16u)));
 }
 std::puts("paired_products=2097152 all_bf16_encodings=65536 controls=16 exact=1");
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-paired-products-') as tmp:
            exe = str(Path(tmp) / 'paired')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=20)


if __name__ == '__main__':
    unittest.main()
