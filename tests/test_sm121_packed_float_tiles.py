"""Validate packed products and certified carries against wide canonical groups."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class PackedFloatTileTests(unittest.TestCase):
    def test_products_and_every_ordered_carry(self):
        code = r'''
#include "strong_float_replay_cases.h"
#include "../../native/providers/moe_accumulator/sm121_packed_float_tiles.h"
#include <cassert>
#include <cstdio>
namespace cases=qrt_strong_replay_cases;
namespace fast=qrt_sm121_packed_tiles;
bool same(fast::Value a,fast::Value b) {return a.significand==b.significand && a.exponent==b.exponent && a.negative==b.negative;}
int main() {
 unsigned long long pairs=0,steps=0,tiles=0,accepted=0,rejected=0;
 // Every BF16 encoding appears on the left, all certified exponent values
 // and a mantissa/sign boundary set on the right; no product may lose bits.
 for(unsigned a=0;a<65536;a++) if(fast::strong::eligible(uint16_t(a)))
  for(unsigned e=84;e<=174;e++) for(unsigned m:{0u,1u,63u,127u,32768u,32895u}) {
   const uint16_t b=uint16_t((e<<7)|m);
   const auto original=fast::strong::prepare(uint16_t(a),b),decoded=fast::unpack(fast::prepare(uint16_t(a),b));
   assert(original.exponent==decoded.exponent);
   assert(std::memcmp(&original.value,&decoded.value,4)==0);++pairs;
  }
 for(unsigned width:{16u,272u,2048u,4096u}) for(unsigned row=0;row<4096;row++) {
  if(!cases::row_certificate(row,width))continue;
  fast::Value carry{0,-133,false};
  for(unsigned base=0;base<width/16;base+=8) {
   const unsigned count=std::min(8u,width/16-base);
   uint32_t packed[8][16]{};fast::Value expected[8],prefixes[8];
   auto reference=carry;
   for(unsigned g=0;g<count;g++) {
    cases::Pair p[16];
    for(unsigned i=0;i<16;i++){p[i]=cases::input(row,base+g,i);packed[g][i]=fast::prepare(p[i].x,p[i].y);}
    reference=cases::reference(reference,p);expected[g]=reference;
   }
   auto output=carry;
   const bool ok=fast::try_tile(carry,packed,count,&output,prefixes);++tiles;
   if(ok) {++accepted;for(unsigned g=0;g<count;g++){assert(same(prefixes[g],expected[g]));++steps;}assert(same(output,reference));}
   else {
    ++rejected;assert(same(output,carry));
    for(unsigned g=0;g<count;g++) {
     fast::strong::Product products[16];for(unsigned i=0;i<16;i++)products[i]=fast::unpack(packed[g][i]);
     output=fast::strong::group(output,products);assert(same(output,expected[g]));++steps;
    }
   }
   carry=output;
  }
 }
 assert(accepted && rejected);
 std::printf("{\"kind\":\"packed_float_tiles_host\",\"packed_product_pairs\":%llu,\"ordered_carries\":%llu,\"tiles\":%llu,\"accepted\":%llu,\"rejected\":%llu,\"raw_mismatches\":0,\"inference_acceptance\":false}\n",pairs,steps,tiles,accepted,rejected);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-packed-tiles-') as tmp:
            exe=str(Path(tmp)/'check')
            subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-Wall','-Wextra','-Werror',
                '-fsanitize=undefined,float-cast-overflow','-fno-sanitize-recover=all','-I',str(ROOT/'tests/native'),
                '-x','c++','-','-o',exe],input=code,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=45)

if __name__=='__main__':
    unittest.main()
