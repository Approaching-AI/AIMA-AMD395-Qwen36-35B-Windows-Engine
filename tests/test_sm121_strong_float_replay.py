"""Check the strong row proof, cancellation counterexamples and ordered values."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class StrongFloatReplayTests(unittest.TestCase):
    def test_actual_arithmetic_against_independent_canonical_groups(self):
        code = r'''
#include "strong_float_replay_cases.h"
#include <cassert>
#include <cstdio>
namespace cases=qrt_strong_replay_cases;
namespace strong=qrt_sm121_strong_float;
int main() {
 unsigned encodings=0,groups=0,certified=0,fallback=0;
 for(unsigned bits=0;bits<65536;bits++) {
  const unsigned exp=(bits>>7)&255;
  const bool expected=(bits&0x7fff)==0 || (exp>=84 && exp<=174);
  assert(strong::eligible(uint16_t(bits))==expected);encodings+=expected;
 }
 assert(encodings==23298);
 for(unsigned exp:{80u,83u}) {
  cases::Pair p[16];for(unsigned i=0;i<16;i++)p[i]=cases::cancellation(exp,i);
  auto carry=cases::reference({0,-133,false},p);
  assert(carry.significand && carry.exponent==(exp==80?-108:-102));
  for(auto& pair:p)pair={0,0};auto next=cases::reference(carry,p);
  assert(next.significand==carry.significand && next.exponent==carry.exponent);
 }
 for(unsigned width:{16u,272u,2048u,4096u}) for(unsigned row=0;row<8192;row++) {
  const bool eligible=cases::row_certificate(row,width);
  eligible ? ++certified : ++fallback;
  cases::original::Value actual{0,-133,false},expected=actual;
  for(unsigned g=0;g<width/16;g++) {
   cases::Pair pairs[16];strong::Product products[16];
   for(unsigned i=0;i<16;i++) {
    pairs[i]=cases::input(row,g,i);
    if(eligible)products[i]=strong::prepare(pairs[i].x,pairs[i].y);
   }
   expected=cases::reference(expected,pairs);
   if(eligible) {
    int maximum=std::max(-133,int(actual.exponent));
    for(auto p:products)maximum=std::max(maximum,p.exponent);
    assert((maximum==-133 && !actual.significand) || (maximum>=-100 && maximum<=108));
    actual=strong::group(actual,products);
    assert(actual.significand==expected.significand && actual.exponent==expected.exponent && actual.negative==expected.negative);
    ++groups;
   }else actual=expected;
  }
 }
 std::printf("{\"kind\":\"strong_replay_host\",\"certified_rows\":%u,\"fallback_rows\":%u,\"certified_ordered_groups\":%u,\"raw_value_mismatches\":0,\"all_bf16_encodings_checked\":65536,\"cancellation_counterexamples\":2,\"inference_acceptance\":false}\n",certified,fallback,groups);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-strong-replay-') as tmp:
            exe=str(Path(tmp)/'check')
            subprocess.run([os.environ.get('CXX','c++'),'-O2','-std=c++17','-Wall','-Wextra','-Werror',
                '-fsanitize=undefined,float-cast-overflow','-fno-sanitize-recover=all','-I',str(ROOT/'tests/native'),
                '-x','c++','-','-o',exe],input=code,text=True,check=True,timeout=30)
            subprocess.run([exe],check=True,timeout=30)
if __name__=='__main__':
    unittest.main()
