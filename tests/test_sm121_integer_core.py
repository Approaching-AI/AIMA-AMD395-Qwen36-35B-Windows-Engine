"""Check sparse integer decomposition against the independent wide K16 sum."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class IntegerCoreTests(unittest.TestCase):
    def test_original_group_boundaries_and_sparse_exceptions(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include "sm121_integer_core.h"
using namespace qrt_sm121_integer_core;
using qrt_q1_moe_hawkeye::Value;
unsigned seed=0x3958192u;
unsigned random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int byte_high(uint16_t v){return v&0x8000u ? int(v>>8u)-256 : int(v>>8u);}
void partials(const Row& a,const Row& b,int32_t (&parts)[4]){
 for(unsigned i=0;i<16;++i){
  const auto x=encode(a.original[i],a.unit),y=encode(b.original[i],b.unit);
  const int ah=byte_high(x),bh=byte_high(y),al=x&255u,bl=y&255u;
  parts[0]+=ah*bh;parts[1]+=ah*bl;parts[2]+=al*bh;parts[3]+=al*bl;
 }
}
Value canonical(qrt_sm121_group16::AlignedSum sum){
 const Value scaled{sum.value.magnitude,int16_t(sum.max_exponent-2),sum.value.negative};
 return qrt_sm121_group16::finish_accumulator(qrt_q1_moe_hawkeye::group_sum<26,-133>(&scaled,1u));
}
int main(){
 unsigned accepted=0,declined=0,exceptions=0,replayed=0,zero_replay=0,large_shift=0,
   over_i32=0,cancellations=0,underflow=0,old_ineligible=0;
 unsigned histogram[17]{};
 for(unsigned group=0;group<500000u;++group){
  Row a{},b{};Value values[17];int32_t parts[4]{};
  const unsigned ae=1u+random_word()%220u,be=1u+random_word()%220u;
  const unsigned spread=1u+group%34u;
  for(unsigned i=0;i<16;++i){
   a.original[i]=uint16_t((random_word()&0x807fu)|((ae+random_word()%spread)<<7u));
   b.original[i]=uint16_t((random_word()&0x807fu)|((be+random_word()%spread)<<7u));
   if(group%11u==0u&&i%3u==0u)a.original[i]=0;
   if(group%13u==0u&&i%3u==1u)b.original[i]=0x8000;
   if(group%17u==0u){ // Conservative row maxima never occur in a pair.
    a.original[i]=uint16_t((random_word()&0x807fu)|((ae+i%9u)<<7u));
    b.original[i]=uint16_t((random_word()&0x807fu)|((be+8u-i%9u)<<7u));
   }
   if(group%19u==0u&&i==2u)a.original[i]=1u;
   if(group%23u==0u&&i==3u)b.original[i]=0x7fc1u;
   if(group%29u==0u){a.original[i]=uint16_t(0x3fffu|((group&1u)<<15u));b.original[i]=0x3fffu;}
   if(group%31u==0u){a.original[i]=uint16_t((0x3f81u+(i/2u%4u)*128u)|((i&1u)<<15u));b.original[i]=0x3f85u;}
   if(group%37u==0u){ // Disjoint nonzero rows, extreme exponent bound.
    a.original[i]=i%2u ? 0x8000u : 0x7f7fu;b.original[i]=i%2u ? 0x7f7fu : 0u;
   }
   if(group%41u==0u){a.original[i]=uint16_t(0x0081u+(i&1u));b.original[i]=0x0081u;}
   if(group%43u==0u){a.original[i]=i==2u?0x7f7fu:uint16_t(0x0081u+(i&1u));b.original[i]=i==2u?0u:0x0081u;}
   values[i+1]=qrt_q1_moe_hawkeye::multiply_bf16(a.original[i],b.original[i],-133);
  }
  uint16_t before_a[18],before_b[18];std::memcpy(before_a,a.original,sizeof(before_a));std::memcpy(before_b,b.original,sizeof(before_b));
  prepare(a);prepare(b);partials(a,b,parts);
  int unused;
  old_ineligible+=qrt_sm121_integer_parts::row_unit_range(a.original,&unused)<0||
    qrt_sm121_integer_parts::row_unit_range(b.original,&unused)<0;
  values[0]={(random_word()&0x7fffffu)|0x800000u,
    int16_t(a.maximum+b.maximum-254+int(group%67u)-26),bool(group&1u)};
  if(group%3u==0u||group%31u==0u||group%37u==0u||group%43u==0u)values[0]={0u,-133,bool(group&1u)};
  if(group%29u==0u)values[0]={0xffffffu,0,bool(group&1u)};
  if(group%41u==0u)values[0]={0u,-133,false};
  const auto expected=qrt_sm121_group16::finish_accumulator(qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u));
  qrt_sm121_group16::AlignedSum actual{{123u,true},777};unsigned pairs=99;
  if(sum(values[0],a,b,parts,&actual,&pairs)){
   ++accepted;const auto result=canonical(actual);
   if(result.significand!=expected.significand||result.exponent!=expected.exponent||result.negative!=expected.negative){
    std::fprintf(stderr,"group=%u units=%d,%d exceptions=%x,%x pairs=%u shift=%d actual=%u,%d,%u expected=%u,%d,%u\n",group,a.unit,b.unit,a.exceptions,b.exceptions,pairs,actual.max_exponent-(a.unit+b.unit-254)-11,result.significand,result.exponent,result.negative,expected.significand,expected.exponent,expected.negative);
    return 1;
   }
   assert(pairs<=16u);++histogram[pairs];replayed+=pairs;zero_replay+=pairs==0;
   exceptions+=bool(a.exceptions|b.exceptions);
   large_shift+=actual.max_exponent-(a.unit+b.unit-254)-11>=30;
   over_i32+=actual.value.magnitude>0x7fffffffu;cancellations+=result.significand==0u;
   underflow+=result.exponent==-126;
   for(const Row* row:{&a,&b})for(unsigned i=0;i<16u;++i){
    const auto v=row->original[i];if(!(v&0x7fffu))continue;
    const double original=std::ldexp(double((v&0x8000u)?-int(128u|(v&127u)):int(128u|(v&127u))),int((v>>7u)&255u)-134);
    const double core=std::ldexp(double(signed_core(v,row->unit)),row->unit-134);
    assert(bool(row->exceptions&(1u<<i))==(original!=core));
   }
  }else{
   ++declined;assert(a.unit<0||b.unit<0);
   assert(actual.value.magnitude==123u&&actual.value.negative&&actual.max_exponent==777&&pairs==99u);
  }
  assert(!std::memcmp(before_a,a.original,sizeof(before_a))&&!std::memcmp(before_b,b.original,sizeof(before_b)));
 }
 assert(accepted>300000u&&declined>10000u&&exceptions>100000u&&old_ineligible>100000u);
 assert(zero_replay>50000u&&large_shift>1000u&&over_i32>1000u&&cancellations>10000u&&underflow>1000u);
 // Independent scalar divisibility checks cover every bit of the packed mask.
 for(unsigned trial=0;trial<100000u;++trial){
  Row a{},b{};const unsigned shift=1u+trial%29u;uint32_t expected=0;
  for(unsigned i=0;i<16u;++i){const unsigned x=random_word()%32u,y=random_word()%32u;
   a.trailing[i/4]|=x<<((i%4)*8);b.trailing[i/4]|=y<<((i%4)*8);if(x+y<shift)expected|=1u<<i;}
  assert(remainder_mask(a,b,shift)==expected);
 }
 std::printf("core_groups=500000 accepted=%u declined=%u exceptions=%u old_ineligible=%u replayed_pairs=%u zero_replay=%u large_shift=%u over_i32=%u zero_sum=%u underflow=%u canonical_exact=1 histogram=",accepted,declined,exceptions,old_ineligible,replayed,zero_replay,large_shift,over_i32,cancellations,underflow);
 for(unsigned i=0;i<=16u;++i)std::printf("%s%u",i?",":"",histogram[i]);std::puts("");
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-integer-core-') as tmp:
            exe = str(Path(tmp) / 'core')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2',
                            '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined,address',
                            '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True,
                           check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=30)


if __name__ == '__main__':
    unittest.main()
