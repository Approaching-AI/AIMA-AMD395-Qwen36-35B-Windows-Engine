"""Prove signed-byte matrix reconstruction and individual K16 truncation."""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]

class IntegerPartsTests(unittest.TestCase):
    def test_range_bound_preserves_canonical_sum_and_falls_back(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include "sm121_integer_parts.h"
using namespace qrt_sm121_integer_parts;
using qrt_q1_moe_hawkeye::Value;
constexpr bool unit_branch=QRT_TEST_INTEGER_UNITS;
unsigned seed=0x3952026u;
unsigned random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int high_byte(uint16_t v){int h=v>>8u;return h>=128?h-256:h;}
Value canonical(qrt_sm121_group16::AlignedSum sum){
 const Value scaled{sum.value.magnitude,int16_t(sum.max_exponent-2),sum.value.negative};
 return qrt_sm121_group16::finish_accumulator(qrt_q1_moe_hawkeye::group_sum<26,-133>(&scaled,1u));
}
int main(){
 unsigned accepted=0,rejected=0,rebound=0,with_carry=0,over_i32=0,zeros=0,underflow=0;
 for(unsigned group=0;group<400000u;++group){
  uint16_t a[18]{},b[18]{};Value values[17];int32_t partials[4]{};uint32_t packed[16],pairs[16],at[4]{},bt[4]{};
  const unsigned ae=1u+random_word()%244u,be=1u+random_word()%244u,spread=1u+group%11u;
  for(unsigned i=0;i<16u;++i){
   a[i]=uint16_t((random_word()&0x807fu)|((ae+random_word()%spread)<<7u));
   b[i]=uint16_t((random_word()&0x807fu)|((be+random_word()%spread)<<7u));
   if(group%13u==0u){ // Opposite extrema make the bound larger than every actual product.
    a[i]=uint16_t((random_word()&0x807fu)|((ae+i%4u)<<7u));
    b[i]=uint16_t((random_word()&0x807fu)|((be+3u-i%4u)<<7u));
   }
   if(group%17u==0u)a[i]=0u;
   if(group%19u==0u)b[i]=0x8000u;
   if(group%23u==0u&&i==2u)a[i]=1u;
   if(group%29u==0u&&i==3u)b[i]=0x7fc1u;
   if(group%31u==0u){ // Near the signed-32-bit overflow bound, both signs.
    a[i]=uint16_t(0x3fffu|((group&1u)<<15u));b[i]=0x3fffu;
   }
   if(group%37u==0u){ // Exact cancellation despite a conservative range bound.
    a[i]=uint16_t((0x3f80u+(i/2u%4u)*128u)|((i&1u)<<15u));
    b[i]=uint16_t(0x3f80u+(3u-i/2u%4u)*128u);
   }
   values[i+1]=qrt_q1_moe_hawkeye::multiply_bf16(a[i],b[i],-133);
   packed[i]=qrt_sm121_group16::pack_product(values[i+1]);
   pairs[i]=uint32_t(a[i])|(uint32_t(b[i])<<16u);
  }
  int amax,bmax;const int amin=unit_branch?row_unit_range(a,&amax):row_range(a,&amax);
  const int bmin=unit_branch?row_unit_range(b,&bmax):row_range(b,&bmax);
  if(!unit_branch)assert(amin==row_minimum(a)&&bmin==row_minimum(b));
  for(unsigned i=0;i<16u;++i){
   const uint16_t x=encode(a[i],amin),y=encode(b[i],bmin);
   const int ah=high_byte(x),bh=high_byte(y),al=x&255u,bl=y&255u;
   partials[0]+=ah*bh;partials[1]+=ah*bl;partials[2]+=al*bh;partials[3]+=al*bl;
   at[i/4u]|=trailing_bits(x)<<((i%4u)*8u);bt[i/4u]|=trailing_bits(y)<<((i%4u)*8u);
  }
  const int product_max=amax+bmax-254;
  values[0]={(random_word()&0x7fffffu)|0x800000u,
             int16_t(product_max+int(group%48u)-16),bool(group&1u)};
  if(group%3u==0u||group%37u==0u)values[0]={0u,-133,bool(group&1u)};
  if(group%5u==0u)values[0].significand&=0xffff00u; // Exact aligned carry beyond two bits.
  if(group%31u==0u)values[0]={0xffffffu,0,bool(group&1u)};
  const auto expected=qrt_sm121_group16::finish_accumulator(qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u));
  qrt_sm121_group16::AlignedSum actual{{123u,true},777};
  if(sum_exact_range(values[0],partials,amin,amax,bmin,bmax,&actual,unit_branch?at:nullptr,unit_branch?bt:nullptr)){
   ++accepted;const auto result=canonical(actual);
   const auto original=qrt_sm121_group16::sum_packed(values[0],packed);
   rebound+=unsigned(actual.max_exponent!=original.max_exponent);
   with_carry+=unsigned(values[0].significand!=0u);
   over_i32+=unsigned(actual.value.magnitude>0x7fffffffu);
   zeros+=unsigned(result.significand==0u);
   underflow+=unsigned(result.exponent==-126);
   assert(result.significand==expected.significand&&result.exponent==expected.exponent&&result.negative==expected.negative);
  }else{
   ++rejected;assert(actual.max_exponent==777&&actual.value.magnitude==123u&&actual.value.negative);
  }
  if(unit_branch&&sum(values[0],pairs,partials,amin,bmin,&actual)){
   const auto compensated=canonical(actual);
   assert(compensated.significand==expected.significand&&compensated.exponent==expected.exponent&&compensated.negative==expected.negative);
  }
 }
 // A carry whose low bit would be lost must decline without modifying output.
 const int32_t parts[4]{};qrt_sm121_group16::AlignedSum untouched{{123u,true},777};
 assert(!sum_exact_range({0x800001u,-3,false},parts,127,127,127,127,&untouched));
 assert(!sum_exact_range({0x800000u,-40,false},parts,127,127,127,127,&untouched));
 assert(untouched.max_exponent==777&&untouched.value.magnitude==123u);
 assert(accepted>50000u&&rejected>50000u&&rebound>1000u&&with_carry>10000u);
 assert(over_i32>1000u&&zeros>1000u&&underflow>10u);
 // Compare the packed four-byte proof to independent scalar column checks.
 for(unsigned trial=0;trial<100000u;++trial){
  uint32_t a[4]{},b[4]{};const unsigned shift=trial%34u;bool expected=shift<=31u;
  for(unsigned i=0;i<16u;++i){const unsigned x=random_word()%32u,y=random_word()%32u;
   a[i/4u]|=x<<((i%4u)*8u);b[i/4u]|=y<<((i%4u)*8u);expected=expected&&x+y>=shift;}
  assert(products_divisible(a,b,shift)==expected);
 }
 std::printf("integer_range_groups=400000 unit_branch=%u accepted=%u rejected=%u rebound=%u carry=%u over_i32=%u zeros=%u underflow=%u canonical_exact=1\n",
  unsigned(unit_branch),accepted,rejected,rebound,with_carry,over_i32,zeros,underflow);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-integer-range-') as tmp:
            exe = str(Path(tmp) / 'range')
            for unit_branch in (0, 1):
                subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                                '-fsanitize=undefined', '-I', str(ROOT / 'native/providers/moe_accumulator'),
                                f'-DQRT_TEST_INTEGER_UNITS={unit_branch}',
                                '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
                subprocess.run([exe], check=True, timeout=20)

    def test_byte_matrix_and_compensation_match_original_groups(self):
        source = r'''
#include <cassert>
#include <cstdio>
#include "sm121_integer_parts.h"
using namespace qrt_sm121_integer_parts;
unsigned seed=0x8191395u;
unsigned random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int high_byte(uint16_t v){int h=v>>8u;return h>=128?h-256:h;}
int main(){
 unsigned accepted=0,rejected=0,shifted=0;
 for(unsigned group=0;group<200000u;++group){
  uint16_t a[18]{},b[18]{};uint32_t pairs[16],original[16];int32_t partials[4]{};
  const unsigned ae=1u+random_word()%238u,be=1u+random_word()%238u;
  const unsigned spread=group%3u==0u?11u:8u;
  for(unsigned i=0;i<16u;++i){
   a[i]=uint16_t((random_word()&0x807fu)|((ae+random_word()%spread)<<7u));
   b[i]=uint16_t((random_word()&0x807fu)|((be+random_word()%spread)<<7u));
   if(group%41u==0u)a[i]=0;
   if(group%47u==0u)b[i]=0x8000;
   if(group%53u==0u&&i==1u)a[i]=1;
   if(group%59u==0u&&i==2u)b[i]=0x7f80;
   if(group%71u==0u){a[i]=uint16_t(0x3fffu|((group&1u)<<15u));b[i]=0x3fffu;}
   pairs[i]=uint32_t(a[i])|(uint32_t(b[i])<<16u);
   original[i]=qrt_sm121_group16::pack_product(qrt_q1_moe_hawkeye::multiply_bf16(a[i],b[i],-133));
  }
  int amin=row_minimum(a),bmin=row_minimum(b);
  for(unsigned i=0;i<16u;++i){
   const uint16_t x=encode(a[i],amin),y=encode(b[i],bmin);
   const int ah=high_byte(x),bh=high_byte(y),al=x&255u,bl=y&255u;
   partials[0]+=ah*bh;partials[1]+=ah*bl;partials[2]+=al*bh;partials[3]+=al*bl;
  }
  const int exponent=int(ae+be)-254+int(random_word()%70u)-4;
  qrt_q1_moe_hawkeye::Value carry{(random_word()&0x7fffffu)|0x800000u,int16_t(exponent),bool(random_word()&1u)};
  if(group%7u==0u)carry={0u,-133,false};
  if(group%71u==0u)carry={0xffffffu,0,bool(group&1u)};
  const auto expected=qrt_sm121_group16::sum_packed(carry,original);
  qrt_sm121_group16::AlignedSum actual{{123u,true},777};
  if(sum(carry,pairs,partials,amin,bmin,&actual)){
   ++accepted;if(exponent-amin-bmin+254>11)++shifted;
   assert(actual.max_exponent==expected.max_exponent);
   assert(actual.value.magnitude==expected.value.magnitude);
   assert(actual.value.negative==expected.value.negative);
  }else{++rejected;assert(actual.max_exponent==777&&actual.value.magnitude==123u&&actual.value.negative);}
 }
 assert(accepted>100000u&&rejected>10000u&&shifted>10000u);
 std::printf("integer_groups=200000 accepted=%u rejected=%u shifted=%u exact=1\n",accepted,rejected,shifted);
}
'''
        with tempfile.TemporaryDirectory(prefix='qrt-integer-parts-') as tmp:
            exe = str(Path(tmp) / 'parts')
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-fsanitize=undefined', '-I', str(ROOT / 'native/providers/moe_accumulator'),
                            '-x', 'c++', '-', '-o', exe], input=source, text=True, check=True, timeout=30)
            subprocess.run([exe], check=True, timeout=15)

if __name__ == '__main__':
    unittest.main()
