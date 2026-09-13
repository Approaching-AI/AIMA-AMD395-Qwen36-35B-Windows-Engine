
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include "sm121_matrix_float_fallback.h"
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
 unsigned paths[3]{};
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
  values[0]={(random_word()&0x7fffffu)|0x800000u,
    int16_t(a.maximum+b.maximum-254+int(group%67u)-26),bool(group&1u)};
  if(group%3u==0u||group%31u==0u||group%37u==0u||group%43u==0u)values[0]={0u,-133,bool(group&1u)};
  if(group%29u==0u)values[0]={0xffffffu,0,bool(group&1u)};
  if(group%41u==0u)values[0]={0u,-133,false};
  const auto expected=qrt_sm121_group16::finish_accumulator(qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u));
  qrt_sm121_group16::AlignedSum actual;
  const bool eligible=qrt_sm121_matrix_float_fallback::eligible(a.original)&&qrt_sm121_matrix_float_fallback::eligible(b.original);
  const auto path=qrt_sm121_matrix_float_fallback::sum(values[0],a,b,parts,eligible,&actual);
  ++paths[unsigned(path)];
  const auto result=canonical(actual);
  if(result.significand!=expected.significand || result.exponent!=expected.exponent || result.negative!=expected.negative){
   std::fprintf(stderr,"group=%u path=%u actual=%u,%d,%u expected=%u,%d,%u\n",group,unsigned(path),result.significand,result.exponent,result.negative,expected.significand,expected.exponent,expected.negative);
   return 1;
  }
  assert(!std::memcmp(before_a,a.original,sizeof(before_a))&&!std::memcmp(before_b,b.original,sizeof(before_b)));
 }
 assert(paths[0]>10000u && paths[1]>10000u && paths[2]>10000u);
 std::printf("{\"groups\":500000,\"fast_integer\":%u,\"scalar_float\":%u,\"original_integer\":%u,\"raw_mismatches\":0,\"immutable_inputs\":true}\n",paths[0],paths[1],paths[2]);
}
