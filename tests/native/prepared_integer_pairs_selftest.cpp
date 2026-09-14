
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include "sm121_integer_core.h"
#include "sm121_prepared_integer_pairs.h"
#include "sm121_canonical_normalize.h"
using namespace qrt_sm121_integer_core;
using qrt_q1_moe_hawkeye::Value;
unsigned seed=0x3958192u;
unsigned random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}

namespace pair = qrt_sm121_prepared_integer_pairs;
void verify_pair(uint32_t left,uint32_t right,uint16_t a,uint16_t b,uint16_t c,uint16_t d) {
 const auto actual=pair::multiply(left,right);
 const auto lo=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133),hi=qrt_q1_moe_hawkeye::multiply_bf16(c,d,-133);
 assert((actual.significands&65535u)==lo.significand>>9u);
 assert((actual.significands>>16u)==hi.significand>>9u);
 assert((actual.exponents&65535u)==unsigned(lo.exponent+254));
 assert((actual.exponents>>16u)==unsigned(hi.exponent+254));
}
int main(){
 unsigned ordered=0u;Value previous{0u,-133,false};
 for(unsigned group=0;group<500000u;++group){
  Row a{},b{};Value values[17];
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
  prepare(a);prepare(b);
  values[0]={(random_word()&0x7fffffu)|0x800000u,
    int16_t(a.maximum+b.maximum-254+int(group%67u)-26),bool(group&1u)};
  if(group%3u==0u||group%31u==0u||group%37u==0u||group%43u==0u)values[0]={0u,-133,bool(group&1u)};
  if(group%29u==0u)values[0]={0xffffffu,0,bool(group&1u)};
  if(group%41u==0u)values[0]={0u,-133,false};
  if(group%4u==1u)values[0]=previous;
  const auto expected_raw=qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
  previous=expected_raw;++ordered;
  const auto pa=pair::prepare(a.original),pb=pair::prepare(b.original),saved_a=pa,saved_b=pb;
  const auto actual=pair::accumulate(values[0],pa,pb);
  if(actual.significand!=expected_raw.significand || actual.exponent!=expected_raw.exponent || actual.negative!=expected_raw.negative) {
   std::fprintf(stderr,"group=%u actual=%u,%d,%u expected=%u,%d,%u\n",group,actual.significand,actual.exponent,actual.negative,expected_raw.significand,expected_raw.exponent,expected_raw.negative);return 1;
  }
  for(unsigned i=0u;i<8u;++i)verify_pair(pa.pairs[i],pb.pairs[i],a.original[2u*i],b.original[2u*i],a.original[2u*i+1u],b.original[2u*i+1u]);
  assert(!std::memcmp(&pa,&saved_a,sizeof(pa))&&!std::memcmp(&pb,&saved_b,sizeof(pb)));
  assert(!std::memcmp(before_a,a.original,sizeof(before_a))&&!std::memcmp(before_b,b.original,sizeof(before_b)));
 }
 const uint16_t controls[]={0,0x8000,1,0x7f,0x80,0x807f,0x3f80,0xbf80,0x3fff,0xbfff,0x7f7f,0xff7f,0x7f80,0xff80,0x7fc1,0xffff};
 for(unsigned a=0u;a<65536u;++a) {
  uint16_t original[16]{};original[0]=uint16_t(a);const auto row=pair::prepare(original);
  const unsigned encoded=row.pairs[0]&65535u,significand=encoded&255u;
  const unsigned exponent=significand<128u?0u:encoded>>8u;
  const unsigned restored=(row.negative&1u?0x8000u:0u)|(exponent<<7u)|(significand&127u);
  assert(restored==a);
  for(uint16_t b:controls) {
   const uint16_t c=uint16_t(random_word()),d=uint16_t(random_word());
   verify_pair(uint32_t(pair::encode(uint16_t(a)))|(uint32_t(pair::encode(c))<<16u),
    uint32_t(pair::encode(b))|(uint32_t(pair::encode(d))<<16u),uint16_t(a),b,c,d);
  }
 }
 for(unsigned i=0u;i<1048576u;++i) {
  const uint32_t a=random_word(),b=random_word();
  const uint32_t low=(a&65535u)>(b&65535u)?a&65535u:b&65535u;
  const uint32_t high=(a>>16u)>(b>>16u)?a>>16u:b>>16u;
  assert(pair::maximum_pair(a,b)==(low|(high<<16u)));
 }
 std::printf("{\"kind\":\"prepared_integer_pairs_host\",\"all_bf16_roundtrips\":65536,\"edge_and_random_product_pairs\":1048576,\"random_packed_max_pairs\":1048576,\"unmodified_raw_carry_states\":%u,\"original_group_products\":8000000,\"raw_mismatches\":0,\"immutable_inputs\":true}\n",ordered);
}
