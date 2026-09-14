
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include "sm121_integer_core.h"
#include "sm121_scaled_half_products.h"
#include "sm121_canonical_normalize.h"
using namespace qrt_sm121_integer_core;
using qrt_q1_moe_hawkeye::Value;
unsigned seed=0x3958192u;
unsigned random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}

namespace pair = qrt_sm121_scaled_half_products;
int main(){
 unsigned ordered=0u,paths[4]{},wide_paths[4]{};Value previous{0u,-133,false};
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
  unsigned path=99u;const auto actual=pair::accumulate(values[0],pa,pb,&path);assert(path<4u);++paths[path];
  if(actual.significand!=expected_raw.significand || actual.exponent!=expected_raw.exponent || actual.negative!=expected_raw.negative) {
   std::fprintf(stderr,"group=%u actual=%u,%d,%u expected=%u,%d,%u\n",group,actual.significand,actual.exponent,actual.negative,expected_raw.significand,expected_raw.exponent,expected_raw.negative);return 1;
  }
  unsigned wide_path=99u;const auto wide=pair::accumulate<true>(values[0],pa,pb,&wide_path);
  assert(wide_path<4u);++wide_paths[wide_path];
  assert(wide.significand==expected_raw.significand && wide.exponent==expected_raw.exponent && wide.negative==expected_raw.negative);
  for(unsigned i=0u;i<16u;++i) {
   assert(pair::original(pa,i)==a.original[i] && pair::original(pb,i)==b.original[i]);
   if(pair::unit(pa)!=-32768 && pair::unit(pb)!=-32768) {
    const float product=i&1u?pair::product<true>(pa.pairs[i/2u],pb.pairs[i/2u]):pair::product<false>(pa.pairs[i/2u],pb.pairs[i/2u]);
    const double restored=std::ldexp(double(product),pair::unit(pa)+pair::unit(pb));
    const auto original=values[i+1u];const double expected=std::ldexp(original.negative?-double(original.significand):double(original.significand),int(original.exponent)-23);
    assert(restored==expected);
   }
  }
  assert(!std::memcmp(&pa,&saved_a,sizeof(pa))&&!std::memcmp(&pb,&saved_b,sizeof(pb)));
  assert(!std::memcmp(before_a,a.original,sizeof(before_a))&&!std::memcmp(before_b,b.original,sizeof(before_b)));
 }
 for(unsigned value=0u;value<65536u;++value)for(unsigned mode=0u;mode<4u;++mode) {
  uint16_t original[16]{};original[0]=uint16_t(value);
  if(mode==1u)original[15]=0x3f80u;
  if(mode==2u)original[15]=0x7f7fu;
  if(mode==3u)original[15]=0x0081u;
  const auto row=pair::prepare(original);
  for(unsigned i=0u;i<16u;++i)assert(pair::original(row,i)==original[i]);
 }
 assert(paths[0] && paths[1] && paths[2]>50000u && paths[3]);
 std::printf("{\"kind\":\"scaled_half_products_host\",\"all_bf16_encodings\":65536,\"roundtrip_contexts\":4,\"unmodified_raw_carry_states\":%u,\"path_fallback_zero_full_partial\":[%u,%u,%u,%u],\"power_of_two_product_reconstruction\":true,\"raw_mismatches\":0,\"immutable_inputs\":true}\n",ordered,paths[0],paths[1],paths[2],paths[3]);
 assert(wide_paths[0]<paths[0] && wide_paths[2]>paths[2]);
 std::printf("{\"kind\":\"scaled_half_integer_carry_host\",\"unmodified_raw_carry_states\":%u,\"path_fallback_zero_full_partial\":[%u,%u,%u,%u],\"raw_mismatches\":0,\"immutable_inputs\":true}\n",ordered,wide_paths[0],wide_paths[1],wide_paths[2],wide_paths[3]);
}
