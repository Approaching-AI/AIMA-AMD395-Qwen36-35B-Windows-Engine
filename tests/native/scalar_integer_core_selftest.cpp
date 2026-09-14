
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include "sm121_scalar_integer_core.h"
#include "sm121_canonical_normalize.h"
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
 unsigned paths[4][4]{},ordered=0u;Value previous{0u,-133,false};
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
  if(group%4u==1u)values[0]=previous;
  const auto expected_raw=qrt_q1_moe_hawkeye::group_sum<26,-133>(values,17u);
  previous=expected_raw;++ordered;
  const int64_t mathematical=int64_t(parts[0])*65536+(int64_t(parts[1])+parts[2])*256+parts[3];

  int64_t independent_product=0;
  for(unsigned i=0u;i<16u;++i)
   independent_product+=int64_t(signed_core(a.original[i],a.unit))*signed_core(b.original[i],b.unit);
  assert(mathematical==independent_product);
  assert(qrt_sm121_scalar_integer_core::product(a,b)==independent_product);
  qrt_sm121_scalar_integer_core::Row scalar_a{},scalar_b{};
  std::memcpy(scalar_a.core.original,before_a,sizeof(before_a));
  std::memcpy(scalar_b.core.original,before_b,sizeof(before_b));
  qrt_sm121_scalar_integer_core::prepare(scalar_a);qrt_sm121_scalar_integer_core::prepare(scalar_b);
  const auto saved_a=scalar_a,saved_b=scalar_b;
  for(unsigned policy=0u;policy<2u;++policy) {
   const auto actual=policy?qrt_sm121_scalar_integer_core::accumulate<true>(values[0],scalar_a,scalar_b):
    qrt_sm121_scalar_integer_core::accumulate<false>(values[0],scalar_a,scalar_b);
   assert(actual.significand==expected_raw.significand && actual.exponent==expected_raw.exponent && actual.negative==expected_raw.negative);
  }
  assert(!std::memcmp(&scalar_a,&saved_a,sizeof(saved_a))&&!std::memcmp(&scalar_b,&saved_b,sizeof(saved_b)));
  const Row before_core_a=a,before_core_b=b;
  const auto pa=qrt_sm121_core_remainder::prepare(a),pb=qrt_sm121_core_remainder::prepare(b);
  const auto before_pa=pa,before_pb=pb;
  for(unsigned i=0u;i<16u;++i) {
   const int x=signed_core(a.original[i],a.unit),y=signed_core(b.original[i],b.unit);
   assert(((pa.magnitudes[i/2u]>>(i%2u*16u))&65535u)==unsigned(x<0?-x:x));
   assert(((pb.magnitudes[i/2u]>>(i%2u*16u))&65535u)==unsigned(y<0?-y:y));
   assert(((pa.negative>>i)&1u)==unsigned((a.original[i]&0x8000u)!=0u));
   assert(((pb.negative>>i)&1u)==unsigned((b.original[i]&0x8000u)!=0u));
  }
  for(unsigned variant=0u;variant<4u;++variant) {
   qrt_sm121_group16::AlignedSum actual{{0xdeadbeefu,true},123};const auto sentinel=actual;
   unsigned path=99u;bool accepted=false;
   if(variant==0u)accepted=qrt_sm121_core_remainder::sum<false,false>(values[0],a,b,pa,pb,mathematical,&actual,&path);
   if(variant==1u)accepted=qrt_sm121_core_remainder::sum<true,false>(values[0],a,b,pa,pb,mathematical,&actual,&path);
   if(variant==2u)accepted=qrt_sm121_core_remainder::sum<false,true>(values[0],a,b,pa,pb,mathematical,&actual,&path);
   if(variant==3u)accepted=qrt_sm121_core_remainder::sum<true,true>(values[0],a,b,pa,pb,mathematical,&actual,&path);
   assert(path<4u);++paths[variant][path];
   if(accepted) {
    const auto v=qrt_sm121_canonical::normalize(actual.value.magnitude,actual.value.negative,actual.max_exponent);
    if(v.significand!=expected_raw.significand||v.exponent!=expected_raw.exponent||v.negative!=expected_raw.negative) {
     std::fprintf(stderr,"group=%u variant=%u path=%u actual=%u,%d,%u expected=%u,%d,%u\n",group,variant,path,v.significand,v.exponent,v.negative,expected_raw.significand,expected_raw.exponent,expected_raw.negative);return 1;
    }
   }else if(path||actual.value.magnitude!=sentinel.value.magnitude||actual.value.negative!=sentinel.value.negative||actual.max_exponent!=sentinel.max_exponent)return 2;
  }
  assert(!std::memcmp(&a,&before_core_a,sizeof(a))&&!std::memcmp(&b,&before_core_b,sizeof(b)));
  assert(!std::memcmp(&pa,&before_pa,sizeof(pa))&&!std::memcmp(&pb,&before_pb,sizeof(pb)));
  assert(!std::memcmp(before_a,a.original,sizeof(before_a))&&!std::memcmp(before_b,b.original,sizeof(before_b)));
 }
 // Exercise both halfwords independently through all low16 operand bits.
 for(unsigned i=0u;i<1048576u;++i) {
  const uint32_t a=random_word(),b=random_word();
  const uint32_t expected=uint32_t(uint16_t(uint64_t(a&65535u)*(b&65535u)))|
   (uint32_t(uint16_t(uint64_t(a>>16u)*(b>>16u)))<<16u);
  assert(qrt_sm121_core_remainder::multiply_low16(a,b)==expected);
 }
 for(unsigned v=0u;v<4u;++v) {
  assert(paths[v][0]+paths[v][1]+paths[v][2]+paths[v][3]==500000u&&paths[v][2]>100000u);
  std::printf("{\"kind\":\"scalar_integer_core_host\",\"variant\":%u,\"groups\":500000,\"unmodified_raw_carry_states\":%u,\"path_unsupported_fast_packed_wide\":[%u,%u,%u,%u],\"packed_halfword_pairs\":1048576,\"raw_mismatches\":0,\"immutable_inputs\":true}\n",v,ordered,paths[v][0],paths[v][1],paths[v][2],paths[v][3]);
 }
 std::printf("{\"kind\":\"scalar_integer_product_and_carry_host\",\"independent_signed16_products\":500000,\"raw_carry_states_per_policy\":500000,\"policies\":2,\"raw_mismatches\":0,\"immutable_inputs\":true}\n");
}
