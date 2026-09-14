
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <initializer_list>
#include "sm121_integer_interval.h"
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
 unsigned paths[3]{},new_certificates=0u,bounded=0u,ordered=0u;Value previous{0u,-133,false};
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
  const auto expected=qrt_sm121_group16::finish_accumulator(expected_raw);
  previous=expected_raw;++ordered;
  const int64_t mathematical=int64_t(parts[0])*65536+(int64_t(parts[1])+parts[2])*256+parts[3];
  qrt_sm121_group16::AlignedSum old_sum;
  const bool old_accepted=a.unit>=0&&b.unit>=0&&!((a.exceptions|b.exceptions)&a.nonzero&b.nonzero)&&
    qrt_sm121_integer_parts::sum_exact_integer_product(values[0],mathematical,a.unit,a.maximum,b.unit,b.maximum,&old_sum,a.trailing,b.trailing);
  qrt_sm121_integer_interval::Bounds interval;
  const bool has_bounds=qrt_sm121_integer_interval::bounds(values[0],a,b,mathematical,&interval);
  if(has_bounds){
   ++bounded;
   // Independent original aligned signed sum using the wide host primitive's
   // Value products. No encoded-pair arithmetic enters this reference.
   int maximum=-133;for(const auto& v:values)if(v.exponent>maximum)maximum=v.exponent;
   int64_t aligned=0;
   for(const auto& v:values){const unsigned shift=unsigned(maximum-v.exponent);const uint64_t n=shift>=64u?0u:(uint64_t(v.significand)<<2u)>>shift;aligned+=v.negative?-int64_t(n):int64_t(n);}
   if(maximum!=interval.maximum||aligned<interval.lower||aligned>interval.upper){
    std::fprintf(stderr,"bound group=%u expected=%lld maximum=%d low=%lld high=%lld maximum=%d\n",group,(long long)aligned,maximum,(long long)interval.lower,(long long)interval.upper,interval.maximum);return 1;
   }
  }
  Value actual{0xdeadbeefu,123,true};const Value sentinel=actual;
  const bool accepted=qrt_sm121_integer_interval::accumulate(values[0],a,b,mathematical,&actual);
  ++paths[!has_bounds?2u:accepted?0u:1u];
  if(accepted){
   if(!qrt_sm121_integer_interval::same(qrt_sm121_group16::finish_accumulator(actual),expected)){
    std::fprintf(stderr,"group=%u actual=%u,%d,%u expected=%u,%d,%u\n",group,actual.significand,actual.exponent,actual.negative,expected.significand,expected.exponent,expected.negative);return 1;
   }
   new_certificates+=!old_accepted;
   // Check all possible integer sums, including any interior sign crossing.
   if(interval.upper-interval.lower>32) return 2;
   for(int64_t sum=interval.lower;sum<=interval.upper;++sum)
    if(!qrt_sm121_integer_interval::same(qrt_sm121_integer_interval::normalize(sum,interval.maximum),actual))return 3;
  }else if(!qrt_sm121_integer_interval::same(actual,sentinel))return 4;
  assert(!std::memcmp(before_a,a.original,sizeof(before_a))&&!std::memcmp(before_b,b.original,sizeof(before_b)));
 }
 assert(paths[0]>10000u && paths[1]>1000u && paths[2]>10000u && new_certificates>1000u);
 std::printf("{\"groups\":500000,\"certified\":%u,\"uncertain\":%u,\"unsupported\":%u,\"new_certificates\":%u,\"independent_bound_checks\":%u,\"carry_states_checked\":%u,\"raw_mismatches\":0,\"immutable_inputs\":true}\n",paths[0],paths[1],paths[2],new_certificates,bounded,ordered);
}
