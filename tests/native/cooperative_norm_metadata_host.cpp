#include "../../native/providers/moe_accumulator/sm121_cooperative_norm_metadata.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
namespace meta=qrt_sm121_cooperative_norm_metadata;
namespace scalar=meta::scalar;
size_t checks=0u,unsupported=0u;
void verify(const std::vector<uint16_t>& input){
 double squares=0.0,maximum=0.0;bool valid=true;
 for(uint16_t word:input){
  valid=valid&&meta::bound::base::eligible(word);
  const double value=scalar::value(uint32_t(word&0x7fffu)<<16u);
  squares+=value*value;maximum=std::max(maximum,value);
 }
 for(const auto value:{meta::simulate<float>(input.data(),unsigned(input.size())),meta::simulate<double>(input.data(),unsigned(input.size()))}){
  if(!valid){assert(std::isinf(value.norm)&&std::isinf(value.maximum));++unsupported;}
  else{
   assert(std::isfinite(value.norm)&&value.norm>=0.0f&&double(value.maximum)==maximum);
   assert(double(value.norm)*double(value.norm)>=squares);
   assert(squares?double(value.norm)<=std::sqrt(squares)*1.00003:value.norm==0.0f);
  }
  ++checks;
 }
}
int main(){
 for(unsigned word=0u;word<65536u;++word)for(unsigned context=0u;context<3u;++context){
  std::vector<uint16_t> input(80u,context==0u?0u:context==1u?uint16_t(80u<<7u):uint16_t((174u<<7u)|127u));
  input[word%input.size()]=uint16_t(word);verify(input);
 }
 uint32_t state=0x3958192u;
 auto random=[&](){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;};
 for(unsigned count=1u;count<=1024u;++count)for(unsigned mode=0u;mode<8u;++mode){
  std::vector<uint16_t> input(count);
  for(unsigned i=0u;i<count;++i){
   unsigned exponent=80u+random()%95u;
   if(mode==1u)exponent=80u;if(mode==2u)exponent=174u;
   if(mode==3u)exponent=i%2u?80u:174u;
   if(mode==4u)exponent=(i%32u)?80u:174u;
   if(mode==5u)exponent=(i/32u)?80u:174u;
   input[i]=uint16_t((exponent<<7u)|(random()&0x807fu));
   if(mode==6u)input[i]=uint16_t((i%2u)<<15u);
   if(mode==7u && i%31u==0u)input[i]=1u;
  }
  verify(input);
 }
 for(const auto value:{meta::simulate<float>(nullptr,32u),meta::simulate<double>(nullptr,32u),meta::simulate<float>(nullptr,0u),meta::simulate<double>(nullptr,1025u)})assert(std::isinf(value.norm)&&std::isinf(value.maximum));
 std::printf("{\"kind\":\"cooperative_norm_metadata_host\",\"distinct_bf16_encodings\":65536,\"encoding_contexts\":3,\"generated_lengths\":1024,\"generated_modes\":8,\"methods\":2,\"metadata_checks\":%zu,\"unsupported_checks\":%zu,\"norm_undercoverage\":0,\"maximum_mismatches\":0,\"native_execution\":false}\n",checks,unsupported);
}
