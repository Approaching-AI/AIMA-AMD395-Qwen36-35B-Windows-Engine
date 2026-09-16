#include "../../native/providers/moe_accumulator/sm121_whole_dot_bound.h"
#include "../../native/providers/moe_accumulator/sm121_cooperative_norm_metadata.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
namespace bound=qrt_sm121_whole_dot_bound;
namespace meta=qrt_sm121_cooperative_norm_metadata;
namespace original=qrt_q1_moe_hawkeye;
uint32_t seed=0x3958192u;
uint32_t random_word(){seed^=seed<<13u;seed^=seed>>17u;seed^=seed<<5u;return seed;}
float f32(uint16_t word){return bound::scalar::value(uint32_t(word)<<16u);}
float add(float a,float b){volatile float result=a+b;return result;}
size_t metadata_checks=0u;
meta::Summary metadata(const uint16_t* words,unsigned count){
 assert(count<=2048u);float sums[32]{};unsigned maxima[32]{},invalid[32]{};
 double squares=0.0,maximum=0.0;bool valid=true;
 for(unsigned k=0u;k<count;++k){const double value=f32(words[k]&0x7fffu);squares+=value*value;maximum=std::max(maximum,value);valid=valid&&meta::bound::base::eligible(words[k]);}
 for(unsigned lane=0u;lane<32u;++lane)for(unsigned k=lane;k<count;k+=32u)meta::add_word(sums[lane],maxima[lane],invalid[lane],words[k]);
 for(unsigned offset=16u;offset;offset/=2u)for(unsigned lane=0u;lane<offset;++lane){
  sums[lane]=add(sums[lane],sums[lane+offset]);maxima[lane]=std::max(maxima[lane],maxima[lane+offset]);invalid[lane]|=invalid[lane+offset];}
 const auto value=meta::finish(sums[0],maxima[0],invalid[0]);++metadata_checks;
 if(valid){assert(double(value.maximum)==maximum&&double(value.norm)*double(value.norm)>=squares);assert(squares?double(value.norm)<=std::sqrt(squares)*1.00003:value.norm==0.0f);}
 else assert(std::isinf(value.norm)&&std::isinf(value.maximum));
 return value;
}
void run(unsigned width){
 constexpr unsigned cases=1024u;const unsigned groups=width/16u;
 size_t interiors=0u,suffixes=0u,selected=0u,stops[4]{},original_groups=0u,executed_groups=0u;
 for(unsigned sample=0u;sample<cases;++sample){
  std::vector<uint16_t> w(width),x(width);std::vector<float> canonical(groups),native[3];
  const unsigned ae=80u+random_word()%87u,be=80u+random_word()%87u;
  original::Value carry{0u,-133,false};
  for(unsigned group=0u;group<groups;++group){
   double dot=0.0,absolute=0.0;original::Value terms[17];terms[0]=carry;
   for(unsigned i=0u;i<16u;++i){
    uint16_t a=uint16_t((random_word()&0x807fu)|((ae+random_word()%9u)<<7u)),b=uint16_t((random_word()&0x807fu)|((be+random_word()%9u)<<7u));
    if(sample%7u==0u){a=uint16_t(0x3f81u|((i&1u)<<15u));b=0x3f85u;}
    if(sample%11u==0u&&i%3u==0u)a=uint16_t((sample&1u)<<15u);
    if(sample%13u==0u)a=b=0x3fffu;
    if(sample%17u==0u){a=uint16_t(0x3f80u|((i&1u)<<15u));b=a;}
    if(sample%19u==0u){a=uint16_t((80u<<7u)|(random_word()&0x807fu));b=uint16_t((174u<<7u)|(random_word()&0x807fu));}
    if(sample%23u==0u)a=0u;
    w[group*16u+i]=a;x[group*16u+i]=b;terms[i+1u]=original::multiply_bf16(a,b,-133);
    const double product=double(f32(a))*double(f32(b));dot+=product;absolute+=std::abs(product);
   }
   carry=original::group_sum<26,-133>(terms,17u);canonical[group]=original::value_to_float(carry);
   for(unsigned mode=0u;mode<3u;++mode){const double perturbation=(mode==0u?0.0:mode==1u?0.75:-0.75)*0x1p-19*absolute;
    const float product=float(dot+perturbation);assert(std::abs(double(product)-dot)<=0x1p-19*absolute);native[mode].push_back(product);}
  }
  float segment_norms[4]{},norm=0.0f,maximum_product=0.0f;
  for(unsigned part=0u;part<4u;++part){const unsigned first=(part*groups/4u)*16u,end=((part+1u)*groups/4u)*16u;
   const auto a=metadata(w.data()+first,end-first),b=metadata(x.data()+first,end-first);
   segment_norms[part]=meta::bound::absolute_bound(a,b);norm=bound::scalar::upper(norm+segment_norms[part]);maximum_product=std::max(maximum_product,meta::bound::product_bound(a,b));
   double absolute=0.0,largest=0.0;for(unsigned k=first;k<end;++k){const double value=std::abs(double(f32(w[k]))*double(f32(x[k])));absolute+=value;largest=std::max(largest,value);}
   assert(double(segment_norms[part])>=absolute&&double(meta::bound::product_bound(a,b))>=largest);
  }
  for(unsigned mode=0u;mode<3u;++mode){
   float sum=0.0f,maximum=0.0f;std::vector<float> prefixes;
   for(float value:native[mode]){sum=add(sum,value);maximum=std::max(maximum,std::abs(sum));prefixes.push_back(sum);}
   const auto envelope=bound::build(groups,sum,maximum,norm,maximum_product);
   assert(bound::scalar::finite(envelope.value.error)&&std::abs(double(sum)-double(canonical.back()))<=double(envelope.value.error));
   for(float value:canonical){assert(std::abs(value)<=envelope.barrier);++interiors;}
   if(bound::base::certified(envelope.value))assert(bound::scalar::bf16(sum)==bound::scalar::bf16(canonical.back()));
   unsigned first_stop=3u;
   for(unsigned stage=0u;stage<3u;++stage){const unsigned end=(stage+1u)*groups/4u;float remaining_norm=0.0f;
    for(unsigned part=stage+1u;part<4u;++part)remaining_norm=bound::scalar::upper(remaining_norm+segment_norms[part]);
    const auto estimate=bound::suffix(sum,end?prefixes[end-1u]:0.0f,end?canonical[end-1u]:0.0f,
     bound::matrix_error(remaining_norm,groups-end),envelope.step_error,groups-end);
    assert(std::abs(double(estimate.center)-double(canonical.back()))<=double(estimate.error));++suffixes;
    if(bound::base::certified(estimate)){assert(bound::scalar::bf16(estimate.center)==bound::scalar::bf16(canonical.back()));if(first_stop==3u)first_stop=stage;}
   }
   if(!bound::base::certified(envelope.value)){++selected;++stops[first_stop];original_groups+=groups;executed_groups+=(first_stop+1u)*groups/4u;}
  }
 }
 std::printf("{\"kind\":\"whole_dot_bound_host\",\"width\":%u,\"cases\":%u,\"native_error_modes\":3,\"interior_carry_checks\":%zu,\"suffix_checks\":%zu,\"selected\":%zu,\"stop_counts\":[%zu,%zu,%zu,%zu],\"original_groups\":%zu,\"executed_groups\":%zu,\"undercoverage\":0,\"false_certificates\":0,\"native_matrix_executed\":false,\"hardware_error_bound_proven\":false}\n",width,cases,interiors,suffixes,selected,stops[0],stops[1],stops[2],stops[3],original_groups,executed_groups);
}
int main(){
 for(unsigned count=0u;count<=2048u;++count)for(unsigned mode=0u;mode<5u;++mode){
  std::vector<uint16_t> input(count);
  for(unsigned i=0u;i<count;++i)input[i]=mode==0u?0u:mode==1u?uint16_t(80u<<7u):mode==2u?uint16_t((174u<<7u)|127u):uint16_t(((i%32u?80u:174u)<<7u)|(random_word()&0x807fu));
  if(mode==4u&&count)input[count/2u]=1u;metadata(input.data(),count);
 }
 const auto before=metadata_checks;
 for(unsigned width:{16u,80u,272u,512u,2048u,4096u,8192u})run(width);
 const float inf=std::numeric_limits<float>::infinity(),nan=std::numeric_limits<float>::quiet_NaN();
 for(float invalid:{-1.0f,inf,-inf,nan}){
  assert(!bound::base::certified(bound::build(1u,0.0f,invalid,0.0f,0.0f).value));
  assert(!bound::base::certified(bound::build(1u,0.0f,0.0f,invalid,0.0f).value));
  assert(!bound::base::certified(bound::build(1u,0.0f,0.0f,0.0f,invalid).value));
  assert(!bound::base::certified(bound::suffix(0.0f,0.0f,0.0f,invalid,0.0f,1u)));
  assert(!bound::base::certified(bound::suffix(0.0f,0.0f,0.0f,0.0f,invalid,1u)));
 }
 for(unsigned invalid:{0u,513u})assert(!bound::base::certified(bound::build(invalid,0.0f,0.0f,0.0f,0.0f).value));
 assert(!bound::base::certified(bound::build(1u,2.0f,1.0f,1.0f,1.0f).value));
 assert(!bound::base::certified(bound::build(1u,nan,1.0f,1.0f,1.0f).value));
 assert(!bound::base::certified(bound::suffix(0.0f,0.0f,0.0f,0.0f,0.0f,0u)));
 std::printf("{\"kind\":\"whole_dot_metadata_host\",\"lengths_through\":2048,\"edge_metadata_checks\":%zu,\"dot_metadata_checks\":%zu,\"norm_undercoverage\":0,\"maximum_mismatches\":0}\n",before,metadata_checks-before);
}
