#include "../../native/providers/moe_accumulator/sm121_prefix_replay_bound.h"
#include "../../native/providers/moe_accumulator/sm121_macro_norm_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
namespace bound=qrt_sm121_coarse_projection_bound;
namespace original=qrt_q1_moe_hawkeye;
namespace observed=qrt_sm121_macro_norm_bound;
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
float f32(uint16_t x){return bound::scalar::value(uint32_t(x)<<16u);}
float add(float a,float b){volatile float value=a+b;return value;}
template<unsigned Groups, unsigned Width> void run(){
 constexpr unsigned cases=4096u,width=Width;size_t checkpoints=0,certificates=0,selected=0,saved_groups=0;size_t stopped[4]{};
 for(unsigned sample=0u;sample<cases;++sample){
  std::vector<float> products[3],positive[3],canonical;std::vector<uint16_t> left,right;
  original::Value carry{0u,-133,false};
  const unsigned ae=80u+random_word()%87u,be=80u+random_word()%87u;
  for(unsigned group=0u;group<width/16u;++group){
   double sum=0.0,absolute=0.0;original::Value terms[17];terms[0]=carry;
   for(unsigned i=0u;i<16u;++i){
    uint16_t a=uint16_t((random_word()&0x807fu)|((ae+random_word()%9u)<<7u));
    uint16_t b=uint16_t((random_word()&0x807fu)|((be+random_word()%9u)<<7u));
    if(sample%7u==0u){a=uint16_t(0x3f81u|((i&1u)<<15u));b=0x3f85u;}
    if(sample%11u==0u && i%3u==0u)a=uint16_t((sample&1u)<<15u);
    if(sample%13u==0u){a=0x3fffu;b=0x3fffu;}
    if(sample%17u==0u){a=uint16_t(0x3f80u|((i&1u)<<15u));b=a;}
    if(sample%19u==0u){a=uint16_t((80u<<7u)|(random_word()&0x807fu));b=uint16_t((174u<<7u)|(random_word()&0x807fu));}
    if(sample%23u==0u)a=0u;
    assert(bound::eligible(a)&&bound::eligible(b));left.push_back(a);right.push_back(b);
    const double value=double(f32(a))*double(f32(b));sum+=value;absolute+=std::abs(value);
    terms[i+1u]=original::multiply_bf16(a,b,-133);
   }
   carry=original::group_sum<26,-133>(terms,17u);canonical.push_back(original::value_to_float(carry));
   for(unsigned mode=0u;mode<3u;++mode){
    const double perturbation=mode==0u?0.0:(mode==1u?0.75:-0.75)*0x1p-19*absolute;
    const float dot=float(sum+perturbation),abs=float(absolute-perturbation);
    assert(std::abs(double(dot)-sum)<=0x1p-19*absolute && std::abs(double(abs)-absolute)<=0x1p-19*absolute);
    products[mode].push_back(dot);positive[mode].push_back(abs);
   }
  }
  for(unsigned mode=0u;mode<3u;++mode){
   bound::State value;std::vector<bound::State> snapshots;
   for(unsigned base=0u;base<width/16u;base+=Groups){
    float dot=0.0f,prefix_max=std::abs(value.center),partial_max=0.0f;
    for(unsigned i=0u;i<Groups;++i){
     dot=add(dot,products[mode][base+i]);
     prefix_max=std::max(prefix_max,std::abs(add(value.center,dot)));
     partial_max=std::max(partial_max,std::abs(dot));
    }
    float barrier=-1.0f;
    const auto a=observed::prepare(left.data()+size_t(base)*16u,Groups*16u),b=observed::prepare(right.data()+size_t(base)*16u,Groups*16u);
    double absolute=0.0,maximum=0.0;
    for(unsigned i=0u;i<Groups*16u;++i){const double term=std::abs(double(f32(left[base*16u+i]))*double(f32(right[base*16u+i])));absolute+=term;maximum=std::max(maximum,term);}
    const float absolute_upper=observed::absolute_bound(a,b),product_upper=observed::product_bound(a,b);
    assert(double(absolute_upper)>=absolute&&double(product_upper)>=maximum);
    value=observed::advance<Groups>(value,dot,absolute_upper,prefix_max,partial_max,product_upper,&barrier);
    for(unsigned i=0u;i<Groups;++i)assert(std::abs(canonical[base+i])<=barrier);
    const double difference=std::abs(double(value.center)-double(canonical[base+Groups-1u]));
    assert(bound::scalar::finite(value.error)&&difference<=double(value.error));++checkpoints;
    if((base+Groups)%(width/64u)==0u)snapshots.push_back(value);
   }
   assert(snapshots.size()==4u);const bool initial=bound::certified(value);selected+=!initial;
   unsigned first=3u;
   for(unsigned stage=0u;stage<3u;++stage){
    const unsigned prefix=(stage+1u)*(width/64u)-1u;
    const auto refined=qrt_sm121_prefix_replay_bound::suffix(snapshots[stage],value,canonical[prefix]);
    assert(bound::scalar::finite(refined.error));
    assert(std::abs(double(refined.center)-double(canonical.back()))<=double(refined.error));
    float representative=123.0f;
    const bool ok=qrt_sm121_prefix_replay_bound::certificate(snapshots[stage],value,canonical[prefix],&representative);
    assert(ok==bound::certified(refined));
    if(ok){assert(bound::scalar::bf16(representative)==bound::scalar::bf16(canonical.back()));++certificates;if(first==3u)first=stage;}
    else assert(representative==123.0f);
   }
   if(!initial){++stopped[first];saved_groups+=(3u-first)*(width/64u);}
  }
 }
 std::printf("{\"kind\":\"macro_norm_bound_host\",\"groups_per_block\":%u,\"width\":%u,\"cases\":%u,\"native_error_modes\":3,\"original_prefix_checkpoints\":%zu,\"suffix_interval_checks\":%u,\"certified_suffixes\":%zu,\"initial_selected\":%zu,\"selected_stop_counts\":[%zu,%zu,%zu,%zu],\"selected_original_groups\":%zu,\"skipped_original_groups\":%zu,\"interior_carry_barriers_checked\":true,\"all_metadata_dot_bounds_checked\":true,\"original_bf16_encodings_checked\":65536,\"undercoverage\":0,\"false_certificates\":0,\"native_matrix_executed\":false,\"hardware_error_bound_proven\":false}\n",Groups,width,cases,checkpoints,cases*9u,certificates,selected,stopped[0],stopped[1],stopped[2],stopped[3],selected*(width/16u),saved_groups);

}
int main(){
 for(unsigned x=0u;x<65536u;++x){const unsigned exponent=(x>>7u)&255u;assert(bound::eligible(uint16_t(x))==(!(x&0x7fffu)||(exponent>=80u&&exponent<=174u)));}
 for(unsigned exponent=1u;exponent<255u;++exponent)for(unsigned mantissa:{0u,1u,0x3fffffu,0x7fffffu}){
  const float value=bound::scalar::value((exponent<<23u)|mantissa);
  for(unsigned fractional:{23u,25u}){
   const double expected=std::ldexp(1.0,int(exponent)-127-int(fractional));
   assert(double(bound::unit(value,fractional))>=expected);
   if(expected>=double(std::numeric_limits<float>::denorm_min()))assert(double(bound::unit(value,fractional))==expected);
  }
 }
 for(unsigned word=0u;word<65536u;++word)for(unsigned mode=0u;mode<3u;++mode){
  uint16_t row[16];for(unsigned i=0u;i<16u;++i)row[i]=mode==0u?0u:mode==1u?uint16_t(80u<<7u):uint16_t((174u<<7u)|127u);
  row[word%16u]=uint16_t(word);const auto value=observed::prepare(row,16u);
  if(!bound::eligible(uint16_t(word))){assert(std::isinf(value.norm)&&std::isinf(value.maximum));continue;}
  double squares=0.0,maximum=0.0;for(uint16_t x:row){const double magnitude=std::abs(double(f32(x)));squares+=magnitude*magnitude;maximum=std::max(maximum,magnitude);}
  assert(double(value.norm)*double(value.norm)>=squares && double(value.maximum)==maximum);
  if(squares)assert(double(value.norm)<=std::sqrt(squares)*1.00003);else assert(value.norm==0.0f);
 }
 assert(std::isinf(observed::prepare(nullptr,16u).norm));
 for(float invalid:{-1.0f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
  float untouched=123.0f;
  assert(!bound::certified(observed::advance<8u>({},0.0f,invalid,0.0f,0.0f,0.0f,&untouched))&&untouched==123.0f);
  assert(!bound::certified(observed::advance<8u>({0.0f,invalid},0.0f,0.0f,0.0f,0.0f,0.0f)));
  assert(!bound::certified(observed::advance<8u>({},0.0f,0.0f,invalid,0.0f,0.0f)));
  assert(!bound::certified(observed::advance<8u>({},0.0f,0.0f,0.0f,invalid,0.0f)));
  assert(!bound::certified(observed::advance<8u>({},0.0f,0.0f,0.0f,0.0f,invalid)));
 }
 assert(!bound::certified(observed::advance<8u>({2.0f,0.0f},-2.0f,2.0f,0.0f,2.0f,2.0f)));
 assert(!bound::certified(observed::advance<8u>({},2.0f,2.0f,1.0f,2.0f,2.0f)));
 assert(!bound::certified(observed::advance<8u>({},2.0f,2.0f,2.0f,1.0f,2.0f)));
 run<8u,1024u>();run<8u,4096u>();run<16u,4096u>();run<32u,4096u>();run<64u,4096u>();
}
