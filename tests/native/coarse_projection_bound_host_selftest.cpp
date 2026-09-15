#include "../../native/providers/moe_accumulator/sm121_coarse_projection_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
namespace bound=qrt_sm121_coarse_projection_bound;
namespace original=qrt_q1_moe_hawkeye;
uint32_t state=0x3958192u;
uint32_t random_word(){state^=state<<13u;state^=state>>17u;state^=state<<5u;return state;}
float f32(uint16_t x){return bound::scalar::value(uint32_t(x)<<16u);}
float add(float a,float b){volatile float value=a+b;return value;}
template<unsigned Groups> void run(){
 constexpr unsigned cases=16384u,width=512u;size_t checkpoints=0,certificates=0;
 for(unsigned sample=0u;sample<cases;++sample){
  std::vector<float> products[3],positive[3],canonical;
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
    assert(bound::eligible(a)&&bound::eligible(b));
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
   bound::State value;
   for(unsigned base=0u;base<width/16u;base+=Groups){
    float dot=0.0f,abs=0.0f;
    for(unsigned i=0u;i<Groups;++i){dot=add(dot,products[mode][base+i]);abs=add(abs,positive[mode][base+i]);}
    value=bound::advance<Groups>(value,dot,abs);
    const double difference=std::abs(double(value.center)-double(canonical[base+Groups-1u]));
    assert(bound::scalar::finite(value.error)&&difference<=double(value.error));++checkpoints;
    if(bound::certified(value)){assert(bound::scalar::bf16(value.center)==bound::scalar::bf16(canonical[base+Groups-1u]));++certificates;}
   }
  }
 }
 std::printf("{\"kind\":\"coarse_projection_bound_host\",\"groups_per_block\":%u,\"cases\":16384,\"native_error_modes\":3,\"original_prefix_checkpoints\":%zu,\"certified_prefixes\":%zu,\"undercoverage\":0,\"false_certificates\":0,\"native_matrix_executed\":false,\"hardware_error_bound_proven\":false}\n",Groups,checkpoints,certificates);
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
 for(float invalid:{-1.0f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
  assert(!bound::certified(bound::advance<4u>({},0.0f,invalid)));
  assert(!bound::certified(bound::advance<4u>({0.0f,invalid},0.0f,0.0f)));
 }
 run<4u>();run<8u>();run<16u>();run<32u>();
}
