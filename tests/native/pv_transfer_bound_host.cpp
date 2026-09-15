#include "../../native/providers/moe_accumulator/sm121_pv_transfer_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
namespace b=qrt_sm121_pv_transfer_bound;
namespace original=qrt_q1_moe_hawkeye;
uint32_t rng=0x3958192u;
uint32_t random_word(){rng^=rng<<13u;rng^=rng>>17u;rng^=rng<<5u;return rng;}
float add(float a,float c){volatile float result=a+c;return result;}
template<unsigned Groups> void run(){
 constexpr unsigned cases=8192u,total_groups=64u;
 size_t checkpoints=0u,certificates=0u,zero_scales=0u;
 for(unsigned sample=0u;sample<cases;++sample){
  std::array<double,total_groups> dots{},absolute{};
  std::array<float,total_groups> canonical{},alpha{};
  float carry=0.0f;
  const unsigned ae=80u+random_word()%87u,be=80u+random_word()%87u;
  for(unsigned group=0u;group<total_groups;++group){
   if(!(group&1u)){
    const unsigned choice=random_word()%7u;
    alpha[group]=choice<3u?1.0f:choice==3u?0.0f:choice==4u?0.5f:
     choice==5u?b::scalar::value(0x3f000000u|(random_word()&0x7fffffu)):
     b::scalar::value(0x00800000u|(random_word()&0x7fffffu));
    carry=b::product(carry,alpha[group]);zero_scales+=alpha[group]==0.0f;
   }
   original::Value terms[17];terms[0]=original::value_from_float(carry,-133);
   for(unsigned i=0u;i<16u;++i){
    uint16_t x=uint16_t((random_word()&0x807fu)|((ae+random_word()%9u)<<7u));
    uint16_t y=uint16_t((random_word()&0x807fu)|((be+random_word()%9u)<<7u));
    if(sample%7u==0u){x=uint16_t(0x3f81u|((i&1u)<<15u));y=0x3f85u;}
    if(sample%11u==0u&&i%3u==0u)x=uint16_t((sample&1u)<<15u);
    if(sample%13u==0u){x=0x3fffu;y=0x3fffu;}
    if(sample%17u==0u){x=uint16_t((80u<<7u)|(random_word()&0x807fu));y=uint16_t((174u<<7u)|(random_word()&0x807fu));}
    assert(b::coarse::eligible(x)&&b::coarse::eligible(y));
    const double product=double(b::scalar::value(uint32_t(x)<<16u))*double(b::scalar::value(uint32_t(y)<<16u));
    dots[group]+=product;absolute[group]+=std::abs(product);
    terms[i+1u]=original::multiply_bf16(x,y,-133);
   }
   carry=original::value_to_float(original::group_sum<26,-133>(terms,17u));
   canonical[group]=carry;
  }
  for(unsigned mode=0u;mode<3u;++mode){
   b::State state{};
   for(unsigned base=0u;base<total_groups;base+=Groups){
    float partial=0.0f,positive=0.0f,entry=state.center;bool erased=false;
    for(unsigned group=base;group<base+Groups;++group){
     if(!(group&1u)){partial=b::product(partial,alpha[group]);entry=b::product(entry,alpha[group]);erased|=alpha[group]==0.0f;}
     const double perturbation=(mode==0u?0.0:mode==1u?0.75:-0.75)*0x1p-19*absolute[group];
     const float d=float(dots[group]+perturbation),a=float(absolute[group]-perturbation);
     assert(std::abs(double(d)-dots[group])<=0x1p-19*absolute[group]);
     assert(std::abs(double(a)-absolute[group])<=0x1p-19*absolute[group]);
     partial=add(partial,d);positive=add(positive,a);
    }
    state=b::advance<Groups>(state,entry,partial,positive,true,erased);
    assert(b::scalar::finite(state.error));
    assert(std::abs(double(state.center)-double(canonical[base+Groups-1u]))<=double(state.error));++checkpoints;
    if(b::scalar::same_bf16(state.center,state.error)){
     assert(b::scalar::bf16(state.center)==b::scalar::bf16(canonical[base+Groups-1u]));++certificates;
    }
   }
  }
 }
 std::printf("{\"kind\":\"pv_transfer_bound_host\",\"groups\":%u,\"cases\":%u,\"native_error_modes\":3,\"original_checkpoints\":%zu,\"certified\":%zu,\"zero_scales\":%zu,\"undercoverage\":0,\"false_certificates\":0,\"hardware_error_bound_proven\":false}\n",Groups,cases,checkpoints,certificates,zero_scales);
}
int main(){
 for(float invalid:{-1.0f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
  assert(!b::alpha_valid(invalid));
  assert(!b::scalar::finite(b::advance<8u>({},0.0f,0.0f,invalid,true,false).error));
  assert(!b::scalar::finite(b::advance<8u>({0.0f,invalid},0.0f,0.0f,0.0f,true,true).error));
 }
 assert(b::alpha_valid(0.0f)&&b::alpha_valid(1.0f));
 assert(!b::alpha_valid(b::scalar::value(0x3f800001u)));
 assert(!b::scalar::finite(b::advance<8u>({},0.0f,0.0f,0.0f,false,false).error));
 run<8u>();run<16u>();run<32u>();
}
