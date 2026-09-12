#include "sm121_dot_certificate.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
using qrt_q1_moe_hawkeye::Value;
Value step(Value carry,const Value* products){Value v[17];v[0]=carry;for(unsigned i=0;i<16;++i)v[i+1]=products[i];return qrt_q1_moe_hawkeye::group_sum<26,-133>(v,17);}
bool equal(Value a,Value b){return a.significand==b.significand && a.exponent==b.exponent && a.negative==b.negative;}
bool certificate(Value initial,const Value (&products)[4][16],unsigned count,Value (&prefix)[4]){
 if(!qrt_sm121_dot_certificate::canonical_normal(initial))return false;
 for(unsigned g=0;g<count;++g)for(auto p:products[g])if(p.exponent>initial.exponent)return false;
 uint32_t partial=initial.negative?0u-initial.significand:initial.significand;bool valid=true;
 for(unsigned g=0;g<count;++g){uint32_t sum=0;
  for(auto p:products[g]){unsigned shift=unsigned(initial.exponent-p.exponent);uint32_t a=shift>=32u?0u:(p.significand<<2u)>>shift;sum+=p.negative?0u-a:a;}
  valid &= qrt_sm121_dot_certificate::advance(partial,sum,initial.negative);
  uint32_t magnitude=initial.negative?0u-partial:partial;
  prefix[g]={magnitude,initial.exponent,initial.negative};
 }
 return valid;
}
uint32_t seed=0x3958192u;
uint32_t random_word(){seed^=seed<<13;seed^=seed>>17;seed^=seed<<5;return seed;}
int main(){uint64_t tiles=0,admitted=0,boundaries=0,bad=0;uint64_t admitted_modes[6]{};
 for(unsigned k:{48u,512u,2048u,2064u})for(unsigned row=0;row<4099u;++row){Value carry{0,-133,false};
  for(unsigned base=0;base<k;base+=64u){Value products[4][16];unsigned count=std::min(4u,(k-base)/16u);
   for(unsigned g=0;g<count;++g)for(unsigned i=0;i<16u;++i){unsigned index=base+g*16u+i,mode=row%6;
    unsigned spread=mode==4?30u:10u,exponent=mode==4?200u:mode==5?1u:119u;
    uint16_t a=uint16_t((random_word()&0x807fu)|((exponent+random_word()%spread)<<7u));
    uint16_t b=uint16_t((random_word()&0x807fu)|((exponent+random_word()%spread)<<7u));
    if(mode==1){a=uint16_t(0x3fffu|((row&1u)<<15u));b=0x3fffu;}
    if(mode==2){a=uint16_t(0x3fffu|((index%128u>=64u)?0x8000u:0u));b=0x3fffu;}
    if(mode==3){a&=0x807fu;b&=0x807fu;}
    products[g][i]=qrt_q1_moe_hawkeye::multiply_bf16(a,b,-133);
   }
   Value prefix[4];bool accepted=certificate(carry,products,count,prefix);++tiles;
   if(accepted){++admitted;++admitted_modes[row%6];}
   for(unsigned g=0;g<count;++g){carry=step(carry,products[g]);if(accepted){++boundaries;if(!equal(carry,prefix[g])){if(bad<4)std::printf("bad k=%u row=%u base=%u group=%u\n",k,row,base,g);++bad;}}}
  }
 }
 std::printf("{\"tiles\":%llu,\"certified_tiles\":%llu,\"certified_intermediate_boundaries\":%llu,\"mismatches\":%llu,\"admitted_modes\":[%llu,%llu,%llu,%llu,%llu,%llu],\"inference_acceptance\":false}\n",(unsigned long long)tiles,(unsigned long long)admitted,(unsigned long long)boundaries,(unsigned long long)bad,(unsigned long long)admitted_modes[0],(unsigned long long)admitted_modes[1],(unsigned long long)admitted_modes[2],(unsigned long long)admitted_modes[3],(unsigned long long)admitted_modes[4],(unsigned long long)admitted_modes[5]);return bad||!admitted;
}
