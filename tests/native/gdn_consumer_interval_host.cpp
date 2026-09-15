#include "../../native/providers/gdn/consumer_interval.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
namespace consumer=qrt_fla_consumer_interval;
namespace interval=qrt_sm121_projection_interval;
namespace bits=qrt_sm121_pv_bound;
uint32_t seed=0x3958192u;
uint32_t random_word(){seed^=seed<<13u;seed^=seed>>17u;return seed^=seed<<5u;}
float multiply(float a,float b){volatile float value=a*b;return value;}
float subtract(float a,float b){volatile float value=a-b;return value;}
float down(float x){return std::nextafter(x,-INFINITY);}
float up(float x){return std::nextafter(x,INFINITY);}
int main(){
 size_t state_admitted=0u,output_admitted=0u,checked_state=0u,checked_output=0u;
 for(unsigned trial=0u;trial<32768u;++trial){
  // Test every representable point in narrow intervals crossing nearby BF16
  // midpoints, including cancellation and original FP32 multiply boundaries.
  const uint32_t x=(random_word()&0x80000000u)|((100u+random_word()%55u)<<23u)|((random_word()&127u)<<16u)|0x8000u;
  const uint32_t y=(random_word()&0x80000000u)|((100u+random_word()%55u)<<23u)|(random_word()&0x7fffffu);
  const float center=bits::value(x),other=bits::value(y);
  float a[7],b[7];a[3]=center;b[3]=other;
  for(unsigned i=1u;i<=3u;++i){a[3u-i]=down(a[4u-i]);a[3u+i]=up(a[2u+i]);b[3u-i]=down(b[4u-i]);b[3u+i]=up(b[2u+i]);}
  const interval::Interval first{a[0],a[6]},second{b[0],b[6]};
  const float u=trial%3u?bits::value((random_word()&0x80000000u)|((100u+random_word()%55u)<<23u)|((random_word()&127u)<<16u)):center;
  const float decay=trial%7u?bits::value(((90u+random_word()%40u)<<23u)|(random_word()&0x7fffffu)):trial%2u?0.0f:1.0f;
  uint16_t updated=0xa5a5u,scaled=0xa5a5u,result=0xa5a5u;
  if(consumer::residual(first,u,decay,&updated,&scaled)){
   ++state_admitted;
   for(float dot:a){const float difference=subtract(u,dot);assert(updated==bits::bf16(difference));assert(scaled==bits::bf16(multiply(difference,decay)));++checked_state;}
  }else assert(updated==0xa5a5u&&scaled==0xa5a5u);
  if(consumer::output(first,second,decay,&result)){
   ++output_admitted;
   for(float old:a)for(float local:b){constexpr float scale=0.08838834764831845f;const float prior=multiply(multiply(old,decay),scale);assert(result==bits::bf16(std::fma(local,scale,prior)));++checked_output;}
  }else assert(result==0xa5a5u);
 }
 assert(state_admitted>1000u&&output_admitted>1000u);
 unsigned rejected=0u;
 for(auto bad:{interval::invalid(),interval::Interval{1.0f,-1.0f},interval::Interval{NAN,0.0f}}){
  uint16_t a=0xa5a5u,b=a;assert(!consumer::rounded(bad,&a));assert(!consumer::residual(bad,1,1,&a,&b));assert(!consumer::output(bad,{0,0},1,&a));assert(!consumer::output({0,0},bad,1,&a));assert(a==0xa5a5u&&b==0xa5a5u);rejected+=4u;
 }
 for(float bad:{-1.0f,-0.0f,INFINITY,NAN}){uint16_t a=0xa5a5u,b=a;assert(!consumer::residual({0,0},1,bad,&a,&b));assert(!consumer::output({0,0},{0,0},bad,&a));assert(a==0xa5a5u&&b==0xa5a5u);rejected+=2u;}
 for(float zero:{0.0f,-0.0f}){
  uint16_t a=0xa5a5u,b=a;if(consumer::residual({-0.0f,0.0f},zero,1,&a,&b))for(float point:{-0.0f,0.0f}){const auto expected=bits::bf16(subtract(zero,point));assert(a==expected&&b==expected);}
 }
 std::printf("{\"cases\":32768,\"state_admitted\":%zu,\"output_admitted\":%zu,\"state_points\":%zu,\"output_points\":%zu,\"exceptional_rejections\":%u,\"false_admissions\":0,\"rejection_outputs_unchanged\":true}\n",state_admitted,output_admitted,checked_state,checked_output,rejected);
}
