#include "triton_moe/down_consumer_interval.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
namespace c = qrt_routed_consumer;
namespace interval = qrt_moe_down_consumer;
constexpr unsigned kHidden=1u,kTopK=8u,kExperts=256u,kFusedCombineWidth=1u;
struct Index { size_t x; } blockIdx{0u},threadIdx{0u},blockDim{1u};
float bf16_to_float(uint16_t x){return c::widen(x);}
uint16_t float_to_bf16(float x){return c::rounded(x);}
float __fmul_rn(float a,float b){volatile float x=a*b;return x;}
float __fadd_rn(float a,float b){volatile float x=a+b;return x;}
float routed_down_contribution_bf16_endpoint(float weight,float down,const uint16_t*,const uint16_t*,size_t,int32_t,uint32_t,uint32_t radius){
 assert(!radius);return bf16_to_float(float_to_bf16(__fmul_rn(weight,down)));
}
#include "moe_down_actual_combine.h"
float production(const float (&contribution)[8],float shared,float hidden){
 float weights[8]={1,1,1,1,1,1,1,1},gate=1.0f,output=0;
 int32_t experts[8]={0,1,2,3,4,5,6,7};uint16_t down=c::rounded(shared);
 full_v3_fused_combine_residual_kernel(contribution,weights,experts,nullptr,nullptr,&down,&gate,&hidden,&output,true,true,true,true,true,false,0u,1u);
 return output;
}
int main(){
 unsigned long long corners=0,certified=0,range_checks=0;
 uint32_t random=0x6d6f6538u;auto next=[&]{random^=random<<13u;random^=random>>17u;random^=random<<5u;return random;};
 for(unsigned fixture=0;fixture<2048u;++fixture){
  float raw[8],error[8],low[8],high[8],native[8];const unsigned mask=fixture&255u;
  bool valid=true;
  for(unsigned route=0;route<8u;++route){
   const unsigned word=next();
   const unsigned magnitude=((112u+(word>>8u)%27u)<<23u)|((word&127u)<<16u)|((fixture%3u==0u)?0x8000u:0x7fffu);
   raw[route]=c::value((word&0x80000000u)|magnitude);
   error[route]=fixture%4u ? 0.0f : std::fabs(raw[route])*0.004f;
   low[route]=high[route]=native[route]=interval::rounded(raw[route]);
   if(mask&(1u<<route)){
    const auto r=c::range(raw[route],error[route]);valid=valid&&r.valid;
    if(r.valid){low[route]=c::widen(c::unordered(r.low));high[route]=c::widen(c::unordered(r.high));
     for(unsigned key=r.low;key<=r.high;++key){assert(c::contains(r,c::widen(c::unordered(uint16_t(key)))));++range_checks;}
    }
   }
  }
  const auto snapshot=interval::enclose(raw,error,mask);
  assert(bool(snapshot.flags&interval::valid_bit)==valid);
  const float shared=interval::rounded(raw[fixture%8u]*(fixture&1u?-0.125f:0.25f));
  const float hidden=interval::rounded(raw[(fixture+3u)%8u]*0.5f);
  const bool stable=interval::combined_constant(snapshot,shared);
  const float baseline=production(native,shared,hidden);
  assert(c::bits(baseline)==c::bits(interval::residual(snapshot.center,shared,hidden)));
  for(unsigned corner=0;corner<256u;++corner){
   float values[8];for(unsigned route=0;route<8u;++route)values[route]=corner&(1u<<route)?high[route]:low[route];
   const float actual=production(values,shared,hidden);
   const float sum=interval::sum(values);
   if(valid)assert(snapshot.low<=sum&&sum<=snapshot.high);
   if(stable){assert(c::bits(actual)==c::bits(baseline));++certified;}
   ++corners;
  }
 }
 float raw[8]={},error[8]={};
 for(float special:{c::value(0x7f800000u),c::value(0x7fc00000u),c::value(1u)}){
  raw[0]=special;const auto s=interval::enclose(raw,error,1u);
  assert(!(s.flags&interval::valid_bit)&&!interval::combined_constant(s,1.0f));
 }
 assert(certified&&range_checks);
 std::printf("moe_down_interval_host corners=%llu certified_original_f32_outputs=%llu enumerated_bf16_endpoints=%llu pass=1\n",corners,certified,range_checks);
}
