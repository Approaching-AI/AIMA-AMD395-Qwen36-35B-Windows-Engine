#include "../../native/providers/moe_accumulator/sm121_exponent_loss_bound.h"
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <cassert>
#include <cmath>
#include <cstdio>
#include <initializer_list>
#include <limits>
namespace loss=qrt_sm121_exponent_loss_bound;
namespace coarse=qrt_sm121_coarse_projection_bound;
namespace original=qrt_q1_moe_hawkeye;
uint32_t rng=0x395916u;
uint32_t random_word(){rng^=rng<<13u;rng^=rng>>17u;rng^=rng<<5u;return rng;}
float f32(uint16_t x){return coarse::scalar::value(uint32_t(x)<<16u);}
float add(float a,float b){volatile float result=a+b;return result;}
unsigned exponent(uint16_t x){return (x>>7u)&255u;}
unsigned exact_below(const uint16_t* x,unsigned length,int threshold){
 unsigned n=0;for(unsigned i=0;i<length;++i)n+=unsigned((x[i]&0x7fffu)&&int(exponent(x[i]))<threshold);return n;
}
int main(){
 uint16_t a[64]{},b[64]{};size_t thresholds=0,union_checks=0;
 for(unsigned x=0;x<65536u;++x){
  a[0]=uint16_t(x);const auto s=loss::summarize(a,1u);
  assert(loss::valid(s)==coarse::eligible(uint16_t(x)));
 }
 for(unsigned sample=0;sample<4096u;++sample){
  const unsigned length=sample%65u;
  for(unsigned i=0;i<64u;++i){
   a[i]=uint16_t(((80u+random_word()%95u)<<7u)|(random_word()&0x807fu));
   b[i]=uint16_t(((80u+random_word()%95u)<<7u)|(random_word()&0x807fu));
   if(sample%3u==0u)a[i]=uint16_t(((120u+i%9u)<<7u)|(random_word()&0x807fu));
   if(i>=length||random_word()%7u==0u)a[i]=uint16_t((i&1u)<<15u);
   if(i>=length||random_word()%11u==0u)b[i]=0u;
  }
  if(sample%5u==0u)for(unsigned i=0;i<length;++i)b[i]=a[length-1u-i];
  const auto left=loss::summarize(a,length),right=loss::summarize(b,length);
  assert(loss::valid(left)&&loss::valid(right));
  for(int t=60;t<=195;++t){
   assert(loss::below(left,t)>=exact_below(a,length,t));
   assert(loss::below(right,t)>=exact_below(b,length,t));thresholds+=2;
  }
  for(unsigned bin=0;bin<8u;++bin){
   assert(loss::count(left,bin)==exact_below(a,length,int(loss::maximum(left))-int(2u*bin)));
   assert(loss::count(right,bin)==exact_below(b,length,int(loss::maximum(right))-int(2u*bin)));
  }
  for(int m=-133;m<=127;++m){
   unsigned possible=0;
   for(unsigned i=0;i<length;++i)possible+=unsigned((a[i]&0x7fffu)&&(b[i]&0x7fffu)&&int(exponent(a[i]))+int(exponent(b[i]))<m+243);
   const unsigned upper=loss::possible_products(left,right,m);
   assert(upper>=possible&&upper<=64u);++union_checks;
  }
 }
 constexpr unsigned cases=16384u,width=512u;
 size_t checkpoints=0,legacy_certificates=0,new_certificates=0,charged=0,legacy_charged=0,actual_loss_terms=0;
 for(unsigned sample=0;sample<cases;++sample){
  uint16_t left[width],right[width];float products[3][width/16],positive[3][width/16],canonical[width/16];
  unsigned actual_losses[width/16]{};int maxima[width/16]{};
  original::Value carry{0u,-133,false};
  const unsigned ae=80u+random_word()%87u,be=80u+random_word()%87u;
  for(unsigned group=0;group<width/16;++group){
   double sum=0.0,absolute=0.0;original::Value terms[17];terms[0]=carry;int maximum=carry.exponent;
   for(unsigned i=0;i<16u;++i){
    uint16_t x=uint16_t((random_word()&0x807fu)|((ae+random_word()%9u)<<7u));
    uint16_t y=uint16_t((random_word()&0x807fu)|((be+random_word()%9u)<<7u));
    if(sample%7u==0u){x=uint16_t(0x3f81u|((i&1u)<<15u));y=0x3f85u;}
    if(sample%11u==0u&&i%3u==0u)x=uint16_t((sample&1u)<<15u);
    if(sample%13u==0u){x=0x3fffu;y=0x3fffu;}
    if(sample%17u==0u){x=uint16_t(0x3f80u|((i&1u)<<15u));y=x;}
    if(sample%19u==0u){x=uint16_t((80u<<7u)|(random_word()&0x807fu));y=uint16_t((174u<<7u)|(random_word()&0x807fu));}
    if(sample%23u==0u)x=0u;
    left[group*16u+i]=x;right[group*16u+i]=y;
    const double value=double(f32(x))*double(f32(y));sum+=value;absolute+=std::abs(value);
    terms[i+1u]=original::multiply_bf16(x,y,-133);maximum=terms[i+1u].exponent>maximum?terms[i+1u].exponent:maximum;
   }
   maxima[group]=maximum;
   for(unsigned i=1;i<17u;++i){
    const uint64_t significand=uint64_t(terms[i].significand)<<2u;
    const unsigned shift=unsigned(maximum-terms[i].exponent);
    actual_losses[group]+=unsigned(shift>=32u?significand!=0u:(significand&((uint64_t(1u)<<shift)-1u))!=0u);
   }
   carry=original::group_sum<26,-133>(terms,17u);canonical[group]=original::value_to_float(carry);
   for(unsigned mode=0;mode<3u;++mode){
    const double perturbation=mode==0u?0.0:(mode==1u?0.75:-0.75)*0x1p-19*absolute;
    products[mode][group]=float(sum+perturbation);positive[mode][group]=float(absolute-perturbation);
    assert(std::abs(double(products[mode][group])-sum)<=0x1p-19*absolute);
    assert(std::abs(double(positive[mode][group])-absolute)<=0x1p-19*absolute);
   }
  }
  loss::Summary lm[width/64],rm[width/64];
  for(unsigned block=0;block<width/64;++block){lm[block]=loss::summarize(left+block*64);rm[block]=loss::summarize(right+block*64);}
  for(unsigned mode=0;mode<3u;++mode){
   coarse::State legacy{},revised{};
   for(unsigned base=0;base<width/16;base+=4u){
    float dot=0.0f,abs=0.0f;unsigned actual_count=0;int maximum=-133;
    for(unsigned i=0;i<4u;++i){dot=add(dot,products[mode][base+i]);abs=add(abs,positive[mode][base+i]);actual_count+=actual_losses[base+i];maximum=maxima[base+i]>maximum?maxima[base+i]:maximum;}
    unsigned possible=64u;
    revised=loss::advance(revised,dot,abs,lm[base/4u],rm[base/4u],&possible);
    legacy=coarse::advance<4u>(legacy,dot,abs);
    assert(coarse::scalar::bits(revised.center)==coarse::scalar::bits(legacy.center));
    assert(coarse::scalar::finite(revised.error)&&revised.error<=legacy.error);
    assert(std::abs(double(revised.center)-double(canonical[base+3u]))<=double(revised.error));
    assert(possible>=actual_count);
    assert(loss::possible_products(lm[base/4u],rm[base/4u],maximum)>=actual_count);
    charged+=possible;legacy_charged+=64u;actual_loss_terms+=actual_count;++checkpoints;
    if(coarse::certified(legacy)){++legacy_certificates;assert(coarse::certified(revised));}
    if(coarse::certified(revised)){++new_certificates;assert(coarse::scalar::bf16(revised.center)==coarse::scalar::bf16(canonical[base+3u]));}
   }
  }
 }
 for(float bad:{-1.0f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}){
  a[0]=b[0]=0x3f80u;const auto x=loss::summarize(a,1),y=loss::summarize(b,1);
  assert(!coarse::certified(loss::advance({},0.0f,bad,x,y)));
  assert(!coarse::certified(loss::advance({0.0f,bad},0.0f,0.0f,x,y)));
  if(!std::isfinite(bad))assert(!coarse::certified(loss::advance({},bad,0.0f,x,y)));
 }
 assert(!loss::valid(loss::summarize(nullptr))&&!loss::valid(loss::summarize(a,65u)));
 assert(!coarse::certified(loss::advance({},0.0f,0.0f,{},{})));
 assert(charged<legacy_charged&&new_certificates>=legacy_certificates);
 std::printf("{\"kind\":\"exponent_loss_bound_host\",\"summary_bytes\":8,\"payloads\":65536,\"threshold_checks\":%zu,\"union_bound_checks\":%zu,\"original_prefix_checkpoints\":%zu,\"legacy_product_charges\":%zu,\"revised_product_charges\":%zu,\"actual_inexact_products\":%zu,\"legacy_certificates\":%zu,\"revised_certificates\":%zu,\"undercoverage\":0,\"false_certificates\":0,\"native_error_coefficient\":0.0000019073486328125,\"hardware_error_bound_proven\":false,\"native_matrix_executed\":false,\"inference_acceptance\":false}\n",thresholds,union_checks,checkpoints,legacy_charged,charged,actual_loss_terms,legacy_certificates,new_certificates);
}
