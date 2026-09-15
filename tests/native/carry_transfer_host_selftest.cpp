#include "sm121_scaled_half_products.h"
#include "sm121_carry_transfer.h"
#include <array>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <stdexcept>
#include <vector>
namespace original=qrt_q1_moe_hawkeye;
namespace half=qrt_sm121_scaled_half_products;
using Value=original::Value;
bool same(Value a,Value b){return a.significand==b.significand&&a.exponent==b.exponent&&a.negative==b.negative;}
bool normal(Value c){return c.significand>=0x800000u&&c.significand<0x1000000u&&c.exponent>=-126&&c.exponent<=127;}
uint32_t rng=0x3958192u;
uint32_t random_word(){rng^=rng<<13u;rng^=rng>>17u;return rng^=rng<<5u;}
struct Statistics{size_t blocks=0,regular=0,dominant=0,transfer=0,prepared_transfer=0,row_bound_transfer=0,final_normalize_transfer=0,final_normalize_prepared=0,verified_groups=0;};
template<unsigned Batch>
Value block(Value carry,const uint16_t* left,const uint16_t* right,Statistics& stats){
 ++stats.blocks;stats.regular+=normal(carry);bool dominated=normal(carry),supported=true,row_bound=true;
 int32_t increments[Batch]{};uint32_t sums[Batch]{};bool signs[Batch]{};Value reference=carry,prefix[Batch];
 for(unsigned group=0u;group<Batch;++group){
  Value values[17];values[0]=reference;uint32_t modulo=0u;
  const auto a=half::prepare(left+group*16u),b=half::prepare(right+group*16u);
  supported=supported&&half::unit(a)!=-32768&&half::unit(b)!=-32768;
  row_bound=row_bound&&half::unit(a)!=-32768&&half::unit(b)!=-32768&&30+half::unit(a)+half::unit(b)<=carry.exponent;
  for(unsigned i=0u;i<16u;++i){
   const auto p=original::multiply_bf16(left[group*16u+i],right[group*16u+i],-133);values[i+1u]=p;
   if(!i)signs[group]=p.negative;
   if(p.exponent>carry.exponent){dominated=false;continue;}
   const unsigned shift=unsigned(carry.exponent-p.exponent);
   const uint32_t magnitude=shift>=32u?0u:(p.significand<<2u)>>shift;
   modulo+=p.negative?0u-magnitude:magnitude;
  }
  // Sixteen bounded products alone fit signed32; the carried significand is
  // excluded here. Explicit magnitude conversion avoids signed overflow.
  const int32_t sum=modulo>0x7fffffffu?-int32_t(0u-modulo):int32_t(modulo);
  sums[group]=modulo;
  increments[group]=carry.negative?(sum>=0?(sum+3)/4:-(-sum/4)):(sum>=0?sum/4:-((-sum+3)/4));
  reference=original::group_sum<26,-133>(values,17u);prefix[group]=reference;++stats.verified_groups;
 }
 stats.dominant+=dominated;
 bool internal=dominated;int32_t before_last=carry.negative?-int32_t(carry.significand):int32_t(carry.significand);
 for(unsigned group=0u;group+1u<Batch;++group){
  if(increments[group]<=-0x800000||increments[group]>=0x800000){internal=false;break;}
  before_last+=increments[group];
  if(!(carry.negative?before_last<=-0x800000&&before_last>-0x1000000:before_last>=0x800000&&before_last<0x1000000)){internal=false;break;}
 }
 const Value marker{0xa5a5a5a5u,-17,true};Value candidate=marker;
 const bool applied=dominated&&qrt_sm121_carry_transfer::apply(carry,sums,signs[Batch-1u],&candidate);
 assert(applied==internal);
 if(applied)assert(same(candidate,reference));else assert(same(candidate,marker));
 if(internal){
  ++stats.final_normalize_transfer;stats.final_normalize_prepared+=supported;
  const uint32_t aligned=uint32_t(before_last)*4u+sums[Batch-1u];
  const auto value=qrt_sm121_group16::decode_modulo_sum(aligned,signs[Batch-1u]);
  const auto result=qrt_sm121_canonical::normalize(value.magnitude,value.negative,carry.exponent);
  assert(same(result,reference));
 }

 bool accepted=dominated;int32_t current=carry.negative?-int32_t(carry.significand):int32_t(carry.significand);
 for(unsigned group=0u;group<Batch;++group){
  const int32_t increment=increments[group];
  if(increment<=-0x800000||increment>=0x800000){accepted=false;break;}
  current+=increment;
  const bool inside=carry.negative?current<=-0x800000&&current>-0x1000000:current>=0x800000&&current<0x1000000;
  if(!inside){accepted=false;break;}
 }
 if(accepted){
  ++stats.transfer;stats.prepared_transfer+=supported;stats.row_bound_transfer+=row_bound;
  int32_t running=carry.negative?-int32_t(carry.significand):int32_t(carry.significand);
  for(unsigned group=0u;group<Batch;++group){running+=increments[group];const Value candidate{uint32_t(running<0?-running:running),carry.exponent,carry.negative};assert(same(candidate,prefix[group]));}
  const Value candidate{uint32_t(current<0?-current:current),carry.exponent,carry.negative};assert(same(candidate,reference));
 }
 return reference;
}
void print(const Statistics& s,unsigned batch,unsigned dots,unsigned width,const char* label){std::printf("{\"kind\":\"carry_transfer_feasibility\",\"label\":\"%s\",\"batch_groups\":%u,\"dots\":%u,\"width\":%u,\"blocks\":%zu,\"normal_entry\":%zu,\"all_product_exponents_dominated\":%zu,\"certified_transfer\":%zu,\"prepared_transfer\":%zu,\"row_bound_transfer\":%zu,\"final_normalize_transfer\":%zu,\"final_normalize_prepared\":%zu,\"canonical_groups\":%zu,\"raw_mismatches\":0,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",label,batch,dots,width,s.blocks,s.regular,s.dominant,s.transfer,s.prepared_transfer,s.row_bound_transfer,s.final_normalize_transfer,s.final_normalize_prepared,s.verified_groups);}
std::vector<uint16_t> read(const char* path,size_t words){std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(words*2u))throw std::runtime_error("input span");std::vector<uint16_t> v(words);f.seekg(0);f.read(reinterpret_cast<char*>(v.data()),v.size()*2u);if(!f)throw std::runtime_error("input read");return v;}
template<unsigned Batch>void capture(const std::vector<uint16_t>& x,const std::vector<uint16_t>& w,unsigned tokens,unsigned rows,unsigned width){
 Statistics stats;rng=0x3958192u;constexpr unsigned samples=8192u;
 for(unsigned sample=0u;sample<samples;++sample){const unsigned token=random_word()%tokens,row=random_word()%rows;Value carry{0u,-133,false};for(unsigned base=0u;base<width;base+=Batch*16u)carry=block<Batch>(carry,x.data()+size_t(token)*width+base,w.data()+size_t(row)*width+base,stats);}
 print(stats,Batch,samples,width,"uniform_captured_cells_not_filtered_candidates");
}
template<unsigned Batch>void generated(){
 Statistics stats;rng=0x3958192u;
 for(unsigned test=0u;test<32768u;++test){
  Value carry{0x800000u|(random_word()&0x7fffffu),int16_t(int(random_word()%201u)-100),bool(test%2u)};
  if(test%29u==0u)carry={0u,-133,false};
  std::array<uint16_t,Batch*16u> a{},b{};
  for(unsigned i=0u;i<Batch*16u;++i){
   if(test%11u==0u){a[i]=uint16_t(random_word());b[i]=uint16_t(random_word());}
   else{const int target=int(carry.exponent)-(test%5u?int(random_word()%12u+4u):0);const int ae=127+target/2,be=254+target-ae;a[i]=uint16_t((unsigned(ae)<<7u)|(random_word()&0x807fu));b[i]=uint16_t((unsigned(be)<<7u)|(random_word()&0x807fu));}
  }
  block<Batch>(carry,a.data(),b.data(),stats);
 }
 assert(stats.transfer>1000u);print(stats,Batch,32768u,Batch*16u,"generated_normal_carries_and_fallbacks");
}
template<unsigned Groups>unsigned rejection_boundaries(){
 unsigned rejected=0u;uint32_t sums[Groups]{};const Value marker{0xa5a5a5a5u,-17,true};
 auto reject=[&](Value carry){Value output=marker;assert(!qrt_sm121_carry_transfer::apply(carry,sums,false,&output));assert(same(output,marker));++rejected;};
 for(Value carry:std::initializer_list<Value>{{0u,0,false},{0x7fffffu,0,false},{0x1000000u,0,false},{0x800000u,-127,false},{0x800000u,128,false}})reject(carry);
 sums[0]=uint32_t(-4);reject({0x800000u,0,false});sums[0]=4u;reject({0x800000u,0,true});
 // A final return to the original binade cannot hide an invalid interior.
 sums[0]=4u;sums[Groups-1u]=uint32_t(-4);reject({0xffffffu,0,false});
 sums[0]=uint32_t(-4);sums[Groups-1u]=4u;reject({0xffffffu,0,true});
 for(auto& sum:sums)sum=0u;sums[0]=0x80000000u;reject({0xc00000u,0,false});
 sums[0]=0u;sums[Groups-1u]=0x80000000u;reject({0xc00000u,0,true});
 sums[Groups-1u]=0u;assert(!qrt_sm121_carry_transfer::apply({0xc00000u,0,false},sums,false,nullptr));++rejected;
 return rejected;
}
int main(int argc,char** argv)try{
 if(argc==1){generated<2>();generated<4>();generated<8>();const unsigned rejected=rejection_boundaries<2>()+rejection_boundaries<4>()+rejection_boundaries<8>();std::printf("{\"kind\":\"carry_transfer_boundaries\",\"rejected\":%u,\"outputs_unchanged\":true}\n",rejected);return 0;}
 if(argc!=6)throw std::runtime_error("requires input, weights, tokens, rows, width");
 const unsigned tokens=unsigned(std::stoul(argv[3])),rows=unsigned(std::stoul(argv[4])),width=unsigned(std::stoul(argv[5]));
 if(!tokens||!rows||(width!=2048u&&width!=4096u))throw std::runtime_error("shape");
 const auto x=read(argv[1],size_t(tokens)*width),w=read(argv[2],size_t(rows)*width);
 capture<2>(x,w,tokens,rows,width);capture<4>(x,w,tokens,rows,width);capture<8>(x,w,tokens,rows,width);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
