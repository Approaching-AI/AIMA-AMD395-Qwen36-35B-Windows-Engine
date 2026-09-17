#include "gdn/conv_consumer_interval.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
namespace a=qrt_conv_consumer;namespace c=qrt_routed_consumer;namespace s=qrt_sm121_silu;
uint32_t rng=0x52361abdu;
unsigned random_word(){rng^=rng<<13;rng^=rng>>17;rng^=rng<<5;return rng;}
float reference_product(uint16_t x,uint16_t w){return c::widen(c::rounded(float(double(c::widen(x))*double(c::widen(w)))));}
float reference_sum(const uint16_t (&x)[4],const uint16_t (&w)[4],unsigned mask){float result=0.0f;for(unsigned t=0;t<4;++t)if(mask&(1u<<t)){volatile float next=result+reference_product(x[t],w[t]);result=next;}return result;}
a::Interval between(uint32_t first,uint32_t last){float x=c::value(first),y=c::value(last);return {std::min(x,y),std::max(x,y),true};}
void wide_check(const unsigned char* table,size_t& endpoints,size_t& constants){
 for(unsigned attempt=0;attempt<128u;++attempt){
  const float raw=std::ldexp((attempt&1u)?-1.25f:1.25f,int(attempt%61u)-30);
  const float error=std::fabs(raw)*(attempt%3u?0.125f:4.0f);
  const auto range=a::endpoint(raw,error,true);assert(range.valid&&unsigned(range.high-range.low)>8u);
  assert(!c::range(raw,error).valid);
  const float magnitude=std::max(std::fabs(c::widen(c::unordered(range.low))),std::fabs(c::widen(c::unordered(range.high))));
  const uint16_t dominant=c::rounded(magnitude*1024.0f),fixed=c::ordered(dominant);
  c::Range inputs[4]={range,{fixed,fixed,true},{c::ordered(0),c::ordered(0),true},{c::ordered(0),c::ordered(0),true}};
  const uint16_t weights[4]={uint16_t(attempt&2u?0xbf80u:0x3f80u),0x3f80u,0x3f80u,0x3f80u};
  const auto cert=a::certify(inputs,weights,15u,table);assert(cert.sum.valid);
  for(unsigned key=range.low;key<=range.high;++key){
   const uint16_t values[4]={c::unordered(uint16_t(key)),dominant,0u,0u};
   const float result=reference_sum(values,weights,15u);
   assert(result>=cert.sum.low&&result<=cert.sum.high);++endpoints;
   if(cert.constant){assert(s::evaluate(table,result)==cert.output);++constants;}
  }
 }
 for(float raw:{c::value(1u),0.0f,c::value(0x7f800000u)})assert(!a::endpoint(raw,1.0f,true).valid);
 assert(!a::endpoint(1.0f,-1.0f,true).valid&&!a::endpoint(1.0f,c::value(0x7f800000u),true).valid);
 assert(endpoints>1000000u&&constants>0u);
}
void graph_check(const unsigned char* table,size_t& changed,size_t& outputs,size_t& protected_halo){
 constexpr unsigned tokens=17u,features=16u;
 for(unsigned attempt=0;attempt<512u;++attempt){
  c::Range projected[tokens][features];uint16_t original[tokens][features],mixed[tokens][features],weights[features][4];bool constant[tokens][features];
  for(unsigned feature=0;feature<features;++feature)for(unsigned tap=0;tap<4;++tap)
   weights[feature][tap]=random_word()%8u==0u?0u:uint16_t((random_word()%2u?0x8000u:0u)+(uint16_t(124u+random_word()%7u)<<7u)+(random_word()%128u));
  for(unsigned token=0;token<tokens;++token)for(unsigned feature=0;feature<features;++feature){
   const uint16_t raw=uint16_t((random_word()%2u?0x8000u:0u)+(uint16_t(124u+random_word()%7u)<<7u)+(random_word()%126u));
   const unsigned key=c::ordered(raw),width=random_word()%2u;
   projected[token][feature]={uint16_t(key),uint16_t(key+width),true};original[token][feature]=mixed[token][feature]=c::unordered(uint16_t(key+width));
  }
  for(unsigned token=0;token<tokens;++token)for(unsigned feature=0;feature<features;++feature){
   c::Range inputs[4]{};unsigned present=0u;
   for(unsigned tap=0;tap<4;++tap)if(token+tap>=3u){inputs[tap]=projected[token+tap-3u][feature];present|=1u<<tap;}
   constant[token][feature]=a::certify(inputs,weights[feature],present,table).constant;
  }
  for(unsigned token=0;token<tokens;++token)for(unsigned feature=0;feature<features;++feature){
   bool following[4]{};for(unsigned j=0;j<4u&&token+j<tokens;++j)following[j]=constant[token+j][feature];
   if(a::can_omit(token,tokens,following))mixed[token][feature]=c::unordered(projected[token][feature].low);
   changed+=mixed[token][feature]!=original[token][feature];
   if(tokens-token<=3u){assert(mixed[token][feature]==original[token][feature]);++protected_halo;}
  }
  for(unsigned token=0;token<tokens;++token)for(unsigned feature=0;feature<features;++feature){
   uint16_t before[4]{},after[4]{};unsigned present=0u;
   for(unsigned tap=0;tap<4;++tap)if(token+tap>=3u){before[tap]=original[token+tap-3u][feature];after[tap]=mixed[token+tap-3u][feature];present|=1u<<tap;}
   assert(s::evaluate(table,reference_sum(before,weights[feature],present))==s::evaluate(table,reference_sum(after,weights[feature],present)));++outputs;
  }
 }
 assert(changed>0u&&outputs==512u*tokens*features&&protected_halo==512u*3u*features);
}
std::vector<unsigned char> synthetic(){
 std::vector<unsigned char> table(s::table_bytes);std::memcpy(table.data(),"QSLUTB1\0",8);
 const uint32_t fields[]={1,16,s::directory_count,s::transition_count,0x7f800000u,0};
 const uint64_t spans[]={s::key_start,s::value_start,s::table_bytes,UINT64_C(4278190080)};
 std::memcpy(table.data()+8,fields,sizeof(fields));std::memcpy(table.data()+32,spans,sizeof(spans));
 auto* d=reinterpret_cast<uint32_t*>(table.data()+64);auto* keys=reinterpret_cast<uint32_t*>(table.data()+s::key_start);auto* values=reinterpret_cast<uint16_t*>(table.data()+s::value_start);
 for(unsigned i=0;i<s::transition_count;++i){const unsigned j=i%(s::transition_count/2);keys[i]=uint32_t(uint64_t(j)*0x7f7fffffu/(s::transition_count/2-1))+(i>=s::transition_count/2?0x80000000u:0u);values[i]=uint16_t(0x3800u+i%4096u);}
 unsigned floor=0;for(unsigned page=0;page<s::directory_count;++page){while(floor+1<s::transition_count&&keys[floor+1]<=uint64_t(page)<<16u)++floor;d[page]=floor;}
 return table;
}
int main(int argc,char** argv){
 assert(argc==1||argc==2);auto table=synthetic();
 if(argc==2){std::ifstream in(argv[1],std::ios::binary);assert(in);table.assign(std::istreambuf_iterator<char>(in),{});}
 assert(s::valid_layout(table.data(),table.size()));
 const auto* keys=reinterpret_cast<const uint32_t*>(table.data()+s::key_start);const auto* values=reinterpret_cast<const uint16_t*>(table.data()+s::value_start);
 size_t segments=0,boundaries=0,interior=0,nonfinite_declined=0;
 for(unsigned i=0;i<s::transition_count;++i){
  const uint32_t first=keys[i],limit=(first&0x80000000u)?0xff7fffffu:0x7f7fffffu;
  const uint32_t last=i+1<s::transition_count?std::min(limit,keys[i+1]-1u):limit;
  uint16_t value=0xffffu;const bool accepted=a::table_constant(table.data(),between(first,last),&value);
  assert(accepted==c::finite(c::widen(values[i])));
  if(accepted){assert(value==values[i]);}else{assert(value==0xffffu);++nonfinite_declined;}
  ++segments;
  for(uint32_t raw:{first,first+(last-first)/2,last}){assert(s::evaluate(table.data(),c::value(raw))==values[i]);++interior;}
  if(i+1<s::transition_count){value=0xffffu;assert(!a::table_constant(table.data(),between(last,keys[i+1]),&value)&&value==0xffffu);++boundaries;}
 }
 for(a::Interval interval: {a::Interval{1,-1,true},a::Interval{-1,1,true},a::Interval{0,c::value(0x7f800000u),true},a::Interval{0,0,false}}){uint16_t out=99;assert(!a::table_constant(table.data(),interval,&out)&&out==99);}
 uint16_t out=99;assert(!a::table_constant(nullptr,{0,0,true},&out)&&!a::table_constant(table.data(),{0,0,true},nullptr));
 size_t combinations=0,admitted=0,admitted_combinations=0;
 for(unsigned case_index=0;case_index<32768;++case_index){
  c::Range input[4];uint16_t weight[4],x[4];unsigned count=1;
  for(unsigned tap=0;tap<4;++tap){
   const uint16_t raw=uint16_t((random_word()%2?0x8000u:0u)+((119u+random_word()%17u)<<7u)+(random_word()%124u));
   const unsigned key=c::ordered(raw),width=case_index%4?random_word()%3u:0u;
   input[tap]={uint16_t(key),uint16_t(key+width),true};count*=width+1u;
   weight[tap]=case_index%16==0?0u:uint16_t((random_word()%2?0x8000u:0u)+((119u+random_word()%17u)<<7u)+(random_word()%128u));
  }
  const unsigned present=case_index%16;const auto cert=a::certify(input,weight,present,table.data());assert(cert.sum.valid);
  admitted+=cert.constant;
  for(unsigned code=0;code<count;++code){unsigned q=code;
   for(unsigned tap=0;tap<4;++tap){const unsigned n=input[tap].high-input[tap].low+1u;x[tap]=c::unordered(uint16_t(input[tap].low+q%n));q/=n;}
   const float value=reference_sum(x,weight,present);assert(value>=cert.sum.low&&value<=cert.sum.high);++combinations;
   if(cert.constant){assert(s::evaluate(table.data(),value)==cert.output);++admitted_combinations;}
  }
 }
 assert(admitted&&admitted_combinations&&combinations>32768);
 c::Range tiny[4]={{c::ordered(1),c::ordered(1),true},{},{},{}};uint16_t weights[4]={0x3f80,0,0,0};
 assert(!a::sum(tiny,weights,1).valid);assert(a::sum(tiny,weights,0).valid);assert(!a::sum(tiny,weights,16).valid);
 size_t halo_cases=0;
 for(unsigned tokens:{0u,1u,2u,3u,4u,17u,8192u})for(unsigned token=0;token<=tokens;++token)for(unsigned mask=0;mask<16;++mask){
  bool fixed[4];for(unsigned i=0;i<4;++i)fixed[i]=(mask>>i)&1u;
  assert(a::can_omit(token,tokens,fixed)==(token<tokens&&tokens-token>3u&&mask==15u));++halo_cases;
 }
 size_t graph_changed=0,graph_outputs=0,graph_halo=0;graph_check(table.data(),graph_changed,graph_outputs,graph_halo);
 size_t wide_endpoints=0,wide_constants=0;wide_check(table.data(),wide_endpoints,wide_constants);
 std::cout<<"{\"kind\":\"conv_consumer_interval_host\",\"actual_sm121_table\":"<<(argc==2?"true":"false")<<",\"segments\":"<<segments<<",\"nonfinite_segments_declined\":"<<nonfinite_declined<<",\"boundary_rejections\":"<<boundaries<<",\"interior_points\":"<<interior<<",\"enumerated_combinations\":"<<combinations<<",\"constant_intervals\":"<<admitted<<",\"constant_combinations\":"<<admitted_combinations<<",\"halo_cases\":"<<halo_cases<<",\"graph_changed_inputs\":"<<graph_changed<<",\"graph_output_checks\":"<<graph_outputs<<",\"graph_halo_checks\":"<<graph_halo<<",\"wide_endpoint_checks\":"<<wide_endpoints<<",\"wide_constant_checks\":"<<wide_constants<<",\"false_certificates\":0,\"model_loaded\":false}\n";
}
