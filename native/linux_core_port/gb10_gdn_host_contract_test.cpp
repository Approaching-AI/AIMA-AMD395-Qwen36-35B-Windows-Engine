// SPDX-License-Identifier: Apache-2.0
#include "gb10_gdn.hip.cpp"
#include <cassert>
#include <iostream>
using namespace aima_port;
static unsigned calls=0;static int returned=1;static bool seeded=false;
static float* expected_state=nullptr;
int cold(const float* raw,const float* gate,float* output,float* state,int decay,void* stream,int32_t tokens) {
  assert(active && raw==active->raw.data && gate==active->gates.data && output==active->output.data);
  assert(state==expected_state && decay==0 && stream==nullptr && tokens==8192);++calls;
  fake_events.push_back("cold");return returned;
}
int seed(const float* raw,const float* gate,float* output,float* state,int decay,void* stream,int32_t tokens) {
  seeded=true;const int result=cold(raw,gate,output,state,decay,stream,tokens);
  fake_events.back()="seeded";return result;
}
const char* failure(){return "injected provider failure";}
void observed(const char*name,const void*pointer,std::size_t bytes,void*context) {
  assert(std::string(name)=="prefill-a-sampled" && pointer==active->output.data);
  assert(bytes==128*32*2 && context==&calls);fake_events.push_back("observed");
}
template<class F> void reject(F fn){bool bad=false;try{fn();}catch(const std::exception&){bad=true;}assert(bad);}
int main(int argc,char**argv) {
  assert(argc==2);
  // Execute the actual conversion kernels with CPU coordinates. T=3 crosses
  // all feature/head/row boundaries; the extra block tests bounds and guards.
  constexpr unsigned T=3,raw_count=T*8192,gate_count=T*32;
  std::vector<uint16_t> conv(raw_count),a(gate_count),b(gate_count),beta(65536);
  std::vector<float> raw(raw_count+256,-111.f),gate(T*64+256,-222.f),lut(32*65536);
  for(unsigned i=0;i<raw_count;++i)conv[i]=uint16_t(0x3f00+i%128);
  for(unsigned i=0;i<gate_count;++i){a[i]=uint16_t(10+i);b[i]=uint16_t(20+i);}
  for(unsigned i=0;i<beta.size();++i)beta[i]=uint16_t(0x3e00+i%256);
  for(unsigned i=0;i<lut.size();++i)lut[i]=float(i);
  blockDim=dim3(256);
  for(unsigned i=0;i<raw_count+256;++i){blockIdx=dim3(i/256);threadIdx=dim3(i%256);prepare_prefill(conv.data(),a.data(),b.data(),raw.data(),gate.data(),lut.data(),beta.data(),T);}
  for(unsigned i=0;i<raw_count;++i)assert(qrt_sm121_exp2::bits(raw[i])==uint32_t(conv[i])<<16);
  for(unsigned i=0;i<gate_count;++i){assert(gate[i/32*64+i%32]==float((i%32)*65536+a[i]));assert(qrt_sm121_exp2::bits(gate[i/32*64+32+i%32])==uint32_t(beta[b[i]])<<16);}
  for(unsigned i=raw_count;i<raw.size();++i)assert(raw[i]==-111.f);
  for(unsigned i=T*64;i<gate.size();++i)assert(gate[i]==-222.f);
  std::vector<float> dr(8192+256,-333.f),ab(64+256,-444.f);
  for(unsigned i=0;i<8192+256;++i){blockIdx=dim3(i/256);threadIdx=dim3(i%256);prepare_decode(conv.data(),a.data(),b.data(),dr.data(),ab.data());}
  for(unsigned i=0;i<8192;++i)assert(qrt_sm121_exp2::bits(dr[i])==uint32_t(conv[i])<<16);
  for(unsigned i=0;i<32;++i){assert(qrt_sm121_exp2::bits(ab[i])==uint32_t(a[i])<<16);assert(qrt_sm121_exp2::bits(ab[32+i])==uint32_t(b[i])<<16);}
  for(unsigned i=8192;i<dr.size();++i)assert(dr[i]==-333.f);
  for(unsigned i=64;i<ab.size();++i)assert(ab[i]==-444.f);
  float values[]={qrt_sm121_exp2::value(0x3f808000),qrt_sm121_exp2::value(0x3f818000),qrt_sm121_exp2::value(0xbf818000)};
  uint16_t out[]={0,0,0,0x1234};
  for(unsigned i=0;i<4;++i){blockIdx=dim3(0);threadIdx=dim3(i);copy_core(values,out,3);}
  assert(out[0]==0x3f80&&out[1]==0x3f82&&out[2]==0xbf82&&out[3]==0x1234);
  std::vector<uint16_t> sample_input(8192*32),samples(128*32+256,0x1234);
  for(unsigned i=0;i<sample_input.size();++i)sample_input[i]=uint16_t(i);
  const auto preserved=sample_input;
  for(unsigned i=0;i<samples.size();++i){blockIdx=dim3(i/256);threadIdx=dim3(i%256);sample_prefill(sample_input.data(),samples.data(),32);}
  for(unsigned i=0;i<128*32;++i)assert(samples[i]==preserved[((i/32+1)*64-1)*32+i%32]);
  for(unsigned i=128*32;i<samples.size();++i)assert(samples[i]==0x1234);
  assert(sample_input==preserved);
  // Exercise the actual wrapper against a recording provider. Scratch owners
  // are deliberately distinct; kernels are recorded rather than GPU-executed.
  State s;for(Device* d:{&s.raw,&s.gates,&s.output,&s.decode_ab,&s.gate[0],&s.beta,&s.prefill_beta,&s.exp2,&s.rsqrt})d->allocate(64);
  reject([&]{gb10_rsqrt_table();});
  s.cold=cold;s.seeded=seed;s.error=failure;active=&s;float state=17;expected_state=&state;
  assert(gb10_rsqrt_table() == s.rsqrt.as<unsigned char>());
  gb10_prefill_gdn(0,conv.data(),a.data(),b.data(),out,&state,8192,false);
  assert(calls==1&&state==17&&fake_events==std::vector<std::string>({"prepare_prefill","cold","copy_core"}));
  fake_events.clear();gb10_prefill_gdn(0,conv.data(),a.data(),b.data(),out,&state,8192,true);
  assert(calls==2&&seeded&&fake_events==std::vector<std::string>({"prepare_prefill","seeded","copy_core"}));
  fake_events.clear();returned=0;reject([&]{gb10_prefill_gdn(0,conv.data(),a.data(),b.data(),out,&state,8192,false);});
  assert(fake_events==std::vector<std::string>({"prepare_prefill","cold"}));returned=1;
  fake_events.clear();fake_stream=reinterpret_cast<void*>(0x10);
  gb10_decode_gdn(0,conv.data(),a.data(),b.data(),out,&state,fake_stream);
  assert(fake_events==std::vector<std::string>({"prepare_decode","qrt_sm121_q1::recurrent","copy_core"}));
  assert(fake_q2_flags_verified);
  reject([&]{gb10_prefill_gdn(0,conv.data(),a.data(),b.data(),out,&state,8191,false);});
  reject([&]{gb10_decode_gdn(3,conv.data(),a.data(),b.data(),out,&state,nullptr);});
  reject([&]{gb10_decode_gdn(40,conv.data(),a.data(),b.data(),out,&state,nullptr);});
  reject([&]{gb10_decode_gdn(0,nullptr,a.data(),b.data(),out,&state,nullptr);});
  reject([&]{Gb10GdnOwner duplicate;});
  fake_stream=nullptr;fake_events.clear();
  set_gdn_prefill_observer(0,observed,&calls);
  observe_gdn_prefill(1,"prefill-a-sampled",a.data(),32,8192);assert(fake_events.empty());
  observe_gdn_prefill(0,"prefill-a-sampled",a.data(),32,8192);
  assert(fake_events==std::vector<std::string>({"sample_prefill","observed"}));
  reject([&]{set_gdn_prefill_observer(0,observed,&calls);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",nullptr,32,8192);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",s.output.data,32,8192);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",a.data(),31,8192);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",a.data(),32,8191);});
  active=nullptr;
  reject([&]{gb10_rsqrt_table();});
  reject([&]{gb10_decode_gdn(0,conv.data(),a.data(),b.data(),out,&state,nullptr);});
  // The production input reader checks exact byte length and SHA, including
  // artifact replacement with unchanged length.
  const std::filesystem::path file=std::filesystem::u8path(argv[1]);
  {std::ofstream f(file,std::ios::binary);f<<"abc";}
  const Asset asset{"fixture",3,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"};
  assert(read(file,asset)==std::vector<unsigned char>({'a','b','c'}));
  {std::ofstream f(file,std::ios::binary);f<<"abd";}reject([&]{read(file,asset);});
  {std::ofstream f(file,std::ios::binary);f<<"ab";}reject([&]{read(file,asset);});
  std::cout<<"{\"conversion_values_checked\":33027,\"sampled_values_checked\":4096,\"sampling_input_unchanged\":true,\"sampling_guards_pass\":true,\"observer_faults_rejected\":5,\"guards_pass\":true,\"provider_order_pass\":true,\"seeded_state_forwarded\":true,\"decode_q2_flags\":true,\"injected_provider_failure_rejected\":true,\"invalid_bindings_rejected\":6,\"artifact_faults_rejected\":2}\n";
}
