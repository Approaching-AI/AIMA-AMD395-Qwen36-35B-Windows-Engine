// SPDX-License-Identifier: Apache-2.0
#include "gb10_gdn.hip.cpp"
#include <cassert>
#include <iostream>
using namespace aima_port;
static unsigned calls=0;static int returned=1;static bool seeded=false;
static float* expected_state=nullptr;
static const void* expected_first64=nullptr;
static unsigned observed_columns=32;
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
  if (std::string(name)=="prefill-a-first64") {
    assert(expected_first64 && pointer==expected_first64 && bytes==64*32*2 && context==&calls);
    fake_events.push_back("observed_first64");return;
  }
  assert(std::string(name)=="prefill-a-sampled" && pointer==active->output.data);
  assert(bytes==128*observed_columns*2 && context==&calls);fake_events.push_back("observed");
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
  std::vector<uint16_t> native_v(T*4096+256,0x1234);
  std::vector<float> native_g(gate_count+256,-555.f),native_beta(gate_count+256,-666.f);
  for(unsigned i=0;i<T*4096+256;++i){
    blockIdx=dim3(i/256);threadIdx=dim3(i%256);
    prepare_native_v_gate(conv.data(),a.data(),b.data(),native_v.data(),native_g.data(),
                         native_beta.data(),lut.data(),beta.data(),T);
  }
  for(unsigned i=0;i<T*4096;++i)assert(native_v[i]==conv[(i/4096)*8192+4096+i%4096]);
  for(unsigned i=0;i<gate_count;++i){
    assert(native_g[i]==gate[i/32*64+i%32]);
    assert(qrt_sm121_exp2::bits(native_beta[i])==qrt_sm121_exp2::bits(gate[i/32*64+32+i%32]));
  }
  for(unsigned i=T*4096;i<native_v.size();++i)assert(native_v[i]==0x1234);
  for(unsigned i=gate_count;i<native_g.size();++i)assert(native_g[i]==-555.f&&native_beta[i]==-666.f);
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
  for (unsigned columns : {1u, 256u, 512u, 16384u}) {
    std::vector<uint16_t> input(8192u*columns), output(128u*columns+256u,0x1234);
    for(unsigned i=0;i<input.size();++i) input[i]=uint16_t((i/columns)*17u+(i%columns)*29u+(i/columns)/256u);
    const auto copy=input;
    for(unsigned i=0;i<output.size();++i){blockIdx=dim3(i/256);threadIdx=dim3(i%256);sample_prefill(input.data(),output.data(),columns);}
    for(unsigned i=0;i<128u*columns;++i)assert(output[i]==copy[((i/columns+1)*64-1)*columns+i%columns]);
    for(unsigned i=128u*columns;i<output.size();++i)assert(output[i]==0x1234);
    assert(input==copy);
  }
  // Exercise the actual wrapper against a recording provider. Scratch owners
  // are deliberately distinct; kernels are recorded rather than GPU-executed.
  State s;for(Device* d:{&s.raw,&s.gates,&s.output,&s.decode_ab,&s.gate[0],&s.beta,&s.prefill_beta,&s.exp2,&s.rsqrt,&s.native_matrix,&s.native_inverse})d->allocate(64);
  reject([&]{gb10_rsqrt_table();});
  reject([&]{gb10_native_gdn_prefill_enabled(8192,false);});
  assert(!native_prefill_setting(nullptr) && !native_prefill_setting("0") && native_prefill_setting("1"));
  for (const char* value : {"", "true", "01", "2", "-1"}) reject([&]{native_prefill_setting(value);});
  s.cold=cold;s.seeded=seed;s.error=failure;active=&s;float state=17;expected_state=&state;
  assert(gb10_rsqrt_table() == s.rsqrt.as<unsigned char>());
  assert(!gb10_native_gdn_prefill_enabled(8192,false));
  s.native_prefill=true;
  assert(gb10_native_gdn_prefill_enabled(8192,false));
  for (std::size_t tokens : {0u,8191u,8193u}) reject([&]{gb10_native_gdn_prefill_enabled(tokens,false);});
  reject([&]{gb10_native_gdn_prefill_enabled(8192,true);});
  reject([&]{gb10_prefill_gdn(0,conv.data(),a.data(),b.data(),out,&state,8192,false);});
  assert(fake_events.empty() && calls==0);
  // These synthetic addresses test the wrapper's complete byte spans only.
  // fake_launch never dereferences them and does not simulate GPU shuffle math.
  std::array<void*,8> pointers{};
  for(unsigned i=0;i<pointers.size();++i)pointers[i]=reinterpret_cast<void*>(0x100000000ull+i*0x10000000ull);
  auto prepare=[&](const std::array<void*,8>& p,unsigned layer=0,unsigned tokens=8192){
    return gb10_prepare_native_gdn(layer,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],tokens);
  };
  unsigned preparation_rejections=0;
  auto bad_preparation=[&](auto fn){reject(fn);++preparation_rejections;assert(fake_events.empty());};
  for(unsigned i=0;i<8;++i){
    auto p=pointers;p[i]=nullptr;bad_preparation([&]{prepare(p);});
    p=pointers;p[i]=reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(p[i])+1);bad_preparation([&]{prepare(p);});
    p=pointers;p[i]=reinterpret_cast<void*>(UINTPTR_MAX-3);bad_preparation([&]{prepare(p);});
    for(unsigned j=0;j<i;++j){p=pointers;p[i]=p[j];bad_preparation([&]{prepare(p);});}
  }
  // Tail overlap must be rejected even when every starting address differs.
  auto overlap=pointers;overlap[3]=reinterpret_cast<void*>(0x100000000ull+8192ull*8192*2-2);
  bad_preparation([&]{prepare(overlap);});
  for(unsigned layer:{3u,40u})bad_preparation([&]{prepare(pointers,layer);});
  for(unsigned tokens:{0u,8191u,8193u})bad_preparation([&]{prepare(pointers,0,tokens);});
  for(Device* d:{&s.gate[0],&s.rsqrt,&s.prefill_beta,&s.native_matrix,&s.native_inverse}){
    void* saved=d->data;d->data=nullptr;bad_preparation([&]{prepare(pointers);});d->data=saved;
  }
  active=nullptr;bad_preparation([&]{prepare(pointers);});active=&s;
  s.native_prefill=false;bad_preparation([&]{prepare(pointers);});s.native_prefill=true;
  const auto matrices=prepare(pointers);
  assert(matrices.matrix_f32==s.native_matrix.data&&matrices.inverse_bf16==s.native_inverse.data);
  assert(fake_events==std::vector<std::string>({"prepare_native_qk","prepare_native_v_gate"}));
  fake_events.clear();assert(preparation_rejections==65);
  // Check production AOT argument widths, chunk strides, scratch lifetimes,
  // and the final resident-state destination without executing GPU arithmetic.
  std::array<void*,9> pipeline_pointers{};
  for(unsigned i=0;i<pipeline_pointers.size();++i)
    pipeline_pointers[i]=reinterpret_cast<void*>(0x400000000ull+i*0x20000000ull);
  std::array<Device*,4> owned{&s.raw,&s.native_matrix,&s.native_inverse,&s.exp2};
  std::array<void*,4> saved{};
  for(unsigned i=0;i<owned.size();++i){
    saved[i]=owned[i]->data;owned[i]->data=reinterpret_cast<void*>(0x800000000ull+i*0x20000000ull);
  }
  auto pipeline=[&](const std::array<void*,9>& p,unsigned tokens=8192){
    return gb10_native_gdn_pipeline(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],tokens);
  };
  unsigned pipeline_rejections=0;
  auto bad_pipeline=[&](auto fn){
    reject(fn);++pipeline_rejections;
    assert(fake_events.empty()&&fake_module_launches.empty()&&fake_zeroes.empty());
  };
  bad_pipeline([&]{pipeline(pipeline_pointers);});
  for(unsigned i=0;i<s.ordered.size();++i){
    const auto& image=ordered_images[i];
    assert(aima::sha256_bytes(image.data,image.bytes)==image.sha256);
    s.ordered[i]=std::make_unique<aima::AotKernel>(
        std::vector<unsigned char>(image.data,image.data+image.bytes),image.name);
  }
  for(unsigned i=0;i<9;++i){
    auto p=pipeline_pointers;p[i]=nullptr;bad_pipeline([&]{pipeline(p);});
    p=pipeline_pointers;p[i]=reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(p[i])+1);bad_pipeline([&]{pipeline(p);});
    p=pipeline_pointers;p[i]=reinterpret_cast<void*>(UINTPTR_MAX-3);bad_pipeline([&]{pipeline(p);});
    for(unsigned j=0;j<i;++j){p=pipeline_pointers;p[i]=p[j];bad_pipeline([&]{pipeline(p);});}
    for(auto owner:owned){p=pipeline_pointers;p[i]=owner->data;bad_pipeline([&]{pipeline(p);});}
  }
  auto pipeline_overlap=pipeline_pointers;
  pipeline_overlap[7]=reinterpret_cast<void*>(reinterpret_cast<std::uintptr_t>(pipeline_pointers[0])+8192ull*2048*2-2);
  bad_pipeline([&]{pipeline(pipeline_overlap);});
  for(auto owner:owned){
    auto value=owner->data;owner->data=nullptr;bad_pipeline([&]{pipeline(pipeline_pointers);});owner->data=value;
  }
  for(auto& kernel:s.ordered){
    auto value=std::move(kernel);bad_pipeline([&]{pipeline(pipeline_pointers);});kernel=std::move(value);
  }
  for(unsigned tokens:{0u,8191u,8193u})bad_pipeline([&]{pipeline(pipeline_pointers,tokens);});
  active=nullptr;bad_pipeline([&]{pipeline(pipeline_pointers);});active=&s;
  s.native_prefill=false;bad_pipeline([&]{pipeline(pipeline_pointers);});s.native_prefill=true;
  assert(pipeline(pipeline_pointers)==389&&fake_module_launches.size()==389);
  auto ptr=[](const void* p){return reinterpret_cast<std::uintptr_t>(p);};
  const auto matrix=ptr(s.native_matrix.data),inverse=ptr(s.native_inverse.data),scratch=ptr(s.raw.data),table=ptr(s.exp2.data);
  const auto q=ptr(pipeline_pointers[0]),k=ptr(pipeline_pointers[1]),v=ptr(pipeline_pointers[2]);
  const auto g=ptr(pipeline_pointers[3]),beta_pointer=ptr(pipeline_pointers[4]);
  const auto w=ptr(pipeline_pointers[5]),u=ptr(pipeline_pointers[6]),core=ptr(pipeline_pointers[7]),final=ptr(pipeline_pointers[8]);
  const std::vector<std::vector<std::uintptr_t>> first={
    {k,k,beta_pointer,g,table,matrix},{matrix,inverse},{inverse,k,beta_pointer,g,table,w},
    {inverse,v,beta_pointer,u,0},{q,k,beta_pointer,g,table,matrix}};
  for(unsigned i=0;i<5;++i){
    const auto& launch=fake_module_launches[i];assert(launch.index==i&&launch.tokens==8192&&launch.pointers==first[i]);
    assert(launch.x==(i==1?128u:1024u)&&launch.y==(i==1?32u:(i==0||i==4)?8u:16u)&&launch.z==(i==1?1u:32u));
  }
  constexpr std::size_t state_extent=32ull*128*128*4,chunk_extent=64ull*32*128*2;
  const auto residual=scratch+2*state_extent,v_new=residual+chunk_extent;
  std::uintptr_t incoming=scratch;
  for(unsigned chunk=0;chunk<128;++chunk){
    const auto& r=fake_module_launches[5+chunk*3];
    const auto& h=fake_module_launches[6+chunk*3];
    const auto& o=fake_module_launches[7+chunk*3];
    const auto first_row=chunk*64ull;
    const auto next=chunk==127?final:scratch+((chunk+1)%2)*state_extent;
    assert(r.index==5&&h.index==6&&o.index==7);
    assert(r.tokens==64&&h.tokens==64&&o.tokens==64);
    assert(r.x==8&&h.x==16&&o.x==8&&r.y==16&&h.y==16&&o.y==16&&r.z==32&&h.z==32&&o.z==32);
    assert(r.pointers==std::vector<std::uintptr_t>({w+first_row*4096*2,u+first_row*4096*2,incoming,g+first_row*32*4,table,v_new,residual}));
    assert(h.pointers==std::vector<std::uintptr_t>({k+first_row*2048*2,residual,incoming,g+first_row*32*4,table,next}));
    assert(o.pointers==std::vector<std::uintptr_t>({q+first_row*2048*2,v_new,incoming,g+first_row*32*4,matrix+first_row*32*64*2,table,core+first_row*4096*2}));
    assert(incoming!=next);incoming=next;
  }
  assert(incoming==final);
  assert((fake_zeroes==std::vector<std::pair<std::uintptr_t,std::size_t>>({{inverse,8192ull*32*64*2},{scratch,state_extent}})));
  assert(fake_events.size()==391&&fake_events[1]=="memset"&&fake_events[6]=="memset");
  for(unsigned i=0;i<owned.size();++i)owned[i]->data=saved[i];
  fake_events.clear();fake_module_launches.clear();fake_zeroes.clear();
  s.native_prefill=false;
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
  observed_columns=16384;fake_events.clear();
  observe_gdn_prefill(0,"prefill-a-sampled",a.data(),16384,8192);
  assert(fake_events==std::vector<std::string>({"sample_prefill","observed"}));
  observed_columns=32;
  reject([&]{set_gdn_prefill_observer(0,observed,&calls);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",nullptr,32,8192);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",s.output.data,32,8192);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",a.data(),31,8192);});
  reject([&]{observe_gdn_prefill(0,"prefill-a-sampled",a.data(),32,8191);});
  s.observer=nullptr;fake_events.clear();expected_first64=sample_input.data();
  const auto first64_input_copy=sample_input;
  set_gdn_prefill_observer(0,observed,&calls,true);
  observe_gdn_prefill(0,"prefill-a-sampled",sample_input.data(),32,8192);
  assert(fake_events==std::vector<std::string>({"sample_prefill","observed","observed_first64"}));
  assert(sample_input==first64_input_copy);
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
  std::cout<<"{\"ordered_pipeline_rejections\":"<<pipeline_rejections<<",\"ordered_pipeline_abi_and_images_pass\":true,\"ordered_pipeline_launches\":389,\"recurrent_chunks_checked\":128,\"native_conversion_values_checked\":12480,\"native_preparation_rejections\":65,\"gpu_qk_norm_tested\":false,\"conversion_values_checked\":33027,\"sampled_values_checked\":2199680,\"sampling_input_unchanged\":true,\"first64_original_pointer_and_extent\":true,\"sampling_guards_pass\":true,\"observer_faults_rejected\":5,\"guards_pass\":true,\"provider_order_pass\":true,\"seeded_state_forwarded\":true,\"decode_q2_flags\":true,\"injected_provider_failure_rejected\":true,\"invalid_bindings_rejected\":6,\"artifact_faults_rejected\":2,\"native_prefill_rejections\":11,\"native_prefill_cold_scope_pass\":true}\n";
}
