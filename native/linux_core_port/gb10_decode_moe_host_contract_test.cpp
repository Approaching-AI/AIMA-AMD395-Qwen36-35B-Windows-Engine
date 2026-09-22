// SPDX-License-Identifier: Apache-2.0
#include "gb10_decode_moe.hip.cpp"
#include <algorithm>
#include <iostream>
namespace aima_port {
bool fake_gdn_alive=true;
uint16_t fake_sigmoid[16]{};
const uint16_t* gb10_sigmoid_table(){if(!fake_gdn_alive)throw std::runtime_error("No GDN owner");return fake_sigmoid;}
unsigned norm_calls=0;
void gb10_residual_norm(const void* input,const void* residual,const void* weight,void* carrier,void* output,size_t rows,void* stream){
 assert(active && !stream && rows==1 && input==active->terminal.data && residual==active->terminal.as<uint16_t>()+2048);
 assert(weight && output && carrier==active->terminal_carrier);++norm_calls;
}
}
using namespace aima_port;
unsigned rejected=0;
template<class F>void reject(F f){bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught);++rejected;}
void setting(const char* name,const char* value){
#ifdef _WIN32
 _putenv_s(name,value);
#else
 setenv(name,value,1);
#endif
}
int main(int argc,char** argv){
 assert(argc==4);
 std::vector<uint16_t> input(2048,0x1234),arena(65536,0x2345),residual(2048,0x3456),norm(2048),weight(2048);
 auto pointer=[](uintptr_t n)->void*{return reinterpret_cast<void*>(n);};
 Gb10DecodeMoeWeights w{pointer(0x100000000),pointer(0x200000000),pointer(0x300000000),pointer(0x400000000),pointer(0x500000000),pointer(0x600000000),pointer(0x800000000)};
 Gb10DecodeMoeBuffers o{arena.data(),arena.data()+2048,arena.data()+4096,arena.data()+6144,
   arena.data()+8192,arena.data()+8448,arena.data()+8464,arena.data()+8704,
   arena.data()+16896,arena.data()+20992,arena.data()+37376,arena.data()+39424};
 auto call=[&](unsigned layer){gb10_decode_moe(layer,input.data(),w,o,nullptr);};
 auto save=[&]{gb10_decode_moe_save_terminal(o.combined,residual.data(),residual.data(),nullptr);};
 auto terminal=[&]{return gb10_decode_moe_terminal_norm(residual.data(),weight.data(),norm.data(),nullptr);};
 reject([&]{call(0);});reject(save);assert(!terminal());
 reject([]{gb10_moe_silu_table();});reject([]{gb10_moe_router_exp_table();});
 setting("AIMA_PORT_DECODE_MOE","");fake_gdn_alive=false;
 {Gb10DecodeMoeOwner disabled;assert(!active && !gb10_decode_moe_enabled());}
 setting("AIMA_PORT_DECODE_MOE","bad");reject([]{Gb10DecodeMoeOwner invalid;});
 setting("AIMA_PORT_DECODE_MOE","1");reject([]{Gb10DecodeMoeOwner missing_gdn;});fake_gdn_alive=true;
 const char* silu="QRT_QWEN36_CUDA_VLLM_SILU_BF16_DOMAIN_LUT_PATH";
 const char* router="QRT_QWEN36_CUDA_ROUTER_EX2_FRACTION_LUT_PATH";
 setting(silu,"");reject([]{Gb10DecodeMoeOwner missing;});
 setting(silu,argv[1]);setting(router,"");reject([]{Gb10DecodeMoeOwner missing;});assert(allocations==0);
 setting(router,argv[2]);
 {std::ofstream f(argv[3],std::ios::binary);f << "bad";}
 setting(silu,argv[3]);reject([]{Gb10DecodeMoeOwner truncated;});
 {std::vector<char> bad(131096);std::ofstream f(argv[3],std::ios::binary);f.write(bad.data(),bad.size());}
 reject([]{Gb10DecodeMoeOwner wrong_hash;});setting(silu,argv[1]);
 for(int n=0;n<4;++n){fail_allocation=allocation_calls+n;reject([]{Gb10DecodeMoeOwner failed;});assert(allocations==0 && !active);}
 fail_allocation=-1;fail_copy=true;reject([]{Gb10DecodeMoeOwner failed;});assert(allocations==0 && !active);fail_copy=false;
 unsigned complete_layers=0;
 {
  Gb10DecodeMoeOwner owner;assert(active && allocations==4);reject([]{Gb10DecodeMoeOwner duplicate;});
  assert(gb10_moe_silu_table()==active->silu.as<uint16_t>()+12);
  assert(gb10_moe_router_exp_table()==active->router_exp.as<uint32_t>());
  reject([&]{call(1);});reject([&]{call(40);});reject(save);
  reject([&]{gb10_decode_moe(0,input.data(),w,o,pointer(1));});
  reject([&]{gb10_decode_moe(0,nullptr,w,o,nullptr);});
  // Every borrowed weight and output field must be live and aligned.
  for(unsigned i=0;i<7;++i){auto broken=w;const void** fields[]={&broken.router,&broken.shared_gate,&broken.shared_gate_projection,&broken.shared_up_projection,&broken.shared_down,&broken.routed_gate_up,&broken.routed_down};
   *fields[i]=nullptr;reject([&]{gb10_decode_moe(0,input.data(),broken,o,nullptr);});}
  for(unsigned i=0;i<12;++i)for(unsigned control=0;control<3;++control){auto broken=o;void** fields[]={&broken.shared_input,&broken.shared_activation,&broken.shared_down,&broken.shared_output,&broken.router,&broken.router_indices,&broken.router_weights,&broken.routed_gate_up,&broken.routed_activation,&broken.routed_weighted,&broken.routed_output,&broken.combined};
   *fields[i]=control==0 ? nullptr : control==1 ? pointer(address(*fields[i])+1) : pointer(UINTPTR_MAX-1);
   reject([&]{gb10_decode_moe(0,input.data(),w,broken,nullptr);});}
  {auto broken=o;broken.shared_down=o.shared_input;reject([&]{gb10_decode_moe(0,input.data(),w,broken,nullptr);});
   broken=o;broken.combined=input.data();reject([&]{gb10_decode_moe(0,input.data(),w,broken,nullptr);});}
  fake_gdn_alive=false;reject([&]{call(0);});fake_gdn_alive=true;assert(launches.empty());
  for(unsigned token=0;token<2;++token){
   const int resets_before=resets,reads_before=device_reads,copies_before=copies;
   for(unsigned layer=0;layer<40;++layer){const size_t before=launches.size();call(layer);++complete_layers;
    assert(launches.size()==before+10 && resets==resets_before+1);
    assert(device_reads==reads_before+(layer==39) && copies==copies_before+(layer==39));
    if(layer==0){
     assert(launches[before].args==std::vector<uintptr_t>({address(w.router),address(input.data()),address(o.router),256,2048,0,256}));
     assert(launches[before+3].args==std::vector<uintptr_t>({address(w.shared_gate_projection),address(input.data()),address(o.shared_input)+2,512,2048,0,512}));
     assert(launches[before+4].args==std::vector<uintptr_t>({address(w.shared_up_projection),address(input.data()),address(o.shared_input)+1026,512,2048,0,512}));
     assert(launches[before+5].args==std::vector<uintptr_t>({address(input.data()),address(w.routed_gate_up),address(o.router_indices),address(o.router_weights),address(o.routed_gate_up),address(active->invalid.data),0,8192}));
     assert(launches[before+8].args==std::vector<uintptr_t>({address(o.routed_activation),address(w.routed_down),address(o.router_indices),address(o.router_weights),address(o.routed_weighted),address(active->invalid.data),0,16384}));
     const auto& f=launches[before+9];assert(f.args.size()==16 && f.args[10]==address(o.combined) && f.args[14]==address(fake_sigmoid) && f.args[15]==1);
    }
   }
   assert(active->needs_terminal_save);reject([&]{call(0);});reject(terminal);
   std::fill_n(static_cast<uint16_t*>(o.combined),2048,0x2345);std::fill(residual.begin(),residual.end(),0x3456);
   save();assert(!active->needs_terminal_save && active->terminal_carrier==residual.data());
   reject(save);reject([&]{call(0);});
   std::fill(residual.begin(),residual.end(),0);std::fill_n(static_cast<uint16_t*>(o.combined),2048,0);
   assert(std::all_of(active->terminal.as<uint16_t>(),active->terminal.as<uint16_t>()+2048,[](uint16_t v){return v==0x2345;}));
   assert(std::all_of(active->terminal.as<uint16_t>()+2048,active->terminal.as<uint16_t>()+4096,[](uint16_t v){return v==0x3456;}));
   reject([&]{gb10_decode_moe_terminal_norm(input.data(),weight.data(),norm.data(),nullptr);});
   reject([&]{gb10_decode_moe_terminal_norm(residual.data(),weight.data(),residual.data(),nullptr);});
   assert(terminal() && !terminal() && !active->terminal_carrier);
  }
 }
 assert(!active && allocations==0 && norm_calls==2 && snapshots==4);
 reject([]{gb10_moe_silu_table();});reject([]{gb10_moe_router_exp_table();});
 for(unsigned stage:{1u,7u,10u}){Gb10DecodeMoeOwner owner;fail_launch=launches.size()+stage;reject([&]{call(0);});assert(active->poisoned);fail_launch=0;reject([&]{call(0);});}
 {Gb10DecodeMoeOwner owner;
  for(unsigned layer=0;layer<39;++layer){inject_flag=layer==5 ? 4u : 0u;call(layer);}inject_flag=0;
  assert(*active->invalid.as<uint32_t>()==4u);reject([&]{call(39);});assert(active->poisoned);reject(save);
 }
 assert(!active && allocations==0);
 std::cout<<"{\"complete_layer_bindings\":"<<complete_layers<<",\"complete_token_lifetimes\":2,\"rejected_controls\":"<<rejected
  <<",\"persistent_device_error_checked\":true,\"terminal_operands_preserved\":true,\"all_allocations_released\":true,\"kernel_arithmetic_executed\":false}"<<std::endl;
}
