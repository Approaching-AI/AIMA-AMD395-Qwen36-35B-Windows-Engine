// Replay original GB10 decode MoE observations against actual model weights.
// Usage: PREFIX WEIGHT_DIR ROWS CUDA_SILU_TABLE SIGMOID_F32_TABLE.
// Each comparison consumes the original operator input; this tool does not
// replace model inputs with expected downstream activations during inference.
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include "../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include "../native/providers/moe_accumulator/sm121_shared_gate.h"
uint32_t bits(float x){uint32_t u;std::memcpy(&u,&x,4);return u;}
float value(uint16_t x){uint32_t u=uint32_t(x)<<16;float f;std::memcpy(&f,&u,4);return f;}
uint16_t bf(float x){uint32_t u=bits(x);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
template<class T>std::vector<T>read(const std::string&p,size_t count){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(count*sizeof(T)))throw std::runtime_error("file shape "+p);std::vector<T>x(count);f.seekg(0);f.read((char*)x.data(),count*sizeof(T));if(!f)throw std::runtime_error("read");return x;}
int main(int argc,char**argv)try{
 if(argc!=6)return 2;std::string dir=argv[1],w=argv[2];size_t rows=std::stoul(argv[3]);if(!rows||rows>2)return 2;
 auto x=read<uint16_t>(dir+"-input-bf16.bin",rows*2048),gu=read<uint16_t>(dir+"-shared-gate-up-bf16.bin",rows*1024),act=read<uint16_t>(dir+"-shared-activated-bf16.bin",rows*512),down=read<uint16_t>(dir+"-shared-down-bf16.bin",rows*2048),gate=read<uint16_t>(dir+"-shared-gate-bf16.bin",rows),shared=read<uint16_t>(dir+"-shared-bf16.bin",rows*2048);
 auto wg=read<uint16_t>(w+"/shared_gate_up_gate.bin",512*2048),wu=read<uint16_t>(w+"/shared_gate_up_up.bin",512*2048),wd=read<uint16_t>(w+"/shared_down.bin",2048*512);auto lut=read<uint16_t>(argv[4],65536+12);auto beta=read<float>(argv[5],65536);
 auto wr=read<uint16_t>(w+"/router.bin",256*2048),router=read<uint16_t>(dir+"-router-bf16.bin",rows*256); auto wscalar=read<uint16_t>(w+"/shared_gate.bin",2048); unsigned rbad=0,rf32bad=0,scalarbad=0;
 unsigned gbad=0,ubad=0,acthost=0,actlut=0,dbad=0,sharedbad=0,hostsharedbad=0;
 for(size_t t=0;t<rows;t++){
  float lanes[16];for(unsigned l=0;l<16;l++)lanes[l]=qrt_sm121_shared_gate::lane_dot(x.data()+t*2048,wscalar.data(),l);for(unsigned offset=8;offset;offset/=2)for(unsigned l=0;l<offset;l++)lanes[l]+=lanes[l+offset];scalarbad+=bf(lanes[0])!=gate[t];
  for(unsigned r=0;r<256;r++){
   rbad+=bf(qrt_q1_moe_hawkeye::dot_bf16_hopper(x.data()+t*2048,wr.data()+r*2048,2048))!=router[t*256+r];
   float sum=0;for(unsigned base=0;base<2048;base+=128){float chunk=0;for(unsigned l=0;l<16;l++){float lane=0;for(unsigned k=l;k<128;k+=16)lane=std::fma(value(x[t*2048+base+k]),value(wr[r*2048+base+k]),lane);chunk+=lane;}sum+=chunk;}rf32bad+=bf(sum)!=router[t*256+r];
  }
  for(unsigned j=0;j<512;j++){
   gbad+=bf(qrt_q1_moe_hawkeye::dot_bf16_hopper(x.data()+t*2048,wg.data()+j*2048,2048))!=gu[t*1024+j];
   ubad+=bf(qrt_q1_moe_hawkeye::dot_bf16_hopper(x.data()+t*2048,wu.data()+j*2048,2048))!=gu[t*1024+512+j];
   uint16_t g=gu[t*1024+j],u=gu[t*1024+512+j];float gf=value(g),uf=value(u);
   acthost+=bf(value(bf(gf/(1.f+std::exp(-gf))))*uf)!=act[t*512+j];
   actlut+=bf(value(lut[12+g])*uf)!=act[t*512+j];
  }
  for(unsigned j=0;j<2048;j++){
   dbad+=bf(qrt_q1_moe_hawkeye::dot_bf16_hopper(act.data()+t*512,wd.data()+j*512,512))!=down[t*2048+j];
   sharedbad+=bf(value(bf(beta[gate[t]]))*value(down[t*2048+j]))!=shared[t*2048+j];
   hostsharedbad+=bf(value(bf(1.f/(1.f+std::exp(-value(gate[t])))))*value(down[t*2048+j]))!=shared[t*2048+j];
  }
 }
 std::cout<<"{\"rows\":"<<rows<<",\"router_k16_bad\":"<<rbad<<",\"router_f32_bad\":"<<rf32bad<<",\"scalar_gate_bad\":"<<scalarbad<<",\"gate_bad\":"<<gbad<<",\"up_bad\":"<<ubad<<",\"activation_host_bad\":"<<acthost<<",\"activation_lut_bad\":"<<actlut<<",\"down_bad\":"<<dbad<<",\"shared_beta_lut_bad\":"<<sharedbad<<",\"shared_host_bad\":"<<hostsharedbad<<"}\n";
}catch(const std::exception&e){std::cerr<<e.what();return 2;}
