#include "native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <stdexcept>
template<class T> std::vector<T> read(const char* path,size_t maximum){
 std::ifstream f(path,std::ios::binary|std::ios::ate);if(!f||f.tellg()<0||size_t(f.tellg())>maximum||size_t(f.tellg())%sizeof(T))throw std::runtime_error("span");
 std::vector<T> a(size_t(f.tellg())/sizeof(T));f.seekg(0);f.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(T));if(!f)throw std::runtime_error("read");return a;
}
float value(uint16_t b){uint32_t bits=uint32_t(b)<<16;float f;std::memcpy(&f,&bits,4);return f;}
uint16_t bf(float f){uint32_t bits;std::memcpy(&bits,&f,4);return uint16_t((bits+0x7fff+((bits>>16)&1))>>16);}
int main(int argc,char** argv)try{
 if(argc!=5)return 2;
 auto w=read<uint16_t>(argv[1],64u<<20);auto x=read<float>(argv[2],8192);auto native=read<float>(argv[3],32768);auto ref=read<uint16_t>(argv[4],32768);
 if(x.size()!=2048||w.size()%2048||native.size()!=w.size()/2048||ref.size()<native.size())throw std::runtime_error("shape");
 std::array<uint16_t,2048> xb{};for(unsigned i=0;i<2048;i++){xb[i]=bf(x[i]);if(value(xb[i])!=x[i])throw std::runtime_error("input not BF16");}
 size_t nd=0,fd=0,hd=0,td=0;std::cout<<"{\"differences\":[";bool comma=false;
 for(size_t r=0;r<native.size();r++){
  const auto* row=w.data()+r*2048;double sum=0;std::array<float,256> partial{};
  for(unsigned k=0;k<2048;k++){sum+=double(value(row[k]))*double(x[k]);partial[k%256]=std::fmaf(value(row[k]),x[k],partial[k%256]);}
  for(unsigned stride=128;stride;stride/=2)for(unsigned i=0;i<stride;i++)partial[i]+=partial[i+stride];
  float h=qrt_q1_moe_hawkeye::dot_bf16_hopper_blackwell(row,xb.data(),2048);auto n=bf(native[r]),d=bf(float(sum)),t=bf(partial[0]),a=bf(h);
  nd+=n!=ref[r];fd+=d!=ref[r];hd+=a!=ref[r];td+=t!=n;
  if((n!=ref[r]||a!=ref[r])&&nd+hd<40){if(comma)std::cout<<',';comma=true;std::cout<<"{\"row\":"<<r<<",\"native_bits\":"<<n<<",\"reference_bits\":"<<ref[r]<<",\"fp64_bits\":"<<d<<",\"group16_bits\":"<<a<<",\"fp32_tree_bits\":"<<t<<'}';}
 }
 std::cout<<"],\"rows\":"<<native.size()<<",\"native_mismatches\":"<<nd<<",\"fp64_mismatches\":"<<fd<<",\"group16_mismatches\":"<<hd<<",\"tree_native_mismatches\":"<<td<<",\"inference_acceptance\":false}\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
