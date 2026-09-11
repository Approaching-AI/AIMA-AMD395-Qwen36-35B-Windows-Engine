#include "../native/providers/moe_accumulator/sm121_router_exp.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <cstdint>
uint32_t bits(float v){uint32_t u;std::memcpy(&u,&v,4);return u;}
float value(uint16_t u){uint32_t v=uint32_t(u)<<16;float f;std::memcpy(&f,&v,4);return f;}
template<class T>std::vector<T>read(const char*p,size_t max){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("file");auto n=size_t(f.tellg());if(n>max||n%sizeof(T))throw std::runtime_error("span");std::vector<T>x(n/sizeof(T));f.seekg(0);f.read((char*)x.data(),n);if(!f)throw std::runtime_error("read");return x;}
float f32(uint32_t u){float f;std::memcpy(&f,&u,4);return f;}
float cuda_exp(float v,const std::vector<uint32_t>&lut){return qrt_sm121_router::exp(v,lut.data());}


int main(int argc,char**argv)try{if(argc!=5)return 2;auto logits=read<uint16_t>(argv[1],1024);auto ids=read<uint32_t>(argv[2],64);auto weights=read<float>(argv[3],64);auto lut=read<uint32_t>(argv[4],1u<<25);if(logits.size()%256||!logits.size()||ids.size()!=logits.size()/32||ids.size()!=weights.size()||lut.size()!=(1u<<23))throw std::runtime_error("shape");std::cout<<"{\"rows\":"<<logits.size()/256<<",\"variants\":[";for(int table=0;table<2;table++){unsigned badids=0,badweights=0;for(size_t t=0;t<logits.size()/256;t++){float probs[256],maximum=-INFINITY;for(unsigned j=0;j<256;j++)maximum=std::max(maximum,value(logits[t*256+j]));float sums[32]={};for(unsigned l=0;l<32;l++)for(unsigned j=0;j<8;j++){unsigned e=l*8+j;float v=value(logits[t*256+e])-maximum;probs[e]=table?cuda_exp(v,lut):std::exp(v);sums[l]+=probs[e];}for(unsigned mask=16;mask;mask/=2){float next[32];for(unsigned l=0;l<32;l++)next[l]=sums[l]+sums[l^mask];std::copy(next,next+32,sums);}float inv=1.f/sums[0];for(float&v:probs)v*=inv;float selected[8],denom=0;for(unsigned k=0;k<8;k++){unsigned best=0;for(unsigned j=1;j<256;j++)if(probs[j]>probs[best])best=j;selected[k]=probs[best];denom+=selected[k];probs[best]=-10000;badids+=best!=ids[t*8+k];}for(unsigned k=0;k<8;k++)badweights+=bits(selected[k]/denom)!=bits(weights[t*8+k]);}if(table)std::cout<<',';std::cout<<"{\"cuda_exp\":"<<(table?"true":"false")<<",\"id_mismatches\":"<<badids<<",\"weight_bit_mismatches\":"<<badweights<<'}';}std::cout<<"]}\n";}catch(const std::exception&e){std::cerr<<e.what();return 2;}
