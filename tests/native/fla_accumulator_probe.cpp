// CPU-only q64 attribution against captured GB10 pre-decay KKT cells.
#include "../../native/providers/moe_accumulator/q1_moe_hawkeye_bf16_accumulator.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

template<class T> bool load(const char* path, std::vector<T>& data) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream || stream.tellg() != static_cast<std::streamoff>(data.size()*sizeof(T))) return false;
    stream.seekg(0); stream.read(reinterpret_cast<char*>(data.data()), data.size()*sizeof(T));
    return static_cast<bool>(stream);
}
float value(uint16_t b) { uint32_t bits=uint32_t(b)<<16; float v; std::memcpy(&v,&bits,4); return v; }
uint16_t bf16(float v) { uint32_t b; std::memcpy(&b,&v,4); return uint16_t((b+0x7fffu+((b>>16)&1u))>>16); }
template<int Width,int Group,int Block> float dot(const uint16_t* a,const uint16_t* b) {
    float result=0;
    for(int base=0;base<128;base+=Block) {
        result += qrt_q1_moe_hawkeye::dot_bf16_impl<Width,Group,-133>(a+base,b+base,Block);
    }
    return result;
}
int main(int argc,char** argv) {
    if(argc!=4 && argc!=5) { std::cerr<<"usage: fla-accumulator-probe <normalized-k-bf16> <beta-bf16> <gb10-a-dot-f32> [tokens]\n"; return 2; }
    int tokens=64;
    if(argc==5){char* end=nullptr;const long parsed=std::strtol(argv[4],&end,10);if(!end||*end||parsed<1||parsed>8192)return 2;tokens=static_cast<int>(parsed);}
    std::vector<uint16_t> k(tokens*16*128), beta(tokens*32), kb(tokens*32*128);
    std::vector<float> reference(tokens*32*64);
    if(!load(argv[1],k)||!load(argv[2],beta)||!load(argv[3],reference)) return 3;
    for(int t=0;t<tokens;++t) for(int h=0;h<32;++h) for(int d=0;d<128;++d) {
        const float product=value(k[(t*16+h/2)*128+d])*value(beta[t*32+h]);
        if(!std::isfinite(product)) return 4;
        kb[(t*32+h)*128+d]=bf16(product);
    }
    struct Variant { const char* name; float(*call)(const uint16_t*,const uint16_t*); };
    const std::array<Variant,6> variants{{
        {"group16_width26_block128",dot<26,16,128>},
        {"group16_width26_block64",dot<26,16,64>},
        {"group16_width26_block32",dot<26,16,32>},
        {"group8_width25_block128",dot<25,8,128>},
        {"group8_width25_block64",dot<25,8,64>},
        {"group16_width25_block128",dot<25,16,128>}
    }};
    std::cout<<std::setprecision(17)<<"{\"kind\":\"cpu_kkt_accumulator_attribution\",\"tokens\":"<<tokens<<",\"sampled\":"<<(tokens>64?"true":"false")<<",\"inference_acceptance\":false,\"variants\":[";
    for(size_t mode=0;mode<variants.size();++mode) {
        uint64_t mismatches=0, compared=0; double max_error=0,error2=0,norm2=0; int first=-1; float first_actual=0,first_expected=0;
        for(int t=1;t<tokens;++t) for(int h=0;h<32;++h) for(int s=0;s<t%64;++s) {
            const int index=(t*32+h)*64+s;
            if(tokens>64 && ((uint32_t(index)*UINT32_C(0x9e3779b9))>>24)!=0)continue;
            const float expected=reference[index];
            if(!std::isfinite(expected)) return 5;
            const float actual=variants[mode].call(&kb[(t*32+h)*128],&k[((t/64*64+s)*16+h/2)*128]);
            const double delta=double(actual)-expected;
            ++compared; norm2+=double(expected)*expected; error2+=delta*delta;
            if(actual!=expected) { ++mismatches; max_error=std::max(max_error,std::abs(delta)); if(first<0){first=index;first_actual=actual;first_expected=expected;} }
        }
        if(mode)std::cout<<',';
        std::cout<<"{\"name\":\""<<variants[mode].name<<"\",\"elements\":"<<compared<<",\"mismatches\":"<<mismatches<<",\"maximum_absolute_error\":"<<max_error<<",\"relative_l2\":"<<std::sqrt(error2/norm2)<<",\"first_index\":"<<first<<",\"first_actual\":"<<first_actual<<",\"first_expected\":"<<first_expected<<'}';
    }
    std::cout<<"]}\n";
}
