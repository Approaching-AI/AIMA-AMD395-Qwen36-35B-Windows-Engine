// Offline comparisons against the original captured operator inputs.
// Weight files are mapped read-only; no GPU or model framework is loaded.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>
#include "sm121_mtp_head_math.h"

struct Mapping {
    int fd=-1; void* data=MAP_FAILED; size_t bytes=0; const uint16_t* values=nullptr;
    Mapping(const char* path,size_t offset,size_t count) {
        fd=open(path,O_RDONLY); struct stat statbuf{};
        if(fd<0||fstat(fd,&statbuf)||statbuf.st_size<=0||uint64_t(statbuf.st_size)>(uint64_t(4)<<30))
            throw std::runtime_error("weight file");
        bytes=size_t(statbuf.st_size);
        if(offset%2||offset>bytes||count>(bytes-offset)/2)throw std::runtime_error("weight extent");
        data=mmap(nullptr,bytes,PROT_READ,MAP_PRIVATE,fd,0);
        if(data==MAP_FAILED)throw std::runtime_error("weight map");
        values=reinterpret_cast<const uint16_t*>(static_cast<const unsigned char*>(data)+offset);
    }
    ~Mapping(){if(data!=MAP_FAILED)munmap(data,bytes);if(fd>=0)close(fd);}
    Mapping(const Mapping&)=delete;Mapping& operator=(const Mapping&)=delete;
};
template<class T> std::vector<T> read(const std::string& path) {
    std::ifstream file(path,std::ios::binary|std::ios::ate);
    if(!file||file.tellg()<=0||file.tellg()>4u*1024u*1024u||file.tellg()%sizeof(T))
        throw std::runtime_error("capture file");
    std::vector<T> result(size_t(file.tellg())/sizeof(T));file.seekg(0);
    file.read(reinterpret_cast<char*>(result.data()),result.size()*sizeof(T));
    if(!file)throw std::runtime_error("capture read");return result;
}
uint16_t bf16(float x){uint32_t u;std::memcpy(&u,&x,4);return uint16_t((u+0x7fffu+((u>>16)&1u))>>16);}
struct Check {
    size_t elements=0,bad=0,first_row=0;unsigned first_route=0,first_channel=0;
    uint16_t first_actual=0,first_expected=0;
    void compare(float value,uint16_t expected,size_t row,unsigned route,unsigned channel) {
        if(!std::isfinite(value)||(expected&0x7f80u)==0x7f80u)throw std::runtime_error("nonfinite projection");
        const uint16_t actual=bf16(value);++elements;
        if(actual==expected)return;
        if(!bad){first_row=row;first_route=route;first_channel=channel;first_actual=actual;first_expected=expected;}
        ++bad;
    }
    void print()const{
        std::cout<<"{\"elements\":"<<elements<<",\"bf16_mismatches\":"<<bad<<",\"first_difference\":";
        if(bad)std::cout<<"{\"row\":"<<first_row<<",\"route\":"<<first_route<<",\"channel\":"<<first_channel
            <<",\"actual_bits\":"<<first_actual<<",\"expected_bits\":"<<first_expected<<'}';
        else std::cout<<"null";std::cout<<'}';
    }
};

float widen(uint16_t x){uint32_t u=uint32_t(x)<<16u;float f;std::memcpy(&f,&u,4);return f;}
int main(int argc,char** argv)try{
    if(argc!=5)throw std::runtime_error("lm_head_shard offset input logits");
    auto input=read<uint16_t>(argv[3]),expected=read<uint16_t>(argv[4]);
    if(input.size()!=2048||expected.size()!=248320)throw std::runtime_error("original draft row shape");
    Mapping weight(argv[1],std::stoull(argv[2]),size_t(248320)*2048);
    Check comparison;qrt_sm121_mtp::HeadBest actual,reference_best;
    double max_error=0;
    for(unsigned token=0;token<248320;++token){
        float partials[16];
        for(unsigned lane=0;lane<16;++lane)partials[lane]=qrt_sm121_shared_gate::lane_dot(input.data(),weight.values+size_t(token)*2048,lane);
        for(unsigned step=8;step;step/=2)for(unsigned lane=0;lane<step;++lane){volatile float sum=partials[lane]+partials[lane+step];partials[lane]=sum;}
        const float value=partials[0];
        comparison.compare(value,expected[token],0,0,token);
        const float rounded=widen(bf16(value)),reference=widen(expected[token]);
        const double error=std::abs(double(rounded)-reference);max_error=error>max_error?error:max_error;
        if(!qrt_sm121_mtp::head_candidate(bf16(value),token,&actual) || !qrt_sm121_mtp::head_candidate(expected[token],token,&reference_best))throw std::runtime_error("nonfinite sampling input");
    }
    std::cout<<"{\"projection\":";comparison.print();
    std::cout<<",\"actual_token\":"<<actual.token<<",\"expected_token\":"<<reference_best.token
        <<",\"actual_logit\":"<<actual.logit<<",\"expected_logit\":"<<reference_best.logit
        <<",\"maximum_bf16_logit_error\":"<<max_error<<",\"gpu_used\":false,\"inference_acceptance\":false}\n";
    return comparison.bad?1:0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
