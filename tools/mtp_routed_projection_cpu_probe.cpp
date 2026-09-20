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
#include "q1_moe_hawkeye_bf16_accumulator.h"

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
int main(int argc,char** argv)try{
    if(argc!=8)throw std::runtime_error("directory gate_up_shard offset down_shard offset first_row rows");
    const std::string root=argv[1];const size_t first=std::stoull(argv[6]),rows=std::stoull(argv[7]);
    if(!rows||rows>4||first>128||first+rows>128)throw std::runtime_error("row bound");
    auto input=read<uint16_t>(root+"/input.bin"),gate_up=read<uint16_t>(root+"/routed-gate-up.bin"),
         activated=read<uint16_t>(root+"/routed-activated.bin"),weighted=read<uint16_t>(root+"/routed-weighted.bin");
    auto ids=read<uint32_t>(root+"/topk-ids.bin");auto weights=read<float>(root+"/topk-weights.bin");
    const size_t total=input.size()/2048;
    if(!total||input.size()!=total*2048||first+rows>total||gate_up.size()!=total*8192||
       activated.size()!=total*4096||weighted.size()!=total*16384||ids.size()!=total*8||weights.size()!=total*8)
        throw std::runtime_error("captured shape");
    for(auto id:ids)if(id>=256)throw std::runtime_error("expert ID");
    for(auto value:weights)if(!std::isfinite(value)||value<0||value>1)throw std::runtime_error("route weight");
    Mapping w1(argv[2],std::stoull(argv[3]),size_t(256)*1024*2048);
    Mapping w2(argv[4],std::stoull(argv[5]),size_t(256)*2048*512);
    Check first_projection,second_projection;
    for(size_t row=first;row<first+rows;++row)for(unsigned route=0;route<8;++route){
        const unsigned id=ids[row*8+route];
        for(unsigned channel=0;channel<1024;++channel){
            const float result=qrt_q1_moe_hawkeye::dot_bf16_hopper_blackwell(input.data()+row*2048,
                w1.values+(size_t(id)*1024+channel)*2048,2048);
            first_projection.compare(result,gate_up[row*8192+route*1024+channel],row,route,channel);
        }
        for(unsigned channel=0;channel<2048;++channel){
            const float result=qrt_q1_moe_hawkeye::dot_bf16_hopper_blackwell(activated.data()+row*4096+route*512,
                w2.values+(size_t(id)*2048+channel)*512,512);
            second_projection.compare(result*weights[row*8+route],weighted[row*16384+route*2048+channel],row,route,channel);
        }
    }
    std::cout<<"{\"first_row\":"<<first<<",\"rows\":"<<rows<<",\"gate_up\":";first_projection.print();
    std::cout<<",\"weighted_down\":";second_projection.print();
    std::cout<<",\"gpu_used\":false,\"inference_acceptance\":false}\n";
    return first_projection.bad||second_projection.bad?1:0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
