// Independent integer-oracle cases run through the actual gfx1151 compiler.
#include <hip/hip_runtime.h>
#include "native/providers/gdn/sm121_bf16_fma.h"
#include <windows.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
__global__ void probe(const uint16_t* cases,uint16_t* results,unsigned count){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    const auto* row=cases+size_t(i)*4u;
    results[size_t(i)*2u]=qrt_sm121_bf16_fma::round(row[0],row[1],row[2]);
    const float value=qrt_sm121_bf16_fma::rounded(qrt_sm121_bf16_fma::widen(row[0]),
        qrt_sm121_bf16_fma::widen(row[1]),qrt_sm121_bf16_fma::widen(row[2]));
    uint32_t bits;std::memcpy(&bits,&value,4);results[size_t(i)*2u+1u]=uint16_t(bits>>16u);
}
void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
int main(int argc,char** argv)try{
    if(argc!=2)throw std::runtime_error("usage: bf16-fma-probe cases.bin");
    char host[256]{};DWORD length=sizeof(host);
    if(!GetComputerNameA(host,&length)||_stricmp(host,"baiying"))throw std::runtime_error("requires baiying");
    std::ifstream f(argv[1],std::ios::binary|std::ios::ate);const std::streamoff size=f.tellg();
    if(!f||size<=0||size>8*1024*1024||size%8)throw std::runtime_error("bounded cases extent");
    std::vector<uint16_t> cases(size_t(size)/2u);f.seekg(0);f.read(reinterpret_cast<char*>(cases.data()),size);
    if(!f)throw std::runtime_error("case read");
    check(hipSetDevice(0));hipDeviceProp_t device{};check(hipGetDeviceProperties(&device,0));
    if(std::string(device.gcnArchName).find("gfx1151")!=0)throw std::runtime_error("requires gfx1151");
    const unsigned count=unsigned(cases.size()/4u);uint16_t *input=nullptr,*output=nullptr;
    check(hipMalloc(reinterpret_cast<void**>(&input),cases.size()*2u));
    check(hipMalloc(reinterpret_cast<void**>(&output),size_t(count)*4u));
    check(hipMemcpy(input,cases.data(),cases.size()*2u,hipMemcpyHostToDevice));
    hipLaunchKernelGGL(probe,dim3((count+255u)/256u),dim3(256u),0u,0,input,output,count);
    check(hipGetLastError());check(hipDeviceSynchronize());
    std::vector<uint16_t> actual(size_t(count)*2u);check(hipMemcpy(actual.data(),output,actual.size()*2u,hipMemcpyDeviceToHost));
    check(hipFree(output));check(hipFree(input));size_t bad=0;unsigned first=count;
    for(unsigned i=0;i<count;++i)for(unsigned j=0;j<2u;++j)if(actual[size_t(i)*2u+j]!=cases[size_t(i)*4u+3u]){++bad;if(first==count)first=i;}
    std::cout<<"{\"kind\":\"single_round_bf16_fma_gpu_probe\",\"host\":\"baiying\",\"cases\":"<<count
        <<",\"checked_endpoints\":"<<size_t(count)*2u<<",\"mismatches\":"<<bad<<",\"first_mismatch\":"<<first
        <<",\"hip_completion_checked\":true,\"inference_acceptance\":false}\n";
    return bad?6:0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
