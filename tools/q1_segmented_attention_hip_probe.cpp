// Original full-shape segmented-attention probe. Expected values stay on the host.
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_q1_attention.h"
#include "native/providers/gdn/sm121_q1_segmented_attention.h"
#include <map>
#include <memory>
#include <chrono>

void check(hipError_t s) { if (s != hipSuccess) throw std::runtime_error(hipGetErrorString(s)); }
std::string sha256(const void* data, size_t bytes) {
    if (bytes > ULONG_MAX) throw std::runtime_error("hash extent");
    BCRYPT_ALG_HANDLE algorithm = nullptr; BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD length = 0, copied = 0; unsigned char digest[32];
    std::vector<unsigned char> object;
    const auto ok = [](NTSTATUS s) { if (s < 0) throw std::runtime_error("BCrypt SHA256"); };
    try {
        ok(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0));
        ok(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH, reinterpret_cast<PUCHAR>(&length), sizeof(length), &copied, 0));
        object.resize(length);
        ok(BCryptCreateHash(algorithm, &hash, object.data(), length, nullptr, 0, 0));
        ok(BCryptHashData(hash, const_cast<PUCHAR>(static_cast<const unsigned char*>(data)), static_cast<ULONG>(bytes), 0));
        ok(BCryptFinishHash(hash, digest, sizeof(digest), 0));
        BCryptDestroyHash(hash); hash = nullptr;
        BCryptCloseAlgorithmProvider(algorithm, 0); algorithm = nullptr;
    } catch (...) {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
    std::ostringstream text;
    for (unsigned char c : digest) text << std::hex << std::setfill('0') << std::setw(2) << unsigned(c);
    return text.str();
}
template<class T> std::vector<T> read(const std::string& path, uint64_t offset, size_t count,
                           const std::string& digest, bool entire = false) {
    const size_t bytes = count * sizeof(T);
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || uint64_t(file.tellg()) > (uint64_t(8) << 30u) ||
        !count || bytes > (uint64_t(1) << 30u) || offset > uint64_t(file.tellg()) ||
        bytes > uint64_t(file.tellg()) - offset || (entire && (offset || bytes != uint64_t(file.tellg()))))
        throw std::runtime_error("input extent: " + path);
    std::vector<T> values(count); file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!file || sha256(values.data(), bytes) != digest) throw std::runtime_error("input hash: " + path);
    return values;
}
uint64_t number(const std::string& s, uint64_t maximum) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) throw std::runtime_error("integer");
    size_t used = 0; const auto value = std::stoull(s, &used);
    if (used != s.size() || value > maximum) throw std::runtime_error("integer bound");
    return value;
}
struct Entry {std::string path,digest;size_t offset=0,bytes=0;};
std::map<std::string,Entry> plan;
template<class T>std::vector<T> read_role(const std::string& name,size_t count){
    const auto& e=plan.at(name);if(e.bytes!=count*sizeof(T))throw std::runtime_error("role extent: "+name);
    return read<T>(e.path,e.offset,count,e.digest,e.offset==0u);
}
struct Buffer {
    unsigned char* raw=nullptr;size_t bytes=0;std::string digest;
    explicit Buffer(size_t count):bytes(count){check(hipMalloc(reinterpret_cast<void**>(&raw),bytes+512u));check(hipMemset(raw,0xa5,bytes+512u));}
    Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(raw&&hipDeviceSynchronize()==hipSuccess)(void)hipFree(raw);}
    void* data()const{return raw+256u;}
    template<class T>void upload(const std::vector<T>& values){
        if(values.size()*sizeof(T)!=bytes)throw std::runtime_error("upload extent");
        digest=sha256(values.data(),bytes);check(hipMemcpy(data(),values.data(),bytes,hipMemcpyHostToDevice));
    }
    std::pair<size_t,size_t> verify()const{
        std::vector<unsigned char> values(bytes+512u);check(hipMemcpy(values.data(),raw,values.size(),hipMemcpyDeviceToHost));
        size_t guards=0;for(size_t i=0;i<256u;++i)guards+=(values[i]!=0xa5u)+(values[bytes+256u+i]!=0xa5u);
        return {guards,sha256(values.data()+256u,bytes)!=digest};
    }
};
struct Inputs {
    std::vector<std::unique_ptr<Buffer>> buffers;
    template<class T>const T* upload(const std::vector<T>& values){
        auto next=std::make_unique<Buffer>(values.size()*sizeof(T));next->upload(values);
        const auto* p=static_cast<const T*>(next->data());buffers.push_back(std::move(next));return p;
    }
    template<class T>const T* role(const std::string& name,size_t count){return upload(read_role<T>(name,count));}
    std::pair<size_t,size_t> verify()const{
        std::pair<size_t,size_t> errors{};for(const auto& b:buffers){auto e=b->verify();errors.first+=e.first;errors.second+=e.second;}return errors;
    }
};
struct Comparison {
    std::string name;size_t elements=0,mismatches=0,first=0;uint32_t actual=0,expected=0;
    void add(uint32_t a,uint32_t e,size_t i){++elements;if(a==e)return;if(!mismatches){first=i;actual=a;expected=e;}++mismatches;}
};
template<class T>Comparison compare(const std::string& name,const T* device,const std::vector<T>& expected,bool key_major=false){
    std::vector<T> actual(expected.size());check(hipMemcpy(actual.data(),device,actual.size()*sizeof(T),hipMemcpyDeviceToHost));
    Comparison result;result.name=name;
    for(size_t i=0;i<expected.size();++i){
        size_t at=i;
        if(key_major){const size_t value=(i/128u)%128u,key=i%128u;at=(i/16384u)*16384u+key*128u+value;}
        uint32_t a=0,e=0;std::memcpy(&a,&actual[at],sizeof(T));std::memcpy(&e,&expected[i],sizeof(T));result.add(a,e,i);
    }
    return result;
}

int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("bound TSV plan history-tokens");
    const unsigned tokens=unsigned(number(argv[2],263680u));
    if(tokens<262144u)throw std::runtime_error("requires actual long shape");
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("plan missing");std::string line;
    while(std::getline(file,line)){
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream row(line);std::string name,path,offset,bytes,digest,extra;
        if(!std::getline(row,name,'\t')||!std::getline(row,path,'\t')||!std::getline(row,offset,'\t')||
           !std::getline(row,bytes,'\t')||!std::getline(row,digest,'\t')||std::getline(row,extra,'\t')||
           digest.size()!=64u||digest.find_first_not_of("0123456789abcdef")!=std::string::npos||
           !plan.emplace(name,Entry{path,digest,size_t(number(offset,uint64_t(8)<<30u)),size_t(number(bytes,uint64_t(1)<<30u))}).second)
            throw std::runtime_error("plan row");
    }

    using namespace qrt_sm121_q1_segmented_attention;
    Inputs original;
    const auto query=read_role<uint16_t>("query",4096u);
    std::vector<float> rope(9216u,0.0f);
    for(unsigned i=0;i<4096u;++i)rope[i]=qrt_sm121_q1::widen(query[i]);
    const auto* device_query=original.upload(rope);
    const auto* keys=original.role<uint16_t>("keys",size_t(tokens)*512u);
    const auto* values=original.role<uint16_t>("values",size_t(tokens)*512u);
    const auto* exp2=original.role<unsigned char>("exp2",qrt_sm121_exp2::table_bytes);
    const auto* reciprocal=original.role<unsigned char>("rcp",qrt_sm121_attention_rcp::table_bytes);
    const auto expected_output=read_role<float>("expected_output",4096u);
    const auto expected_acc=read_role<float>("expected_segment_output",output_elements);
    const auto expected_max=read_role<float>("expected_segment_max",scalar_elements);
    const auto expected_sum=read_role<float>("expected_segment_sum",scalar_elements);
    const unsigned stride=tokens+17u;
    Buffer scores(size_t(16u)*stride*4u),acc(output_elements*4u),maxima(scalar_elements*4u),sums(scalar_elements*4u),output(4096u*4u);
    // Guard fills above use the default stream. Complete initialization before
    // the independent nonblocking compute stream can write the same storage.
    check(hipDeviceSynchronize());
    hipStream_t stream=nullptr;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    std::vector<Comparison> comparisons;size_t padding_errors=0;
    for(unsigned prefix:{tokens-1u,262144u,tokens/2u,1u}){
        const auto started=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(qrt_sm121_q1_attention::scores,dim3(stride),dim3(256u),0u,stream,
            device_query,keys,keys+size_t(prefix)*512u,static_cast<float*>(scores.data()),prefix,tokens,stride);
        check(hipGetLastError());
        check(launch(static_cast<const float*>(scores.data()),values,values+size_t(prefix)*512u,
            static_cast<float*>(acc.data()),static_cast<float*>(maxima.data()),static_cast<float*>(sums.data()),
            static_cast<float*>(output.data()),prefix,tokens,stride,exp2,reciprocal,stream));
        check(hipStreamSynchronize(stream));
        const std::string name="prefix-"+std::to_string(prefix)+"/";
        comparisons.push_back(compare(name+"segment_output",static_cast<const float*>(acc.data()),expected_acc));
        comparisons.push_back(compare(name+"segment_max",static_cast<const float*>(maxima.data()),expected_max));
        comparisons.push_back(compare(name+"segment_sum",static_cast<const float*>(sums.data()),expected_sum));
        comparisons.push_back(compare(name+"context",static_cast<const float*>(output.data()),expected_output));
        for(unsigned head=0;head<16u;++head){
            float padding[17];check(hipMemcpy(padding,static_cast<const float*>(scores.data())+size_t(head)*stride+tokens,sizeof(padding),hipMemcpyDeviceToHost));
            for(float value:padding)padding_errors+=qrt_sm121_exp2::bits(value)!=0xff800000u;
        }
        std::cerr<<"completed split "<<prefix<<" in "<<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count()<<"s\n"<<std::flush;
    }
    check(hipStreamDestroy(stream));
    auto checks=original.verify();size_t guard_errors=checks.first;
    for(const auto* buffer:{&scores,&acc,&maxima,&sums,&output})guard_errors+=buffer->verify().first;
    size_t elements=0,mismatches=0;unsigned count=0;
    std::cout<<"{\"kind\":\"original_long_segmented_attention\",\"tokens\":"<<tokens<<",\"segments\":16,\"tile_tokens\":16,\"prefix_layouts\":4,\"comparisons\":[";
    for(const auto& s:comparisons){
        if(count++)std::cout<<',';elements+=s.elements;mismatches+=s.mismatches;
        std::cout<<"{\"stage\":\""<<s.name<<"\",\"elements\":"<<s.elements<<",\"bit_mismatches\":"<<s.mismatches
            <<",\"first_difference\":["<<s.first<<','<<s.actual<<','<<s.expected<<"]}";
    }
    const bool passed=!mismatches&&!guard_errors&&!checks.second&&!padding_errors;
    std::cout<<"],\"compared_elements\":"<<elements<<",\"bit_mismatches\":"<<mismatches
        <<",\"guard_errors\":"<<guard_errors<<",\"immutable_digest_errors\":"<<checks.second
        <<",\"score_padding_errors\":"<<padding_errors<<",\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"model_loaded\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
