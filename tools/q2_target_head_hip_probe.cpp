// Complete private two-row target output head. Original expected tensors stay on the host.
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
#include "native/providers/gdn/sm121_q2_head.h"
#include <map>
#include <memory>

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
struct Buffer {
    unsigned char* raw = nullptr; size_t bytes = 0;
    std::vector<unsigned char> original;
    explicit Buffer(size_t size) : bytes(size) {
        check(hipMalloc(reinterpret_cast<void**>(&raw), bytes + 512u)); reset();
    }
    Buffer(const Buffer&) = delete; Buffer& operator=(const Buffer&) = delete;
    ~Buffer() { if (raw && hipDeviceSynchronize() == hipSuccess) (void)hipFree(raw); }
    void* data() const { return raw + 256u; }
    void reset() { check(hipMemset(raw, 0xa5, bytes + 512u)); }
    template<class T> void upload(const std::vector<T>& values) {
        if (values.size() * sizeof(T) != bytes) throw std::runtime_error("upload extent");
        original.resize(bytes + 512u, 0xa5);
        std::memcpy(original.data() + 256u, values.data(), bytes);
        check(hipMemcpy(raw, original.data(), original.size(), hipMemcpyHostToDevice));
    }
    std::vector<unsigned char> copy() const {
        std::vector<unsigned char> host(bytes + 512u);
        check(hipMemcpy(host.data(), raw, host.size(), hipMemcpyDeviceToHost)); return host;
    }
    size_t changed() const {
        if (original.empty()) throw std::runtime_error("missing immutable input");
        const auto after = copy(); size_t bad = 0;
        for (size_t i = 0; i < after.size(); ++i) bad += after[i] != original[i];
        return bad;
    }
    size_t guards(const std::vector<unsigned char>& host, size_t used) const {
        size_t bad = 0; if (used > bytes) throw std::runtime_error("output extent");
        for (size_t i = 0; i < 256u; ++i) bad += host[i] != 0xa5u;
        for (size_t i = 256u + used; i < host.size(); ++i) bad += host[i] != 0xa5u;
        return bad;
    }
};
struct Entry { std::string path, digest; size_t offset, bytes; };
std::map<std::string,Entry> plan;
template<class T> std::vector<T> read_role(const std::string& name, size_t count) {
    const auto& e=plan.at(name);
    if(e.bytes!=count*sizeof(T))throw std::runtime_error("role extent: "+name);
    return read<T>(e.path,e.offset,count,e.digest,e.offset==0u);
}
struct Inputs {
    std::vector<std::unique_ptr<Buffer>> buffers;
    template<class T> const T* upload(const std::string& role,size_t count) {
        const auto values=read_role<T>(role,count);
        auto buffer=std::make_unique<Buffer>(values.size()*sizeof(T));buffer->upload(values);
        const auto* result=static_cast<const T*>(buffer->data());buffers.push_back(std::move(buffer));return result;
    }
    size_t changed()const{size_t bad=0;for(const auto& p:buffers)bad+=p->changed();return bad;}
};
struct Comparison {
    std::string name;size_t elements=0,mismatches=0,first=0;uint32_t actual=0,expected=0;
    void add(uint32_t a,uint32_t e,size_t i){++elements;if(a==e)return;if(!mismatches){first=i;actual=a;expected=e;}++mismatches;}
};
std::map<std::string,Comparison> comparisons;
size_t guards=0,immutable=0;
template<class T> void compare(const std::string& name,const Buffer& buffer,bool key_major=false) {
    const auto expected=read_role<T>("expected_"+name,buffer.bytes/sizeof(T));
    const auto host=buffer.copy();guards+=buffer.guards(host,buffer.bytes);
    auto& c=comparisons[name];c.name=name;
    for(size_t i=0;i<expected.size();++i){
        size_t at=i;
        if(key_major){const size_t value=(i/128u)%128u,k=i%128u;at=(i/16384u)*16384u+k*128u+value;}
        uint32_t a=0,e=0;std::memcpy(&a,host.data()+256u+at*sizeof(T),sizeof(T));std::memcpy(&e,&expected[i],sizeof(T));c.add(a,e,i);
    }
}
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("bound TSV plan first-position");
    const unsigned position=static_cast<unsigned>(number(argv[2],263678u));
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("plan missing");std::string line;
    while(std::getline(file,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back();
        std::istringstream input(line);std::vector<std::string> fields;std::string part;
        while(std::getline(input,part,'\t'))fields.push_back(part);
        if(fields.size()!=5u || plan.count(fields[0]) || plan.size()>=16u)throw std::runtime_error("plan fields");
        plan.emplace(fields[0],Entry{fields[1],fields[4],size_t(number(fields[2],uint64_t(8)<<30u)),
            size_t(number(fields[3],uint64_t(1)<<30u))});
    }
    constexpr unsigned vocab=qrt_sm121_mtp::head_vocabulary;
    Inputs inputs;const auto* weights=inputs.upload<uint16_t>("weights",size_t(vocab)*2048u);
    const auto* input=inputs.upload<uint16_t>("input",4096u);
    auto expected=read_role<uint16_t>("expected_0",vocab);const auto second=read_role<uint16_t>("expected_1",vocab);
    expected.insert(expected.end(),second.begin(),second.end());
    qrt_sm121_mtp::HeadBest golden[2];
    for(unsigned row=0;row<2u;++row)for(unsigned token=0;token<vocab;++token)
        if(!qrt_sm121_mtp::head_candidate(expected[size_t(row)*vocab+token],token,&golden[row]))
            throw std::runtime_error("invalid original logit");
    Comparison logits_check;logits_check.name="logits";size_t invalid_flags=0,sampling_errors=0;
    std::vector<qrt_sm121_mtp::HeadBest> measured;
    for(unsigned blocks:{257u,1024u,4096u}){
        Buffer logits(size_t(2u)*vocab*2u),tokens(8u),values(8u),invalid(4u);
        check(qrt_sm121_q2::launch_target_head(weights,input,static_cast<uint16_t*>(logits.data()),
            static_cast<uint32_t*>(tokens.data()),static_cast<float*>(values.data()),static_cast<uint32_t*>(invalid.data()),blocks));
        check(hipDeviceSynchronize());
        const auto output=logits.copy(),ids=tokens.copy(),best=values.copy(),flag=invalid.copy();
        guards+=logits.guards(output,logits.bytes)+tokens.guards(ids,tokens.bytes)+values.guards(best,values.bytes)+invalid.guards(flag,invalid.bytes);
        for(size_t i=0;i<expected.size();++i){uint16_t value;std::memcpy(&value,output.data()+256u+i*2u,2u);logits_check.add(value,expected[i],i);}
        for(unsigned row=0;row<2u;++row){uint32_t id;float value;std::memcpy(&id,ids.data()+256u+row*4u,4u);std::memcpy(&value,best.data()+256u+row*4u,4u);
            sampling_errors+=id!=golden[row].token || value!=golden[row].logit;measured.push_back({id,value});}
        uint32_t bad;std::memcpy(&bad,flag.data()+256u,4u);invalid_flags+=bad!=0u;
    }
    immutable+=inputs.changed();const bool passed=!logits_check.mismatches&&!guards&&!immutable&&!invalid_flags&&!sampling_errors;
    std::cout<<"{\"kind\":\"original_q2_target_head\",\"first_position\":"<<position
        <<",\"configurations\":3,\"maximum_blocks\":[257,1024,4096],\"compared_elements\":"<<logits_check.elements
        <<",\"bit_mismatches\":"<<logits_check.mismatches<<",\"first_difference\":["<<logits_check.first<<','<<logits_check.actual<<','<<logits_check.expected<<']'
        <<",\"guard_errors\":"<<guards<<",\"immutable_input_errors\":"<<immutable<<",\"invalid_flag_errors\":"<<invalid_flags
        <<",\"sampling_errors\":"<<sampling_errors<<",\"expected_samples\":[";
    for(unsigned row=0;row<2u;++row){if(row)std::cout<<',';std::cout<<"{\"token\":"<<golden[row].token<<",\"logit\":"<<golden[row].logit<<'}';}
    std::cout<<"],\"actual_samples\":[";
    for(size_t i=0;i<measured.size();++i){if(i)std::cout<<',';std::cout<<"{\"token\":"<<measured[i].token<<",\"logit\":"<<measured[i].logit<<'}';}
    std::cout<<"],\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"resident_cache_published\":false,\"model_loaded\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
