// Compare independent one-row output-head GEMVs and BF16 greedy sampling.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_mtp_head.h"

void check(hipError_t status) {
    if (status != hipSuccess) throw std::runtime_error(hipGetErrorString(status));
}
std::vector<unsigned char> region(const std::string& path, uint64_t offset, size_t bytes) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || uint64_t(file.tellg()) > (uint64_t(4) << 30u) ||
        offset > uint64_t(file.tellg()) || !bytes || bytes > (size_t(1) << 30u) ||
        bytes > uint64_t(file.tellg()) - offset) throw std::runtime_error("input extent: " + path);
    std::vector<unsigned char> data(bytes);
    file.seekg(static_cast<std::streamoff>(offset));
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (!file) throw std::runtime_error("short read: " + path);
    return data;
}
std::vector<unsigned char> whole(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || file.tellg() > (16u << 20u))
        throw std::runtime_error("captured file extent: " + path);
    return region(path, 0u, static_cast<size_t>(file.tellg()));
}
struct Input {
    unsigned char* raw = nullptr;
    unsigned char* data = nullptr;
    size_t bytes = 0;
    std::string path;
    uint64_t offset = 0;
};
struct Scratch {
    std::vector<void*> allocations;
    std::vector<Input> inputs;
    ~Scratch() {
        if (hipDeviceSynchronize() != hipSuccess) return;
        for (void* pointer : allocations) (void)hipFree(pointer);
    }
    unsigned char* allocate(size_t bytes) {
        unsigned char* pointer = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&pointer), bytes));
        allocations.push_back(pointer);
        return pointer;
    }
    void* upload(const std::string& path, uint64_t offset, size_t bytes) {
        auto host = region(path, offset, bytes);
        auto* raw = allocate(bytes + 512u);
        check(hipMemset(raw, 0xa5, 256u));
        check(hipMemset(raw + 256u + bytes, 0xa5, 256u));
        check(hipMemcpy(raw + 256u, host.data(), bytes, hipMemcpyHostToDevice));
        inputs.push_back({raw, raw + 256u, bytes, path, offset});
        return raw + 256u;
    }
    size_t immutable_mismatches() const {
        size_t bad = 0;
        for (const auto& input : inputs) {
            for (size_t first = 0; first < input.bytes; first += 4u << 20u) {
                const size_t count = std::min(size_t(4u << 20u), input.bytes - first);
                const auto expected = region(input.path, input.offset + first, count);
                auto actual = expected;
                check(hipMemcpy(actual.data(), input.data + first, count, hipMemcpyDeviceToHost));
                for (size_t i = 0; i < count; ++i) bad += actual[i] != expected[i];
            }
            unsigned char guard[512];
            check(hipMemcpy(guard, input.raw, 256u, hipMemcpyDeviceToHost));
            check(hipMemcpy(guard + 256u, input.data + input.bytes, 256u, hipMemcpyDeviceToHost));
            for (unsigned char value : guard) bad += value != 0xa5u;
        }
        return bad;
    }
};
struct Comparison {
    size_t elements = 0, mismatches = 0, first_row = 0;
    unsigned first_column = 0, first_actual = 0, first_expected = 0;
    void add(uint32_t actual, uint32_t expected, size_t row, unsigned column) {
        ++elements;
        if (actual == expected) return;
        if (!mismatches) { first_row = row; first_column = column; first_actual = actual; first_expected = expected; }
        ++mismatches;
    }
};


struct Guarded {
    unsigned char* raw;unsigned char* data;size_t bytes;
    Guarded(Scratch& owner,size_t count):raw(owner.allocate(count+512u)),data(raw+256u),bytes(count){}
    void reset(){check(hipMemset(raw,0xa5,bytes+512u));}
    std::vector<unsigned char> copy()const {
        std::vector<unsigned char> host(bytes+512u);check(hipMemcpy(host.data(),raw,host.size(),hipMemcpyDeviceToHost));return host;
    }
    size_t guards_and_unused(const std::vector<unsigned char>& host,size_t used)const{
        if(used>bytes)throw std::runtime_error("output extent");size_t bad=0;
        for(size_t i=0;i<256u;++i)bad+=host[i]!=0xa5u;
        for(size_t i=256u+used;i<host.size();++i)bad+=host[i]!=0xa5u;
        return bad;
    }
};
int main(int argc,char** argv)try{
    if(argc!=4)throw std::runtime_error("directory lm_head_shard absolute_offset");
    using namespace qrt_sm121_mtp;
    const std::string root=argv[1];const auto input=whole(root+"/input.bin");
    const auto expected=whole(root+"/logits.bin");
    const unsigned rows=static_cast<unsigned>(input.size()/4096u);
    if(!rows||rows>32u||input.size()!=size_t(rows)*4096u||expected.size()!=size_t(rows)*head_vocabulary*2u)
        throw std::runtime_error("original head rows");
    Scratch scratch;
    const auto* weights=static_cast<const uint16_t*>(scratch.upload(argv[2],std::stoull(argv[3]),size_t(head_vocabulary)*4096u));
    const auto* hidden=static_cast<const uint16_t*>(scratch.upload(root+"/input.bin",0u,input.size()));
    Guarded logits(scratch,size_t(2u)*head_vocabulary*2u),tokens(scratch,8u),values(scratch,8u),invalid(scratch,4u);
    const auto invoke=[&](const uint16_t* x,unsigned count,unsigned blocks){
        return launch_head(weights,x,reinterpret_cast<uint16_t*>(logits.data),reinterpret_cast<uint32_t*>(tokens.data),
            reinterpret_cast<float*>(values.data),reinterpret_cast<uint32_t*>(invalid.data),count,blocks);
    };
    for(auto* b:{&logits,&tokens,&values,&invalid})b->reset();
    unsigned rejected=0;
    const auto reject=[&](hipError_t result){if(result!=hipErrorInvalidValue)throw std::runtime_error("bad head contract accepted");++rejected;};
    reject(invoke(hidden,0u,1024u));reject(invoke(hidden,3u,1024u));
    reject(invoke(hidden,1u,0u));reject(invoke(hidden,1u,4097u));reject(invoke(nullptr,1u,1024u));
    reject(invoke(reinterpret_cast<const uint16_t*>(logits.data+128u),1u,1024u));
    reject(launch_head(nullptr,hidden,reinterpret_cast<uint16_t*>(logits.data),reinterpret_cast<uint32_t*>(tokens.data),
        reinterpret_cast<float*>(values.data),reinterpret_cast<uint32_t*>(invalid.data),1u));
    reject(launch_head(weights,hidden,reinterpret_cast<uint16_t*>(logits.data),reinterpret_cast<uint32_t*>(logits.data+16u),
        reinterpret_cast<float*>(values.data),reinterpret_cast<uint32_t*>(invalid.data),1u));
    reject(launch_head(weights,hidden,reinterpret_cast<uint16_t*>(logits.data),reinterpret_cast<uint32_t*>(tokens.data),
        reinterpret_cast<float*>(values.data),reinterpret_cast<uint32_t*>(tokens.data),1u));
    check(hipDeviceSynchronize());
    for(auto* b:{&logits,&tokens,&values,&invalid})
        if(b->guards_and_unused(b->copy(),0u))throw std::runtime_error("rejected head submission wrote output");
    Comparison projection,selected_ids,selected_logits;size_t guards=0,flags=0;unsigned calls=0;
    for(unsigned blocks:{7u,1024u})for(unsigned batch:{1u,2u})for(unsigned first=0;first<rows;){
        const unsigned count=std::min(batch,rows-first);
        for(auto* b:{&logits,&tokens,&values,&invalid})b->reset();
        check(invoke(hidden+size_t(first)*2048u,count,blocks));check(hipDeviceSynchronize());++calls;
        const auto scores=logits.copy(),ids=tokens.copy(),raw_values=values.copy(),flag=invalid.copy();
        guards+=logits.guards_and_unused(scores,size_t(count)*head_vocabulary*2u);
        guards+=tokens.guards_and_unused(ids,count*4u)+values.guards_and_unused(raw_values,count*4u);
        guards+=invalid.guards_and_unused(flag,4u);uint32_t device_flag=0;
        std::memcpy(&device_flag,flag.data()+256u,4u);flags+=device_flag!=0u;
        for(unsigned row=0;row<count;++row){
            HeadBest actual_best,wanted_best;
            for(unsigned token=0;token<head_vocabulary;++token){
                uint16_t actual=0,wanted=0;
                std::memcpy(&actual,scores.data()+256u+(size_t(row)*head_vocabulary+token)*2u,2u);
                std::memcpy(&wanted,expected.data()+(size_t(first+row)*head_vocabulary+token)*2u,2u);
                projection.add(actual,wanted,first+row,token);
                if(!head_candidate(actual,token,&actual_best)||!head_candidate(wanted,token,&wanted_best))
                    throw std::runtime_error("nonfinite original or generated logits");
            }
            uint32_t actual_id=0,actual_bits=0,wanted_bits=0;
            std::memcpy(&actual_id,ids.data()+256u+row*4u,4u);
            std::memcpy(&actual_bits,raw_values.data()+256u+row*4u,4u);
            std::memcpy(&wanted_bits,&wanted_best.logit,4u);
            selected_ids.add(actual_id,wanted_best.token,first+row,0u);
            selected_logits.add(actual_bits,wanted_bits,first+row,0u);
            uint32_t own_bits=0;std::memcpy(&own_bits,&actual_best.logit,4u);
            if(actual_id!=actual_best.token||actual_bits!=own_bits)throw std::runtime_error("device argmax differs from actual produced logits");
        }
        first+=count;
    }
    const size_t input_bad=scratch.immutable_mismatches();
    const bool passed=!(projection.mismatches||selected_ids.mismatches||selected_logits.mismatches||guards||flags||input_bad);
    std::cout<<"{\"kind\":\"original_mtp_head_native\",\"rows\":"<<rows<<",\"calls\":"<<calls
        <<",\"configurations\":4,\"row_batch_configurations\":[1,2],\"maximum_blocks\":[7,1024],\"projection_elements\":"<<projection.elements
        <<",\"projection_mismatches\":"<<projection.mismatches<<",\"first_projection_difference\":";
    if(projection.mismatches)std::cout<<"{\"row\":"<<projection.first_row<<",\"token\":"<<projection.first_column
        <<",\"actual\":"<<projection.first_actual<<",\"expected\":"<<projection.first_expected<<'}';
    else std::cout<<"null";
    std::cout<<",\"selected_id_mismatches\":"<<selected_ids.mismatches<<",\"selected_logit_mismatches\":"<<selected_logits.mismatches
        <<",\"selected_rows_compared\":"<<selected_ids.elements<<",\"guard_and_unused_mismatches\":"<<guards
        <<",\"invalid_device_flags\":"<<flags<<",\"immutable_input_byte_mismatches\":"<<input_bad
        <<",\"rejected_launch_cases\":"<<rejected<<",\"passed\":"<<(passed?"true":"false")
        <<",\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
