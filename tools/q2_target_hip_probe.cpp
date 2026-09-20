// Complete continuous two-row target probe. Original expectations stay on the host.
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
#include "native/providers/gdn/sm121_q2_target.h"
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
struct OriginalWeights final:qrt_sm121_mtp::ModelWeightSource {
    Inputs inputs;
    std::array<qrt_sm121_mtp::ModelTensorView,qrt_sm121_q2::target_weight_count> views{};
    explicit OriginalWeights(){
        size_t index=0;
        for(const auto& s:qrt_sm121_q2::target_weight_specs()){
            const auto* p=inputs.role<uint16_t>("weight/"+s.name,s.bytes()/2u);
            views[index++]={s.name.c_str(),p,s.rank,s.shape,s.bytes(),1u,true,true};
            if(index%40u==0u)std::cerr<<"loaded original tensors "<<index<<"/633\n"<<std::flush;
        }
    }
    uint64_t epoch()const noexcept override{return 1u;}
    bool tensor(const char* name,qrt_sm121_mtp::ModelTensorView* output)const override{
        for(const auto& v:views)if(!std::strcmp(v.name,name)){*output=v;return true;}return false;
    }
};
struct TableOwner {
    Inputs inputs;
    qrt_sm121_q2::TargetTables tables;
    TableOwner(){
        auto& t=tables;
        t.beta=inputs.role<float>("beta_table",65536u);t.gated_silu=inputs.role<float>("gated_silu",65536u);
        t.convolution_silu=inputs.role<unsigned char>("conv_silu",qrt_sm121_silu::table_bytes);
        t.attention.rsqrt=inputs.role<unsigned char>("rsqrt",qrt_sm121_rsqrt::table_bytes);
        t.attention.exp2=inputs.role<unsigned char>("exp2",qrt_sm121_exp2::table_bytes);
        t.attention.reciprocal=inputs.role<unsigned char>("rcp",qrt_sm121_attention_rcp::table_bytes);
        const auto& rope=plan.at("rope");
        if(rope.bytes%128u||rope.bytes/128u>qrt_sm121_q2::target_context_limit)throw std::runtime_error("RoPE extent");
        t.attention.rope_rows=unsigned(rope.bytes/128u);t.attention.rope=inputs.role<uint16_t>("rope",rope.bytes/2u);
        t.attention.sigmoid=inputs.role<uint16_t>("sigmoid",65536u);
        t.moe={inputs.role<uint16_t>("silu",65536u+12u)+12u,t.attention.sigmoid,inputs.role<uint32_t>("router_exp",1u<<23u)};
        for(unsigned layer=0;layer<40u;++layer)if(layer%4u!=3u)t.g[layer]=inputs.role<float>("layer-"+std::to_string(layer)+"/g_table",32u*65536u);
    }
};
struct OriginalState final:qrt_sm121_q2::TargetStateSource {
    Inputs inputs;
    std::shared_ptr<const TableOwner> table_owner;
    qrt_sm121_q2::TargetSnapshot state;
    mutable bool isolated=false;
    OriginalState(std::shared_ptr<const TableOwner> tables,unsigned position,uint32_t current,bool fp32,bool key_major)
        :table_owner(std::move(tables)){
        using namespace qrt_sm121_q2;
        state.owner=this;state.generation=1;state.model_epoch=1;state.processed_tokens=position;state.current_token=current;
        state.tables=table_owner->tables;
        for(unsigned layer=0;layer<40u;++layer){
            const std::string prefix="layer-"+std::to_string(layer)+"/";
            if(layer%4u==3u){
                const auto k=read_role<uint16_t>(prefix+"history_k",size_t(position+1u)*512u);
                const auto v=read_role<uint16_t>(prefix+"history_v",size_t(position+1u)*512u);
                const auto bind=[&](const auto& keys,const auto& values){
                    const auto* kp=inputs.upload(keys);const auto* vp=inputs.upload(values);
                    const unsigned split=position/2u;auto& a=state.attention[layer];
                    a.prefix={kp,vp,split,split,512u,unsigned(sizeof(*kp))};
                    if(!split)a.prefix={};
                    a.decoded={kp+size_t(split)*512u,vp+size_t(split)*512u,position-split,position+1u-split,512u,unsigned(sizeof(*kp))};
                };
                if(fp32){
                    std::vector<float> keys(k.size()),values(v.size());
                    for(size_t i=0;i<k.size();++i){keys[i]=qrt_sm121_q1::widen(k[i]);values[i]=qrt_sm121_q1::widen(v[i]);}
                    bind(keys,values);
                }else bind(k,v);
            }else{
                const auto original=read_role<float>(prefix+"initial_state",state_elements);
                std::vector<float> state_values(state_elements);
                for(unsigned h=0;h<32u;++h)for(unsigned value=0;value<128u;++value)for(unsigned key=0;key<128u;++key)
                    state_values[state_offset(h,value,key_major)+key*(key_major?128u:1u)]=original[(h*128u+value)*128u+key];
                const auto ring=read_role<uint16_t>(prefix+"initial_ring",ring_elements);
                auto& linear=state.linear[layer];linear.state=inputs.upload(state_values);linear.key_major=key_major;
                linear.ring_element_bytes=fp32?4u:2u;
                if(fp32){std::vector<float> values(ring.size());for(size_t i=0;i<ring.size();++i)values[i]=qrt_sm121_q1::widen(ring[i]);linear.ring=inputs.upload(values);}
                else linear.ring=inputs.upload(ring);
            }
        }
    }
    bool snapshot(qrt_sm121_q2::TargetSnapshot* output)const override{*output=state;return !isolated;}
    bool matches(const qrt_sm121_q2::TargetSnapshot& other)const noexcept override{
        return !isolated&&other.owner==this&&other.generation==state.generation&&other.model_epoch==1u&&
            other.processed_tokens==state.processed_tokens&&other.current_token==state.current_token;
    }
    void quarantine()const noexcept override{isolated=true;}
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
struct Configuration {
    bool fp32=false,key_major=false;double seconds=0;
    size_t workspace_bytes=0,guards=0,immutable=0,sampling_errors=0;
    std::vector<Comparison> comparisons;
    std::array<uint32_t,2> tokens{};std::array<float,2> logits{};
};
Configuration evaluate(qrt_sm121_q2::Target& target,const qrt_sm121_q2::ModelWeightBinding& binding,
    const std::shared_ptr<const OriginalState>& source,const std::array<uint32_t,2>& inputs,bool fp32,bool key_major){
    using namespace qrt_sm121_q2;
    Configuration result;result.fp32=fp32;result.key_major=key_major;
    TargetResult evaluated;const auto start=std::chrono::steady_clock::now();
    const auto step=target.evaluate(binding,source,inputs,&evaluated,1024u);
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    if(step.status!=hipSuccess)throw std::runtime_error(std::string(step.stage)+": "+hipGetErrorString(step.status));
    if(!evaluated.ready())throw std::runtime_error("target completion receipt");
    result.workspace_bytes=evaluated.allocated_bytes();
    for(unsigned layer=0;layer<40u;++layer){
        const std::string prefix="layer-"+std::to_string(layer)+"/";
        if(layer%4u==3u){
            const auto qkv=read_role<uint16_t>(prefix+"expected_qkv",18432u);
            const auto keys=read_role<uint16_t>(prefix+"expected_k_rope",1024u);
            std::vector<uint16_t> expected(2048u);
            for(unsigned row=0;row<2u;++row){
                std::copy_n(keys.data()+row*512u,512u,expected.data()+row*1024u);
                std::copy_n(qkv.data()+row*9216u+8704u,512u,expected.data()+row*1024u+512u);
            }
            const auto selected=evaluated.attention(layer,2u);
            if(selected.rows!=2u||selected.first_position!=source->state.processed_tokens)throw std::runtime_error("attention selection");
            result.comparisons.push_back(compare(prefix+"kv",selected.key_values,expected));
        }else{
            const auto one=evaluated.linear(layer,1u),two=evaluated.linear(layer,2u);
            if(one.rows!=1u||two.rows!=2u||two.state!=one.state+state_elements||one.ring_element_bytes!=(fp32?4u:2u)||
                static_cast<const unsigned char*>(two.ring)!=static_cast<const unsigned char*>(one.ring)+ring_elements*one.ring_element_bytes)
                throw std::runtime_error("linear selection");
            result.comparisons.push_back(compare(prefix+"states",one.state,read_role<float>(prefix+"expected_states",2u*state_elements),key_major));
            const auto rings=read_role<uint16_t>(prefix+"expected_rings",2u*ring_elements);
            if(fp32){
                std::vector<float> expected(rings.size());for(size_t i=0;i<rings.size();++i)expected[i]=qrt_sm121_q1::widen(rings[i]);
                result.comparisons.push_back(compare(prefix+"rings",static_cast<const float*>(one.ring),expected));
            }else result.comparisons.push_back(compare(prefix+"rings",static_cast<const uint16_t*>(one.ring),rings));
        }
    }
    const auto expected_norm=read_role<uint16_t>("expected_final_norm",4096u);
    Comparison norm;norm.name="final_norm";
    const auto& host=*evaluated.host();for(size_t i=0;i<4096u;++i)norm.add(host.normalized[i],expected_norm[i],i);
    result.comparisons.push_back(norm);
    const auto expected_logits=read_role<uint16_t>("expected_logits",size_t(2u)*qrt_sm121_mtp::head_vocabulary);
    result.comparisons.push_back(compare("logits",evaluated.vocabulary_logits(),expected_logits));
    result.tokens=host.tokens;result.logits=host.logits;
    for(unsigned row=0;row<2u;++row){
        qrt_sm121_mtp::HeadBest best;
        for(unsigned token=0;token<qrt_sm121_mtp::head_vocabulary;++token)
            if(!qrt_sm121_mtp::head_candidate(expected_logits[size_t(row)*qrt_sm121_mtp::head_vocabulary+token],token,&best))
                throw std::runtime_error("nonfinite original logits");
        result.sampling_errors+=(host.tokens[row]!=best.token)+(host.logits[row]!=best.logit);
    }
    const auto checks=source->inputs.verify();result.guards=checks.first;result.immutable=checks.second;
    return result;
}
int main(int argc,char** argv)try{
    if(argc!=5)throw std::runtime_error("bound TSV plan first-position first-input second-input");
    const unsigned position=unsigned(number(argv[2],263678u));
    const std::array<uint32_t,2> inputs{uint32_t(number(argv[3],248319u)),uint32_t(number(argv[4],248319u))};
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
    auto original=std::make_shared<OriginalWeights>();qrt_sm121_q2::ModelWeights model;
    const auto prepared=model.prepare(original,1u);if(prepared.status!=hipSuccess)throw std::runtime_error(prepared.stage);
    auto binding=model.binding(1u);auto tables=std::make_shared<TableOwner>();qrt_sm121_q2::Target target;
    std::vector<Configuration> configurations;
    for(bool fp32:{false,true})for(bool key_major:{false,true}){
        auto source=std::make_shared<OriginalState>(tables,position,inputs[0],fp32,key_major);
        configurations.push_back(evaluate(target,binding,source,inputs,fp32,key_major));
        std::cerr<<"completed target configuration "<<configurations.size()<<"/4\n"<<std::flush;
    }
    auto guards=original->inputs.verify();auto table_guards=tables->inputs.verify();
    size_t guard_errors=guards.first+table_guards.first,immutable_errors=guards.second+table_guards.second;
    size_t elements=0,mismatches=0,sampling_errors=0;unsigned count=0;
    std::cout<<std::setprecision(17)<<"{\"kind\":\"original_q2_continuous_target\",\"first_position\":"<<position
        <<",\"scheduled_inputs\":["<<inputs[0]<<','<<inputs[1]<<"],\"original_weight_count\":633,\"configurations\":[";
    for(const auto& c:configurations){
        if(count++)std::cout<<',';guard_errors+=c.guards;immutable_errors+=c.immutable;sampling_errors+=c.sampling_errors;
        std::cout<<"{\"fp32_carriers\":"<<(c.fp32?"true":"false")<<",\"key_major\":"<<(c.key_major?"true":"false")
            <<",\"execution_seconds\":"<<c.seconds<<",\"workspace_bytes\":"<<c.workspace_bytes<<",\"samples\":[";
        for(unsigned row=0;row<2u;++row){if(row)std::cout<<',';std::cout<<"{\"token\":"<<c.tokens[row]<<",\"logit\":"<<c.logits[row]<<'}';}
        std::cout<<"],\"stages\":[";unsigned n=0;
        for(const auto& s:c.comparisons){
            if(n++)std::cout<<',';elements+=s.elements;mismatches+=s.mismatches;
            std::cout<<"{\"stage\":\""<<s.name<<"\",\"elements\":"<<s.elements<<",\"bit_mismatches\":"<<s.mismatches
                <<",\"first_difference\":["<<s.first<<','<<s.actual<<','<<s.expected<<"]}";
        }
        std::cout<<"]}";
    }
    const bool passed=!guard_errors&&!immutable_errors&&!mismatches&&!sampling_errors;
    std::cout<<"],\"compared_elements\":"<<elements<<",\"bit_mismatches\":"<<mismatches
        <<",\"guard_errors\":"<<guard_errors<<",\"immutable_digest_errors\":"<<immutable_errors
        <<",\"sampling_errors\":"<<sampling_errors<<",\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"all_target_weights_loaded\":true,\"resident_cache_published\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
