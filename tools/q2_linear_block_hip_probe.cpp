// Complete private two-row linear block. Original expected tensors stay on the host.
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
#include "native/providers/gdn/sm121_q2_linear_block.h"
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
        !count || bytes > (256u << 20u) || offset > uint64_t(file.tellg()) ||
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
void compare_float_rings(const Buffer& buffer){
    const auto expected=read_role<uint16_t>("expected_rings",2u*qrt_sm121_q2::ring_elements);
    const auto host=buffer.copy();guards+=buffer.guards(host,buffer.bytes);auto& c=comparisons["rings"];c.name="rings";
    for(size_t i=0;i<expected.size();++i){uint32_t a;std::memcpy(&a,host.data()+256u+i*4u,4u);c.add(a,uint32_t(expected[i])<<16u,i);}
}
template<class Element> void execute(const qrt_sm121_q2::LinearBlockViews<Element>& borrowed,
    const qrt_sm121_q2::LinearBlockTables& tables,bool key_major,size_t position){
    using namespace qrt_sm121_q2;
    auto v=borrowed;const auto state=read_role<float>("initial_state",state_elements);
    const auto ring=read_role<uint16_t>("initial_ring",ring_elements);
    std::vector<float> staged_input(state_elements);std::vector<Element> ring_input(ring_elements);
    for(unsigned h=0;h<32u;++h)for(unsigned value=0;value<128u;++value)for(unsigned k=0;k<128u;++k)
        staged_input[state_offset(h,value,key_major)+k*(key_major?128u:1u)]=state[(h*128u+value)*128u+k];
    for(size_t i=0;i<ring.size();++i)ring_input[i]=ring_element<Element>(ring[i]);
    Buffer initial(staged_input.size()*4u),initial_ring(ring_input.size()*sizeof(Element));
    initial.upload(staged_input);initial_ring.upload(ring_input);
    Buffer qkv(2u*8192u*2u),z(2u*4096u*2u),a(2u*32u*2u),b(2u*32u*2u),gated(2u*4096u*2u),out(2u*2048u*2u);
    Buffer rings(2u*ring_elements*sizeof(Element)),conv(2u*8192u*2u),states(staged_state_bytes),core(staged_core_bytes);
    v.qkv=static_cast<uint16_t*>(qkv.data());v.z=static_cast<uint16_t*>(z.data());v.a=static_cast<uint16_t*>(a.data());v.b=static_cast<uint16_t*>(b.data());
    v.gated=static_cast<uint16_t*>(gated.data());v.output=static_cast<uint16_t*>(out.data());
    v.convolution.qkv=v.qkv;v.convolution.initial_ring=static_cast<const Element*>(initial_ring.data());
    v.convolution.staged_rings=static_cast<Element*>(rings.data());v.convolution.staged_convolution=static_cast<uint16_t*>(conv.data());v.convolution.first_position=position;
    v.recurrent={v.convolution.staged_convolution,v.a,v.b,static_cast<const float*>(initial.data()),static_cast<float*>(states.data()),static_cast<uint16_t*>(core.data()),key_major};
    check(launch_linear_block(v,tables));check(hipDeviceSynchronize());
    compare<uint16_t>("qkv",qkv);compare<uint16_t>("z",z);compare<uint16_t>("a",a);compare<uint16_t>("b",b);
    compare<uint16_t>("convolution",conv);compare<float>("states",states,key_major);compare<uint16_t>("core",core);
    compare<uint16_t>("gated",gated);compare<uint16_t>("output",out);
    if constexpr(std::is_same<Element,uint16_t>::value)compare<uint16_t>("rings",rings);else compare_float_rings(rings);
    for(unsigned accepted:{1u,2u}){
        const auto selected=accepted_linear(v.convolution,v.recurrent,accepted);
        if(selected.state!=v.recurrent.staged_states+(accepted-1u)*state_elements ||
           selected.ring!=v.convolution.staged_rings+(accepted-1u)*ring_elements || selected.rows!=accepted)
            throw std::runtime_error("accepted private selection");
    }
    immutable+=initial.changed()+initial_ring.changed();
}
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("bound TSV plan first-position");
    const size_t position=number(argv[2],263678u);
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("plan missing");std::string line;
    while(std::getline(file,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back();
        std::istringstream input(line);std::vector<std::string> f;std::string part;
        while(std::getline(input,part,'\t'))f.push_back(part);
        if(f.size()!=5u || plan.count(f[0]) || plan.size()>=40u)throw std::runtime_error("plan fields");
        plan.emplace(f[0],Entry{f[1],f[4],size_t(number(f[2],uint64_t(8)<<30u)),size_t(number(f[3],256u<<20u))});
    }
    Inputs inputs;
    const auto* x=inputs.upload<uint16_t>("input",4096u);
    const auto* qkv=inputs.upload<uint16_t>("qkv_weights",8192u*2048u);
    const auto* z=inputs.upload<uint16_t>("z_weights",4096u*2048u);
    const auto* a=inputs.upload<uint16_t>("a_weights",32u*2048u);
    const auto* b=inputs.upload<uint16_t>("b_weights",32u*2048u);
    const auto* output=inputs.upload<uint16_t>("output_weights",2048u*4096u);
    const auto* norm=inputs.upload<uint16_t>("norm_weights",128u);
    const auto* conv_weights=inputs.upload<uint16_t>("conv_weights",8192u*4u);
    const auto* conv_silu=inputs.upload<unsigned char>("conv_silu",qrt_sm121_silu::table_bytes);
    using namespace qrt_sm121_q2;
    const LinearBlockTables tables{{inputs.upload<float>("g_table",32u*65536u),inputs.upload<float>("beta_table",65536u),
        inputs.upload<unsigned char>("exp2",qrt_sm121_exp2::table_bytes),inputs.upload<unsigned char>("rsqrt",qrt_sm121_rsqrt::table_bytes)},
        inputs.upload<float>("gated_silu",65536u)};
    const auto run=[&](auto element,bool key_major){
        LinearBlockViews<decltype(element)> v;v.normalized_input=x;v.qkv_weights=qkv;v.z_weights=z;v.a_weights=a;v.b_weights=b;
        v.output_weights=output;v.norm_weights=norm;v.convolution.weights=conv_weights;v.convolution.silu=conv_silu;
        execute(v,tables,key_major,position);
    };
    run(float{},false);run(float{},true);run(uint16_t{},false);run(uint16_t{},true);
    immutable+=inputs.changed();size_t total=0,mismatches=0;unsigned count=0;
    std::cout<<"{\"kind\":\"original_q2_complete_linear_block\",\"first_position\":"<<position<<",\"configurations\":4,\"stages\":[";
    for(const auto& entry:comparisons){const auto& c=entry.second;if(count++)std::cout<<',';total+=c.elements;mismatches+=c.mismatches;
        std::cout<<"{\"stage\":\""<<c.name<<"\",\"elements\":"<<c.elements<<",\"bit_mismatches\":"<<c.mismatches
            <<",\"first_difference\":["<<c.first<<','<<c.actual<<','<<c.expected<<"]}";
    }
    const bool passed=!mismatches&&!guards&&!immutable;
    std::cout<<"],\"compared_elements\":"<<total<<",\"bit_mismatches\":"<<mismatches<<",\"guard_errors\":"<<guards
        <<",\"immutable_input_errors\":"<<immutable<<",\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"resident_cache_published\":false,\"model_loaded\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
