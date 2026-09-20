// Complete private two-row target linear layer. Original expected tensors stay on the host.
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
#include "native/providers/gdn/sm121_q2_linear_layer.h"
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
void compare_float_rings(const Buffer& buffer){
    const auto expected=read_role<uint16_t>("expected_rings",2u*qrt_sm121_q2::ring_elements);
    const auto host=buffer.copy();guards+=buffer.guards(host,buffer.bytes);auto& c=comparisons["rings"];c.name="rings";
    for(size_t i=0;i<expected.size();++i){uint32_t a;std::memcpy(&a,host.data()+256u+i*4u,4u);c.add(a,uint32_t(expected[i])<<16u,i);}
}
size_t invalid_flags=0;
void compare_moe(const Buffer& workspace) {
    using namespace qrt_sm121_mtp;
    MoeBuffers b;
    if(!bind_moe_buffers(workspace.data(),workspace.bytes,2u,&b))throw std::runtime_error("MoE binding");
    const auto host=workspace.copy();guards+=workspace.guards(host,workspace.bytes);
    std::vector<unsigned char> written(workspace.bytes,0u);
    const auto mark=[&](const void* pointer,size_t bytes){
        const size_t at=static_cast<const unsigned char*>(pointer)-static_cast<const unsigned char*>(workspace.data());
        if(at>workspace.bytes || bytes>workspace.bytes-at)throw std::runtime_error("MoE subspan");
        std::fill(written.begin()+at,written.begin()+at+bytes,1u);return at;
    };
    const auto compare_bf16=[&](const char* name,const uint16_t* pointer,size_t count){
        const size_t at=mark(pointer,count*2u);const auto expected=read_role<uint16_t>(std::string("expected_")+name,count);
        auto& c=comparisons[name];c.name=name;
        for(size_t i=0;i<count;++i){uint16_t value;std::memcpy(&value,host.data()+256u+at+i*2u,2u);c.add(value,expected[i],i);}
    };
    compare_bf16("moe_router",b.router,512u);compare_bf16("moe_shared_gate",b.shared_gate,2u);
    compare_bf16("moe_shared_gate_up",b.shared_gate_up,2048u);compare_bf16("moe_shared_activated",b.shared_activated,1024u);
    compare_bf16("moe_shared_down",b.shared_down,4096u);compare_bf16("moe_shared",b.shared,4096u);
    compare_bf16("moe_routed",b.routed,4096u);compare_bf16("moe_output",b.output,4096u);
    // The original target capture retains the routed sum, not individual routed
    // expert activations. Mark their exact private extents; compare the resulting
    // sum and final output without feeding any reference into these producers.
    mark(b.routed_gate_up,2u*8192u*2u);mark(b.routed_activated,2u*4096u*2u);mark(b.routed_weighted,2u*16384u*2u);
    const auto ids=read_role<uint32_t>("expected_moe_topk_ids",16u);
    const auto weights=read_role<float>("expected_moe_topk_weights",16u);
    const size_t ids_at=mark(b.topk_ids,64u),weights_at=mark(b.topk_weights,64u);
    for(size_t i=0;i<16u;++i){
        uint32_t actual_id,actual_weight,expected_weight;
        std::memcpy(&actual_id,host.data()+256u+ids_at+i*4u,4u);
        std::memcpy(&actual_weight,host.data()+256u+weights_at+i*4u,4u);
        std::memcpy(&expected_weight,&weights[i],4u);
        auto& id=comparisons["moe_topk_ids"];id.name="moe_topk_ids";id.add(actual_id,ids[i],i);
        auto& weight=comparisons["moe_topk_weights"];weight.name="moe_topk_weights";weight.add(actual_weight,expected_weight,i);
    }
    uint32_t invalid;const size_t at=mark(b.invalid,4u);std::memcpy(&invalid,host.data()+256u+at,4u);
    invalid_flags+=invalid!=0u;
    for(size_t i=0;i<workspace.bytes;++i)if(!written[i])guards+=host[256u+i]!=0xa5u;
}

template<class Element> void execute(const qrt_sm121_q2::LinearLayerViews<Element>& borrowed,
    const qrt_sm121_q2::LinearLayerTables& tables,bool key_major,size_t position) {
    using namespace qrt_sm121_q2;
    auto v=borrowed;auto& l=v.linear;
    const auto state=read_role<float>("initial_state",state_elements);
    const auto ring=read_role<uint16_t>("initial_ring",ring_elements);
    std::vector<float> staged_input(state_elements);std::vector<Element> ring_input(ring_elements);
    for(unsigned h=0;h<32u;++h)for(unsigned value=0;value<128u;++value)for(unsigned k=0;k<128u;++k)
        staged_input[state_offset(h,value,key_major)+k*(key_major?128u:1u)]=state[(h*128u+value)*128u+k];
    for(size_t i=0;i<ring.size();++i)ring_input[i]=ring_element<Element>(ring[i]);
    Buffer initial(staged_input.size()*4u),initial_ring(ring_input.size()*sizeof(Element));
    initial.upload(staged_input);initial_ring.upload(ring_input);
    Buffer input_norm(8192u),input_residual(8192u),moe_input(8192u),output_residual(8192u);
    Buffer qkv(2u*8192u*2u),z(2u*4096u*2u),a(2u*32u*2u),b(2u*32u*2u),gated(2u*4096u*2u),out(8192u);
    Buffer rings(2u*ring_elements*sizeof(Element)),conv(2u*8192u*2u),states(staged_state_bytes),core(staged_core_bytes);
    Buffer moe(qrt_sm121_mtp::moe_workspace_bytes(2u));
    v.normalized_input=static_cast<uint16_t*>(input_norm.data());v.input_residual=static_cast<uint16_t*>(input_residual.data());
    v.moe_input=static_cast<uint16_t*>(moe_input.data());v.output_residual=static_cast<uint16_t*>(output_residual.data());
    v.moe_workspace=moe.data();v.moe_workspace_bytes=moe.bytes;l.normalized_input=v.normalized_input;
    l.qkv=static_cast<uint16_t*>(qkv.data());l.z=static_cast<uint16_t*>(z.data());l.a=static_cast<uint16_t*>(a.data());l.b=static_cast<uint16_t*>(b.data());
    l.gated=static_cast<uint16_t*>(gated.data());l.output=static_cast<uint16_t*>(out.data());
    l.convolution.qkv=l.qkv;l.convolution.initial_ring=static_cast<const Element*>(initial_ring.data());
    l.convolution.staged_rings=static_cast<Element*>(rings.data());l.convolution.staged_convolution=static_cast<uint16_t*>(conv.data());l.convolution.first_position=position;
    l.recurrent={l.convolution.staged_convolution,l.a,l.b,static_cast<const float*>(initial.data()),static_cast<float*>(states.data()),static_cast<uint16_t*>(core.data()),key_major};
    check(launch_linear_layer(v,tables));check(hipDeviceSynchronize());
    compare<uint16_t>("input_norm",input_norm);guards+=input_residual.guards(input_residual.copy(),input_residual.bytes);
    compare<uint16_t>("qkv",qkv);compare<uint16_t>("z",z);compare<uint16_t>("a",a);compare<uint16_t>("b",b);
    compare<uint16_t>("convolution",conv);compare<float>("states",states,key_major);compare<uint16_t>("core",core);
    compare<uint16_t>("gated",gated);compare<uint16_t>("output",out);
    if constexpr(std::is_same<Element,uint16_t>::value)compare<uint16_t>("rings",rings);else compare_float_rings(rings);
    compare<uint16_t>("moe_input",moe_input);compare<uint16_t>("output_residual",output_residual);compare_moe(moe);
    for(unsigned accepted:{1u,2u}){
        const auto selected=accepted_linear(l.convolution,l.recurrent,accepted);
        if(selected.state!=l.recurrent.staged_states+(accepted-1u)*state_elements ||
           selected.ring!=l.convolution.staged_rings+(accepted-1u)*ring_elements || selected.rows!=accepted)
            throw std::runtime_error("accepted private selection");
    }
    immutable+=initial.changed()+initial_ring.changed();
}

int main(int argc,char** argv)try {
    if(argc!=3)throw std::runtime_error("bound TSV plan first-position");
    const size_t position=number(argv[2],263678u);
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("plan missing");std::string line;
    while(std::getline(file,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back();
        std::istringstream input(line);std::vector<std::string> fields;std::string part;
        while(std::getline(input,part,'\t'))fields.push_back(part);
        if(fields.size()!=5u || plan.count(fields[0]) || plan.size()>=64u)throw std::runtime_error("plan fields");
        plan.emplace(fields[0],Entry{fields[1],fields[4],size_t(number(fields[2],uint64_t(8)<<30u)),
            size_t(number(fields[3],uint64_t(1)<<30u))});
    }
    Inputs inputs;
    const uint16_t* hidden=nullptr;
    if(plan.count("hidden"))hidden=inputs.upload<uint16_t>("hidden",4096u);
    else {
        auto values=read_role<uint16_t>("embedding_0",2048u);
        const auto second=read_role<uint16_t>("embedding_1",2048u);
        values.insert(values.end(),second.begin(),second.end());
        auto owner=std::make_unique<Buffer>(8192u);owner->upload(values);
        hidden=static_cast<const uint16_t*>(owner->data());inputs.buffers.push_back(std::move(owner));
    }
    const auto* residual=inputs.upload<uint16_t>("residual",4096u);
    const auto* input_norm=inputs.upload<uint16_t>("input_norm_weights",2048u);
    const auto* post_norm=inputs.upload<uint16_t>("post_norm_weights",2048u);
    const auto* qkv=inputs.upload<uint16_t>("qkv_weights",8192u*2048u);
    const auto* z=inputs.upload<uint16_t>("z_weights",4096u*2048u);
    const auto* a=inputs.upload<uint16_t>("a_weights",32u*2048u);
    const auto* b=inputs.upload<uint16_t>("b_weights",32u*2048u);
    const auto* output=inputs.upload<uint16_t>("output_weights",2048u*4096u);
    const auto* norm=inputs.upload<uint16_t>("norm_weights",128u);
    const auto* conv_weights=inputs.upload<uint16_t>("conv_weights",8192u*4u);
    const auto* conv_silu=inputs.upload<unsigned char>("conv_silu",qrt_sm121_silu::table_bytes);
    const auto* router=inputs.upload<uint16_t>("router_weights",256u*2048u);
    const auto* shared_gate=inputs.upload<uint16_t>("shared_gate_weights",2048u);
    auto gate_up=read_role<uint16_t>("shared_gate_up_gate_weights",512u*2048u);
    const auto up=read_role<uint16_t>("shared_gate_up_up_weights",512u*2048u);
    gate_up.insert(gate_up.end(),up.begin(),up.end());
    auto shared_gate_up_owner=std::make_unique<Buffer>(gate_up.size()*2u);shared_gate_up_owner->upload(gate_up);
    const auto* shared_gate_up=static_cast<const uint16_t*>(shared_gate_up_owner->data());
    inputs.buffers.push_back(std::move(shared_gate_up_owner));
    const auto* shared_down=inputs.upload<uint16_t>("shared_down_weights",2048u*512u);
    const auto* routed_gate_up=inputs.upload<uint16_t>("routed_gate_up_weights",size_t(256u)*1024u*2048u);
    const auto* routed_down=inputs.upload<uint16_t>("routed_down_weights",size_t(256u)*2048u*512u);
    using namespace qrt_sm121_q2;
    const LinearLayerTables tables{
        {{inputs.upload<float>("g_table",32u*65536u),inputs.upload<float>("beta_table",65536u),
          inputs.upload<unsigned char>("exp2",qrt_sm121_exp2::table_bytes),
          inputs.upload<unsigned char>("rsqrt",qrt_sm121_rsqrt::table_bytes)},
         inputs.upload<float>("gated_silu",65536u)},
        {inputs.upload<uint16_t>("silu",65536u+12u)+12u,inputs.upload<uint16_t>("sigmoid",65536u),
         inputs.upload<uint32_t>("router_exp",1u<<23u)}};
    const auto run=[&](auto element,bool key_major){
        LinearLayerViews<decltype(element)> v;
        v.hidden=hidden;v.residual=residual;v.input_norm_weights=input_norm;v.post_norm_weights=post_norm;
        v.linear.qkv_weights=qkv;v.linear.z_weights=z;v.linear.a_weights=a;v.linear.b_weights=b;
        v.linear.output_weights=output;v.linear.norm_weights=norm;
        v.linear.convolution.weights=conv_weights;v.linear.convolution.silu=conv_silu;
        v.moe_weights={router,shared_gate,shared_gate_up,shared_down,routed_gate_up,routed_down};
        execute(v,tables,key_major,position);
    };
    run(float{},false);run(float{},true);run(uint16_t{},false);run(uint16_t{},true);
    immutable+=inputs.changed();size_t total=0,mismatches=0;unsigned count=0;
    std::cout<<"{\"kind\":\"original_q2_complete_linear_layer\",\"first_position\":"<<position<<",\"configurations\":4,\"stages\":[";
    for(const auto& entry:comparisons){
        const auto& c=entry.second;if(count++)std::cout<<',';total+=c.elements;mismatches+=c.mismatches;
        std::cout<<"{\"stage\":\""<<c.name<<"\",\"elements\":"<<c.elements<<",\"bit_mismatches\":"<<c.mismatches
            <<",\"first_difference\":["<<c.first<<','<<c.actual<<','<<c.expected<<"]}";
    }
    const bool passed=!mismatches&&!guards&&!immutable&&!invalid_flags;
    std::cout<<"],\"compared_elements\":"<<total<<",\"bit_mismatches\":"<<mismatches<<",\"guard_errors\":"<<guards
        <<",\"immutable_input_errors\":"<<immutable<<",\"invalid_flag_errors\":"<<invalid_flags
        <<",\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"resident_cache_published\":false,\"model_loaded\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
