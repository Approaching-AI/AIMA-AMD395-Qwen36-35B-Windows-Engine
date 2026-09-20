// Complete private two-row target attention layer. Original expected tensors stay on the host.
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
#include "native/providers/gdn/sm121_q2_attention_layer.h"
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
size_t invalid_flags=0,mask_errors=0,nonfinite=0;
void compare_values(const std::string& name,const std::vector<uint16_t>& actual,const std::vector<uint16_t>& expected){
    if(actual.size()!=expected.size())throw std::runtime_error("comparison shape");
    auto& c=comparisons[name];c.name=name;
    for(size_t i=0;i<actual.size();++i){c.add(actual[i],expected[i],i);
        nonfinite+=!qrt_sm121_mtp::moe_finite(qrt_sm121_q1::widen(actual[i]));}
}
std::vector<uint16_t> bf16_values(const Buffer& buffer){
    const auto host=buffer.copy();guards+=buffer.guards(host,buffer.bytes);
    std::vector<uint16_t> values(buffer.bytes/2u);std::memcpy(values.data(),host.data()+256u,buffer.bytes);return values;
}
void compare_moe(const Buffer& workspace){
    using namespace qrt_sm121_mtp;
    MoeBuffers b;if(!bind_moe_buffers(workspace.data(),workspace.bytes,2u,&b))throw std::runtime_error("MoE binding");
    const auto host=workspace.copy();guards+=workspace.guards(host,workspace.bytes);
    std::vector<unsigned char> written(workspace.bytes,0u);
    const auto mark=[&](const void* pointer,size_t bytes){
        const size_t at=static_cast<const unsigned char*>(pointer)-static_cast<const unsigned char*>(workspace.data());
        if(at>workspace.bytes || bytes>workspace.bytes-at)throw std::runtime_error("MoE subspan");
        std::fill(written.begin()+at,written.begin()+at+bytes,1u);return at;
    };
    const size_t output_at=mark(b.output,8192u);std::vector<uint16_t> output(4096u);
    std::memcpy(output.data(),host.data()+256u+output_at,8192u);
    compare_values("moe_output",output,read_role<uint16_t>("expected_moe_output",4096u));
    // The original layer-3 capture retains final MoE output, not its internal
    // expert frontiers. Guard all exact workspaces without inventing an oracle.
    mark(b.router,1024u);mark(b.shared_gate,4u);mark(b.shared_gate_up,4096u);
    mark(b.shared_activated,2048u);mark(b.shared_down,8192u);mark(b.shared,8192u);mark(b.routed,8192u);
    mark(b.routed_gate_up,2u*8192u*2u);mark(b.routed_activated,2u*4096u*2u);mark(b.routed_weighted,2u*16384u*2u);
    mark(b.topk_ids,64u);mark(b.topk_weights,64u);
    uint32_t invalid;const size_t at=mark(b.invalid,4u);std::memcpy(&invalid,host.data()+256u+at,4u);invalid_flags+=invalid!=0u;
    for(size_t i=0;i<workspace.bytes;++i)if(!written[i])guards+=host[256u+i]!=0xa5u;
}
void execute(const qrt_sm121_q2::AttentionLayerViews& borrowed,
    const qrt_sm121_q2::AttentionLayerTables& tables,unsigned position,uint16_t poison){
    using namespace qrt_sm121_q2;auto v=borrowed;auto& a=v.attention;
    const auto keys=read_role<uint16_t>("history_k",size_t(position+2u)*512u);
    const auto values=read_role<uint16_t>("history_v",size_t(position+2u)*512u);
    std::vector<uint16_t> history(size_t(position+2u)*1024u,poison);
    for(unsigned token=0;token<position;++token){
        std::copy_n(keys.data()+size_t(token)*512u,512u,history.data()+size_t(token)*1024u);
        std::copy_n(values.data()+size_t(token)*512u,512u,history.data()+size_t(token)*1024u+512u);
    }
    Buffer retained(history.size()*2u);retained.upload(history);
    a.history=static_cast<const uint16_t*>(retained.data());a.history_capacity=position+2u;a.first_position=position;
    a.score_stride=(position+33u)&~31u;
    Buffer norm(8192u),residual(8192u),moe_input(8192u),output_residual(8192u),moe(qrt_sm121_mtp::moe_workspace_bytes(2u));
    Buffer qp(32768u),kvp(4096u),qn(16384u),kn(2048u),q(16384u),gate(16384u),kv(4096u);
    Buffer scores(size_t(32u)*a.score_stride*4u),fp_context(32768u),context(16384u),gated(16384u),out(8192u);
    v.normalized_input=static_cast<uint16_t*>(norm.data());v.input_residual=static_cast<uint16_t*>(residual.data());
    v.moe_input=static_cast<uint16_t*>(moe_input.data());v.output_residual=static_cast<uint16_t*>(output_residual.data());
    v.moe_workspace=moe.data();v.moe_workspace_bytes=moe.bytes;a.normalized_input=v.normalized_input;
    a.q_projected=static_cast<uint16_t*>(qp.data());a.kv_projected=static_cast<uint16_t*>(kvp.data());
    a.q_norm=static_cast<uint16_t*>(qn.data());a.k_norm=static_cast<uint16_t*>(kn.data());
    a.queries=static_cast<uint16_t*>(q.data());a.gates=static_cast<uint16_t*>(gate.data());a.staged_kv=static_cast<uint16_t*>(kv.data());
    a.scores=static_cast<float*>(scores.data());a.float_context=static_cast<float*>(fp_context.data());
    a.context=static_cast<uint16_t*>(context.data());a.gated=static_cast<uint16_t*>(gated.data());a.output=static_cast<uint16_t*>(out.data());
    check(launch_attention_layer(v,tables));check(hipDeviceSynchronize());
    compare<uint16_t>("input_norm",norm);compare<uint16_t>("moe_input",moe_input);compare<uint16_t>("output_residual",output_residual);compare_moe(moe);
    const auto q_projection=bf16_values(qp),kv_projection=bf16_values(kvp);std::vector<uint16_t> qkv(18432u);
    for(unsigned row=0;row<2u;++row){
        std::copy_n(q_projection.data()+row*8192u,8192u,qkv.data()+row*9216u);
        std::copy_n(kv_projection.data()+row*1024u,1024u,qkv.data()+row*9216u+8192u);
    }
    const auto expected_qkv=read_role<uint16_t>("expected_qkv",18432u);
    compare_values("qkv",qkv,expected_qkv);
    compare<uint16_t>("q_norm",qn);compare<uint16_t>("k_norm",kn);compare<uint16_t>("q_rope",q);
    const auto actual_kv=bf16_values(kv),expected_keys=read_role<uint16_t>("expected_k_rope",1024u);
    std::vector<uint16_t> expected_kv(2048u),actual_keys(1024u);
    for(unsigned row=0;row<2u;++row){
        std::copy_n(expected_keys.data()+row*512u,512u,expected_kv.data()+row*1024u);
        std::copy_n(expected_qkv.data()+row*9216u+8704u,512u,expected_kv.data()+row*1024u+512u);
        std::copy_n(actual_kv.data()+row*1024u,512u,actual_keys.data()+row*512u);
    }
    compare_values("k_rope",actual_keys,expected_keys);compare_values("staged_kv",actual_kv,expected_kv);
    compare<uint16_t>("context",context);compare<uint16_t>("gated",gated);compare<uint16_t>("output",out);
    const auto score_host=scores.copy();guards+=scores.guards(score_host,scores.bytes);
    for(unsigned row=0;row<2u;++row)for(unsigned head=0;head<16u;++head)for(unsigned token=0;token<a.score_stride;++token){
        uint32_t bits;const size_t i=(size_t(row)*16u+head)*a.score_stride+token;
        std::memcpy(&bits,score_host.data()+256u+i*4u,4u);
        if(token>position+row)mask_errors+=bits!=0xff800000u;
        else nonfinite+=(bits&0x7f800000u)==0x7f800000u;
    }
    const auto raw_context=fp_context.copy();guards+=fp_context.guards(raw_context,fp_context.bytes);
    const auto published=bf16_values(context);
    for(size_t i=0;i<8192u;++i){float value;std::memcpy(&value,raw_context.data()+256u+i*4u,4u);
        nonfinite+=!qrt_sm121_mtp::moe_finite(value);mask_errors+=qrt_sm121_q1::bf16(value)!=published[i];}
    guards+=residual.guards(residual.copy(),residual.bytes)+gate.guards(gate.copy(),gate.bytes);
    for(unsigned rows:{1u,2u}){const auto selected=accepted_attention(a,rows);
        if(selected.key_values!=a.staged_kv || selected.first_position!=position || selected.rows!=rows)
            throw std::runtime_error("private accepted tail");}
    immutable+=retained.changed();
}
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("bound TSV plan first-position");
    const unsigned position=static_cast<unsigned>(number(argv[2],263678u));
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("plan missing");std::string line;
    while(std::getline(file,line)){
        if(!line.empty() && line.back()=='\r')line.pop_back();
        std::istringstream input(line);std::vector<std::string> fields;std::string part;
        while(std::getline(input,part,'\t'))fields.push_back(part);
        if(fields.size()!=5u || plan.count(fields[0]) || plan.size()>=64u)throw std::runtime_error("plan fields");
        plan.emplace(fields[0],Entry{fields[1],fields[4],size_t(number(fields[2],uint64_t(8)<<30u)),
            size_t(number(fields[3],uint64_t(1)<<30u))});
    }
    using namespace qrt_sm121_q2;Inputs inputs;AttentionLayerViews v;
    v.hidden=inputs.upload<uint16_t>("hidden",4096u);v.residual=inputs.upload<uint16_t>("residual",4096u);
    v.input_norm_weights=inputs.upload<uint16_t>("input_norm_weights",2048u);
    v.post_norm_weights=inputs.upload<uint16_t>("post_norm_weights",2048u);
    auto& a=v.attention;
    a.q_weights=inputs.upload<uint16_t>("q_weight",8192u*2048u);
    a.k_weights=inputs.upload<uint16_t>("k_weight",512u*2048u);a.v_weights=inputs.upload<uint16_t>("v_weight",512u*2048u);
    a.output_weights=inputs.upload<uint16_t>("out_weight",2048u*4096u);
    a.q_norm_weights=inputs.upload<uint16_t>("q_norm_weight",256u);a.k_norm_weights=inputs.upload<uint16_t>("k_norm_weight",256u);
    const auto* router=inputs.upload<uint16_t>("router_weights",256u*2048u);
    const auto* shared_gate=inputs.upload<uint16_t>("shared_gate_weights",2048u);
    auto gate_up=read_role<uint16_t>("shared_gate_up_gate_weights",512u*2048u);
    const auto up=read_role<uint16_t>("shared_gate_up_up_weights",512u*2048u);
    gate_up.insert(gate_up.end(),up.begin(),up.end());
    auto gate_up_owner=std::make_unique<Buffer>(gate_up.size()*2u);gate_up_owner->upload(gate_up);
    const auto* shared_gate_up=static_cast<const uint16_t*>(gate_up_owner->data());inputs.buffers.push_back(std::move(gate_up_owner));
    const auto* shared_down=inputs.upload<uint16_t>("shared_down_weights",2048u*512u);
    const auto* routed_gate_up=inputs.upload<uint16_t>("routed_gate_up_weights",size_t(256u)*1024u*2048u);
    const auto* routed_down=inputs.upload<uint16_t>("routed_down_weights",size_t(256u)*2048u*512u);
    v.moe_weights={router,shared_gate,shared_gate_up,shared_down,routed_gate_up,routed_down};
    const auto* sigmoid=inputs.upload<uint16_t>("sigmoid",65536u);
    const size_t rope_bytes=plan.at("rope").bytes;
    if(rope_bytes%128u || rope_bytes/128u>target_context_limit)throw std::runtime_error("rope extent");
    const AttentionLayerTables tables{
        {inputs.upload<unsigned char>("rsqrt",qrt_sm121_rsqrt::table_bytes),
         inputs.upload<unsigned char>("exp2",qrt_sm121_exp2::table_bytes),
         inputs.upload<unsigned char>("rcp",qrt_sm121_attention_rcp::table_bytes),
         inputs.upload<uint16_t>("rope",rope_bytes/2u),unsigned(rope_bytes/128u),sigmoid},
        {inputs.upload<uint16_t>("silu",65536u+12u)+12u,sigmoid,inputs.upload<uint32_t>("router_exp",1u<<23u)}};
    // Neither uncommitted historical tail may be consumed or changed. Repeat
    // with NaN and finite poison while recomputing actual candidate K/V privately.
    execute(v,tables,position,0x7fc1u);execute(v,tables,position,0x3f80u);
    immutable+=inputs.changed();size_t total=0,mismatches=0;unsigned count=0;
    std::cout<<"{\"kind\":\"original_q2_complete_attention_layer\",\"first_position\":"<<position
        <<",\"configurations\":2,\"history_tail_poison_bf16\":[32705,16256],\"stages\":[";
    for(const auto& entry:comparisons){const auto& c=entry.second;if(count++)std::cout<<',';total+=c.elements;mismatches+=c.mismatches;
        std::cout<<"{\"stage\":\""<<c.name<<"\",\"elements\":"<<c.elements<<",\"bit_mismatches\":"<<c.mismatches
            <<",\"first_difference\":["<<c.first<<','<<c.actual<<','<<c.expected<<"]}";}
    const bool passed=!mismatches&&!guards&&!immutable&&!invalid_flags&&!mask_errors&&!nonfinite;
    std::cout<<"],\"compared_elements\":"<<total<<",\"bit_mismatches\":"<<mismatches<<",\"guard_errors\":"<<guards
        <<",\"immutable_input_errors\":"<<immutable<<",\"invalid_flag_errors\":"<<invalid_flags
        <<",\"mask_or_publication_errors\":"<<mask_errors<<",\"nonfinite\":"<<nonfinite
        <<",\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"resident_cache_published\":false,\"model_loaded\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
