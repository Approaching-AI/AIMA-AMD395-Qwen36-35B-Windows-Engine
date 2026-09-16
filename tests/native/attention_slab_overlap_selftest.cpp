#include "../../native/providers/ck_fmha/prepared_decoded_qk.h"
#include "../../native/providers/ck_fmha/attention_slab_schedule.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
namespace attention = qrt_blackwell_attention;
namespace prepared = qrt_prepared_decoded_qk;
namespace schedule = qrt_attention_slab_schedule;
using Clock = std::chrono::steady_clock;
constexpr unsigned guard = 64u, batch = 128u;
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void check(hipError_t status) { require(status == hipSuccess, hipGetErrorString(status)); }
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::milli>(Clock::now()-start).count();
}
struct Device {
    void* pointer = nullptr;
    size_t bytes;
    explicit Device(size_t count) : bytes(count) { check(hipMalloc(&pointer, bytes)); }
    Device(const Device&) = delete;
    ~Device() { if (pointer) (void)hipFree(pointer); }
    template<class T> T* as() { return static_cast<T*>(pointer); }
    void reset(unsigned char value = 0xa5u) { check(hipMemset(pointer,value,bytes)); }
};
template<class T> std::vector<T> read(const char* name, size_t count) {
    std::ifstream stream(name,std::ios::binary|std::ios::ate);
    require(bool(stream) && stream.tellg()==std::streamoff(count*sizeof(T)),"input size");
    std::vector<T> values(count); stream.seekg(0);
    require(bool(stream.read(reinterpret_cast<char*>(values.data()),std::streamsize(count*sizeof(T)))),"input read");
    return values;
}
template<class T> void upload(Device& target, const std::vector<T>& values) {
    require(values.size()*sizeof(T)==target.bytes,"upload size");
    check(hipMemcpy(target.pointer,values.data(),target.bytes,hipMemcpyHostToDevice));
}
template<class T> std::vector<T> download(Device& source) {
    require(source.bytes%sizeof(T)==0u,"download size");
    std::vector<T> values(source.bytes/sizeof(T));
    check(hipMemcpy(values.data(),source.pointer,source.bytes,hipMemcpyDeviceToHost)); return values;
}
template<class T> void unchanged(Device& source, const std::vector<T>& values) {
    const auto actual=download<T>(source);
    require(actual.size()==values.size() && !std::memcmp(actual.data(),values.data(),source.bytes),"immutable input/encoding");
}
void guards(Device& source, size_t words) {
    uint32_t values[2u*guard];
    check(hipMemcpy(values,source.pointer,guard*4u,hipMemcpyDeviceToHost));
    check(hipMemcpy(values+guard,source.as<uint32_t>()+guard+words,guard*4u,hipMemcpyDeviceToHost));
    for(uint32_t value:values) require(value==0xa5a5a5a5u,"arena redzone");
}
struct Slot {
    Device scratch;
    hipStream_t stream = nullptr;
    explicit Slot(size_t words) : scratch((words+2u*guard)*4u) {
        check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    }
    ~Slot() {
        if(stream) { (void)hipStreamSynchronize(stream); (void)hipStreamDestroy(stream); }
    }
};
bool complete(hipStream_t stream) {
    const auto start=Clock::now();
    for(;;) {
        const auto status=hipStreamQuery(stream);
        if(status==hipSuccess)return true;
        if(status!=hipErrorNotReady || elapsed(start)>30000.0)return false;
        std::this_thread::yield();
    }
}
void pad(std::vector<uint16_t>& values,unsigned columns,unsigned original,unsigned total) {
    require(values.size()==size_t(original)*columns && total>=original,"capture extension");
    const std::vector<uint16_t> last(values.end()-columns,values.end());
    for(unsigned i=original;i<total;++i)values.insert(values.end(),last.begin(),last.end());
    values.insert(values.begin(),guard,0x5a5au); values.insert(values.end(),guard,0x5a5au);
}
struct Inputs {
    unsigned tokens;
    std::vector<uint16_t> q,k,v,kt,vt;
    std::vector<uint32_t> encoding;
    Device dq,dk,dv,dkt,dvt,dp;
    prepared::Workspace workspace;
    double preparation_ms;
    Inputs(unsigned n,std::vector<uint16_t> query,std::vector<uint16_t> key,std::vector<uint16_t> value)
      : tokens(n),q(std::move(query)),k(std::move(key)),v(std::move(value)),
        kt(k.size(),0x5a5au),vt(v.size(),0x5a5au),encoding(prepared::workspace_words+2u*guard,0xa5a5a5a5u),
        dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dkt(kt.size()*2u),dvt(vt.size()*2u),dp(encoding.size()*4u),
        workspace{dp.as<uint32_t>()+guard,n},preparation_ms(0.0) {
        upload(dq,q);upload(dk,k);upload(dv,v);upload(dkt,kt);upload(dvt,vt);upload(dp,encoding);
        check(hipDeviceSynchronize()); const auto begin=Clock::now();
        check(hipError_t(prepared::prepare_workspace(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,
            dkt.as<uint16_t>()+guard,workspace,nullptr)));
        check(hipError_t(attention::transpose_keys(dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,
            size_t(n)*512u,n,nullptr)));
        require(complete(nullptr),"preparation completion");preparation_ms=elapsed(begin);
        for(unsigned token=0u;token<n;++token)for(unsigned feature=0u;feature<512u;++feature) {
            kt[guard+size_t(feature)*n+token]=k[guard+size_t(token)*512u+feature];
            vt[guard+size_t(feature)*n+token]=v[guard+size_t(token)*512u+feature];
        }
        for(unsigned key_kind=0u;key_kind<2u;++key_kind) {
            const unsigned heads=key_kind?2u:16u;
            const auto& words=key_kind?k:q;
            const size_t data_base=guard+(key_kind?prepared::query_words:0u);
            const size_t flag_base=guard+prepared::query_words+prepared::key_words+
                (key_kind?prepared::query_flag_words:0u);
            for(unsigned row=0u;row<n*heads;++row) {
                bool valid=true;
                for(unsigned c=0u;c<256u;++c) {
                    const uint16_t x=words[guard+size_t(row)*256u+c];
                    const unsigned exponent=(x>>7u)&255u;
                    valid &= !(x&0x7fffu) || (exponent>=64u && exponent<=190u);
                    const size_t index=key_kind?(size_t(row%heads)*256u+c)*n+row/heads:size_t(row)*256u+c;
                    encoding[data_base+index]=(uint32_t(x)<<16u)|uint16_t((x&0x7fffu)?int(exponent)-127:-512);
                }
                encoding[flag_base+row]=unsigned(valid);
            }
        }
        verify();
    }
    void verify() { unchanged(dq,q);unchanged(dk,k);unchanged(dv,v);unchanged(dkt,kt);unchanged(dvt,vt);unchanged(dp,encoding); }
};
void compare(unsigned tokens,unsigned query_start,unsigned queries,unsigned mode,
    std::vector<uint16_t> q,std::vector<uint16_t> k,std::vector<uint16_t> v,
    const std::vector<uint16_t>& golden,Device& exp,Device& rcp,bool capture) {
    Inputs inputs(tokens,std::move(q),std::move(k),std::move(v));
    const size_t words=attention::split_scratch_elements(std::min(batch,queries),query_start+queries,22u);
    require(words!=0u,"scratch shape");
    // Output offset3 tests the ownership boundary independently of query_start.
    constexpr unsigned output_start=3u;
    const size_t output_words=size_t(queries+output_start+2u)*4096u;
    Device output((output_words+2u*guard)*4u);
    std::array<std::unique_ptr<Slot>,4> slots;
    for(auto& slot:slots)slot=std::make_unique<Slot>(words);
    attention::SplitQkProducer producer{&inputs.workspace,prepared::launch_workspace};
    std::vector<uint32_t> expected;
    double samples[4][3]{},maximum_window[4]{};
    unsigned expected_submitted[4]{},expected_completions[4]{},expected_windows[4]{};
    const unsigned rounds=capture?4u:2u;
    const unsigned external_tokens=capture?std::min(7169u,queries):0u;
    for(unsigned round=0u;round<rounds;++round)for(unsigned order=0u;order<4u;++order) {
        const unsigned variant=round?(order+round-1u)%4u:order;
        const unsigned slot_count=variant<2u?1u:variant==2u?2u:4u;
        output.reset();for(auto& slot:slots)slot->scratch.reset();check(hipDeviceSynchronize());
        const auto begin=Clock::now();auto window_begin=begin;
        bool after_drain=false;
        const auto result=schedule::run({queries,batch,slot_count,variant?8u:1u},
            [&](unsigned slot,unsigned offset,unsigned count) {
                if(after_drain){window_begin=Clock::now();after_drain=false;}
                const auto stream=variant?slots[slot]->stream:nullptr;
                return attention::launch_queries(inputs.dq.as<uint16_t>()+guard,inputs.dk.as<uint16_t>()+guard,
                    inputs.dv.as<uint16_t>()+guard,output.as<float>()+guard,stream,query_start+offset,count,
                    output_start+offset,exp.as<unsigned char>(),nullptr,nullptr,true,rcp.as<unsigned char>(),22u,
                    slots[slot]->scratch.as<float>()+guard,words,nullptr,nullptr,
                    inputs.dkt.as<uint16_t>()+guard,tokens,false,nullptr,nullptr,0u,nullptr,nullptr,
                    inputs.dvt.as<uint16_t>()+guard,tokens,1u,1u,true,true,true,0u,false,&producer,false)==int(hipSuccess);
            },
            [&](unsigned slot) {
                const bool okay=complete(variant?slots[slot]->stream:nullptr);
                maximum_window[variant]=std::max(maximum_window[variant],elapsed(window_begin));
                after_drain=true;return okay;
            },
            [&] { return elapsed(begin)<=30000.0 && elapsed(window_begin)<=30000.0; });
        const double ms=elapsed(begin);
        require(result.error==0u && result.drained,"slab submission/completion/deadline");
        if(round)samples[variant][round-1u]=ms;
        expected_submitted[variant]=result.submitted;expected_completions[variant]=result.completion_calls;
        expected_windows[variant]=result.windows;
        const auto actual=download<uint32_t>(output);
        if(expected.empty()) { require(variant==0u,"original control must run first");expected=actual; }
        require(actual==expected,"raw attention output mismatch");
        for(size_t cell=0u;cell<actual.size();++cell) {
            const bool live=cell>=guard+size_t(output_start)*4096u &&
                cell<guard+size_t(output_start+queries)*4096u;
            if(!live)require(actual[cell]==0xa5a5a5a5u,"output guard or unused rows");
            if(live && !capture && mode==2u) require(actual[cell]==0u,"independent zero-V result");
        }
        for(size_t cell=0u;cell<size_t(external_tokens)*4096u;++cell) {
            float value;const uint32_t raw=actual[guard+size_t(output_start)*4096u+cell];
            std::memcpy(&value,&raw,4u);
            require(qrt_sm121_pv_bound::bf16(value)==golden[cell],"GB10 context mismatch");
        }
        for(auto& slot:slots)guards(slot->scratch,words);
        inputs.verify();
    }
    for(unsigned variant=0u;variant<4u;++variant) {
        const unsigned slot_count=variant<2u?1u:variant==2u?2u:4u;
        double sorted[3]={samples[variant][0],samples[variant][1],samples[variant][2]};
        std::sort(sorted,sorted+3u);const double median=capture?sorted[1]:samples[variant][0];
        std::printf("{\"kind\":\"attention_slab_overlap_%s\",\"tokens\":%u,\"query_start\":%u,\"queries\":%u,\"mode\":%u,\"variant\":%u,\"streams\":%u,\"window_slabs\":%u,\"query_batch\":128,\"submitted_slabs\":%u,\"completion_calls\":%u,\"completed_windows\":%u,\"unique_output_cells\":%zu,\"external_gb10_cells\":%zu,\"compared_attempts\":%u,\"samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_ms\":%.9f,\"common_preparation_ms\":%.9f,\"preparation_plus_median_ms\":%.9f,\"maximum_completion_window_ms\":%.9f,\"scratch_bytes_per_stream\":%zu,\"active_scratch_bytes\":%zu,\"qk_prepared_bytes\":%zu,\"allocated_scratch_arenas\":4,\"raw_bit_mismatches\":0,\"external_bf16_mismatches\":0,\"redzones_pass\":true,\"immutable_inputs\":true,\"complete_cpu_encoding_check\":true,\"all_streams_drained\":true,\"all_attempts_verified\":true,\"source_capture_tokens\":%u,\"repeated_last_rows\":%u,\"real_model_prompt\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            capture?"capture":"safety",tokens,query_start,queries,mode,variant,slot_count,variant?8u:1u,
            expected_submitted[variant],expected_completions[variant],expected_windows[variant],size_t(queries)*4096u,
            size_t(external_tokens)*4096u,rounds,samples[variant][0],samples[variant][1],samples[variant][2],
            median,inputs.preparation_ms,inputs.preparation_ms+median,maximum_window[variant],words*4u,
            words*4u*slot_count,prepared::workspace_words*4u,capture?7169u:0u,capture?tokens-7169u:0u);
        std::fflush(stdout);
    }
}
void generated(unsigned start,unsigned count,unsigned mode,Device& exp,Device& rcp) {
    const unsigned tokens=start+count;
    std::vector<uint16_t> q(size_t(tokens)*4096u),k(size_t(tokens)*512u),v(k.size());
    for(size_t i=0u;i<q.size();++i)q[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((122u+i%4u)<<7u));
    for(size_t i=0u;i<k.size();++i) {
        k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%5u)<<7u));
        v[i]=mode==2u?0u:uint16_t(((i*31u+i/17u)&0x807fu)|((122u+i%6u)<<7u));
    }
    if(mode==1u) { for(size_t i=0u;i<q.size();i+=211u)q[i]=1u;for(size_t i=0u;i<k.size();i+=107u)k[i]=0x8001u; }
    pad(q,4096u,tokens,tokens);pad(k,512u,tokens,tokens);pad(v,512u,tokens,tokens);
    compare(tokens,start,count,mode,std::move(q),std::move(k),std::move(v),{},exp,rcp,false);
}
} // namespace
int main(int argc,char** argv) try {
    require(argc==4 || argc==8,"use --selftest EXP RCP or --q7169/--q8192 Q K V GB10 EXP RCP");
    hipDeviceProp_t properties{};check(hipGetDeviceProperties(&properties,0));
    require(!std::strncmp(properties.gcnArchName,"gfx1151",7u),"requires gfx1151");
    const auto hexp=read<unsigned char>(argv[argc-2],attention::exp2_backend::table_bytes);
    const auto hrcp=read<unsigned char>(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    require(attention::exp2_backend::valid_layout(hexp.data(),hexp.size()),"exp2 layout");
    require(qrt_sm121_attention_rcp::valid_layout(hrcp.data(),hrcp.size()),"rcp layout");
    Device exp(hexp.size()),rcp(hrcp.size());upload(exp,hexp);upload(rcp,hrcp);
    if(argc==4 && !std::strcmp(argv[1],"--selftest")) {
        for(auto shape:{std::array<unsigned,2>{0u,1u},{31u,129u},{17u,257u},{0u,1025u}})
            for(unsigned mode=0u;mode<3u;++mode)generated(shape[0],shape[1],mode,exp,rcp);
    } else {
        require(argc==8 && (!std::strcmp(argv[1],"--q7169") || !std::strcmp(argv[1],"--q8192")),"capture arguments");
        const unsigned tokens=!std::strcmp(argv[1],"--q7169")?7169u:8192u;
        auto q=read<uint16_t>(argv[2],size_t(7169u)*4096u);
        auto k=read<uint16_t>(argv[3],size_t(7169u)*512u),v=read<uint16_t>(argv[4],size_t(7169u)*512u);
        const auto golden=read<uint16_t>(argv[5],size_t(7169u)*4096u);
        pad(q,4096u,7169u,tokens);pad(k,512u,7169u,tokens);pad(v,512u,7169u,tokens);
        compare(tokens,0u,tokens,0u,std::move(q),std::move(k),std::move(v),golden,exp,rcp,true);
    }
    unchanged(exp,hexp);unchanged(rcp,hrcp);return 0;
} catch(const std::exception& error) {
    std::fprintf(stderr,"attention_slab_overlap_error=%s\n",error.what());return 2;
}
