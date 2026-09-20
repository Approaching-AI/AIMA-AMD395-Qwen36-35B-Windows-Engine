// Component replay of the actual drafter owner from original target hidden rows.
// Reference row selection drives only this probe; it is not a target acceptance implementation.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_mtp_drafter.h"

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
    if (!file || file.tellg() <= 0 || file.tellg() > (128u << 20u))
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

struct Transaction {
    unsigned first,append_rows,query_first,query_rows,expected_first;
    std::string hidden_file,ids_file;
    const uint16_t* hidden=nullptr;
    const uint32_t* ids=nullptr;
};
struct Stage {const char* label;unsigned width,element_bytes;const void* data;};
std::vector<Stage> stages(const qrt_sm121_mtp::DrafterObservation& view) {
    using namespace qrt_sm121_mtp;
    MoeBuffers m;
    if(view.rows && !bind_moe_buffers(const_cast<void*>(view.moe_workspace),view.moe_bytes,view.rows,&m))
        throw std::runtime_error("completed MoE view");
    return {{"query-projection",8192,2,view.query_projection},{"queries",4096,2,view.queries},
        {"gates",4096,2,view.gates},{"context",4096,2,view.context},{"gated",4096,2,view.gated_context},
        {"attention-output",2048,2,view.output_projection},{"post-attention-norm",2048,2,view.post_attention},
        {"attention-residual",2048,2,view.attention_residual},{"final-norm",2048,2,view.final_hidden},
        {"router",256,2,m.router},{"shared-gate",1,2,m.shared_gate},{"shared-gate-up",1024,2,m.shared_gate_up},
        {"shared-activated",512,2,m.shared_activated},{"shared-down",2048,2,m.shared_down},{"shared",2048,2,m.shared},
        {"routed-gate-up",8192,2,m.routed_gate_up},{"routed-activated",4096,2,m.routed_activated},
        {"routed-weighted",16384,2,m.routed_weighted},{"expert-part-1",2048,2,m.routed},{"moe-output",2048,2,m.output},
        {"topk-ids",8,4,m.topk_ids},{"topk-weights",8,4,m.topk_weights}};
}
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("case_directory original_weight_and_table_plan");
    using namespace qrt_sm121_mtp;
    const std::filesystem::path root(argv[1]);Scratch scratch;
    std::ifstream plan(argv[2]);if(!plan)throw std::runtime_error("weight plan");
    struct Bound {void* data;size_t bytes;};std::map<std::string,Bound> bound;
    std::string label,path;uint64_t offset=0,bytes=0;
    while(plan>>label>>std::quoted(path)>>offset>>bytes){
        if(label.size()>64u || path.size()>2048u || !bytes || bytes>(uint64_t(1)<<30u) || bound.count(label))
            throw std::runtime_error("weight/table plan extent");
        bound.emplace(label,Bound{scratch.upload(path,offset,size_t(bytes)),size_t(bytes)});
    }
    if(!plan.eof() || bound.size()!=26u)throw std::runtime_error("weight/table plan completeness");
    const auto weight=[&](const char* name,size_t count){
        const auto item=bound.at(name);if(item.bytes!=count*2u)throw std::runtime_error("weight shape");
        return static_cast<const uint16_t*>(item.data);
    };
    const auto table=[&](const char* name,size_t count){
        const auto item=bound.at(name);if(item.bytes!=count)throw std::runtime_error("table layout");
        return static_cast<const unsigned char*>(item.data);
    };
    DrafterWeights weights;
    weights.prompt={weight("embeddings",size_t(head_vocabulary)*2048u),weight("embedding-norm",2048u),
        weight("hidden-norm",2048u),weight("fusion",2048u*4096u),weight("input-norm",2048u),
        weight("kv-projection",1024u*2048u),weight("key-norm",256u)};
    weights.query=weight("query",8192u*2048u);weights.query_norm=weight("query-norm",256u);
    weights.output=weight("output",2048u*4096u);weights.post_attention_norm=weight("post-attention-norm",2048u);
    weights.moe={weight("router",256u*2048u),weight("shared-gate",2048u),weight("shared-gate-up",1024u*2048u),
        weight("shared-down",2048u*512u),weight("routed-gate-up",size_t(256u)*1024u*2048u),
        weight("routed-down",size_t(256u)*2048u*512u)};
    weights.final_norm=weight("final-norm",2048u);weights.lm_head=weight("lm-head",size_t(head_vocabulary)*2048u);
    DrafterTables tables{table("rsqrt",17301808u),weight("rope",262144u*64u),262144u,
        table("exp2",183174448u),table("reciprocal",8388640u),
        {weight("silu",65536u+12u)+12u,weight("sigmoid",65536u),reinterpret_cast<const uint32_t*>(table("router-exp",8388608u*4u))}};
    std::ifstream schedule(root/"transactions.txt");unsigned transaction_count=0,expected_rows=0,history_rows=0;
    if(!(schedule>>transaction_count>>expected_rows>>history_rows) || !transaction_count || transaction_count>32u ||
        expected_rows!=33u || history_rows>16384u)throw std::runtime_error("original transaction header");
    std::vector<Transaction> transactions(transaction_count);unsigned expected_cursor=0,retained=0;
    for(unsigned i=0;i<transaction_count;++i){
        auto& t=transactions[i];
        if(!(schedule>>t.first>>t.append_rows>>t.query_first>>t.query_rows>>t.expected_first>>std::quoted(t.hidden_file)>>std::quoted(t.ids_file)))
            throw std::runtime_error("original transaction record");
        if(!t.append_rows || t.append_rows>8192u || !t.query_rows || t.query_rows>2u || t.first!=retained ||
            t.query_first<t.first || t.query_first-t.first>=t.append_rows ||
            t.query_rows>t.append_rows-(t.query_first-t.first) || t.expected_first!=expected_cursor ||
            t.first+t.append_rows>history_rows+1u || (i && (t.append_rows!=2u || t.query_first!=t.first)) ||
            (!i && (t.first || (t.append_rows!=7169u && t.append_rows!=8192u) || t.query_rows!=1u || t.query_first+1u!=t.append_rows)))
            throw std::runtime_error("original accepted/provisional extent");
        for(const auto& name:{t.hidden_file,t.ids_file})
            if(std::filesystem::path(name).filename()!=std::filesystem::path(name))throw std::runtime_error("input path");
        t.hidden=static_cast<const uint16_t*>(scratch.upload((root/t.hidden_file).string(),0u,size_t(t.append_rows)*4096u));
        t.ids=static_cast<const uint32_t*>(scratch.upload((root/t.ids_file).string(),0u,size_t(t.append_rows)*4u));
        expected_cursor+=t.query_rows;retained=t.query_first+t.query_rows;
    }
    schedule>>std::ws;if(!schedule.eof() || expected_cursor!=expected_rows || retained!=history_rows)
        throw std::runtime_error("original schedule final extent");
    std::map<std::string,std::vector<unsigned char>> expected;
    for(const auto& stage:stages({})){
        auto data=whole((root/(std::string(stage.label)+".bin")).string());
        if(data.size()!=size_t(expected_rows)*stage.width*stage.element_bytes)throw std::runtime_error("original frontier shape");
        expected.emplace(stage.label,std::move(data));
    }
    const auto expected_k=whole((root/"history-k.bin").string()),expected_v=whole((root/"history-v.bin").string());
    const auto expected_logits=whole((root/"logits.bin").string());
    if(expected_k.size()!=size_t(history_rows)*1024u || expected_v.size()!=expected_k.size() ||
        expected_logits.size()!=size_t(transaction_count)*head_vocabulary*2u)throw std::runtime_error("original history/logit shape");
    std::map<std::string,Comparison> comparisons;Comparison vocabulary,ids,sampled_values,cache;
    unsigned proposals=0,truncations=0,rejected_contracts=0;size_t maximum_owner_bytes=0;
    for(unsigned maximum_rows:{1u,2u}){
        Drafter drafter;
        check(drafter.reserve(history_rows+1u,transactions[0].append_rows));
        if(!drafter.bind(weights,tables,1u))throw std::runtime_error("model binding");
        maximum_owner_bytes=std::max(maximum_owner_bytes,drafter.allocated_bytes());
        if(drafter.propose(0u,1u,1u).status!=hipErrorInvalidValue ||
           drafter.append_target(transactions[0].hidden,transactions[0].ids,0u,transactions[0].append_rows,true,2u).status!=hipErrorInvalidValue)
            throw std::runtime_error("empty/stale contract accepted");
        rejected_contracts+=2u;
        for(unsigned ordinal=0;ordinal<transaction_count;++ordinal){
            const auto& t=transactions[ordinal];
            if(drafter.retained_tokens()!=t.first)throw std::runtime_error("retained prefix before append");
            const auto append=drafter.append_target(t.hidden,t.ids,t.first,t.append_rows,ordinal==0u,1u);
            if(append.status!=hipSuccess || append.completion_unknown){
                std::cerr<<"append transaction "<<ordinal<<" stage "<<append.stage<<'\n';
                check(append.status);throw std::runtime_error("append completion unknown");
            }
            if(drafter.retained_tokens()!=t.first+t.append_rows)throw std::runtime_error("provisional KV extent");
            for(unsigned first=0;first<t.query_rows;){
                const unsigned rows=std::min(maximum_rows,t.query_rows-first);
                const auto draft=drafter.propose(t.query_first+first,rows,1u);++proposals;
                if(draft.status!=hipSuccess || draft.completion_unknown){
                    std::cerr<<"proposal transaction "<<ordinal<<" stage "<<draft.stage<<'\n';
                    check(draft.status);throw std::runtime_error("proposal completion unknown");
                }
                const auto view=drafter.observation(1u);
                if(draft.rows!=rows || view.rows!=rows || draft.first_position!=t.query_first+first || view.first_position!=draft.first_position)
                    throw std::runtime_error("completed proposal extent");
                for(const auto& stage:stages(view)){
                    std::vector<unsigned char> actual(size_t(rows)*stage.width*stage.element_bytes);
                    check(hipMemcpy(actual.data(),stage.data,actual.size(),hipMemcpyDeviceToHost));
                    for(unsigned row=0;row<rows;++row)for(unsigned column=0;column<stage.width;++column){
                        uint32_t actual_bits=0,wanted_bits=0;
                        std::memcpy(&actual_bits,actual.data()+(size_t(row)*stage.width+column)*stage.element_bytes,stage.element_bytes);
                        std::memcpy(&wanted_bits,expected.at(stage.label).data()+
                            (size_t(t.expected_first+first+row)*stage.width+column)*stage.element_bytes,stage.element_bytes);
                        comparisons[stage.label].add(actual_bits,wanted_bits,t.expected_first+first+row,column);
                    }
                }
                const unsigned sampled=t.query_rows-1u;
                if(sampled>=first && sampled<first+rows){
                    const unsigned row=sampled-first;
                    std::vector<uint16_t> actual(head_vocabulary);
                    check(hipMemcpy(actual.data(),view.vocabulary_logits+size_t(row)*head_vocabulary,
                        actual.size()*2u,hipMemcpyDeviceToHost));
                    HeadBest expected_best,actual_best;
                    for(unsigned token=0;token<head_vocabulary;++token){
                        uint16_t wanted=0;std::memcpy(&wanted,expected_logits.data()+(size_t(ordinal)*head_vocabulary+token)*2u,2u);
                        vocabulary.add(actual[token],wanted,ordinal,token);
                        if(!head_candidate(wanted,token,&expected_best)||!head_candidate(actual[token],token,&actual_best))
                            throw std::runtime_error("nonfinite sampled logits");
                    }
                    ids.add(draft.tokens[row],expected_best.token,ordinal,0u);
                    uint32_t actual_bits=0,wanted_bits=0;
                    std::memcpy(&actual_bits,&draft.logits[row],4u);std::memcpy(&wanted_bits,&expected_best.logit,4u);
                    sampled_values.add(actual_bits,wanted_bits,ordinal,0u);
                    if(draft.tokens[row]!=actual_best.token || draft.logits[row]!=actual_best.logit)
                        throw std::runtime_error("published candidate differs from generated logits");
                }
                first+=rows;
            }
            const unsigned accepted=t.query_first+t.query_rows;
            if(!drafter.truncate(accepted,1u)||drafter.retained_tokens()!=accepted)throw std::runtime_error("actual cache truncation");
            ++truncations;
            if(accepted<t.first+t.append_rows && drafter.observation(1u).rows)
                throw std::runtime_error("truncated observation remained valid");
        }
        // Final complete KV history detects damage to any earlier prompt or
        // accepted decode row, including overwritten rejected padding.
        std::vector<uint16_t> window(1024u*1024u);
        for(unsigned first=0;first<history_rows;first+=1024u){
            const unsigned count=std::min(1024u,history_rows-first);
            check(hipMemcpy(window.data(),drafter.cache_data()+size_t(first)*1024u,size_t(count)*2048u,hipMemcpyDeviceToHost));
            for(unsigned row=0;row<count;++row)for(unsigned column=0;column<1024u;++column){
                const auto& source=column<512u?expected_k:expected_v;uint16_t wanted=0;
                std::memcpy(&wanted,source.data()+(size_t(first+row)*512u+column%512u)*2u,2u);
                cache.add(window[size_t(row)*1024u+column],wanted,first+row,column);
            }
        }
        if(drafter.quarantined()||drafter.retained_tokens()!=history_rows)throw std::runtime_error("completed owner state");
    }
    const size_t input_bad=scratch.immutable_mismatches();
    size_t mismatch_count=vocabulary.mismatches+ids.mismatches+sampled_values.mismatches+cache.mismatches+input_bad;
    std::cout<<"{\"kind\":\"original_complete_mtp_drafter_component_replay\",\"transactions\":"<<transaction_count
        <<",\"original_prompt_rows\":"<<transactions[0].append_rows<<",\"accepted_query_rows\":"<<expected_rows
        <<",\"history_rows\":"<<history_rows<<",\"proposal_batch_configurations\":[1,2],\"configurations\":2"
        <<",\"proposal_calls\":"<<proposals<<",\"cache_truncations\":"<<truncations<<",\"maximum_owner_bytes\":"<<maximum_owner_bytes
        <<",\"rejected_contracts\":"<<rejected_contracts<<",\"checks\":{";
    bool comma=false;
    for(const auto& entry:comparisons){
        if(comma)std::cout<<',';comma=true;const auto& c=entry.second;mismatch_count+=c.mismatches;
        std::cout<<'"'<<entry.first<<"\":{\"elements\":"<<c.elements<<",\"mismatches\":"<<c.mismatches<<",\"first_difference\":";
        if(c.mismatches)std::cout<<"{\"row\":"<<c.first_row<<",\"column\":"<<c.first_column
            <<",\"actual_bits\":"<<c.first_actual<<",\"expected_bits\":"<<c.first_expected<<'}';
        else std::cout<<"null";std::cout<<'}';
    }
    std::cout<<"},\"vocabulary_elements\":"<<vocabulary.elements<<",\"vocabulary_mismatches\":"<<vocabulary.mismatches
        <<",\"sampled_tokens_compared\":"<<ids.elements<<",\"sampled_token_mismatches\":"<<ids.mismatches
        <<",\"sampled_logit_mismatches\":"<<sampled_values.mismatches<<",\"cache_elements\":"<<cache.elements
        <<",\"cache_mismatches\":"<<cache.mismatches<<",\"immutable_input_byte_mismatches\":"<<input_bad
        <<",\"passed\":"<<(mismatch_count?"false":"true")
        <<",\"original_schedule_drives_probe_only\":true,\"expected_frontiers_are_compute_input\":false"
        <<",\"model_inference\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n";
    return mismatch_count?1:0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
