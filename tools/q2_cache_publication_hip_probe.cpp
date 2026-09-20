// Original continuous target followed by accepted cache publication. Reference
// expectations stay on the host; no expected activation feeds the GPU graph.
#define main qrt_private_target_probe_main
#include "q2_target_hip_probe.cpp"
#undef main
#include "native/providers/gdn/sm121_q2_publication.h"

struct PublicationState final : qrt_sm121_q2::TargetCacheOwner {
    OriginalState original;
    Inputs tails;
    qrt_sm121_q2::TargetSnapshot state;
    mutable bool invalid=false;
    PublicationState(std::shared_ptr<const TableOwner> tables,unsigned position,uint32_t current,bool fp32,bool key_major)
        :original(std::move(tables),position,current,fp32,key_major),state(original.state) {
        state.owner=this;
        // New mutable tails are separate from the complete original input
        // buffers. Five extra rows expose rejected-row and padding overwrites.
        for(unsigned layer=3u;layer<40u;layer+=4u) {
            const std::string prefix="layer-"+std::to_string(layer)+"/";
            const unsigned split=state.attention[layer].prefix.tokens;
            const unsigned committed=position-split,capacity=committed+5u;
            auto keys=read_role<uint16_t>(prefix+"history_k",size_t(position+1u)*512u);
            auto values=read_role<uint16_t>(prefix+"history_v",size_t(position+1u)*512u);
            keys.erase(keys.begin(),keys.begin()+size_t(split)*512u);
            values.erase(values.begin(),values.begin()+size_t(split)*512u);
            keys.resize(size_t(committed)*512u);values.resize(size_t(committed)*512u);
            keys.resize(size_t(capacity)*512u,0xa5a5u);values.resize(size_t(capacity)*512u,0xa5a5u);
            auto& tail=state.attention[layer].decoded;
            tail={nullptr,nullptr,committed,capacity,512u,fp32?4u:2u};
            if(fp32) {
                std::vector<float> k(keys.size()),v(values.size());
                for(size_t i=0;i<k.size();++i){k[i]=qrt_sm121_q1::widen(keys[i]);v[i]=qrt_sm121_q1::widen(values[i]);}
                tail.keys=tails.upload(k);tail.values=tails.upload(v);
            } else {tail.keys=tails.upload(keys);tail.values=tails.upload(values);}
        }
    }
    bool snapshot(qrt_sm121_q2::TargetSnapshot* output)const override {
        if(!output||invalid)return false;*output=state;return true;
    }
    bool matches(const qrt_sm121_q2::TargetSnapshot& other)const noexcept override {
        return !invalid&&other.owner==this&&other.generation==state.generation&&other.model_epoch==state.model_epoch&&
            other.processed_tokens==state.processed_tokens&&other.current_token==state.current_token;
    }
    bool prepare_publication(const qrt_sm121_q2::TargetSnapshot& snapshot,unsigned rows)const override {
        return matches(snapshot)&&rows>=1u&&rows<=2u;
    }
    void invalidate()const noexcept override {invalid=true;}
    void quarantine()const noexcept override {invalid=true;}
};

struct PublicationConfiguration : Configuration {
    unsigned accepted_rows=0;
    double publication_seconds=0;
};

PublicationConfiguration publish_case(qrt_sm121_q2::Target& target,const qrt_sm121_q2::ModelWeightBinding& binding,
    const std::shared_ptr<PublicationState>& source,const std::array<uint32_t,2>& inputs,
    bool fp32,bool key_major,unsigned rows) {
    using namespace qrt_sm121_q2;
    PublicationConfiguration result;result.fp32=fp32;result.key_major=key_major;result.accepted_rows=rows;
    check(hipDeviceSynchronize()); // Order all default-stream guard initialization.
    hipStream_t stream=nullptr;check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    TargetResult evaluated;
    const auto start=std::chrono::steady_clock::now();
    const auto computed=target.evaluate(binding,source,inputs,&evaluated,1024u,stream);
    if(computed.status!=hipSuccess)throw std::runtime_error(std::string(computed.stage)+": "+hipGetErrorString(computed.status));
    result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    result.workspace_bytes=evaluated.allocated_bytes();
    const auto host=*evaluated.host();result.tokens=host.tokens;result.logits=host.logits;
    const auto expected_norm=read_role<uint16_t>("expected_final_norm",4096u);
    Comparison norm;norm.name="private_final_norm";
    for(size_t i=0;i<4096u;++i)norm.add(host.normalized[i],expected_norm[i],i);
    result.comparisons.push_back(norm);
    const auto logits=read_role<uint16_t>("expected_logits",size_t(2u)*qrt_sm121_mtp::head_vocabulary);
    result.comparisons.push_back(compare("private_logits",evaluated.vocabulary_logits(),logits));
    for(unsigned row=0;row<2u;++row) {
        qrt_sm121_mtp::HeadBest best;
        for(unsigned token=0;token<qrt_sm121_mtp::head_vocabulary;++token)
            if(!qrt_sm121_mtp::head_candidate(logits[size_t(row)*qrt_sm121_mtp::head_vocabulary+token],token,&best))
                throw std::runtime_error("nonfinite original logits");
        result.sampling_errors+=(best.token!=host.tokens[row])+(best.logit!=host.logits[row]);
    }
    const auto publication_start=std::chrono::steady_clock::now();
    const auto published=CachePublisher::publish(evaluated,*source,rows,stream);
    if(published.status!=hipSuccess||!evaluated.cache_published())
        throw std::runtime_error(std::string(published.stage)+": "+hipGetErrorString(published.status));
    result.publication_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-publication_start).count();
    check(hipStreamDestroy(stream));stream=nullptr;
    const unsigned position=source->state.processed_tokens;
    for(unsigned layer=0;layer<40u;++layer) {
        const std::string prefix="layer-"+std::to_string(layer)+"/";
        if(layer%4u==3u) {
            const auto& view=source->state.attention[layer];
            const auto qkv=read_role<uint16_t>(prefix+"expected_qkv",18432u);
            const auto rope=read_role<uint16_t>(prefix+"expected_k_rope",1024u);
            for(bool value:{false,true}) {
                auto expected=read_role<uint16_t>(prefix+(value?"history_v":"history_k"),size_t(position+1u)*512u);
                expected.erase(expected.begin(),expected.begin()+size_t(view.prefix.tokens)*512u);
                expected.resize(size_t(view.decoded.tokens)*512u);
                expected.resize(size_t(view.decoded.capacity)*512u,0xa5a5u);
                for(unsigned row=0;row<rows;++row)
                    std::copy_n(value?qkv.data()+row*9216u+8704u:rope.data()+row*512u,512u,
                        expected.data()+size_t(view.decoded.tokens+row)*512u);
                const auto* pointer=value?view.decoded.values:view.decoded.keys;
                const auto name=prefix+(value?"published_v_tail":"published_k_tail");
                if(fp32) {
                    std::vector<float> widened(expected.size());
                    for(size_t i=0;i<expected.size();++i)widened[i]=qrt_sm121_q1::widen(expected[i]);
                    result.comparisons.push_back(compare(name,static_cast<const float*>(pointer),widened));
                } else result.comparisons.push_back(compare(name,static_cast<const uint16_t*>(pointer),expected));
            }
            // Every original prefix allocation (including the unused original
            // tail extent) remains byte-identical.
            for(unsigned plane=0;plane<2u;++plane)
                result.immutable+=source->original.inputs.buffers[layer*2u+plane]->verify().second;
        } else {
            auto states=read_role<float>(prefix+"expected_states",2u*state_elements);
            states.erase(states.begin(),states.begin()+size_t(rows-1u)*state_elements);states.resize(state_elements);
            result.comparisons.push_back(compare(prefix+"published_state",source->state.linear[layer].state,states,key_major));
            auto rings=read_role<uint16_t>(prefix+"expected_rings",2u*ring_elements);
            rings.erase(rings.begin(),rings.begin()+size_t(rows-1u)*ring_elements);rings.resize(ring_elements);
            const auto* pointer=source->state.linear[layer].ring;
            if(fp32) {
                std::vector<float> expected(rings.size());for(size_t i=0;i<rings.size();++i)expected[i]=qrt_sm121_q1::widen(rings[i]);
                result.comparisons.push_back(compare(prefix+"published_ring",static_cast<const float*>(pointer),expected));
            } else result.comparisons.push_back(compare(prefix+"published_ring",static_cast<const uint16_t*>(pointer),rings));
        }
    }
    result.guards=source->original.inputs.verify().first+source->tails.verify().first;
    // Publication is exactly once; a second invocation must submit no writes.
    if(CachePublisher::publish(evaluated,*source,rows).status==hipSuccess)throw std::runtime_error("duplicate publication accepted");
    return result;
}

int main(int argc,char** argv)try {
    if(argc!=5)throw std::runtime_error("bound TSV plan first-position first-input second-input");
    const unsigned position=unsigned(number(argv[2],263673u));
    const std::array<uint32_t,2> inputs{uint32_t(number(argv[3],248319u)),uint32_t(number(argv[4],248319u))};
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("plan missing");std::string line;
    while(std::getline(file,line)) {
        if(!line.empty()&&line.back()=='\r')line.pop_back();
        std::istringstream row(line);std::string name,path,offset,bytes,digest,extra;
        if(!std::getline(row,name,'\t')||!std::getline(row,path,'\t')||!std::getline(row,offset,'\t')||
           !std::getline(row,bytes,'\t')||!std::getline(row,digest,'\t')||std::getline(row,extra,'\t')||
           digest.size()!=64u||digest.find_first_not_of("0123456789abcdef")!=std::string::npos||
           !plan.emplace(name,Entry{path,digest,size_t(number(offset,uint64_t(8)<<30u)),size_t(number(bytes,uint64_t(1)<<30u))}).second)
            throw std::runtime_error("plan row");
    }
    auto original=std::make_shared<OriginalWeights>();qrt_sm121_q2::ModelWeights weights;
    const auto prepared=weights.prepare(original,1u);if(prepared.status!=hipSuccess)throw std::runtime_error(prepared.stage);
    const auto binding=weights.binding(1u);auto tables=std::make_shared<TableOwner>();qrt_sm121_q2::Target target;
    std::vector<PublicationConfiguration> configurations;
    for(bool fp32:{false,true})for(bool key_major:{false,true})for(unsigned rows:{1u,2u}) {
        auto source=std::make_shared<PublicationState>(tables,position,inputs[0],fp32,key_major);
        configurations.push_back(publish_case(target,binding,source,inputs,fp32,key_major,rows));
        std::cerr<<"completed publication configuration "<<configurations.size()<<"/8\n"<<std::flush;
    }
    const auto model_guards=original->inputs.verify(),table_guards=tables->inputs.verify();
    size_t guards=model_guards.first+table_guards.first,immutable=model_guards.second+table_guards.second;
    size_t elements=0,mismatches=0,sampling=0;unsigned count=0;
    std::cout<<std::setprecision(17)<<"{\"kind\":\"original_q2_cache_publication\",\"first_position\":"<<position
        <<",\"scheduled_inputs\":["<<inputs[0]<<','<<inputs[1]<<"],\"original_weight_count\":633,\"configurations\":[";
    for(const auto& c:configurations) {
        if(count++)std::cout<<',';guards+=c.guards;immutable+=c.immutable;sampling+=c.sampling_errors;
        std::cout<<"{\"fp32_carriers\":"<<(c.fp32?"true":"false")<<",\"key_major\":"<<(c.key_major?"true":"false")
            <<",\"accepted_rows\":"<<c.accepted_rows<<",\"execution_seconds\":"<<c.seconds
            <<",\"publication_seconds\":"<<c.publication_seconds<<",\"workspace_bytes\":"<<c.workspace_bytes<<",\"samples\":[";
        for(unsigned row=0;row<2u;++row){if(row)std::cout<<',';std::cout<<"{\"token\":"<<c.tokens[row]<<",\"logit\":"<<c.logits[row]<<'}';}
        std::cout<<"],\"stages\":[";unsigned n=0;
        for(const auto& s:c.comparisons) {
            if(n++)std::cout<<',';elements+=s.elements;mismatches+=s.mismatches;
            std::cout<<"{\"stage\":\""<<s.name<<"\",\"elements\":"<<s.elements<<",\"bit_mismatches\":"<<s.mismatches
                <<",\"first_difference\":["<<s.first<<','<<s.actual<<','<<s.expected<<"]}";
        }
        std::cout<<"]}";
    }
    const bool passed=!guards&&!immutable&&!mismatches&&!sampling;
    std::cout<<"],\"compared_elements\":"<<elements<<",\"bit_mismatches\":"<<mismatches
        <<",\"guard_errors\":"<<guards<<",\"immutable_digest_errors\":"<<immutable<<",\"sampling_errors\":"<<sampling
        <<",\"passed\":"<<(passed?"true":"false")
        <<",\"native_execution\":true,\"accepted_cache_published\":true,\"actual_model_request\":false,\"inference_acceptance\":false}\n";
    return passed?0:1;
} catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 2;}
