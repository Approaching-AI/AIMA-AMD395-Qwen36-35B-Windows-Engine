// Queued host checks exercise production ownership, not device arithmetic.
#include <algorithm>
#include <cassert>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>
#include "native/providers/gdn/sm121_mtp_moe_layout.h"
#include "native/providers/gdn/sm121_mtp_head_math.h"
using hipError_t = int;
using hipStream_t = void*;
constexpr int hipSuccess=0, hipErrorInvalidValue=1, hipErrorOutOfMemory=2;
constexpr int hipMemcpyDeviceToHost=3, injected=99;
constexpr int hipMemcpyDeviceToDevice=4;
constexpr int hipMemcpyHostToDevice=5;
struct Allocation { size_t bytes; bool host; };
static std::map<void*,Allocation> allocations;
static std::vector<std::function<void()>> queued;
static std::vector<std::string> stages;
static unsigned allocation_call=0, fail_allocation=0, sync_call=0, fail_sync=0, copies=0;
static unsigned fail_stage_occurrence=1, failing_stage_seen=0;
static std::string fail_stage;
static bool invalid_input=false, invalid_moe=false, invalid_head=false, invalid_id=false, invalid_logit=false;
static hipStream_t expected_stream=reinterpret_cast<void*>(0x1234);
static const uint16_t *fusion_weight=reinterpret_cast<uint16_t*>(0x10000),
    *query_weight=reinterpret_cast<uint16_t*>(0x20000), *output_weight=reinterpret_cast<uint16_t*>(0x30000),
    *post_weight=reinterpret_cast<uint16_t*>(0x40000), *final_weight=reinterpret_cast<uint16_t*>(0x50000);
static unsigned attention_tokens=0, attention_first=0, attention_rows=0;
static std::vector<uint16_t> expected_hidden;
static std::vector<uint32_t> expected_ids;
static bool expected_split=true;
static uint16_t cache_fill=55u;
static std::function<void()> completion_hook;
static std::function<void(const uint16_t*,const uint32_t*,unsigned,bool)> input_observer;
static std::function<void(const uint16_t*,unsigned)> cache_observer;
static hipError_t allocate(void** pointer,size_t bytes,bool host) {
    if(++allocation_call==fail_allocation)return hipErrorOutOfMemory;
    assert(!posix_memalign(pointer,256u,(bytes+255u)&~size_t(255u)));
    allocations.emplace(*pointer,Allocation{bytes,host});std::memset(*pointer,0xa5,bytes);return hipSuccess;
}
static hipError_t hipMalloc(void** p,size_t n){return allocate(p,n,false);}
static hipError_t hipHostMalloc(void** p,size_t n){return allocate(p,n,true);}
static hipError_t release(void* pointer,bool host) {
    assert(allocations.count(pointer)&&allocations.at(pointer).host==host);
    allocations.erase(pointer);std::free(pointer);return hipSuccess;
}
static hipError_t hipFree(void* p){return release(p,false);}
static hipError_t hipHostFree(void* p){return release(p,true);}
static hipError_t enqueue(const std::string& stage,hipStream_t stream,std::function<void()> action) {
    assert(stream==expected_stream);stages.push_back(stage);queued.push_back(action);
    return fail_stage==stage && ++failing_stage_seen==fail_stage_occurrence?injected:hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t stream) {
    assert(stream==expected_stream);if(++sync_call==fail_sync)return injected;
    auto work=std::move(queued);queued.clear();for(auto& action:work)action();
    if(completion_hook){auto hook=std::move(completion_hook);completion_hook={};hook();}
    return hipSuccess;
}
static hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t stream) {
    return enqueue("clear",stream,[=]{std::memset(p,value,bytes);});
}
static hipError_t hipMemcpyAsync(void* p,const void* q,size_t bytes,int kind,hipStream_t stream) {
    if(kind==hipMemcpyDeviceToDevice)
        return enqueue("weight_copy"+std::to_string(++copies),stream,[=]{std::memcpy(p,q,bytes);});
    if(kind==hipMemcpyHostToDevice){
        assert(allocations.count(const_cast<void*>(q))&&allocations.at(const_cast<void*>(q)).host);
        assert(allocations.at(const_cast<void*>(q)).bytes>=bytes);
        assert(allocations.count(p)&&!allocations.at(p).host&&allocations.at(p).bytes>=bytes);
        return enqueue("input_copy",stream,[=]{std::memcpy(p,q,bytes);});
    }
    assert(kind==hipMemcpyDeviceToHost);bool pinned=false;
    for(const auto& pair:allocations) {
        auto begin=reinterpret_cast<uintptr_t>(pair.first),address=reinterpret_cast<uintptr_t>(p);
        if(pair.second.host && address>=begin && address-begin<=pair.second.bytes &&
           bytes<=pair.second.bytes-(address-begin))pinned=true;
    }
    assert(pinned);return enqueue("copy"+std::to_string(++copies),stream,[=]{std::memcpy(p,q,bytes);});
}
namespace qrt_sm121_mtp {
static hipError_t launch_fusion_inputs(const uint16_t*,const uint16_t* hidden,const uint32_t* ids,const uint16_t*,
    const uint16_t*,const unsigned char*,unsigned rows,uint16_t* out,uint32_t* invalid,hipStream_t stream,bool split) {
    return enqueue("gather",stream,[=]{
        if(input_observer)input_observer(hidden,ids,rows,split);
        if(!expected_hidden.empty()){
            assert(expected_hidden.size()==size_t(rows)*2048u&&expected_ids.size()==rows&&split==expected_split);
            assert(std::equal(expected_hidden.begin(),expected_hidden.end(),hidden));
            assert(std::equal(expected_ids.begin(),expected_ids.end(),ids));
        }
        *invalid=invalid_input?1u:0u;std::fill_n(out,rows*4096u,11u);});
}
static hipError_t launch_normalize(const uint16_t* in,const uint16_t*,const unsigned char*,unsigned rows,
    uint16_t* out,hipStream_t stream) {
    return enqueue("prompt_norm",stream,[=]{assert(in[0]==22u);std::fill_n(out,rows*2048u,33u);});
}
static hipError_t launch_key_values(const uint16_t* in,const uint16_t*,const unsigned char*,const uint16_t*,
    unsigned,unsigned first,unsigned rows,unsigned capacity,uint16_t* out,uint16_t*,hipStream_t stream) {
    const auto value=cache_fill;
    assert(first+rows<=capacity);return enqueue("cache",stream,[=]{assert(in[0]==44u);
        std::fill_n(out+first*1024u,rows*1024u,value);});
}
static hipError_t launch_projection(const uint16_t* weights,const uint16_t* in,uint16_t* out,
    unsigned output,unsigned input,unsigned rows,unsigned maximum_blocks,hipStream_t stream) {
    assert(maximum_blocks==7u || maximum_blocks==1024u);
    const bool fc=weights==fusion_weight,q=weights==query_weight,o=weights==output_weight;
    assert((fc||o)?(output==2048u&&input==4096u):(q?(output==8192u&&input==2048u):(output==1024u&&input==2048u)));
    return enqueue(fc?"fusion":q?"query":o?"output":"kv",stream,[=]{
        assert(in[0]==(fc?11u:q?33u:o?71u:33u));std::fill_n(out,rows*output,fc?22u:q?66u:o?72u:44u);});
}
static hipError_t launch_queries(const uint16_t* in,const uint16_t*,const unsigned char*,const uint16_t*,
    unsigned,unsigned,unsigned rows,uint16_t* q,uint16_t* gate,uint16_t*,hipStream_t stream) {
    return enqueue("queries",stream,[=]{assert(in[0]==66u);std::fill_n(q,rows*4096u,67u);std::fill_n(gate,rows*4096u,68u);});
}
static hipError_t launch_attention(const uint16_t* q,const uint16_t* cache,unsigned tokens,unsigned first,
    unsigned rows,const unsigned char*,const unsigned char*,float*,unsigned stride,float*,uint16_t* out,hipStream_t stream) {
    attention_tokens=tokens;attention_first=first;attention_rows=rows;
    assert(first+rows<=tokens&&stride>=first+rows&&stride%32u==0u);
    return enqueue("attention",stream,[=]{assert(q[0]==67u);
        if(cache_observer)cache_observer(cache,tokens);
        else for(unsigned i=0;i<tokens;++i)assert(cache[i*1024u]==55u);
        std::fill_n(out,rows*4096u,70u);});
}
static hipError_t launch_gate(const uint16_t* in,const uint16_t* gate,const uint16_t*,uint16_t* out,
    unsigned rows,hipStream_t stream) {
    return enqueue("gate",stream,[=]{assert(in[0]==70u&&gate[0]==68u);std::fill_n(out,rows*4096u,71u);});
}
static hipError_t launch_residual_normalize(const uint16_t* in,const uint16_t* residual,const uint16_t* weight,
    const unsigned char*,unsigned rows,uint16_t* out,uint16_t* residual_out,hipStream_t stream) {
    bool post=weight==post_weight;assert(post||weight==final_weight);
    return enqueue(post?"post_norm":"final_norm",stream,[=]{assert(in[0]==(post?72u:75u));
        assert(residual[0]==(post?22u:74u));std::fill_n(out,rows*2048u,post?73u:76u);
        std::fill_n(residual_out,rows*2048u,post?74u:77u);});
}
struct MoeWeights {const uint16_t *router,*shared_gate,*shared_gate_up,*shared_down,*routed_gate_up,*routed_down;};
struct MoeTables {const uint16_t *silu,*sigmoid;const uint32_t* router_exp_fraction;};
static hipError_t launch_moe(const uint16_t* in,const MoeWeights&,const MoeTables&,void* workspace,
    size_t bytes,unsigned rows,unsigned,hipStream_t stream) {
    MoeBuffers b;assert(bind_moe_buffers(workspace,bytes,rows,&b));
    return enqueue("moe",stream,[=]{assert(in[0]==73u);std::fill_n(b.output,rows*2048u,75u);*b.invalid=invalid_moe?1u:0u;});
}
static hipError_t launch_head(const uint16_t*,const uint16_t* in,uint16_t* logits,uint32_t* ids,float* values,
    uint32_t* invalid,unsigned rows,unsigned,hipStream_t stream) {
    return enqueue("head",stream,[=]{assert(in[0]==76u);std::fill_n(logits,rows*head_vocabulary,0x4000u);
        *invalid=invalid_head?1u:0u;for(unsigned row=0;row<rows;++row){ids[row]=invalid_id?head_vocabulary:200u+row;
        values[row]=invalid_logit?INFINITY:12.5f+row;}});
}
}
#include "sm121_mtp_prompt_cache.h"
#include "sm121_mtp_model_weights.h"
#include "sm121_mtp_cache_snapshot.h"
#include "sm121_mtp_drafter.h"
#include "sm121_mtp_target_inputs.h"
#include "sm121_mtp_request.h"
namespace qrt_sm121_mtp_runtime {
static bool fail_tables=false;
static hipError_t prepare(qrt_sm121_mtp::DrafterTables* out,unsigned last) {
    assert(last<262144u);if(fail_tables)return injected;
    const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
    const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
    *out={t,p,262144u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};return hipSuccess;
}
}
#include "sm121_mtp_prefill_probe.h"
#include "sm121_mtp_request_seed.h"
#include "sm121_mtp_prefix_seed.h"
struct LeasedWeights final:qrt_sm121_mtp::ModelWeightSource {
    std::array<std::vector<uint16_t>,4> packed_parts;
    uint64_t generation=10;
    LeasedWeights() {for(auto& part:packed_parts)part.assign(1048576u,0x3f80u);}
    uint64_t epoch()const noexcept override{return generation;}
    bool tensor(const char* name,qrt_sm121_mtp::ModelTensorView* out)const override {
        using namespace qrt_sm121_mtp;
        const auto found=std::find_if(model_weight_specs.begin(),model_weight_specs.end(),
            [&](const auto& spec){return std::strcmp(spec.name,name)==0;});
        if(found==model_weight_specs.end())return false;
        auto* p=reinterpret_cast<const uint16_t*>(0x60000);
        switch(static_cast<ModelWeight>(found-model_weight_specs.begin())){
        case ModelWeight::Fusion:p=fusion_weight;break;
        case ModelWeight::Query:p=query_weight;break;
        case ModelWeight::Output:p=output_weight;break;
        case ModelWeight::PostAttentionNorm:p=post_weight;break;
        case ModelWeight::FinalNorm:p=final_weight;break;
        case ModelWeight::Key:p=packed_parts[0].data();break;
        case ModelWeight::Value:p=packed_parts[1].data();break;
        case ModelWeight::SharedGateProjection:p=packed_parts[2].data();break;
        case ModelWeight::SharedUpProjection:p=packed_parts[3].data();break;
        default:break;
        }
        *out={found->name,p,found->rank,found->shape,found->bytes(),generation,true,true};return true;
    }
};
static void reset() {
    assert(queued.empty());allocation_call=fail_allocation=sync_call=fail_sync=copies=0;
    stages.clear();fail_stage.clear();invalid_input=invalid_moe=invalid_head=invalid_id=invalid_logit=false;
    expected_hidden.clear();expected_ids.clear();expected_split=true;
    cache_fill=55u;completion_hook={};input_observer={};cache_observer={};
    fail_stage_occurrence=1;failing_stage_seen=0;
}
static bool bind(qrt_sm121_mtp::Drafter& d,uint64_t epoch=10u) {
    using namespace qrt_sm121_mtp;
    const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
    const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
    DrafterWeights w{{p,p,p,fusion_weight,p,p,p},query_weight,p,output_weight,post_weight,
        {p,p,p,p,p,p},final_weight,p};
    DrafterTables tables{t,p,262144u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
    return d.bind(w,tables,epoch);
}
static qrt_sm121_mtp::PromptStep append(qrt_sm121_mtp::Drafter& d,unsigned first,unsigned rows,uint64_t epoch=10u) {
    static uint16_t hidden=0;static uint32_t id=1;
    return d.append_target(&hidden,&id,first,rows,true,epoch,expected_stream,7u);
}
static qrt_sm121_mtp::DraftStep propose(qrt_sm121_mtp::Drafter& d,unsigned first,unsigned rows,uint64_t epoch=10u) {
    return d.propose(first,rows,epoch,expected_stream,7u);
}
static void late_completion() {
    assert(!allocations.empty()&&!queued.empty());fail_sync=0;
    assert(hipStreamSynchronize(expected_stream)==hipSuccess);
    while(!allocations.empty())release(allocations.begin()->first,allocations.begin()->second.host);
}
static std::unique_ptr<qrt_mtp_target_rows::PrefillRows> target_batch(unsigned rows=2u) {
    std::vector<uint32_t> ids(rows);for(unsigned i=0;i<rows;++i)ids[i]=100u+i;
    auto batch=std::make_unique<qrt_mtp_target_rows::PrefillRows>(ids.data(),ids.size(),0u,rows);
    std::vector<float> hidden(size_t(rows)*2048u);
    for(size_t i=0;i<hidden.size();++i)hidden[i]=1.0f+float(i%127)/128.0f;
    assert(batch->stage(batch->local_rows(),hidden,999u)&&batch->publish(999u));
    expected_hidden=batch->hidden();expected_ids=batch->shifted_tokens();return batch;
}
static void test_target_inputs() {
    using namespace qrt_sm121_mtp;
    reset();{Drafter d;TargetInputs inputs;uint32_t token=100;
        qrt_mtp_target_rows::PrefillRows unpublished(&token,1u,0u,1u);
        assert(inputs.append(d,unpublished,true,10u,expected_stream).status==hipErrorInvalidValue);
        assert(!inputs.allocated_bytes()&&allocations.empty()&&stages.empty());
    }
    for(unsigned rows:{2u,8192u}){
        reset();{Drafter d;TargetInputs inputs;assert(bind(d)&&d.reserve(rows,rows)==hipSuccess);
            auto batch=target_batch(rows);
            assert(inputs.append(d,*batch,true,10u,expected_stream).status==hipSuccess);
            assert(d.retained_tokens()==rows&&inputs.allocated_bytes()==size_t(rows)*8200u);
            assert(propose(d,rows-1u,1u).rows==1u);
            assert(d.truncate(0u,10u));
            const auto calls=allocation_call;expected_split=false;
            assert(inputs.append(d,*batch,false,10u,expected_stream).status==hipSuccess);
            assert(allocation_call==calls&&queued.empty());
        }assert(allocations.empty());
    }
    // Upload failure, stale model and failed drafter entry cannot leave H2D
    // work behind or expose a new completed KV prefix.
    for(unsigned failure=0;failure<5u;++failure){
        reset();{Drafter d;TargetInputs inputs;assert(bind(d)&&d.reserve(2,2)==hipSuccess);
            auto batch=target_batch();allocation_call=0;
            if(failure<2)fail_allocation=failure+1u;
            if(failure==2)fail_stage="input_copy";
            if(failure==4)fail_stage="gather";
            const auto result=inputs.append(d,*batch,true,failure==3?9u:10u,expected_stream);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&!inputs.quarantined());
            assert(!d.retained_tokens()&&queued.empty());
            if(failure<2)assert(!inputs.allocated_bytes());
            else assert(inputs.allocated_bytes()==16400u);
            reset();batch=target_batch();
            assert(inputs.append(d,*batch,true,10u,expected_stream).status==hipSuccess);
        }assert(allocations.empty());
    }
    // Destroy the original host batch and both request owners before the
    // delayed copy/kernel is allowed to finish. ASan observes actual reads.
    for(unsigned failure=0;failure<4u;++failure){
        reset();{Drafter d;TargetInputs inputs;assert(bind(d)&&d.reserve(2,2)==hipSuccess);
            auto batch=target_batch();sync_call=0;fail_sync=failure<2?1u:2u;
            if(failure==1)fail_stage="input_copy";
            if(failure==3)fail_stage="gather";
            const auto result=inputs.append(d,*batch,true,10u,expected_stream);
            assert(result.status==injected&&result.completion_unknown&&inputs.quarantined());
            assert(d.quarantined()==(failure>=2)&&!d.retained_tokens());
            const auto calls=stages.size();batch->discard();
            assert(inputs.append(d,*batch,true,10u,expected_stream).completion_unknown&&stages.size()==calls);
        }
        assert(allocations.size()==(failure<2?2u:27u));late_completion();assert(allocations.empty());
    }
}
static void test_prefill_probe(const std::string& directory) {
    using namespace qrt_sm121_mtp;
    using namespace qrt_sm121_mtp_runtime;
    for(unsigned rows:{7169u,8192u}){
        reset();expected_stream=nullptr;{
            auto batch=target_batch(rows);std::string stage,failure;
            auto source=std::make_shared<LeasedWeights>();
            const std::string prefix=directory+"/prefill-"+std::to_string(rows);
            assert(probe_prefill(*batch,source,prefix,stage,failure));
            assert(allocations.empty()&&queued.empty());
            // A repeated diagnostic cannot replace any prior capture file.
            assert(!probe_prefill(*batch,source,prefix,stage,failure));
            assert(stage=="mtp_prefill_capture_path"&&allocations.empty());
            if(rows==7169u){
                fail_tables=true;
                assert(!probe_prefill(*batch,source,prefix+"-no-table",stage,failure));
                assert(stage=="mtp_prefill_probe_tables"&&allocations.empty());fail_tables=false;
                assert(!probe_prefill(*batch,nullptr,prefix+"-no-source",stage,failure));
                assert(stage=="mtp_prefill_probe_contract"&&allocations.empty());
            }
        }assert(allocations.empty());expected_stream=reinterpret_cast<void*>(0x1234);
    }
    // Actual observer copies: reject a partial transfer, drain every queued
    // read, and leave the completed proposal usable on known completion.
    for(unsigned index=0;index<=25u;++index){
        reset();{Drafter d;assert(bind(d)&&d.reserve(2,2)==hipSuccess&&append(d,0,2).status==hipSuccess);
            const auto proposal=propose(d,1,1);auto batch=target_batch();reset();
            if(!index)fail_allocation=1;else fail_stage="copy"+std::to_string(index);
            std::string stage,failure;const std::string prefix=directory+"/capture-copy-"+std::to_string(index);
            assert(!prefill_probe_detail::capture(d,proposal,10u,*batch,prefix,0u,0u,1u,stage,failure,expected_stream));
            assert(stage==(index?"mtp_prefill_capture_copy":"mtp_prefill_capture_allocation"));
            assert(!d.quarantined()&&d.observation(10u).rows==1u&&queued.empty()&&allocations.size()==25u);
            assert(!std::filesystem::exists(prefix+".json"));
        }assert(allocations.empty());
    }
    // Every failed-enqueue boundary may still have a partial read in flight.
    // Retain pinned outputs plus ALL borrowed KV/proposal buffers after scope.
    for(unsigned index=0;index<=25u;++index){
        reset();{Drafter d;assert(bind(d)&&d.reserve(2,2)==hipSuccess&&append(d,0,2).status==hipSuccess);
            const auto proposal=propose(d,1,1);auto batch=target_batch();reset();fail_sync=1;
            if(index)fail_stage="copy"+std::to_string(index);
            std::string stage,failure;const std::string prefix=directory+"/capture-sync-"+std::to_string(index);
            assert(!prefill_probe_detail::capture(d,proposal,10u,*batch,prefix,0u,0u,1u,stage,failure,expected_stream));
            assert(stage=="mtp_prefill_capture_completion"&&d.quarantined()&&!d.observation(10u).rows);
            assert(!std::filesystem::exists(prefix+".json"));
        }assert(allocations.size()==26u);late_completion();assert(allocations.empty());
    }
}
static void seed_leased(qrt_sm121_mtp::Drafter& drafter,const std::shared_ptr<LeasedWeights>& source) {
    using namespace qrt_sm121_mtp;
    ModelWeights model;assert(model.prepare(source,10u,expected_stream).status==hipSuccess);
    const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
    const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
    DrafterTables tables{t,p,262144u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
    assert(drafter.reserve(8u,4u)==hipSuccess&&drafter.bind(model.binding(10u),tables,10u));
    assert(append(drafter,0u,2u).status==hipSuccess);
}
static void test_checkpoints() {
    using namespace qrt_sm121_mtp;
    reset();{Drafter raw;DrafterCheckpoint absent;
        assert(bind(raw)&&raw.reserve(4,2)==hipSuccess&&append(raw,0,2).status==hipSuccess);
        reset();assert(raw.checkpoint(&absent,10u,expected_stream).status==hipErrorInvalidValue);
        assert(!absent.tokens()&&stages.empty()); // A bare epoch is not a model lease.
    }assert(allocations.empty());
    reset();{
        DrafterCheckpoint saved;std::weak_ptr<LeasedWeights> borrowed;
        {auto source=std::make_shared<LeasedWeights>();borrowed=source;Drafter original;
            seed_leased(original,source);assert(original.checkpoint(&saved,10u,expected_stream).status==hipSuccess);
            assert(saved.tokens()==2u&&saved.epoch()==10u&&saved.allocated_bytes()==4096u);
            assert(saved.cache_data(10u)!=original.cache_data());
            assert(original.truncate(0u,10u));cache_fill=99u;
            assert(append(original,0u,2u).status==hipSuccess&&original.cache_data()[0]==99u);
            assert(std::all_of(saved.cache_data(10u),saved.cache_data(10u)+2048u,[](auto x){return x==55u;}));
        }
        assert(allocations.size()==2u&&!borrowed.expired());
        auto alias=saved;saved={};assert(alias.valid(10u)&&!alias.valid(11u));
        reset();{Drafter restored;assert(restored.restore(alias,8u,2u,10u,expected_stream).status==hipSuccess);
            assert(restored.retained_tokens()==2u&&restored.cache_data()!=alias.cache_data(10u));
            assert(!restored.observation(10u).rows&&propose(restored,1u,1u).status==hipErrorInvalidValue);
            assert(append(restored,2u,2u).status==hipSuccess&&propose(restored,3u,1u).rows==1u);
            assert(alias.tokens()==2u&&alias.cache_data(10u)[0]==55u);
        }
        assert(allocations.size()==2u);alias={};assert(borrowed.expired());
    }assert(allocations.empty());
    for(unsigned kind=0;kind<3u;++kind){
        reset();{auto source=std::make_shared<LeasedWeights>();Drafter d;seed_leased(d,source);
            DrafterCheckpoint saved;assert(d.checkpoint(&saved,10u,expected_stream).status==hipSuccess);
            const auto* original=saved.cache_data(10u);reset();
            if(!kind)fail_allocation=1;
            if(kind==1)fail_stage="weight_copy1";
            if(kind==2)completion_hook=[&]{source->generation=11u;};
            const auto result=d.checkpoint(&saved,10u,expected_stream);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&!d.quarantined()&&queued.empty());
            assert(allocations.size()==27u&&saved.tokens()==2u);source->generation=10u;
            assert(saved.cache_data(10u)==original);
        }assert(allocations.empty());
    }
    for(unsigned kind=0;kind<3u;++kind){
        reset();{auto source=std::make_shared<LeasedWeights>();DrafterCheckpoint saved;
            {Drafter d;seed_leased(d,source);assert(d.checkpoint(&saved,10u,expected_stream).status==hipSuccess);}
            assert(allocations.size()==2u);Drafter restored;reset();
            assert(restored.restore(saved,1u,2u,10u,expected_stream).status==hipErrorInvalidValue&&stages.empty());
            assert(restored.restore(saved,8u,2u,11u,expected_stream).status==hipErrorInvalidValue&&stages.empty());
            if(!kind)fail_allocation=1;
            if(kind==1)fail_stage="weight_copy1";
            if(kind==2)completion_hook=[&]{source->generation=11u;};
            const auto result=restored.restore(saved,8u,2u,10u,expected_stream);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&!restored.quarantined());
            assert(!restored.retained_tokens()&&queued.empty());source->generation=10u;reset();
            assert(saved.valid(10u)&&restored.restore(saved,8u,2u,10u,expected_stream).status==hipSuccess);
            assert(restored.retained_tokens()==2u&&saved.cache_data(10u)[0]==55u);
        }assert(allocations.empty());
    }
    for(bool partial:{false,true}){
        reset();{auto source=std::make_shared<LeasedWeights>();Drafter d;seed_leased(d,source);
            DrafterCheckpoint saved;reset();fail_sync=1;if(partial)fail_stage="weight_copy1";
            const auto result=d.checkpoint(&saved,10u,expected_stream);
            assert(result.completion_unknown&&d.quarantined()&&!saved.valid(10u));
        }
        assert(allocations.size()==27u);late_completion();assert(allocations.empty());
    }
    for(bool partial:{false,true}){
        reset();{auto source=std::make_shared<LeasedWeights>();DrafterCheckpoint saved;
            {Drafter d;seed_leased(d,source);assert(d.checkpoint(&saved,10u,expected_stream).status==hipSuccess);}
            Drafter restored;reset();fail_sync=1;if(partial)fail_stage="weight_copy1";
            const auto result=restored.restore(saved,8u,2u,10u,expected_stream);
            assert(result.completion_unknown&&restored.quarantined()&&!saved.valid(10u));
        }
        assert(allocations.size()==27u);late_completion();assert(allocations.empty());
    }
}
#include "mtp_request_host.inc"
#include "mtp_chunked_request_host.inc"
#include "mtp_retired_request_host.inc"
#include "mtp_prefix_request_host.inc"
#include "mtp_prefix_runtime_host.inc"
int main(int argc,char** argv){
    assert(argc==2);
    using qrt_sm121_mtp::Drafter;
    test_target_inputs();
    test_prefill_probe(argv[1]);
    test_checkpoints();
    test_request();
    test_prefill_request_seed();
    test_chunked_request();
    test_retired_request();
    test_chunked_seed_runtime();
    test_explicit_prefill_norm_order();
    test_prefix_request();
    test_prefix_runtime();
    test_prefix_entry();
    test_prefill_request_probe(argv[1]);
    for(unsigned fail=1;fail<=25u;++fail){
        reset();{Drafter d;assert(d.reserve(8,2)==hipSuccess&&allocations.size()==25u);auto* old=d.cache_data();
            reset();fail_allocation=fail;assert(d.reserve(16,4)==hipErrorOutOfMemory);
            assert(d.capacity()==8u&&d.cache_data()==old&&allocations.size()==25u);}
        assert(allocations.empty());
    }
    reset();{Drafter d;
        assert(!bind(d,0u)&&bind(d));assert(d.reserve(16,4)==hipSuccess);
        assert(append(d,0,4,9).status==hipErrorInvalidValue&&stages.empty());
        assert(append(d,0,4).status==hipSuccess&&d.retained_tokens()==4u);
        assert(!bind(d)&&d.reserve(17,4)==hipErrorInvalidValue);
        reset();assert(propose(d,3,1,11).status==hipErrorInvalidValue&&stages.empty());
        assert(propose(d,4,1).status==hipErrorInvalidValue&&stages.empty());
        auto r=propose(d,3,1);assert(r.status==hipSuccess&&r.rows==1&&r.tokens[0]==200u&&r.logits[0]==12.5f);
        assert(attention_tokens==4u&&attention_first==3u&&attention_rows==1u&&queued.empty());
        auto view=d.observation(10u);assert(view.rows==1u&&view.context[0]==70u&&view.gated_context[0]==71u);
        assert(!d.observation(9u).rows);reset();r=propose(d,2,2);
        assert(r.rows==2&&r.tokens[1]==201u&&r.logits[1]==13.5f&&r.first_position==2u);
        assert(d.observation(10u).generation!=view.generation);
        assert(!d.truncate(5u,10u)&&!d.truncate(3u,9u));assert(d.truncate(4u,10u)&&d.observation(10u).rows);
        assert(d.truncate(3u,10u)&&!d.observation(10u).rows);
        assert(propose(d,2u,1u).status==hipErrorInvalidValue);
        assert(append(d,3u,2u).status==hipSuccess&&d.retained_tokens()==5u);
        assert(propose(d,4u,1u).status==hipSuccess&&attention_tokens==5u);
        assert(d.truncate(0u,10u)&&bind(d,11u));assert(propose(d,0,1,10u).status==hipErrorInvalidValue);
    }assert(allocations.empty());
    // The real Drafter overload retains and checks the model source after the
    // temporary resolver and all external binding handles are destroyed.
    reset();{
        using namespace qrt_sm121_mtp;
        Drafter d;std::weak_ptr<LeasedWeights> borrowed;
        const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
        const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
        DrafterTables tables{t,p,32u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
        {ModelWeights owner;auto model=std::make_shared<LeasedWeights>();borrowed=model;
            assert(owner.prepare(model,10u,expected_stream).status==hipSuccess);
            assert(d.reserve(8,2)==hipSuccess&&d.bind(owner.binding(10u),tables,10u));
        }
        assert(!borrowed.expired()&&allocations.size()==26u);
        reset();borrowed.lock()->generation=11u;
        assert(append(d,0,2).status==hipErrorInvalidValue&&stages.empty());
        borrowed.lock()->generation=10u;
        assert(append(d,0,2).status==hipSuccess&&propose(d,1,1).rows==1u);
        borrowed.lock()->generation=11u;reset();
        assert(propose(d,1,1).status==hipErrorInvalidValue&&!d.observation(10u).rows);
        assert(!d.truncate(0u,10u)&&stages.empty());borrowed.lock()->generation=10u;
        assert(d.truncate(0u,10u)&&bind(d));assert(borrowed.expired()&&allocations.size()==25u);
    }assert(allocations.empty());
    reset();std::weak_ptr<LeasedWeights> quarantined_model;
    {using namespace qrt_sm121_mtp;
        Drafter d;
        const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
        const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
        DrafterTables tables{t,p,32u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
        {ModelWeights owner;auto model=std::make_shared<LeasedWeights>();quarantined_model=model;
            assert(owner.prepare(model,10u,expected_stream).status==hipSuccess);
            assert(d.reserve(8,2)==hipSuccess&&d.bind(owner.binding(10u),tables,10u));
        }
        assert(append(d,0,2).status==hipSuccess);reset();fail_sync=1;
        assert(propose(d,1,1).completion_unknown&&d.quarantined());
    }
    assert(!quarantined_model.expired()&&allocations.size()==26u);late_completion();
    assert(allocations.empty()&&!quarantined_model.expired());
    for(const char* stage:{"query","queries","attention","gate","output","post_norm","moe","final_norm","head",
                          "copy1","copy2","copy3","copy4"}){
        reset();{Drafter d;assert(bind(d)&&d.reserve(8,2)==hipSuccess&&append(d,0,2).status==hipSuccess);
            reset();fail_stage=stage;auto r=propose(d,0,2);
            assert(r.status==injected&&!r.rows&&!r.completion_unknown&&!d.quarantined());
            assert(d.retained_tokens()==2u&&queued.empty()&&!d.observation(10u).rows);
            reset();assert(propose(d,1,1).rows==1u);
        }assert(allocations.empty());
    }
    for(unsigned kind=0;kind<4;++kind){
        reset();{Drafter d;assert(bind(d)&&d.reserve(8,2)==hipSuccess&&append(d,0,2).status==hipSuccess);
            reset();invalid_moe=kind==0;invalid_head=kind==1;invalid_id=kind==2;invalid_logit=kind==3;
            auto r=propose(d,0,2);assert(r.status==hipErrorInvalidValue&&!r.rows&&!r.completion_unknown);
            assert(!d.observation(10u).rows&&queued.empty());
            reset();assert(propose(d,1,1).rows==1);
        }assert(allocations.empty());
    }
    // Proposal fence failure must retain both previously completed KV and
    // newly queued scratch/host destinations, including after destruction.
    for(unsigned failure=0;failure<3;++failure){
        reset();{Drafter d;assert(bind(d)&&d.reserve(8,2)==hipSuccess);
            if(failure!=2)assert(append(d,0,2).status==hipSuccess);
            reset();fail_sync=1;if(failure==1)fail_stage="copy3";
            bool unknown=failure==2?append(d,0,2).completion_unknown:propose(d,0,2).completion_unknown;
            assert(unknown&&d.quarantined()&&!d.cache_data()&&!d.observation(10u).rows);
            auto count=stages.size();assert(propose(d,0,1).completion_unknown&&append(d,0,1).completion_unknown);
            assert(!d.truncate(0,10u)&&!bind(d)&&d.reserve(8,2)==injected&&stages.size()==count);
        }assert(allocations.size()==25u);late_completion();assert(allocations.empty());
    }
}
