// Queued host checks exercise production ownership, not device arithmetic.
#include <algorithm>
#include <cassert>
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
struct Allocation { size_t bytes; bool host; };
static std::map<void*,Allocation> allocations;
static std::vector<std::function<void()>> queued;
static std::vector<std::string> stages;
static unsigned allocation_call=0, fail_allocation=0, sync_call=0, fail_sync=0, copies=0;
static std::string fail_stage;
static bool invalid_input=false, invalid_moe=false, invalid_head=false, invalid_id=false, invalid_logit=false;
static hipStream_t expected_stream=reinterpret_cast<void*>(0x1234);
static const uint16_t *fusion_weight=reinterpret_cast<uint16_t*>(0x10000),
    *query_weight=reinterpret_cast<uint16_t*>(0x20000), *output_weight=reinterpret_cast<uint16_t*>(0x30000),
    *post_weight=reinterpret_cast<uint16_t*>(0x40000), *final_weight=reinterpret_cast<uint16_t*>(0x50000);
static unsigned attention_tokens=0, attention_first=0, attention_rows=0;
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
    return fail_stage==stage?injected:hipSuccess;
}
static hipError_t hipStreamSynchronize(hipStream_t stream) {
    assert(stream==expected_stream);if(++sync_call==fail_sync)return injected;
    auto work=std::move(queued);queued.clear();for(auto& action:work)action();return hipSuccess;
}
static hipError_t hipMemsetAsync(void* p,int value,size_t bytes,hipStream_t stream) {
    return enqueue("clear",stream,[=]{std::memset(p,value,bytes);});
}
static hipError_t hipMemcpyAsync(void* p,const void* q,size_t bytes,int kind,hipStream_t stream) {
    assert(kind==hipMemcpyDeviceToHost);bool pinned=false;
    for(const auto& pair:allocations) {
        auto begin=reinterpret_cast<uintptr_t>(pair.first),address=reinterpret_cast<uintptr_t>(p);
        if(pair.second.host && address>=begin && address-begin<=pair.second.bytes &&
           bytes<=pair.second.bytes-(address-begin))pinned=true;
    }
    assert(pinned);return enqueue("copy"+std::to_string(++copies),stream,[=]{std::memcpy(p,q,bytes);});
}
namespace qrt_sm121_mtp {
static hipError_t launch_fusion_inputs(const uint16_t*,const uint16_t*,const uint32_t*,const uint16_t*,
    const uint16_t*,const unsigned char*,unsigned rows,uint16_t* out,uint32_t* invalid,hipStream_t stream,bool) {
    return enqueue("gather",stream,[=]{*invalid=invalid_input?1u:0u;std::fill_n(out,rows*4096u,11u);});
}
static hipError_t launch_normalize(const uint16_t* in,const uint16_t*,const unsigned char*,unsigned rows,
    uint16_t* out,hipStream_t stream) {
    return enqueue("prompt_norm",stream,[=]{assert(in[0]==22u);std::fill_n(out,rows*2048u,33u);});
}
static hipError_t launch_key_values(const uint16_t* in,const uint16_t*,const unsigned char*,const uint16_t*,
    unsigned,unsigned first,unsigned rows,unsigned capacity,uint16_t* out,uint16_t*,hipStream_t stream) {
    assert(first+rows<=capacity);return enqueue("cache",stream,[=]{assert(in[0]==44u);
        std::fill_n(out+first*1024u,rows*1024u,55u);});
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
    return enqueue("attention",stream,[=]{assert(q[0]==67u);for(unsigned i=0;i<tokens;++i)assert(cache[i*1024u]==55u);
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
#include "sm121_mtp_drafter.h"
static void reset() {
    assert(queued.empty());allocation_call=fail_allocation=sync_call=fail_sync=copies=0;
    stages.clear();fail_stage.clear();invalid_input=invalid_moe=invalid_head=invalid_id=invalid_logit=false;
}
static bool bind(qrt_sm121_mtp::Drafter& d,uint64_t epoch=10u) {
    using namespace qrt_sm121_mtp;
    const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
    const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
    DrafterWeights w{{p,p,p,fusion_weight,p,p,p},query_weight,p,output_weight,post_weight,
        {p,p,p,p,p,p},final_weight,p};
    DrafterTables tables{t,p,32u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
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
int main(){
    using qrt_sm121_mtp::Drafter;
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
