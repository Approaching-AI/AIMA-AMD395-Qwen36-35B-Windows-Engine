#include "target_test_runtime.h"
#include "sm121_q2_resident_cache.h"
#include "resident_session_types.h"
#include "resident_table_types.h"
#include <climits>
using ResidentOwner = ResidentCacheOwner<Qwen36ResidentSessionState,Qwen36ResidentSessionElementKind>;

struct ResidentFixture {
    Qwen36ResidentSessionState session;
    std::atomic<uint64_t> epoch{7u};
    TargetTables tables;
    std::shared_ptr<Qwen36TargetRollbackPermit> permit = std::make_shared<Qwen36TargetRollbackPermit>();
    std::vector<void*> buffers;
    std::vector<std::pair<unsigned char*,size_t>> guards;
    explicit ResidentFixture(bool fp32=false,bool contiguous=false) {
        Source source; tables=source.state.tables;
        session.owner_engine=fake<qrt_engine_t>(9990);session.model_dir="original-model";
        session.generation=11u;session.prefix_tokens=7u;session.committed_decode_token_count=3u;
        session.current_token_id=144u;session.current_token_valid=true;
        session.valid=session.route_active=session.provider_completed=true;
        session.native_mtp_processed_inputs={1,2,3,4,5,6,7,8,9,10};
        session.native_mtp_checkpoint.owner=std::make_shared<unsigned>(42u);
        session.prefix_checkpoints=std::make_shared<Qwen36ResidentPrefixCheckpointStore>();
        permit->owner=&session;permit->active=true;
        const unsigned bytes=fp32?4u:2u;
        const auto kind=fp32?Qwen36ResidentSessionElementKind::kF32:Qwen36ResidentSessionElementKind::kBf16;
        for(unsigned layer=0;layer<40u;++layer) {
            if(layer%4u==3u) {
                auto& a=session.full_attention_layers[layer];
                a.k_bytes=a.v_bytes=7u*512u*bytes;
                a.decode_tail_k_bytes=a.decode_tail_v_bytes=8u*512u*bytes;
                a.decode_tail_capacity_tokens=8u;a.decode_tail_token_count=3u;a.history_tokens=7u;
                a.decode_tail_contiguous=contiguous;a.element_kind=kind;a.valid=true;
                auto* base=buffer(a.k_bytes+a.v_bytes+(contiguous?a.decode_tail_k_bytes+a.decode_tail_v_bytes:0u));
                a.device_allocation=base;a.device_k=base;
                a.device_v=base+a.k_bytes+(contiguous?a.decode_tail_k_bytes:0u);
                if(contiguous){a.device_decode_tail_k=base+a.k_bytes;a.device_decode_tail_v=static_cast<unsigned char*>(a.device_v)+a.v_bytes;}
                else {
                    auto* tail=buffer(a.decode_tail_k_bytes+a.decode_tail_v_bytes);
                    a.device_decode_tail_allocation=tail;a.device_decode_tail_k=tail;a.device_decode_tail_v=tail+a.decode_tail_k_bytes;
                }
            } else {
                auto& a=session.linear_layers[layer];
                a.recurrent_state_bytes=state_elements*4u;a.qkv_ring_bytes=ring_elements*bytes;
                auto* base=buffer(a.recurrent_state_bytes+a.qkv_ring_bytes);
                a.device_allocation=base;a.device_recurrent_state=reinterpret_cast<float*>(base);
                a.device_qkv_ring=base+a.recurrent_state_bytes;
                a.prefix_tokens=7u;a.decode_qkv_token_count=a.decode_recurrent_token_count=3u;
                a.qkv_element_kind=kind;a.recurrent_state_key_major=bool(layer%2u);a.valid=true;
            }
        }
    }
    unsigned char* buffer(size_t bytes) {
        void* p=nullptr;assert(hipMalloc(&p,bytes+512u)==hipSuccess);buffers.push_back(p);
        std::memset(p,0x5a,bytes+512u);guards.push_back({static_cast<unsigned char*>(p),bytes});
        return static_cast<unsigned char*>(p)+256u;
    }
    std::shared_ptr<ResidentOwner> bind() {
        return ResidentOwner::create(session,epoch,tables,std::make_shared<Qwen36TargetCacheBorrow>(permit));
    }
    ~ResidentFixture() {
        assert(!g_qwen36_target_cache_borrows&&pending.empty());
        for(const auto& g:guards)for(size_t i=0;i<256u;++i)assert(g.first[i]==0x5a&&g.first[g.second+256u+i]==0x5a);
        for(auto* b:buffers)(void)hipFree(b);
    }
    void verify_published(unsigned rows,unsigned bytes) const {
        for(unsigned layer=0;layer<40u;++layer) {
            if(layer%4u==3u) {
                const auto& a=session.full_attention_layers[layer];
                for(const auto* p:{a.device_k,a.device_v})
                    for(size_t i=0;i<a.k_bytes;++i)assert(static_cast<const unsigned char*>(p)[i]==0x5au);
                for(const auto* p:{a.device_decode_tail_k,a.device_decode_tail_v})
                    for(unsigned token=0;token<8u;++token)for(unsigned channel=0;channel<512u;++channel) {
                        const size_t offset=(size_t(token)*512u+channel)*bytes;
                        if(token<3u||token>=3u+rows) {
                            for(unsigned i=0;i<bytes;++i)assert(static_cast<const unsigned char*>(p)[offset+i]==0x5au);
                        } else {
                            uint32_t actual=0;std::memcpy(&actual,static_cast<const unsigned char*>(p)+offset,bytes);
                            const uint32_t expected=layer*2u+(token-3u)+1u;
                            assert(actual==(bytes==2u?expected:expected<<16u));
                        }
                    }
            } else {
                const auto& a=session.linear_layers[layer];
                for(size_t i=0;i<state_elements;++i)assert(a.device_recurrent_state[i]==float(layer*2u+rows));
                for(size_t i=0;i<ring_elements;++i) {
                    if(bytes==2u)assert(static_cast<const uint16_t*>(a.device_qkv_ring)[i]==layer*2u+rows);
                    else assert(static_cast<const float*>(a.device_qkv_ring)[i]==float(layer*2u+rows));
                }
            }
        }
    }
};

static Qwen36ResidentSessionState* factory_session=nullptr;
#define g_qwen36_resident_session (*factory_session)
static std::atomic<uint64_t> g_qwen36_mtp_weight_storage_epoch{7u};
static struct {bool valid=false;} g_qwen36_resident_dual_attention_state,g_qwen36_q1024_suffix_cache;
static TargetTables factory_tables;
static unsigned table_calls=0,table_failure=0,table_position=0;
static hipError_t table_status(){return ++table_calls==table_failure?fault:hipSuccess;}
static const char* hipGetErrorString(hipError_t){return "injected table error";}
class ScopedQwen36ResidentSessionShadowTransaction {
public:
    explicit ScopedQwen36ResidentSessionShadowTransaction(ResidentFixture& f):fixture(f){}
    std::shared_ptr<const ResidentCacheLifetime> acquire_target_cache_lifetime()const{
        if(g_qwen36_target_cache_borrows)return {};
        return std::make_shared<Qwen36TargetCacheBorrow>(fixture.permit);
    }
private:ResidentFixture& fixture;
};
namespace qrt_sm121_q1_full_runtime {
static hipError_t prepare(Tables* output,size_t position){
    table_position=unsigned(position);const auto& t=factory_tables;
    *output={{t.attention.exp2,t.attention.rsqrt,t.convolution_silu,t.beta},t.attention.rope,t.attention.rope_rows};
    return table_status();
}}
namespace qrt_sm121_q1_moe_runtime {
static hipError_t prepare(Tables* output){
    const auto& t=factory_tables;
    *output={{t.attention.exp2,t.attention.rsqrt,t.convolution_silu,t.beta},t.moe.router_exp_fraction,t.moe.silu};
    return table_status();
}}
namespace qrt_sm121_q1_attention_runtime {
static hipError_t prepare(const unsigned char** output){*output=factory_tables.attention.reciprocal;return table_status();}
}
namespace qrt_sm121_mtp_runtime {
static hipError_t prepare_sigmoid(const char*,const uint16_t** output){*output=factory_tables.attention.sigmoid;return table_status();}
}
static bool load_gb10_gated_silu_f32_lut(const float** output,std::string*){*output=factory_tables.gated_silu;return table_status()==hipSuccess;}
static bool load_q1_sm121_gate_table(unsigned layer,const float** output,std::string*){
    assert(layer%4u!=3u);*output=factory_tables.g[layer];return table_status()==hipSuccess;
}
#include "resident_cache_factory.h"
#undef g_qwen36_resident_session

static void verify_factory() {
    ResidentFixture f;factory_session=&f.session;factory_tables=f.tables;
    factory_tables.moe.sigmoid=factory_tables.attention.sigmoid;
    ScopedQwen36ResidentSessionShadowTransaction tx(f);std::string stage,error;
    for(table_failure=1u;table_failure<=35u;++table_failure) {
        table_calls=0;auto source=acquire_qwen36_target_cache_owner(tx,&stage,&error);
        assert(!source&&table_calls==table_failure&&!g_qwen36_target_cache_borrows&&f.session.valid);
    }
    table_failure=table_calls=0;
    {
        auto source=acquire_qwen36_target_cache_owner(tx,&stage,&error);TargetSnapshot snapshot;
        assert(source&&source->snapshot(&snapshot)&&table_calls==35u&&table_position==11u);
        auto expected=snapshot;expected.tables=factory_tables;
        assert(resident_cache_detail::same_snapshot(snapshot,expected));
        assert(!acquire_qwen36_target_cache_owner(tx,&stage,&error)&&g_qwen36_target_cache_borrows==1u);
    }
    for(unsigned problem=0;problem<5u;++problem) {
        f.session.activation_workspace.in_use=problem==0;
        f.session.activation_workspace.phase=problem==1?Qwen36ResidentDecodeActivationWorkspacePhase::kBusy:Qwen36ResidentDecodeActivationWorkspacePhase::kIdle;
        g_qwen36_resident_dual_attention_state.valid=problem==2;
        g_qwen36_q1024_suffix_cache.valid=problem==3;
        f.session.valid=problem!=4;
        table_calls=0;assert(!acquire_qwen36_target_cache_owner(tx,&stage,&error)&&!table_calls&&!g_qwen36_target_cache_borrows);
    }
    g_qwen36_resident_dual_attention_state.valid=g_qwen36_q1024_suffix_cache.valid=false;
    factory_session=nullptr;
}

static void verify_factory_extended_rope(const ModelWeightBinding& binding) {
    ResidentFixture f;factory_session=&f.session;factory_tables=f.tables;
    factory_tables.moe.sigmoid=factory_tables.attention.sigmoid;
    ScopedQwen36ResidentSessionShadowTransaction tx(f);std::string stage,error;
    // The verified runtime allocation includes suffix/rollback capacity beyond
    // the target's logical context. Exercise the actual factory AND Target.
    for(unsigned allocated_rows:{12u,262144u,263680u,264736u}) {
        factory_tables.attention.rope_rows=allocated_rows;table_failure=table_calls=0;
        auto owner=acquire_qwen36_target_cache_owner(tx,&stage,&error);assert(owner);
        Target target;TargetResult result;reset();
        assert(target.evaluate(binding,owner,{144,255},&result,1024,wanted_stream).status==hipSuccess);
        assert(result.ready()&&result.frontier()->tables.attention.rope==factory_tables.attention.rope);
        assert(result.frontier()->tables.attention.rope_rows==std::min(allocated_rows,target_context_limit));
        assert(factory_tables.attention.rope_rows==allocated_rows);
    }
    for(unsigned allocated_rows:{0u,10u,11u}) {
        factory_tables.attention.rope_rows=allocated_rows;table_failure=table_calls=0;reset();
        assert(!acquire_qwen36_target_cache_owner(tx,&stage,&error)&&!calls&&pending.empty());
        assert(!g_qwen36_target_cache_borrows&&f.session.valid);
    }
    factory_session=nullptr;
}

int main() {
    verify_factory();
    {
        auto weights=std::make_shared<Weights>();ModelWeights model;
        assert(model.prepare(weights,7,wanted_stream).status==hipSuccess);const auto binding=model.binding(7);
        verify_factory_extended_rope(binding);
        for(bool fp32:{false,true})for(bool contiguous:{false,true})for(unsigned rows:{1u,2u}) {
            reset();ResidentFixture f(fp32,contiguous);auto owner=f.bind();assert(owner&&g_qwen36_target_cache_borrows==1u);
            TargetSnapshot before;assert(owner->snapshot(&before)&&before.processed_tokens==10u&&before.owner==&f.session);
            std::weak_ptr<unsigned> checkpoint=f.session.native_mtp_checkpoint.owner;
            {
                Target target;TargetResult result;reset();
                assert(target.evaluate(binding,owner,{144,255},&result,257,wanted_stream).status==hipSuccess);
                assert(!owner->commit_metadata(result,rows,{255,82}));
                reset();assert(CachePublisher::publish(result,*owner,rows,wanted_stream).status==hipSuccess&&calls==80u);
                assert(f.session.native_mtp_processed_inputs.size()==10u&&f.session.committed_decode_token_count==3u);
                f.verify_published(rows,fp32?4u:2u);
                assert(!owner->commit_metadata(result,rows,{256,82}));
                assert(owner->commit_metadata(result,rows,{255,82}));
                assert(!result.ready()&&!owner->commit_metadata(result,rows,{255,82}));
                assert(!owner->matches(before));TargetSnapshot after;assert(owner->snapshot(&after));
                assert(after.processed_tokens==10u+rows&&after.current_token==(rows==1u?255u:82u));
                assert(f.session.native_mtp_processed_inputs[10u]==144u);
                if(rows==2u)assert(f.session.native_mtp_processed_inputs[11u]==255u);
                assert(f.session.committed_decode_token_count==3u+rows&&f.session.mtp_target_hidden_position==9u+rows);
                for(const auto value:f.session.mtp_target_hidden_bf16)assert(value==(rows==1u?0x3f80u:0x4000u));
                assert(!f.session.native_mtp_checkpoint.owner&&!checkpoint.expired());
                for(unsigned i=0;i<40u;++i) {
                    if(i%4u==3u)assert(f.session.full_attention_layers[i].decode_tail_token_count==3u+rows);
                    else assert(f.session.linear_layers[i].decode_qkv_token_count==3u+rows&&
                                f.session.linear_layers[i].decode_recurrent_token_count==3u+rows);
                }
                auto foreign=after;foreign.tables.g[0]=fake<float>(9998);assert(!owner->matches(foreign));
                ++f.epoch;assert(!owner->snapshot(&after));--f.epoch;
                f.session.native_mtp_processed_inputs[2]^=1u;assert(!owner->snapshot(&after));
                f.session.native_mtp_processed_inputs[2]^=1u;assert(owner->snapshot(&after));
                f.permit->active=false;assert(!owner->snapshot(&after));f.permit->active=true;
                owner.reset();assert(g_qwen36_target_cache_borrows==1u&&!checkpoint.expired());
            }
            assert(!g_qwen36_target_cache_borrows&&checkpoint.expired());
        }
        // Corrupt every layer's physical ownership/count contract. No work may
        // be queued, and the temporarily acquired actual pin must be released.
        {
            ResidentFixture f;const auto saved=f.session;
            for(unsigned layer=0;layer<40u;++layer)for(unsigned kind=0;kind<4u;++kind) {
                f.session=saved;
                if(layer%4u==3u) {
                    auto& a=f.session.full_attention_layers[layer];
                    if(kind==0u)++a.decode_tail_token_count;
                    if(kind==1u)a.device_v=a.device_k;
                    if(kind==2u)a.prefill_reserved_tokens=1u;
                    if(kind==3u)a.decode_tail_k_bytes=SIZE_MAX;
                } else {
                    auto& a=f.session.linear_layers[layer];
                    if(kind==0u)++a.decode_recurrent_token_count;
                    if(kind==1u)a.device_qkv_ring=a.device_allocation;
                    if(kind==2u)a.recurrent_state_bytes=SIZE_MAX;
                    if(kind==3u)a.qkv_element_kind=Qwen36ResidentSessionElementKind::kNone;
                }
                reset();assert(!f.bind()&&!calls&&!g_qwen36_target_cache_borrows);
            }
            f.session=saved;
            f.session.linear_layers[1]=f.session.linear_layers[0];assert(!f.bind());
        }
        for(unsigned failure:{1u,2u,7u,8u,40u,79u,80u}) {
            reset();ResidentFixture f;auto owner=f.bind();Target target;TargetResult result;
            assert(target.evaluate(binding,owner,{144,255},&result,1024,wanted_stream).status==hipSuccess);
            reset();fail_at=failure;const auto status=CachePublisher::publish(result,*owner,2u,wanted_stream);
            assert(status.status==fault&&!status.completion_unknown&&pending.empty());
            assert(!f.session.valid&&!f.session.route_active&&!g_qwen36_resident_completion_unknown);
            assert(f.session.native_mtp_processed_inputs.size()==10u&&f.session.committed_decode_token_count==3u);
            assert(!owner->commit_metadata(result,2u,{255,82}));
        }
    }
    assert(allocations.empty());
    {
        ResidentFixture f;std::weak_ptr<ResidentOwner> weak;
        {
            reset();auto weights=std::make_shared<Weights>();ModelWeights model;
            assert(model.prepare(weights,7,wanted_stream).status==hipSuccess);
            auto owner=f.bind();weak=owner;Target target;TargetResult result;reset();
            assert(target.evaluate(model.binding(7),owner,{144,255},&result,1024,wanted_stream).status==hipSuccess);
            reset();failed_fence=true;const auto status=CachePublisher::publish(result,*owner,2u,wanted_stream);
            assert(status.completion_unknown&&g_qwen36_resident_completion_unknown&&!f.session.valid);
        }
        assert(!weak.expired()&&g_qwen36_target_cache_borrows==1u&&!pending.empty());
        recover_test_quarantine();assert(weak.expired()&&!g_qwen36_target_cache_borrows);
        g_qwen36_resident_completion_unknown=false;
    }
    assert(allocations.empty());
    std::cout<<"actual resident cache metadata and rollback pins pass; 8 layouts/selections, 160 invalid layers, partial writes and unknown completion checked\n";
}
