// Actual prefix seed, whole-provider admission and publication. The target
// backend supplies synthetic completed hidden rows; no model arithmetic or
// real-token correctness claim is made by this queued host test.
#include "native/src/qrt.h"
struct PrefixRuntimeSession {
    bool valid=true,provider_completed=true,current_token_valid=true,route_active=true;
    const void* owner_engine=nullptr;
    uint64_t generation=17u;
    size_t prefix_tokens=0u,committed_decode_token_count=0u,last_decode_top2_position=0u;
    bool last_decode_top2_valid=false;
    uint32_t current_token_id=999u;
    std::string model_dir="actual-model";
    qrt_sm121_mtp::RequestCheckpoint native_mtp_checkpoint;
    std::vector<uint32_t> native_mtp_processed_inputs;
} g_qwen36_resident_session;
static bool g_qwen36_resident_completion_unknown=false;
static std::shared_ptr<const qrt_sm121_mtp::ModelWeightSource> prefix_runtime_source;
static auto acquire_qwen36_mtp_model_weight_source(const std::string& model,
    std::string* stage,std::string* failure) {
    assert(model=="actual-model");
    if(!prefix_runtime_source){*stage="prefix_test_source";*failure="missing actual source";}
    return prefix_runtime_source;
}
#include "mtp_prefix_provider_functions.inc"

struct ScopedQwen36PrefixBatchSuffix {
    inline static ScopedQwen36PrefixBatchSuffix* active=nullptr;
    size_t prefix;
};
static size_t g_qwen36_chunked_prefill_total_tokens=0u;
static bool direct_provider_orchestration=true,fused_layer_stack_provider=true;
static bool env_flag_enabled(const char* name) {
    return std::strcmp(name,"QRT_QWEN36_MTP_NATIVE_REQUEST_SEED")!=0;
}
static bool raw_env_flag_enabled(const char* name){return !std::strcmp(name,"QRT_QWEN36_MTP_NATIVE_DECODE");}
struct PrefixPrefillRequest {
    const uint32_t* input_tokens;
    size_t input_token_count;
    const void* resident_engine;
    unsigned output_token_capacity=1u;
};
struct PrefixPrefillResult {std::string stage;};
static void qrt_qwen36_whole_provider_set_failure(PrefixPrefillResult* result,
    const std::string& stage,const std::string&,uint64_t) {result->stage=stage;}
static bool invoke_prefix_prefill_admission(const PrefixPrefillRequest* request,
    PrefixPrefillResult* out_result,bool target_ok=true) {
    constexpr uint64_t start_ns=0u;
#include "mtp_prefix_prefill_admission.inc"
    assert(mtp_prefix_inner&&!mtp_native_request_seed&&!mtp_chunked_inner&&!mtp_chunked_outer);
    if(!target_ok)return false;
    auto* batch=qrt_mtp_target_rows::Scope::active;
    assert(batch);
    std::vector<float> hidden(batch->rows()*2048u,1.25f);
    assert(batch->stage(batch->local_rows(),hidden,888u)&&batch->publish(888u));
    mtp_rows_publication.complete();
    return true;
}
static qrt_qwen36_whole_provider_prefix_request_v1_t prefix_runtime_request(
    const PrefixRequestFixture& fixture,const std::vector<uint32_t>& prompt,unsigned capacity=32u) {
    qrt_qwen36_whole_provider_prefix_request_v1_t request{};
    request.expected_session_generation=17u;
    request.expected_prefix_token_count=static_cast<uint32_t>(fixture.owner.prompt.size());
    request.input_tokens=prompt.data();request.input_token_count=static_cast<uint32_t>(prompt.size());
    request.suffix_tokens=prompt.data()+fixture.owner.prompt.size();
    request.suffix_token_count=static_cast<uint32_t>(prompt.size()-fixture.owner.prompt.size());
    request.output_token_capacity=capacity;
    auto& session=g_qwen36_resident_session;session={};
    session.owner_engine=fixture.cached().owner;session.prefix_tokens=fixture.owner.prompt.size();
    session.native_mtp_checkpoint=fixture.checkpoint;
    session.native_mtp_processed_inputs=fixture.owner.prompt;
    prefix_runtime_source=fixture.owner.source;
    g_qwen36_resident_completion_unknown=false;
    return request;
}
static void prefix_runtime_clear() {
    g_qwen36_resident_session={};prefix_runtime_source.reset();
    g_qwen36_resident_completion_unknown=false;
    ScopedQwen36PrefixBatchSuffix::active=nullptr;
    assert(!qrt_mtp_target_rows::Scope::active);
}
static void prefix_runtime_target(const qrt_qwen36_whole_provider_prefix_request_v1_t& request,
    bool success=true) {
    ScopedQwen36PrefixBatchSuffix suffix{request.expected_prefix_token_count};
    ScopedQwen36PrefixBatchSuffix::active=&suffix;
    const PrefixPrefillRequest target{request.suffix_tokens,request.suffix_token_count,
        g_qwen36_resident_session.owner_engine};
    PrefixPrefillResult result;
    assert(invoke_prefix_prefill_admission(&target,&result,success)==success);
    ScopedQwen36PrefixBatchSuffix::active=nullptr;
    if(!success)return;
    auto& session=g_qwen36_resident_session;
    session.committed_decode_token_count=request.suffix_token_count;
    session.current_token_id=888u;session.last_decode_top2_valid=true;
    session.last_decode_top2_position=request.input_token_count-1u;
}
static void test_prefix_runtime() {
    using qrt_sm121_mtp_runtime::PrefixPrefillSeed;
    using namespace qrt_sm121_mtp;
    const auto prior_stream=expected_stream;expected_stream=nullptr;
    assert(!unsetenv("QRT_QWEN36_MTP_TARGET_ROWS_DUMP_PREFIX"));
    assert(!unsetenv("QRT_QWEN36_MTP_NATIVE_PREFILL_PROBE_PREFIX"));
    unsigned successes=0u,declines=0u,publication_failures=0u,fences=0u;
    unsigned finish_fences=0u;
    // Both real retained norm profiles and both eligible owner lengths.
    for(unsigned prefix:{8192u,16384u})for(bool split:{false,true}){
        reset();{
            PrefixRequestFixture fixture(prefix,split);auto prompt=fixture.extended();
            const auto request=prefix_runtime_request(fixture,prompt,512u);
            PrefixPrefillSeed seed;std::string stage,failure;
            assert(prepare_qwen36_native_mtp_prefix(request,true,&seed,&stage,&failure));
            auto* rows=qrt_mtp_target_rows::Scope::active;assert(rows);
            assert(PrefixPrefillSeed::captures(fixture.cached().owner,request.suffix_tokens,1024u,prefix));
            assert(!PrefixPrefillSeed::captures(&request,request.suffix_tokens,1024u,prefix));
            assert(!PrefixPrefillSeed::captures(fixture.cached().owner,request.suffix_tokens,1023u,prefix));
            assert(!PrefixPrefillSeed::captures(fixture.cached().owner,request.suffix_tokens,1024u,prefix+1u));
            PrefixPrefillSeed nested;
            assert(nested.begin(fixture.checkpoint,fixture.owner.source,fixture.cached(),prompt.data(),
                prompt.size(),static_cast<unsigned>(prompt.size()+512u)).status==hipErrorInvalidValue);
            assert(qrt_mtp_target_rows::Scope::active==rows);
            prefix_runtime_target(request);assert(rows->published());
            reset();cache_fill=77u;
            auto observed=expect_prefix_inputs(fixture,*rows);expect_prefix_cache(fixture,prompt);
            assert(finish_qwen36_native_mtp_prefix(request,seed,&stage,&failure));
            assert(*observed==prefix/8192u+1u&&!qrt_mtp_target_rows::Scope::active);
            finish_fences=sync_call;
            auto actual=fixture.actual(prompt,static_cast<unsigned>(prompt.size()));actual.current_token=888u;
            const auto& session=g_qwen36_resident_session;
            assert(session.native_mtp_checkpoint.matches(actual)&&session.native_mtp_processed_inputs==prompt);
            Request restored;assert(restored.restore(session.native_mtp_checkpoint,actual,
                static_cast<unsigned>(prompt.size()+512u),expected_stream).status==hipSuccess);
            TargetBatch plan;assert(restored.begin(actual,1u,&plan)&&plan.inputs[0]==888u&&plan.inputs[1]==200u);
            assert(restored.abort(10u));
            assert(!finish_qwen36_native_mtp_prefix(request,seed,&stage,&failure));
            reset();fixture.unchanged();prefix_runtime_clear();++successes;
        }assert(allocations.empty()&&queued.empty());
    }
    // Owner continuation uses its real next token without recapturing or
    // reseeding MTP; differing teacher input must be rejected before work.
    reset();{
        PrefixRequestFixture fixture;auto prompt=fixture.owner.prompt;prompt.push_back(999u);
        const auto request=prefix_runtime_request(fixture,prompt);
        PrefixPrefillSeed seed;std::string stage,failure;const auto submitted=stages.size();
        assert(prepare_qwen36_native_mtp_prefix(request,false,&seed,&stage,&failure));
        assert(!qrt_mtp_target_rows::Scope::active&&stages.size()==submitted);
        Request resumed;assert(resumed.restore(g_qwen36_resident_session.native_mtp_checkpoint,
            fixture.cached(),16416u,expected_stream).status==hipSuccess);
        TargetBatch plan;assert(resumed.begin(fixture.cached(),1u,&plan)&&plan.inputs[0]==999u);
        assert(resumed.abort(10u));prompt.back()=998u;
        assert(!prepare_qwen36_native_mtp_prefix(request,false,&seed,&stage,&failure));
        assert(stage=="mtp_native_prefix_suffix");prefix_runtime_clear();++successes;
    }assert(allocations.empty());
    reset();{
        PrefixRequestFixture fixture;auto prompt=fixture.extended();
        for(unsigned bad=0u;bad<20u;++bad){
            auto request=prefix_runtime_request(fixture,prompt);auto& session=g_qwen36_resident_session;
            switch(bad){
                case 0:session.valid=false;break;case 1:session.provider_completed=false;break;
                case 2:session.current_token_valid=false;break;case 3:session.generation++;break;
                case 4:session.prefix_tokens--;break;case 5:session.committed_decode_token_count=1u;break;
                case 6:request.input_tokens=nullptr;break;case 7:request.suffix_tokens=nullptr;break;
                case 8:session.native_mtp_processed_inputs.pop_back();break;
                case 9:request.input_token_count--;break;case 10:request.output_token_capacity=0u;break;
                case 11:request.output_token_capacity=513u;break;case 12:session.native_mtp_processed_inputs[0]++;break;
                case 13:session.current_token_id++;break;case 14:session.owner_engine=&request;break;
                case 15:session.native_mtp_checkpoint={};break;case 16:prefix_runtime_source.reset();break;
                case 17:fixture.owner.source->generation++;break;
                case 18:request.suffix_token_count=1u;request.input_token_count=session.prefix_tokens+1u;break;
                case 19:prompt.back()=qrt_mtp_target_rows::vocabulary;break;
            }
            PrefixPrefillSeed seed;std::string stage,failure;const auto submitted=stages.size();
            const auto allocated=allocation_call;
            assert(!prepare_qwen36_native_mtp_prefix(request,true,&seed,&stage,&failure));
            assert(!qrt_mtp_target_rows::Scope::active&&stages.size()==submitted&&allocation_call==allocated);
            if(bad==17u)fixture.owner.source->generation--;
            if(bad==19u)prompt.back()=777u;
            prefix_runtime_clear();++declines;
        }
    }assert(allocations.empty());
    // Target admission is bound to this capture, engine and real suffix.
    // An ordinary external Scope cannot authorize native prefix seeding.
    reset();{
        PrefixRequestFixture fixture;auto prompt=fixture.extended();
        for(unsigned bad=0u;bad<6u;++bad){
            const auto request=prefix_runtime_request(fixture,prompt);
            PrefixPrefillSeed seed;std::string stage,failure;
            assert(prepare_qwen36_native_mtp_prefix(request,true,&seed,&stage,&failure));
            ScopedQwen36PrefixBatchSuffix suffix{request.expected_prefix_token_count};
            ScopedQwen36PrefixBatchSuffix::active=&suffix;
            PrefixPrefillRequest target{request.suffix_tokens,1024u,g_qwen36_resident_session.owner_engine};
            if(bad==0u)target.resident_engine=&request;
            if(bad==1u)suffix.prefix++;
            if(bad==2u)target.input_token_count--;
            if(bad==3u)prompt.back()++;
            if(bad>=4u)seed.end_capture();
            std::unique_ptr<qrt_mtp_target_rows::PrefillRows> unrelated_rows;
            std::unique_ptr<qrt_mtp_target_rows::Scope> unrelated_scope;
            if(bad==5u){
                unrelated_rows=std::make_unique<qrt_mtp_target_rows::PrefillRows>(
                    prompt.data(),prompt.size(),request.expected_prefix_token_count,1024u);
                unrelated_scope=std::make_unique<qrt_mtp_target_rows::Scope>(unrelated_rows.get());
            }
            PrefixPrefillResult result;
            assert(!invoke_prefix_prefill_admission(&target,&result)&&result.stage=="mtp_native_prefill_request");
            if(bad==3u)prompt.back()--;
            unrelated_scope.reset();
            seed.end_capture();prefix_runtime_clear();++declines;
        }
    }assert(allocations.empty());
    // The target's Publication retracts rows on failure. Every later target
    // metadata mismatch is rejected before any MTP GPU submission.
    for(unsigned bad=0u;bad<10u;++bad){
        reset();{
            PrefixRequestFixture fixture;auto prompt=fixture.extended();
            auto request=prefix_runtime_request(fixture,prompt);PrefixPrefillSeed seed;
            std::string stage,failure;
            assert(prepare_qwen36_native_mtp_prefix(request,true,&seed,&stage,&failure));
            prefix_runtime_target(request,bad!=0u);auto& session=g_qwen36_resident_session;
            switch(bad){
                case 1:session.generation++;break;case 2:session.prefix_tokens--;break;
                case 3:session.committed_decode_token_count--;break;case 4:session.current_token_id++;break;
                case 5:session.last_decode_top2_valid=false;break;case 6:session.last_decode_top2_position--;break;
                case 7:session.owner_engine=&request;break;case 8:fixture.owner.source->generation++;break;
                case 9:prompt[0]++;break;
            }
            const auto submitted=stages.size();const auto allocated=allocation_call;
            assert(!finish_qwen36_native_mtp_prefix(request,seed,&stage,&failure));
            assert(!qrt_mtp_target_rows::Scope::active&&stages.size()==submitted&&allocation_call==allocated);
            assert(session.native_mtp_processed_inputs==fixture.owner.prompt);
            if(bad==8u)fixture.owner.source->generation--;
            assert(session.native_mtp_checkpoint.matches(fixture.cached()));
            prefix_runtime_clear();++publication_failures;
        }assert(allocations.empty()&&queued.empty());
    }
    // Extend and final paired save can each lose completion. The actual
    // provider invalidates the target, preserves the old published values,
    // and private owners survive until the queued producers really finish.
    assert(finish_fences>11u);
    for(unsigned failed=1u;failed<=finish_fences;++failed){
        reset();{
            PrefixRequestFixture fixture;auto prompt=fixture.extended();
            const auto request=prefix_runtime_request(fixture,prompt);PrefixPrefillSeed seed;
            std::string stage,failure;
            assert(prepare_qwen36_native_mtp_prefix(request,true,&seed,&stage,&failure));
            prefix_runtime_target(request);auto* rows=qrt_mtp_target_rows::Scope::active;
            reset();cache_fill=77u;expect_prefix_inputs(fixture,*rows);expect_prefix_cache(fixture,prompt);
            fail_sync=failed;
            assert(!finish_qwen36_native_mtp_prefix(request,seed,&stage,&failure));
            const auto& session=g_qwen36_resident_session;
            assert(g_qwen36_resident_completion_unknown&&!session.valid&&!session.route_active);
            assert(session.native_mtp_checkpoint.tokens()==fixture.owner.prompt.size());
            assert(session.native_mtp_processed_inputs==fixture.owner.prompt&&!qrt_mtp_target_rows::Scope::active);
            prefix_runtime_clear();++fences;
        }
        late_completion();assert(allocations.empty()&&queued.empty());
    }
    reset();expected_stream=prior_stream;
    std::cout<<"native MTP prefix runtime cases="<<successes<<" invalid_frontiers="<<declines
        <<" failed_publications="<<publication_failures<<" completion_fences="<<fences<<" pass\n";
}

// Exercise the real entry branch around the two real runtime helpers. This
// host target models completed suffix metadata and a saved metadata rollback;
// the separate model-prefix suite exercises the actual GPU Shadow algorithm.
struct PrefixEntryResult {uint32_t teacher_forced_prediction_tokens[1024]{};};
static const PrefixRequestFixture* prefix_entry_fixture=nullptr;
static const std::vector<uint32_t>* prefix_entry_prompt=nullptr;
static unsigned prefix_entry_mode=0u,prefix_entry_target_calls=0u;
static bool run_qwen36_resident_batch_suffix(const qrt_qwen36_whole_provider_prefix_request_v1_t& request,
    uint32_t* predictions,std::string* stage,std::string* failure) {
    assert(prefix_entry_fixture&&prefix_entry_prompt);++prefix_entry_target_calls;
    prefix_runtime_target(request,prefix_entry_mode!=1u);
    if(prefix_entry_mode==1u){*stage="prefix_test_target";*failure="completed target failure";return false;}
    auto* rows=qrt_mtp_target_rows::Scope::active;assert(rows&&rows->published());
    reset();cache_fill=77u;expect_prefix_inputs(*prefix_entry_fixture,*rows);
    expect_prefix_cache(*prefix_entry_fixture,*prefix_entry_prompt);
    if(prefix_entry_mode==2u){fail_stage="kv";fail_stage_occurrence=1u;}
    std::fill_n(predictions,1024u,888u);
    return true;
}
static int invoke_prefix_entry(const qrt_qwen36_whole_provider_prefix_request_v1_t* request,
    PrefixEntryResult* out_result,bool batch_suffix_requested,bool gb10_continuation_teacher_forced,
    std::string* failed_stage) {
    const auto original=g_qwen36_resident_session;
    std::string failure_stage,failure;
    const auto rollback_failure=[&](qrt_status_t,const std::string& stage,const std::string&) {
        *failed_stage=stage;g_qwen36_resident_session=original;return 0;
    };
#include "mtp_prefix_entry_branch.inc"
    return 1;
}
static void test_prefix_entry() {
    const auto prior_stream=expected_stream;expected_stream=nullptr;
    for(unsigned mode=0u;mode<5u;++mode){
        reset();{
            PrefixRequestFixture fixture;auto prompt=fixture.extended();
            if(mode==3u){prompt=fixture.owner.prompt;prompt.push_back(999u);}
            const auto request=prefix_runtime_request(fixture,prompt);
            prefix_entry_fixture=&fixture;prefix_entry_prompt=&prompt;prefix_entry_mode=mode;
            prefix_entry_target_calls=0u;PrefixEntryResult result;std::string stage;
            const bool success=mode==0u||mode==3u;
            assert(invoke_prefix_entry(&request,&result,mode!=3u,mode==4u,&stage)==success);
            assert(!qrt_mtp_target_rows::Scope::active);
            const auto& session=g_qwen36_resident_session;
            assert(prefix_entry_target_calls==(mode<3u?1u:0u));
            if(mode==0u){
                auto actual=fixture.actual(prompt,static_cast<unsigned>(prompt.size()));actual.current_token=888u;
                assert(session.native_mtp_checkpoint.matches(actual)&&session.native_mtp_processed_inputs==prompt);
                assert(session.committed_decode_token_count==1024u&&result.teacher_forced_prediction_tokens[1023]==888u);
            }else{
                assert(session.native_mtp_checkpoint.matches(fixture.cached()));
                assert(session.native_mtp_processed_inputs==fixture.owner.prompt&&session.committed_decode_token_count==0u);
            }
            if(mode==1u)assert(stage=="prefix_test_target");
            if(mode==4u)assert(stage=="mtp_native_prefix_reference_input");
            reset();fixture.unchanged();prefix_runtime_clear();
        }assert(allocations.empty()&&queued.empty());
    }
    prefix_entry_fixture=nullptr;prefix_entry_prompt=nullptr;expected_stream=prior_stream;
    std::cout<<"native MTP actual prefix entry cases=5 pass\n";
}
