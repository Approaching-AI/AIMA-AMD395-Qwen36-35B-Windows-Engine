#pragma once
// Included inside the provider's private namespace, after the unchanged
// segment arithmetic. Only host ownership and submission order live here.
struct SegmentBinding {
    SegmentStorage& storage;
    explicit SegmentBinding(SegmentStorage& value):storage(value) {
        std::swap(static_cast<SegmentStorage&>(g_state),storage);
    }
    ~SegmentBinding() { std::swap(static_cast<SegmentStorage&>(g_state),storage); }
    SegmentBinding(const SegmentBinding&)=delete;
    SegmentBinding& operator=(const SegmentBinding&)=delete;
};

struct SegmentPipeline {
    std::array<SegmentStorage,3> storage{};
    std::array<hipStream_t,3> streams{};
    hipEvent_t input=nullptr;
    std::array<std::array<hipEvent_t,3>,8> begin{},end{};
    bool ready=false;
};
SegmentPipeline g_pipeline;

void drain_pipeline() {
    for(auto stream:g_pipeline.streams)if(stream)(void)hipStreamSynchronize(stream);
}
void release_pipeline() {
    drain_pipeline();
    for(auto& slot:g_pipeline.storage) { SegmentBinding binding(slot);release_scratch(); }
    for(auto& row:g_pipeline.begin)for(auto event:row)if(event)(void)hipEventDestroy(event);
    for(auto& row:g_pipeline.end)for(auto event:row)if(event)(void)hipEventDestroy(event);
    if(g_pipeline.input)(void)hipEventDestroy(g_pipeline.input);
    for(auto stream:g_pipeline.streams)if(stream)(void)hipStreamDestroy(stream);
    g_pipeline=SegmentPipeline{};
}
bool ensure_pipeline() {
    if(g_pipeline.ready)return true;
    hipError_t status=hipSuccess;
    for(auto& stream:g_pipeline.streams) {
        status=hipStreamCreateWithFlags(&stream,hipStreamNonBlocking);
        if(status!=hipSuccess)break;
    }
    if(status==hipSuccess)status=hipEventCreateWithFlags(&g_pipeline.input,hipEventDisableTiming);
    for(auto* events:{&g_pipeline.begin,&g_pipeline.end})for(auto& row:*events)
        for(auto& event:row)if(status==hipSuccess)status=hipEventCreate(&event);
    if(status!=hipSuccess) {set_error("pipeline events/streams",status);release_pipeline();return false;}
    bool allocated=true;
    for(auto& slot:g_pipeline.storage) {
        SegmentBinding binding(slot);
        if(!ensure_scratch(kSegmentTokens) || !ensure_blackwell_state_scratch()) {allocated=false;break;}
    }
    if(!allocated) {release_pipeline();return false;}
    g_pipeline.ready=true;return true;
}

bool pipeline_compatible() {
    const auto is=[](const char* name,const char* expected) {
        const char* value=std::getenv(name);return value && !std::strcmp(value,expected);
    };
    const auto disabled=[](const char* name) {
        const char* value=std::getenv(name);return !value || !*value || !std::strcmp(value,"0");
    };
    return blackwell_batched_enabled() && blackwell_state_enabled() &&
        qrt_fla_blackwell_cooperative::enabled() &&
        is("QRT_FLA_GDN_SCALAR_FLOAT_MATRICES","1") && is("QRT_FLA_GDN_SCALAR_FLOAT_STATE","8") &&
        is("QRT_FLA_GDN_PAIRED_SCORE_ARENAS","1") &&
        is("QRT_FLA_GDN_NORM_BLACKWELL","1") && is("QRT_FLA_GDN_KKT_BLACKWELL","1") &&
        is("QRT_FLA_GDN_INVERSE_BLACKWELL","1") &&
        disabled("QRT_FLA_GDN_COARSE_INTERVAL") && disabled("QRT_FLA_GDN_FUSED_STATE_OUTPUT") &&
        disabled("QRT_FLA_GDN_SYNC_EACH_STAGE") && disabled("QRT_FLA_GDN_PROFILE_COMPLETED_STAGES");
}
bool pipeline_diagnostic() {
    for(const char* name:{"QRT_FLA_GDN_CAPTURE_FIRST_DIR","QRT_FLA_GDN_DUMP_Q64_DIR"}) {
        const char* value=std::getenv(name);if(value && *value)return true;
    }
    return false;
}

bool launch_pipeline_window(const float* raw,const float* gates,float* output,float* state,
                            hipStream_t caller,unsigned tokens,bool reset,int mode) {
    if(!tokens || tokens>8192u || tokens%64u || (mode!=1 && mode!=2) || !ensure_pipeline())return false;
    auto& owner=g_pipeline;
    const hipStream_t producer=owner.streams[0],recurrence=owner.streams[mode==2?1:0];
    const hipStream_t consumer=owner.streams[mode==2?2:1];
    auto checked=[](hipError_t status,const char* stage) {
        if(status==hipSuccess)return true;set_error(stage,status);return false;
    };
    auto execute=[&] {
        // The caller may have just written inputs or the seeded state.
        if(!checked(hipEventRecord(owner.input,caller),"pipeline input record") ||
           !checked(hipStreamWaitEvent(producer,owner.input,0u),"pipeline input wait"))return false;
        const unsigned segments=(tokens+1023u)/1024u;
        unsigned submissions=0u;
        for(unsigned segment=0u;segment<segments;++segment) {
            const unsigned offset=segment*1024u,count=(std::min)(1024u,tokens-offset);
            const int prior=qrt_fla_pipeline_policy::prior_consumer(segment);
            if(prior>=0 && !checked(hipStreamWaitEvent(producer,owner.end[unsigned(prior)][2],0u),
                                   "pipeline storage reuse wait"))return false;
            SegmentBinding binding(owner.storage[qrt_fla_pipeline_policy::slot(segment)]);
            const hipStream_t streams[]={producer,recurrence,consumer};
            for(unsigned phase=0u;phase<3u;++phase) {
                const auto stream=streams[phase];
                if(phase && !checked(hipStreamWaitEvent(stream,owner.end[segment][phase-1u],0u),
                                      "pipeline phase wait"))return false;
                if(!checked(hipEventRecord(owner.begin[segment][phase],stream),"pipeline phase begin"))return false;
                {
                    BlackwellSegmentGuard deferred(stream);
                    if(!launch_segment_async(raw+size_t(offset)*kQkvRows,gates+size_t(offset)*kGateRows,
                        output+size_t(offset)*kValueFeatures,state,stream,int32_t(count),
                        reset && !segment,0,{},phase+1u))return false;
                    submissions+=deferred.operations;
                }
                if(!checked(hipEventRecord(owner.end[segment][phase],stream),"pipeline phase end"))return false;
            }
        }
        // The last consumer covers every state/producer transitively. Complete
        // it before timestamps, event reuse, caller return or scratch release.
        const auto done=owner.end[segments-1u][2];
        if(!checked(hipStreamWaitEvent(caller,done,0u),"pipeline caller join") ||
           !checked(hipEventSynchronize(done),"pipeline completion"))return false;
        float maximum=0.0f;
        for(unsigned segment=0u;segment<segments;++segment)for(unsigned phase=0u;phase<3u;++phase) {
            float elapsed=0.0f;
            if(!checked(hipEventElapsedTime(&elapsed,owner.begin[segment][phase],owner.end[segment][phase]),
                        "pipeline completed phase time"))return false;
            if(!(elapsed>=0.0f && elapsed<=100.0f)) {
                set_error_text("Completed GDN pipeline phase exceeded its 100 ms guard");return false;
            }
            maximum=(std::max)(maximum,elapsed);
        }
        std::fprintf(stderr,"FLA_PIPELINED_SEGMENTS mode=%d tokens=%u segments=%u storage_slots=3 native_math_submissions=%u max_phase_ms=%.6f state_ordered=1 caller_joined=1 completed=1\n",
            mode,tokens,segments,submissions,static_cast<double>(maximum));
        return true;
    };
    const bool ok=execute();
    // Partial submission errors must drain every reader, not just the stream
    // on which the error occurred, before a caller can free its own surfaces.
    if(!ok)drain_pipeline();
    return ok;
}
