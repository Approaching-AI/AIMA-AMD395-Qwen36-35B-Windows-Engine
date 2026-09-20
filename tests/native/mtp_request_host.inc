// Included after the queued HIP/kernel doubles; exercises the actual Request.
struct RequestFixture {
    std::shared_ptr<LeasedWeights> source = std::make_shared<LeasedWeights>();
    qrt_sm121_mtp::Request request;
    std::vector<uint32_t> history{100u,101u};
    uint32_t current = 999u;
    qrt_sm121_mtp::TargetFrontier frontier() const {
        return {this,7u,10u,history.data(),history.size(),current};
    }
    explicit RequestFixture(unsigned capacity=16u) {
        using namespace qrt_sm121_mtp;
        ModelWeights model;assert(model.prepare(source,10u,expected_stream).status==hipSuccess);
        const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
        const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
        DrafterTables tables{t,p,262144u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
        auto batch=target_batch();
        assert(request.seed(*batch,frontier(),model.binding(10u),tables,capacity,expected_stream).status==hipSuccess);
        expected_hidden.clear();expected_ids.clear();
        assert(request.matches(frontier())&&request.committed_tokens()==2u);
    }
    qrt_sm121_mtp::TargetBatch begin(size_t remaining=4u) {
        qrt_sm121_mtp::TargetBatch plan;
        assert(request.begin(frontier(),remaining,&plan));
        assert(plan.first_position==history.size()&&plan.rows==2u&&plan.inputs[0]==current&&plan.inputs[1]==200u);
        return plan;
    }
    qrt_sm121_mtp::PromptStep prepare(const qrt_sm121_mtp::TargetBatch& plan,
        bool accepted,size_t remaining,qrt_sm121_mtp::AcceptedTarget* result) {
        const uint32_t samples[]={accepted?200u:201u,77u};
        const unsigned rows=accepted&&remaining>1?2u:1u;
        std::vector<float> hidden(4096u,2.0f);
        std::fill(hidden.begin()+2048u,hidden.end(),3.0f);
        expected_hidden.assign(2048u,0x4000u);
        if(rows==2u)expected_hidden.insert(expected_hidden.end(),2048u,0x4040u);
        expected_ids.assign(samples,samples+rows);expected_split=false;
        return request.prepare(plan.inputs.data(),samples,2u,{0u,1u},hidden,10u,result,expected_stream);
    }
    void advance(const qrt_sm121_mtp::TargetBatch& plan,const qrt_sm121_mtp::AcceptedTarget& accepted) {
        history.insert(history.end(),plan.inputs.begin(),plan.inputs.begin()+accepted.rows);
        current=accepted.outputs[accepted.rows-1u];
    }
};
static void test_request() {
    using namespace qrt_sm121_mtp;
    // Rejected, accepted and capacity-clipped q2 blocks retain only actual
    // committed target rows. No KV snapshot or device allocation per step.
    for(unsigned mode=0;mode<3u;++mode){
        reset();{
            RequestFixture fixture;
            RequestCheckpoint initial;
            assert(fixture.request.save(&initial,fixture.frontier(),expected_stream).status==hipSuccess);
            const size_t remaining=mode==2u?1u:4u;
            auto plan=fixture.begin(remaining);auto old=fixture.frontier();
            TargetBatch untouched{123u,1u,{88u,89u}};
            assert(!fixture.request.begin(old,remaining,&untouched)&&untouched.first_position==123u);
            AcceptedTarget accepted;
            const auto allocated=allocation_call;stages.clear();
            assert(fixture.prepare(plan,mode!=0u,remaining,&accepted).status==hipSuccess);
            assert(accepted.rows==(mode==1u?2u:1u)&&!accepted.drafter_retired);
            assert(fixture.request.committed_tokens()==2u&&fixture.request.retained_tokens()==2u+accepted.rows);
            assert(fixture.request.matches(old)&&initial.matches(old));
            assert(allocation_call==allocated);
            assert(std::none_of(stages.begin(),stages.end(),[](const auto& s){return s.rfind("weight_copy",0)==0;}));
            assert(fixture.request.save(&initial,old,expected_stream).status==hipErrorInvalidValue);
            fixture.advance(plan,accepted);
            auto receipt=fixture.frontier();
            assert(!fixture.request.commit(old)); // Old target did not commit.
            auto bad=receipt;bad.owner=&plan;assert(!fixture.request.commit(bad));
            bad=receipt;bad.generation++;assert(!fixture.request.commit(bad));
            bad=receipt;bad.model_epoch++;assert(!fixture.request.commit(bad));
            bad=receipt;bad.current_token++;assert(!fixture.request.commit(bad));
            bad=receipt;bad.processed_count--;assert(!fixture.request.commit(bad));
            for(size_t index:{size_t(0),fixture.history.size()-1u}){
                fixture.history[index]++;assert(!fixture.request.commit(receipt));fixture.history[index]--;
            }
            assert(fixture.request.commit(receipt)&&!fixture.request.pending());
            assert(fixture.request.matches(receipt)&&!initial.matches(receipt)&&!fixture.request.retired());
            assert(!fixture.request.commit(receipt));
            RequestCheckpoint saved;
            assert(fixture.request.save(&saved,receipt,expected_stream).status==hipSuccess);
            assert(saved.tokens()==receipt.processed_count&&saved.matches(receipt));
            Request restored;
            assert(restored.restore(saved,receipt,16u,expected_stream).status==hipSuccess);
            TargetBatch resumed;
            assert(restored.begin(receipt,1u,&resumed));
            assert(resumed.first_position==receipt.processed_count&&resumed.inputs[0]==receipt.current_token&&resumed.inputs[1]==200u);
            assert(restored.abort(10u));
        }assert(allocations.empty());
    }
    // Abort after a completed private append preserves the prior proposal;
    // retry does not read invalidated scratch and cannot emit rejected input.
    reset();{
        RequestFixture fixture;const auto original=fixture.frontier();
        auto plan=fixture.begin();AcceptedTarget accepted;
        assert(fixture.prepare(plan,true,4u,&accepted).status==hipSuccess);
        assert(fixture.request.retained_tokens()==4u&&fixture.request.abort(10u));
        assert(fixture.request.retained_tokens()==2u&&fixture.request.matches(original));
        RequestCheckpoint saved;
        assert(fixture.request.save(&saved,original,expected_stream).status==hipSuccess);
        plan=fixture.begin();
        assert(fixture.prepare(plan,false,4u,&accepted).status==hipSuccess&&accepted.rows==1u);
        fixture.advance(plan,accepted);assert(fixture.request.commit(fixture.frontier()));
        assert(fixture.history.back()==999u&&fixture.current==201u);
    }assert(allocations.empty());
    // Invalid or out-of-order execution cannot consume the pending target
    // operation, write outputs, queue work, or change the paired frontier.
    reset();{
        RequestFixture fixture;auto original=fixture.frontier();auto bad=original;
        TargetBatch plan;
        bad.generation++;assert(!fixture.request.begin(bad,4u,&plan));
        bad=original;bad.current_token++;assert(!fixture.request.begin(bad,4u,&plan));
        bad=original;bad.processed_count--;assert(!fixture.request.begin(bad,4u,&plan));
        assert(!fixture.request.begin(original,0u,&plan));
        assert(fixture.request.begin(original,4u,&plan));
        AcceptedTarget output{1u,{555u,556u},false};
        uint32_t inputs[]={999u,200u},samples[]={200u,77u};
        std::vector<float> hidden(4096u,1.0f);
        stages.clear();inputs[0]++;
        assert(fixture.request.prepare(inputs,samples,2u,{0u,1u},hidden,10u,&output,expected_stream).status==hipErrorInvalidValue);
        inputs[0]--;
        assert(fixture.request.prepare(inputs,samples,2u,{1u,0u},hidden,10u,&output,expected_stream).status==hipErrorInvalidValue);
        assert(fixture.request.prepare(inputs,samples,2u,{0u,1u},hidden,11u,&output,expected_stream).status==hipErrorInvalidValue);
        hidden[0]=INFINITY;
        assert(fixture.request.prepare(inputs,samples,2u,{0u,1u},hidden,10u,&output,expected_stream).status==hipErrorInvalidValue);
        assert(output.rows==1u&&output.outputs[0]==555u&&stages.empty());
        assert(fixture.request.pending()&&fixture.request.matches(original)&&fixture.request.abort(10u));
    }assert(allocations.empty());
    // A too-small live cache rejects before target work; a clipped final
    // block requires only its one accepted row.
    reset();{
        RequestFixture fixture(3u);TargetBatch plan;
        assert(!fixture.request.begin(fixture.frontier(),2u,&plan));
        plan=fixture.begin(1u);AcceptedTarget accepted;
        assert(fixture.prepare(plan,true,1u,&accepted).status==hipSuccess&&accepted.rows==1u);
        fixture.advance(plan,accepted);assert(fixture.request.commit(fixture.frontier()));
        assert(!fixture.request.begin(fixture.frontier(),1u,&plan));
    }assert(allocations.empty());
    for(const char* failure:{"input_copy","gather","kv","query","moe","copy1","copy2"}){
        reset();{
            RequestFixture fixture;auto old=fixture.frontier();auto plan=fixture.begin();reset();
            fail_stage=failure;AcceptedTarget output{1u,{555u,556u},false};
            const auto result=fixture.prepare(plan,true,4u,&output);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&queued.empty());
            assert(!fixture.request.pending()&&fixture.request.matches(old)&&fixture.request.retained_tokens()==2u);
            assert(output.outputs[0]==555u&&!fixture.request.quarantined());
            reset();plan=fixture.begin();
            assert(fixture.prepare(plan,false,4u,&output).status==hipSuccess);
            fixture.advance(plan,output);assert(fixture.request.commit(fixture.frontier()));
        }assert(allocations.empty());
    }
    // Failed save leaves the previous immutable checkpoint available.
    for(unsigned failure=0;failure<2u;++failure){
        reset();{
            RequestFixture fixture;RequestCheckpoint saved;
            assert(fixture.request.save(&saved,fixture.frontier(),expected_stream).status==hipSuccess);
            reset();if(failure)fail_stage="weight_copy1";else fail_allocation=1;
            assert(fixture.request.save(&saved,fixture.frontier(),expected_stream).status!=hipSuccess);
            assert(saved.matches(fixture.frontier())&&fixture.request.matches(fixture.frontier())&&queued.empty());
        }assert(allocations.empty());
    }
    // A copyable checkpoint pins the model and exact token history after the
    // original request dies. A same-length different target branch is refused.
    reset();{
        RequestCheckpoint saved;std::weak_ptr<LeasedWeights> borrowed;
        TargetFrontier actual;std::vector<uint32_t> retained;
        {
            RequestFixture fixture;borrowed=fixture.source;
            assert(fixture.request.save(&saved,fixture.frontier(),expected_stream).status==hipSuccess);
            actual=fixture.frontier();retained=fixture.history;
        }
        actual.processed_inputs=retained.data();assert(!borrowed.expired()&&saved.matches(actual));
        Request restored;retained[0]++;
        assert(restored.restore(saved,actual,16u,expected_stream).status==hipErrorInvalidValue);
        retained[0]--;auto alias=saved;saved={};
        assert(restored.restore(alias,actual,16u,expected_stream).status==hipSuccess);
        alias={};assert(!borrowed.expired());
        TargetBatch plan;assert(restored.begin(actual,2u,&plan)&&plan.inputs[1]==200u);
        assert(restored.abort(10u));
        borrowed.lock()->generation=11u;
        assert(!restored.begin(actual,2u,&plan)&&!restored.matches(actual));
    }assert(allocations.empty());
    // Unknown upload, append and proposal completion retains all borrowed
    // storage after destroying the request, reference views and model owner.
    for(unsigned fence=1u;fence<=4u;++fence){
        reset();{
            RequestFixture fixture;auto plan=fixture.begin();reset();fail_sync=fence;
            AcceptedTarget output{1u,{555u,556u},false};
            const auto result=fixture.prepare(plan,true,4u,&output);
            assert(result.status==injected&&result.completion_unknown&&fixture.request.quarantined());
            assert(output.outputs[0]==555u&&!fixture.request.abort(10u));
        }
        late_completion();assert(allocations.empty());
    }
    // Epoch invalidation at a completed upload/append cannot be committed
    // under the earlier target identity and never publishes prepared output.
    reset();{
        RequestFixture fixture;auto plan=fixture.begin();reset();
        completion_hook=[&]{fixture.source->generation=11u;};
        AcceptedTarget output{1u,{555u,556u},false};
        const auto result=fixture.prepare(plan,true,4u,&output);
        assert(result.status==hipErrorInvalidValue&&!result.completion_unknown&&queued.empty());
        assert(output.outputs[0]==555u&&!fixture.request.matches(fixture.frontier()));
        assert(!fixture.request.quarantined());
    }assert(allocations.empty());
    // Both request-level save and restore keep late copy sources/destinations
    // alive even after every public checkpoint and request has been dropped.
    for(bool restore_failure:{false,true}){
        reset();{
            RequestFixture fixture;RequestCheckpoint saved;
            if(restore_failure)
                assert(fixture.request.save(&saved,fixture.frontier(),expected_stream).status==hipSuccess);
            reset();fail_sync=1u;
            if(restore_failure){
                Request restored;
                assert(restored.restore(saved,fixture.frontier(),16u,expected_stream).completion_unknown);
                assert(restored.quarantined());
            }else{
                assert(fixture.request.save(&saved,fixture.frontier(),expected_stream).completion_unknown);
                assert(fixture.request.quarantined()&&!saved.tokens());
            }
        }
        late_completion();assert(allocations.empty());
    }
}

static void test_prefill_request_probe(const std::string& directory) {
    using namespace qrt_sm121_mtp;
    using namespace qrt_sm121_mtp_runtime;
    for(unsigned rows:{7169u,8192u}){
        reset();expected_stream=nullptr;{
            auto batch=target_batch(rows);auto source=std::make_shared<LeasedWeights>();
            std::vector<uint32_t> ids(rows);for(unsigned i=0;i<rows;++i)ids[i]=100u+i;
            const TargetFrontier actual{source.get(),17u,10u,ids.data(),rows,999u};
            RequestCheckpoint saved;std::string stage,failure;
            const std::string prefix=directory+"/request-"+std::to_string(rows);
            assert(probe_prefill_request(*batch,source,actual,rows+32u,prefix,&saved,stage,failure));
            assert(saved.matches(actual)&&saved.tokens()==rows&&saved.allocated_bytes()==size_t(rows)*2048u);
            assert(saved.model_pack_bytes()==8388608u);
            assert(allocations.size()==2u&&queued.empty());
            assert(!probe_prefill_request(*batch,source,actual,rows+32u,prefix,&saved,stage,failure));
            assert(stage=="mtp_request_probe_contract"&&saved.matches(actual));
            Request restored;assert(restored.restore(saved,actual,rows+32u).status==hipSuccess);
            assert(restored.proposal(10u).tokens[0]==200u&&!restored.observation(10u).rows);
            TargetBatch plan;assert(restored.begin(actual,32u,&plan)&&plan.first_position==rows&&plan.inputs[0]==999u&&plan.inputs[1]==200u);
            assert(!restored.proposal(10u).rows&&!restored.cache_data()&&restored.abort(10u));
        }assert(allocations.empty());expected_stream=reinterpret_cast<void*>(0x1234);
    }
    for(unsigned fence:{6u,7u}){
        reset();expected_stream=nullptr;{
            auto batch=target_batch(7169u);auto source=std::make_shared<LeasedWeights>();
            std::vector<uint32_t> ids(7169u);for(unsigned i=0;i<ids.size();++i)ids[i]=100u+i;
            const TargetFrontier actual{source.get(),17u,10u,ids.data(),ids.size(),999u};
            RequestCheckpoint saved;std::string stage,failure;
            const std::string prefix=directory+"/request-unknown-"+std::to_string(fence);
            fail_sync=fence;
            assert(!probe_prefill_request(*batch,source,actual,7201u,prefix,&saved,stage,failure));
            assert(stage==(fence==6u?"mtp_prefill_capture_completion":"cache_checkpoint_completion"));
            assert(!saved.tokens()&&!std::filesystem::exists(prefix+".request.json"));
        }
        late_completion();assert(allocations.empty());expected_stream=reinterpret_cast<void*>(0x1234);
    }
}

static void test_prefill_request_seed() {
    using namespace qrt_sm121_mtp;
    using namespace qrt_sm121_mtp_runtime;
    unsigned completion_fences=0;
    for(unsigned rows:{1u,7u,256u,7169u,8192u}){
        reset();expected_stream=nullptr;{
            auto batch=target_batch(rows);auto source=std::make_shared<LeasedWeights>();
            std::vector<uint32_t> ids(rows);for(unsigned i=0;i<rows;++i)ids[i]=100u+i;
            const TargetFrontier actual{source.get(),17u,10u,ids.data(),rows,999u};
            RequestCheckpoint saved;
            const auto result=seed_prefill_request(*batch,source,actual,rows+32u,&saved);
            assert(result.status==hipSuccess&&!result.completion_unknown&&saved.matches(actual));
            assert(saved.tokens()==rows&&saved.allocated_bytes()==size_t(rows)*2048u);
            completion_fences=sync_call;
            Request restored;assert(restored.restore(saved,actual,rows+32u).status==hipSuccess);
            TargetBatch next;assert(restored.begin(actual,32u,&next));
            assert(next.first_position==rows&&next.inputs[0]==999u&&next.inputs[1]==200u);
            assert(restored.abort(10u));
        }
        assert(allocations.empty()&&queued.empty());
    }
    // Every completion failure must leave the caller's output unpublished
    // and preserve queued producer, input and original model lifetimes.
    for(unsigned fence=1u;fence<=completion_fences;++fence){
        reset();expected_stream=nullptr;{
            auto batch=target_batch(7u);auto source=std::make_shared<LeasedWeights>();
            const std::vector<uint32_t> ids{100u,101u,102u,103u,104u,105u,106u};
            const TargetFrontier actual{source.get(),17u,10u,ids.data(),ids.size(),999u};
            RequestCheckpoint saved;fail_sync=fence;
            const auto result=seed_prefill_request(*batch,source,actual,39u,&saved);
            assert(result.status!=hipSuccess&&result.completion_unknown&&!saved.tokens());
        }
        late_completion();assert(allocations.empty());
    }
    reset();expected_stream=nullptr;{
        auto batch=target_batch(7u);auto source=std::make_shared<LeasedWeights>();
        std::vector<uint32_t> ids{100u,101u,102u,103u,104u,105u,106u};
        const TargetFrontier actual{source.get(),17u,10u,ids.data(),ids.size(),999u};
        RequestCheckpoint saved;
        assert(seed_prefill_request(*batch,source,actual,39u,&saved).status==hipSuccess);
        const auto calls=allocation_call;
        for(unsigned invalid=0;invalid<8u;++invalid){
            auto frontier=actual;unsigned capacity=39u;auto model=source;
            switch(invalid){
                case 0:frontier.owner=nullptr;break;case 1:frontier.generation=0u;break;
                case 2:frontier.model_epoch++;break;case 3:frontier.current_token++;break;
                case 4:frontier.processed_count--;break;case 5:capacity=6u;break;
                case 6:capacity=262145u;break;case 7:model.reset();break;
            }
            assert(seed_prefill_request(*batch,model,frontier,capacity,&saved).status!=hipSuccess);
            assert(allocation_call==calls&&saved.matches(actual));
        }
        fail_tables=true;
        assert(seed_prefill_request(*batch,source,actual,39u,&saved).status!=hipSuccess);
        assert(saved.matches(actual));fail_tables=false;
    }
    assert(allocations.empty()&&queued.empty());expected_stream=reinterpret_cast<void*>(0x1234);
    std::cout<<"native MTP plain seed cases=5 completion_fences="<<completion_fences<<" pass\n";
}
