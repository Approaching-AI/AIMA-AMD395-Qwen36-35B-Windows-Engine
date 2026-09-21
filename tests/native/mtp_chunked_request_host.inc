// Real Request/PrefillRows with the same queued HIP producers as the drafter
// suite. These checks establish ownership and identity, not GPU arithmetic.
static std::unique_ptr<qrt_mtp_target_rows::PrefillRows> chunk_batch(
    const std::vector<uint32_t>& prompt,unsigned first,unsigned rows,uint32_t sampled=999u) {
    auto batch=std::make_unique<qrt_mtp_target_rows::PrefillRows>(prompt.data(),prompt.size(),first,rows);
    std::vector<float> hidden(size_t(rows)*2048u,1.25f);
    assert(batch->stage(batch->local_rows(),hidden,sampled)&&batch->publish(sampled));
    expected_hidden=batch->hidden();expected_ids=batch->shifted_tokens();expected_split=true;
    assert(expected_ids.back()==(batch->discarded_prefill()?prompt.back():sampled));
    return batch;
}
struct ChunkedRequestFixture {
    std::shared_ptr<LeasedWeights> source=std::make_shared<LeasedWeights>();
    qrt_sm121_mtp::ModelWeights model;
    qrt_sm121_mtp::DrafterTables tables;
    qrt_sm121_mtp::Request request;
    std::vector<uint32_t> prompt;
    explicit ChunkedRequestFixture(unsigned total=7u):prompt(total){
        for(unsigned i=0;i<total;++i)prompt[i]=100u+i;
        assert(model.prepare(source,10u,expected_stream).status==hipSuccess);
        const auto* p=reinterpret_cast<const uint16_t*>(0x60000);
        const auto* t=reinterpret_cast<const unsigned char*>(0x70000);
        tables={t,p,262144u,t,t,{p,p,reinterpret_cast<const uint32_t*>(0x80000)}};
    }
    qrt_sm121_mtp::TargetFrontier frontier(unsigned count,uint32_t current=999u)const{
        return {this,17u,10u,prompt.data(),count,current};
    }
    void first(unsigned rows=3u){
        auto batch=chunk_batch(prompt,0u,rows);
        assert(request.seed_prefill_chunks(*batch,frontier(rows),model.binding(10u),tables,
            static_cast<unsigned>(prompt.size()+32u),expected_stream).status==hipSuccess);
    }
};
static void test_chunked_request() {
    using namespace qrt_sm121_mtp;
    for(unsigned total:{7u,9216u,16384u,17408u,131072u}){
        reset();{
            ChunkedRequestFixture fixture(total);
            unsigned first=0u,chunk=0u;
            const unsigned width=total==7u?3u:8192u;
            while(first<total){
                const unsigned count=std::min(width,total-first);
                auto batch=chunk_batch(fixture.prompt,first,count);
                const auto actual=fixture.frontier(first+count);
                const auto result=first?fixture.request.append_prefill_chunk(*batch,actual,expected_stream):
                    fixture.request.seed_prefill_chunks(*batch,actual,fixture.model.binding(10u),fixture.tables,
                        total+32u,expected_stream);
                assert(result.status==hipSuccess&&!result.completion_unknown);
                assert(fixture.request.committed_tokens()==first+count&&fixture.request.retained_tokens()==first+count);
                RequestCheckpoint saved;TargetBatch plan{55u,1u,{77u,88u}};
                if(first+count<total){
                    assert(!fixture.request.matches(actual)&&!fixture.request.begin(actual,16u,&plan));
                    assert(plan.first_position==55u&&!fixture.request.proposal(10u).rows&&!fixture.request.cache_data());
                    assert(fixture.request.save(&saved,actual,expected_stream).status==hipErrorInvalidValue&&!saved.tokens());
                    assert(std::count(stages.begin(),stages.end(),"head")==0);
                }else{
                    assert(fixture.request.matches(actual)&&fixture.request.proposal(10u).rows==1u);
                    assert(std::count(stages.begin(),stages.end(),"head")==1);
                    assert(attention_tokens==total&&attention_first==total-1u&&attention_rows==1u);
                    assert(fixture.request.save(&saved,actual,expected_stream).status==hipSuccess&&saved.matches(actual));
                    Request restored;assert(restored.restore(saved,actual,total+32u,expected_stream).status==hipSuccess);
                    assert(restored.begin(actual,16u,&plan)&&plan.first_position==total&&plan.inputs[0]==999u&&plan.inputs[1]==200u);
                    assert(restored.abort(10u));
                    assert(fixture.request.append_prefill_chunk(*batch,actual,expected_stream).status==hipErrorInvalidValue);
                }
                first+=count;++chunk;
            }
            assert(chunk>1u);
        }assert(allocations.empty()&&queued.empty());
    }
    // Complete prompt identity includes future tokens and the discarded
    // chunk's shifted last token, not just the prefix consumed so far.
    reset();{
        ChunkedRequestFixture fixture;fixture.first();
        auto batch=chunk_batch(fixture.prompt,3u,4u);auto actual=fixture.frontier(7u);
        for(unsigned bad=0u;bad<8u;++bad){
            auto target=actual;
            switch(bad){
                case 0:target.owner=batch.get();break;case 1:target.generation++;break;
                case 2:target.model_epoch++;break;case 3:target.current_token++;break;
                case 4:target.processed_count--;break;case 5:target.processed_inputs=nullptr;break;
                case 6:fixture.prompt[0]++;break;case 7:fixture.prompt[6]++;break;
            }
            const auto calls=allocation_call;const auto submitted=stages.size();
            assert(fixture.request.append_prefill_chunk(*batch,target,expected_stream).status==hipErrorInvalidValue);
            assert(allocation_call==calls&&stages.size()==submitted&&fixture.request.committed_tokens()==3u);
            if(bad==6u)fixture.prompt[0]--;
            if(bad==7u)fixture.prompt[6]--;
        }
        auto changed=fixture.prompt;changed.back()++;
        auto alien=chunk_batch(changed,3u,4u);
        auto target=actual;target.processed_inputs=changed.data();
        const auto submitted=stages.size();
        assert(fixture.request.append_prefill_chunk(*alien,target,expected_stream).status==hipErrorInvalidValue);
        assert(stages.size()==submitted);
        auto duplicate=chunk_batch(fixture.prompt,0u,3u);
        assert(fixture.request.append_prefill_chunk(*duplicate,fixture.frontier(3u),expected_stream).status==hipErrorInvalidValue);
        batch=chunk_batch(fixture.prompt,3u,4u);
        assert(fixture.request.append_prefill_chunk(*batch,actual,expected_stream).status==hipSuccess);
    }assert(allocations.empty());
    // A completed failure in the final chunk restores the private prefix.
    // Retrying replays the real final rows and publishes only one proposal.
    for(const char* stage:{"input_copy","gather","fusion","cache","query","head","copy2"}){
        reset();{
            ChunkedRequestFixture fixture;fixture.first();
            auto batch=chunk_batch(fixture.prompt,3u,4u);reset();
            expected_hidden=batch->hidden();expected_ids=batch->shifted_tokens();expected_split=true;
            fail_stage=stage;
            auto result=fixture.request.append_prefill_chunk(*batch,fixture.frontier(7u),expected_stream);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&queued.empty());
            assert(fixture.request.committed_tokens()==3u&&fixture.request.retained_tokens()==3u);
            assert(!fixture.request.matches(fixture.frontier(7u))&&!fixture.request.proposal(10u).rows);
            reset();expected_hidden=batch->hidden();expected_ids=batch->shifted_tokens();expected_split=true;
            result=fixture.request.append_prefill_chunk(*batch,fixture.frontier(7u),expected_stream);
            assert(result.status==hipSuccess&&fixture.request.matches(fixture.frontier(7u)));
        }assert(allocations.empty());
    }
    for(bool first:{true,false})for(unsigned fence=1u;fence<=(first?3u:4u);++fence){
        reset();{
            ChunkedRequestFixture fixture;
            if(!first)fixture.first();
            auto batch=chunk_batch(fixture.prompt,first?0u:3u,first?3u:4u);reset();
            expected_hidden=batch->hidden();expected_ids=batch->shifted_tokens();expected_split=true;
            fail_sync=fence;
            const auto actual=fixture.frontier(first?3u:7u);
            auto result=first?fixture.request.seed_prefill_chunks(*batch,actual,fixture.model.binding(10u),fixture.tables,39u,expected_stream):
                fixture.request.append_prefill_chunk(*batch,actual,expected_stream);
            assert(result.status!=hipSuccess&&result.completion_unknown&&fixture.request.quarantined());
            RequestCheckpoint saved;assert(fixture.request.save(&saved,actual,expected_stream).completion_unknown&&!saved.tokens());
            const auto calls=stages.size();
            assert(fixture.request.append_prefill_chunk(*batch,actual,expected_stream).completion_unknown&&stages.size()==calls);
        }
        late_completion();assert(allocations.empty());
    }
    std::cout<<"native MTP chunked seed cases=5 invalid_frontiers=10 retry_failures=7 completion_fences=7 pass\n";
}

static void test_chunked_seed_runtime() {
    using namespace qrt_sm121_mtp;
    using namespace qrt_sm121_mtp_runtime;
    unsigned fences=0;
    // The complete runtime helper includes actual model preparation, table
    // admission, both real chunks and the final immutable device checkpoint.
    for(unsigned failed=0u;failed<=9u;++failed){
        reset();expected_stream=nullptr;{
            const std::vector<uint32_t> prompt{10u,11u,12u,13u,14u,15u,16u};
            auto source=std::make_shared<LeasedWeights>();ChunkedPrefillSeed seed;
            auto first=chunk_batch(prompt,0u,3u);
            expected_split=false;
            TargetFrontier actual{&seed,17u,10u,prompt.data(),3u,999u};
            fail_sync=failed;RequestCheckpoint checkpoint;
            auto result=seed.begin(*first,source,actual,39u);
            if(result.status==hipSuccess){
                auto last=chunk_batch(prompt,3u,4u);actual.processed_count=7u;
                expected_split=false;
                result=seed.append(*last,actual);
                if(result.status==hipSuccess)result=seed.save(&checkpoint,actual);
            }
            if(!failed){
                assert(result.status==hipSuccess&&checkpoint.matches(actual)&&seed.epoch()==10u);
                fences=sync_call;assert(fences==9u);
            }else{
                assert(result.status!=hipSuccess&&result.completion_unknown&&!checkpoint.tokens());
            }
        }
        if(failed)late_completion();
        assert(allocations.empty()&&queued.empty());
    }
    reset();expected_stream=nullptr;{
        const std::vector<uint32_t> prompt{10u,11u,12u,13u,14u,15u,16u};
        auto source=std::make_shared<LeasedWeights>();auto first=chunk_batch(prompt,0u,3u);
        ChunkedPrefillSeed seed;TargetFrontier actual{&seed,17u,10u,prompt.data(),3u,999u};
        const auto calls=allocation_call;auto wrong=actual;wrong.current_token++;
        assert(seed.begin(*first,source,wrong,39u).status==hipErrorInvalidValue&&allocation_call==calls);
        fail_tables=true;auto result=seed.begin(*first,source,actual,39u);
        assert(result.status!=hipSuccess&&!result.completion_unknown);fail_tables=false;
        RequestCheckpoint checkpoint;assert(seed.save(&checkpoint,actual).status==hipErrorInvalidValue&&!checkpoint.tokens());
    }assert(allocations.empty());
    expected_stream=reinterpret_cast<void*>(0x1234);
    std::cout<<"native MTP chunked runtime completion_fences="<<fences<<" pass\n";
}

static void test_explicit_prefill_norm_order() {
    using namespace qrt_sm121_mtp;
    for(unsigned total:{7u,9216u})for(bool split:{false,true}){
        reset();{
            ChunkedRequestFixture fixture(total);
            for(unsigned first=0u;first<total;){
                const unsigned rows=std::min(8192u,total-first);
                auto batch=chunk_batch(fixture.prompt,first,rows);
                expected_split=split;
                const auto actual=fixture.frontier(first+rows);
                const auto result=first?fixture.request.append_prefill_chunk(*batch,actual,expected_stream):
                    total>8192u?fixture.request.seed_prefill_chunks(*batch,actual,fixture.model.binding(10u),
                        fixture.tables,total+32u,expected_stream,1024u,split):
                    fixture.request.seed(*batch,actual,fixture.model.binding(10u),fixture.tables,
                        total+32u,expected_stream,1024u,split);
                assert(result.status==hipSuccess&&!result.completion_unknown);
                first+=rows;
            }
            assert(fixture.request.matches(fixture.frontier(total)));
            assert(std::count(stages.begin(),stages.end(),"head")==1);
        }assert(allocations.empty()&&queued.empty());
    }
    std::cout<<"native MTP explicit prefill norm cases=2 profiles=2 pass\n";
}
