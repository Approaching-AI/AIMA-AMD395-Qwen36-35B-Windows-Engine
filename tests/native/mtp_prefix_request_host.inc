// Actual Request prefix forks, using deferred producer writes and distinct
// old/new cache bytes. Numerical comparisons belong to the native probe.
struct PrefixRequestFixture {
    ChunkedRequestFixture owner;
    qrt_sm121_mtp::RequestCheckpoint checkpoint;
    std::vector<std::array<uint16_t,2048u>> hidden;
    bool split;
    explicit PrefixRequestFixture(unsigned tokens=16384u,bool profile=false):owner(tokens),split(profile){
        for(unsigned first=0u;first<tokens;first+=8192u){
            qrt_mtp_target_rows::PrefillRows batch(owner.prompt.data(),tokens,first,8192u);
            std::vector<float> values(size_t(8192u)*2048u,1.25f);
            for(unsigned h=0;h<2048u;++h)
                values[size_t(8191u)*2048u+h]=1.5f+float(first/8192u)/16.0f+float(h%13u)/128.0f;
            assert(batch.stage(batch.local_rows(),values,999u)&&batch.publish(999u));
            expected_hidden=batch.hidden();expected_ids=batch.shifted_tokens();expected_split=split;
            std::array<uint16_t,2048u> last{};
            std::copy_n(batch.hidden().end()-2048u,2048u,last.begin());hidden.push_back(last);
            const auto actual=owner.frontier(first+8192u);
            const auto result=tokens==8192u?owner.request.seed(batch,actual,owner.model.binding(10u),owner.tables,
                tokens+32u,expected_stream,1024u,split):first?
                owner.request.append_prefill_chunk(batch,actual,expected_stream):
                owner.request.seed_prefill_chunks(batch,actual,owner.model.binding(10u),owner.tables,
                    tokens+32u,expected_stream,1024u,split);
            assert(result.status==hipSuccess);
        }
        assert(owner.request.save(&checkpoint,cached(),expected_stream).status==hipSuccess);
    }
    qrt_sm121_mtp::TargetFrontier cached()const{return owner.frontier(static_cast<unsigned>(owner.prompt.size()));}
    std::vector<uint32_t> extended(unsigned suffix=1024u,uint32_t tail=777u)const{
        auto result=owner.prompt;result.resize(result.size()+suffix,321u);result.back()=tail;return result;
    }
    qrt_sm121_mtp::TargetFrontier actual(const std::vector<uint32_t>& prompt,unsigned count)const{
        auto result=cached();result.processed_inputs=prompt.data();result.processed_count=count;return result;
    }
    void unchanged()const{
        assert(checkpoint.matches(cached())&&owner.request.matches(cached()));
        const auto* cache=owner.request.cache_data();assert(cache);
        assert(std::all_of(cache,cache+owner.prompt.size()*1024u,[](uint16_t v){return v==55u;}));
        qrt_sm121_mtp::Request copy;
        assert(copy.restore(checkpoint,cached(),static_cast<unsigned>(owner.prompt.size()+32u),expected_stream).status==hipSuccess);
        cache=copy.cache_data();assert(cache);
        assert(std::all_of(cache,cache+owner.prompt.size()*1024u,[](uint16_t v){return v==55u;}));
    }
};
struct PrefixInputExpected {
    std::vector<uint16_t> hidden;
    std::vector<uint32_t> ids;
};
static std::shared_ptr<size_t> expect_prefix_inputs(const PrefixRequestFixture& fixture,
    const qrt_mtp_target_rows::PrefillRows& batch){
    auto inputs=std::make_shared<std::vector<PrefixInputExpected>>();
    for(unsigned i=0u;i<fixture.hidden.size();++i){
        const uint32_t old=i+1u==fixture.hidden.size()?999u:fixture.owner.prompt.back();
        if(old==batch.prompt().back())continue;
        inputs->push_back({std::vector<uint16_t>(fixture.hidden[i].begin(),fixture.hidden[i].end()),
            {batch.prompt().back()}});
    }
    inputs->push_back({batch.hidden(),batch.shifted_tokens()});
    auto index=std::make_shared<size_t>(0u);const bool split=fixture.split;
    expected_hidden.clear();expected_ids.clear();
    input_observer=[inputs,index,split](const uint16_t* hidden,const uint32_t* ids,unsigned rows,bool profile){
        assert(*index<inputs->size());const auto& want=inputs->at((*index)++);
        assert(profile==split&&want.ids.size()==rows&&want.hidden.size()==size_t(rows)*2048u);
        assert(std::equal(want.hidden.begin(),want.hidden.end(),hidden));
        assert(std::equal(want.ids.begin(),want.ids.end(),ids));
    };
    return index;
}
static void expect_prefix_cache(const PrefixRequestFixture& fixture,const std::vector<uint32_t>& prompt){
    const auto old=fixture.owner.prompt;
    cache_observer=[old,prompt](const uint16_t* cache,unsigned tokens){
        assert(tokens==prompt.size());
        for(unsigned row=0u;row<tokens;++row){
            const bool seam=row<old.size()&&(row+1u)%8192u==0u;
            const uint32_t previous=row+1u==old.size()?999u:old.back();
            const uint16_t expected=row>=old.size()||(seam&&previous!=prompt.back())?77u:55u;
            for(unsigned c=0u;c<1024u;++c)assert(cache[size_t(row)*1024u+c]==expected);
        }
    };
}
static void test_prefix_request(){
    using namespace qrt_sm121_mtp;
    unsigned successes=0u,declines=0u,failures=0u,fences=0u;
    for(unsigned prefix:{8192u,16384u})for(bool split:{false,true})for(unsigned unchanged:{0u,1u}){
        reset();{
            PrefixRequestFixture fixture(prefix,split);
            const uint32_t tail=unchanged?(prefix==8192u?999u:fixture.owner.prompt.back()):777u;
            auto prompt=fixture.extended(1024u,tail);
            auto batch=chunk_batch(prompt,prefix,1024u);reset();cache_fill=77u;
            auto observed=expect_prefix_inputs(fixture,*batch);expect_prefix_cache(fixture,prompt);
            Request fork;const auto target=fixture.actual(prompt,static_cast<unsigned>(prompt.size()));
            assert(fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*batch,target,
                static_cast<unsigned>(prompt.size()+32u),split,expected_stream).status==hipSuccess);
            const unsigned repairs=unchanged?(prefix==8192u?0u:1u):prefix/8192u;
            assert(*observed==repairs+1u&&std::count(stages.begin(),stages.end(),"head")==1u);
            assert(fork.matches(target)&&fork.proposal(10u).rows==1u);
            cache_observer(fork.cache_data(),static_cast<unsigned>(prompt.size()));
            RequestCheckpoint saved;assert(fork.save(&saved,target,expected_stream).status==hipSuccess&&saved.matches(target));
            TargetBatch plan;assert(fork.begin(target,16u,&plan)&&plan.inputs[0]==999u&&plan.inputs[1]==200u);
            assert(fork.abort(10u));reset();fixture.unchanged();++successes;
        }assert(allocations.empty()&&queued.empty());
    }
    reset();{
        PrefixRequestFixture fixture;auto prompt=fixture.extended(9216u);
        auto batch=chunk_batch(prompt,16384u,8192u);reset();cache_fill=77u;
        auto observed=expect_prefix_inputs(fixture,*batch);expect_prefix_cache(fixture,prompt);
        Request fork;auto target=fixture.actual(prompt,24576u);
        assert(fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*batch,target,
            25632u,false,expected_stream).status==hipSuccess&&*observed==3u);
        RequestCheckpoint saved;TargetBatch next;
        assert(!fork.cache_data()&&!fork.proposal(10u).rows&&!fork.begin(target,16u,&next));
        assert(fork.save(&saved,target,expected_stream).status==hipErrorInvalidValue&&!saved.tokens());
        assert(std::count(stages.begin(),stages.end(),"head")==0);
        input_observer={};auto last=chunk_batch(prompt,24576u,1024u);expected_split=false;
        target.processed_count=25600u;
        assert(fork.append_prefill_chunk(*last,target,expected_stream).status==hipSuccess&&fork.matches(target));
        assert(std::count(stages.begin(),stages.end(),"head")==1);
        assert(fork.save(&saved,target,expected_stream).status==hipSuccess&&saved.matches(target));
        cache_observer(fork.cache_data(),25600u);reset();fixture.unchanged();++successes;
    }assert(allocations.empty()&&queued.empty());
    reset();{
        PrefixRequestFixture fixture;auto prompt=fixture.extended();auto batch=chunk_batch(prompt,16384u,1024u);
        for(unsigned bad=0u;bad<13u;++bad){
            auto cached=fixture.cached();auto actual=fixture.actual(prompt,17408u);
            unsigned capacity=17440u,blocks=1024u;bool profile=false;
            switch(bad){
                case 0:cached.generation++;break;case 1:cached.current_token++;break;
                case 2:actual.owner=batch.get();break;case 3:actual.generation++;break;
                case 4:actual.model_epoch++;break;case 5:actual.processed_count--;break;
                case 6:actual.processed_inputs=nullptr;break;case 7:actual.current_token++;break;
                case 8:capacity=17407u;break;case 9:profile=true;break;
                case 10:blocks=0u;break;case 11:blocks=4097u;break;case 12:prompt[0]++;break;
            }
            Request fork;const auto calls=allocation_call;const auto submitted=stages.size();
            assert(fork.extend_prefill_prefix(fixture.checkpoint,cached,*batch,actual,capacity,profile,
                expected_stream,blocks).status==hipErrorInvalidValue);
            assert(allocation_call==calls&&stages.size()==submitted&&!fork.committed_tokens());
            if(bad==12u)prompt[0]--;++declines;
        }
        auto alien=prompt;alien[99]++;auto other=chunk_batch(alien,16384u,1024u);
        Request fork;const auto submitted=stages.size();
        assert(fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*other,
            fixture.actual(alien,17408u),17440u,false,expected_stream).status==hipErrorInvalidValue);
        assert(stages.size()==submitted);++declines;reset();fixture.unchanged();
    }assert(allocations.empty());
    // Both changed seams may fail after partial producer writes. Each retry
    // starts from the same immutable checkpoint; no failed fork is visible.
    for(unsigned occurrence:{1u,2u})for(const char* stage:{"input_copy","gather","fusion","prompt_norm","kv","cache"}){
        reset();{
            PrefixRequestFixture fixture;auto prompt=fixture.extended();auto batch=chunk_batch(prompt,16384u,1024u);
            reset();cache_fill=77u;expect_prefix_inputs(fixture,*batch);expect_prefix_cache(fixture,prompt);
            fail_stage=stage;fail_stage_occurrence=occurrence;Request fork;
            const auto target=fixture.actual(prompt,17408u);
            const auto result=fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*batch,target,
                17440u,false,expected_stream);
            assert(result.status!=hipSuccess&&!result.completion_unknown&&!fork.committed_tokens()&&!fork.cache_data());
            assert(std::count(stages.begin(),stages.end(),"head")==0&&queued.empty());
            reset();fixture.unchanged();reset();cache_fill=77u;
            expect_prefix_inputs(fixture,*batch);expect_prefix_cache(fixture,prompt);
            assert(fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*batch,target,
                17440u,false,expected_stream).status==hipSuccess&&fork.matches(target));
            reset();fixture.unchanged();++failures;
        }assert(allocations.empty()&&queued.empty());
    }
    // Restore, both seam uploads/norms/KV writes, final actual append and head.
    for(unsigned failed=1u;failed<=11u;++failed){
        reset();{
            PrefixRequestFixture fixture;auto prompt=fixture.extended();auto batch=chunk_batch(prompt,16384u,1024u);
            reset();cache_fill=77u;expect_prefix_inputs(fixture,*batch);expect_prefix_cache(fixture,prompt);
            fail_sync=failed;Request fork;const auto target=fixture.actual(prompt,17408u);
            const auto result=fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*batch,target,
                17440u,false,expected_stream);
            assert(result.status!=hipSuccess&&result.completion_unknown&&fork.quarantined()&&!fork.cache_data());
            RequestCheckpoint output;const auto calls=stages.size();
            assert(fork.save(&output,target,expected_stream).completion_unknown&&!output.tokens());
            assert(fork.extend_prefill_prefix(fixture.checkpoint,fixture.cached(),*batch,target,
                17440u,false,expected_stream).completion_unknown&&stages.size()==calls);++fences;
        }
        late_completion();assert(allocations.empty()&&queued.empty());
    }
    reset();
    std::cout<<"native MTP prefix repairs cases="<<successes<<" invalid_frontiers="<<declines
        <<" retry_failures="<<failures<<" completion_fences="<<fences<<" pass\n";
}
