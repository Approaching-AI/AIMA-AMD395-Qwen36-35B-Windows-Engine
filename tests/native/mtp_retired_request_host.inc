// Exercise the actual Request and immutable checkpoints at the real limit.
// Kernel doubles produce synthetic rows; none of these are model goldens.
static void test_retired_request(){
    using namespace qrt_sm121_mtp;
    reset();{
        constexpr unsigned prefix=262141u;
        ChunkedRequestFixture fixture(prefix);
        for(unsigned i=0u;i<prefix;++i)fixture.prompt[i]=100u+i%1000u;
        for(unsigned first=0u;first<prefix;first+=8192u){
            const unsigned rows=std::min(8192u,prefix-first);
            auto batch=chunk_batch(fixture.prompt,first,rows);
            const auto actual=fixture.frontier(first+rows);
            const auto result=first?fixture.request.append_prefill_chunk(*batch,actual,expected_stream):
                fixture.request.seed_prefill_chunks(*batch,actual,fixture.model.binding(10u),fixture.tables,
                    262144u,expected_stream);
            assert(result.status==hipSuccess&&!result.completion_unknown);
        }
        RequestCheckpoint original;
        assert(fixture.request.save(&original,fixture.frontier(prefix),expected_stream).status==hipSuccess);
        assert(!original.retired(fixture.frontier(prefix)));
        unsigned crossings=0u,continuations=0u,invalid=0u;
        for(unsigned mode=0u;mode<5u;++mode){
            Request live;auto history=fixture.prompt;uint32_t current=999u;
            auto actual=[&]{return TargetFrontier{&fixture,17u,10u,history.data(),history.size(),current};};
            assert(live.restore(original,actual(),262144u,expected_stream).status==hipSuccess);
            for(unsigned transaction=0u;transaction<2u;++transaction){
                const bool accept=transaction?(mode%2u==0u):(mode<2u);
                const unsigned remaining=mode==4u&&transaction?1u:4u;
                TargetBatch batch;assert(live.begin(actual(),remaining,&batch));
                assert(batch.rows==2u&&batch.first_position==history.size());
                const uint32_t samples[]={accept?batch.inputs[1]:201u,77u};
                const unsigned rows=accept&&remaining>1u?2u:1u;
                std::vector<float> hidden(4096u,2.0f);std::fill(hidden.begin()+2048u,hidden.end(),3.0f);
                expected_hidden.assign(2048u,0x4000u);
                if(rows==2u)expected_hidden.insert(expected_hidden.end(),2048u,0x4040u);
                expected_ids.assign(samples,samples+rows);expected_split=false;
                const auto submissions=stages.size();
                const auto allocations_before=allocation_call,synchronizations=sync_call;
                AcceptedTarget accepted;
                assert(live.prepare(batch.inputs.data(),samples,2u,{0u,1u},hidden,10u,&accepted,expected_stream).status==hipSuccess);
                assert(accepted.rows==rows&&accepted.drafter_retired==bool(transaction));
                if(transaction)assert(stages.size()==submissions&&allocation_call==allocations_before&&sync_call==synchronizations);
                history.insert(history.end(),batch.inputs.begin(),batch.inputs.begin()+rows);current=samples[rows-1u];
                assert(live.commit(actual())&&live.retired()==bool(transaction));
            }
            const auto before=actual();
            const auto submissions=stages.size();
            const auto allocations_before=allocation_call,synchronizations=sync_call;
            RequestCheckpoint retired;
            assert(live.save(&retired,before,expected_stream).status==hipSuccess);
            assert(retired.matches(before)&&retired.retired(before)&&!retired.allocated_bytes());
            assert(stages.size()==submissions&&allocation_call==allocations_before&&sync_call==synchronizations);
            assert(live.abort(10u)&&live.retired());TargetBatch unused;
            assert(!live.begin(before,4u,&unused));
            Request cannot_restore;
            assert(cannot_restore.restore(retired,before,262144u,expected_stream).status==hipErrorInvalidValue);
            for(unsigned count:{1u,63u,512u}){
                auto inputs=history;inputs.push_back(current);
                for(unsigned i=1u;i<count;++i)inputs.push_back(700u+i);
                TargetFrontier after{&fixture,17u,10u,inputs.data(),inputs.size(),177u};
                RequestCheckpoint next;
                assert(retired.advance_retired(before,after,&next).status==hipSuccess);
                assert(next.retired(after)&&!next.allocated_bytes()&&retired.retired(before)&&!retired.matches(after));
                ++continuations;
                if(count!=63u)continue;
                for(unsigned bad=0u;bad<14u;++bad){
                    auto source=before,target=after;auto altered=inputs;target.processed_inputs=altered.data();
                    switch(bad){
                    case 0:source.owner=nullptr;break;case 1:source.generation++;break;
                    case 2:source.model_epoch++;break;case 3:source.current_token++;break;
                    case 4:source.processed_count--;break;case 5:target.owner=nullptr;break;
                    case 6:target.generation++;break;case 7:target.model_epoch++;break;
                    case 8:target.processed_count=before.processed_count;break;
                    case 9:target.processed_count=before.processed_count+513u;break;
                    case 10:altered[0]++;break;case 11:altered[before.processed_count]++;break;
                    case 12:altered.back()=qrt_mtp_target_rows::vocabulary;break;
                    case 13:target.current_token=qrt_mtp_target_rows::vocabulary;break;
                    }
                    assert(retired.advance_retired(source,target,&next).status==hipErrorInvalidValue);
                    assert(next.retired(after)&&retired.retired(before));++invalid;
                }
            }
            fixture.source->generation=11u;
            assert(!retired.matches(before)&&!retired.retired(before));
            fixture.source->generation=10u;
            assert(retired.retired(before));++crossings;
        }
        assert(crossings==5u&&continuations==15u&&invalid==70u);
        std::printf("native MTP retired request crossings=%u continuations=%u invalid=%u zero_retired_gpu_work=1 pass\n",crossings,continuations,invalid);
    }
    assert(allocations.empty()&&queued.empty());reset();
}
