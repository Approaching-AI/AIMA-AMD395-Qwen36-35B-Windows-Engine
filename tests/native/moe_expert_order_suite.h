#pragma once
namespace moe_batch_test {
void check_expert_order(unsigned columns,unsigned capacity,unsigned selected,unsigned mode) {
    namespace order=qrt_moe_expert_order;
    constexpr unsigned sentinel=0x5a5a5a5au,routes=65536u;
    const unsigned first=capacity*3u;
    std::vector<uint32_t> input(capacity+2u*kGuard,sentinel),count(1u+2u*kGuard,sentinel);
    std::vector<int32_t> ids(routes+2u*kGuard,int32_t(sentinel));
    std::vector<uint32_t> storage(capacity+order::metadata_words+2u*kGuard,sentinel);
    std::array<unsigned,order::experts> expected_counts{};
    std::vector<unsigned char> present(capacity,0u);
    for(unsigned route=0u;route<routes;++route)
        ids[kGuard+route]=mode==0u?17:mode==1u?int32_t((route*73u+(route/8u)*17u)&255u):
            route%19u?0:int32_t((route&1u)?255u:201u);
    for(unsigned slot=0u;slot<selected;++slot){
        const unsigned offset=(slot*2654435761u)&(capacity-1u),cell=first+offset;
        require(cell/columns<routes,"expert order generated route span");
        input[kGuard+slot]=cell;present[offset]=1u;++expected_counts[unsigned(ids[kGuard+cell/columns])];
    }
    count[kGuard]=selected;
    Device<uint32_t> di(input),dc(count),dw(storage);Device<int32_t> dt(ids);
    hipStream_t stream=nullptr;hip_ok(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"expert order stream");
    hip_ok(order::launch(di.data(),dc.data(),dt.data(),columns,{dw.data(),capacity},stream),"expert candidate permutation");
    hipEvent_t event=nullptr;hip_ok(hipEventCreate(&event),"expert order event");
    hip_ok(hipEventRecord(event,stream),"expert order completion");
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
    for(;;){const auto status=hipEventQuery(event);if(status==hipSuccess)break;
        require(status==hipErrorNotReady&&std::chrono::steady_clock::now()<deadline,"expert order completion deadline");std::this_thread::yield();}
    const auto actual=dw.read(storage.size());unsigned previous=0u;
    for(unsigned slot=0u;slot<selected;++slot){
        const unsigned cell=actual[kGuard+slot];require(cell>=first&&cell<first+capacity,"expert order cell domain");
        const unsigned offset=cell-first;require(present[offset]==1u,"expert order duplicate or foreign candidate");present[offset]=2u;
        const unsigned expert=unsigned(ids[kGuard+cell/columns]);require(!slot||expert>=previous,"expert order monotonic buckets");previous=expert;
    }
    for(unsigned char value:present)require(value!=1u,"expert order missing candidate");
    const size_t counts=kGuard+capacity,offsets=counts+order::experts,cursors=offsets+order::experts+1u;
    unsigned total=0u;
    for(unsigned expert=0u;expert<order::experts;++expert){
        require(actual[counts+expert]==expected_counts[expert]&&actual[offsets+expert]==total,"expert order histogram/prefix");
        total+=expected_counts[expert];require(actual[cursors+expert]==total,"expert order cursor endpoint");
    }
    require(total==selected&&actual[offsets+order::experts]==selected,"expert order total");
    for(size_t i=0u;i<actual.size();++i)
        if(i<kGuard||(i>=kGuard+selected&&i<counts)||i>=cursors+order::experts)
            require(actual[i]==sentinel,"expert order workspace guard or unused indices");
    require(di.read(input.size())==input&&dc.read(count.size())==count&&dt.read(ids.size())==ids,"expert order immutable input");
    hip_ok(hipEventDestroy(event),"expert order event release");hip_ok(hipStreamDestroy(stream),"expert order stream release");
    std::printf("{\"kind\":\"moe_expert_order_permutation\",\"columns\":%u,\"capacity\":%u,\"selected\":%u,\"mode\":%u,\"experts\":256,\"workspace_bytes\":%zu,\"exact_permutation\":true,\"monotonic_experts\":true,\"all_metadata_checked\":true,\"redzones_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
        columns,capacity,selected,mode,order::bytes(capacity));std::fflush(stdout);
}
void expert_order_suite(){
    for(unsigned columns:{512u,2048u})for(unsigned capacity:{256u,4096u,4194304u})
        for(unsigned selected:{0u,1u,capacity/2u+17u,capacity})for(unsigned mode:{0u,1u,2u})
            check_expert_order(columns,capacity,selected,mode);
}
} // namespace moe_batch_test
