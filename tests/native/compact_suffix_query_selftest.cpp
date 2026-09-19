// Reuse the established complete-surface observer without changing its tests.
#define QRT_LONG_FINAL_PV_NO_MAIN
#include "long_final_pv_capture.cpp"

namespace {
void compact_case(const std::vector<uint16_t>& q,const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& v,unsigned origin,unsigned queries,unsigned family,
    const std::vector<uint16_t>& golden,const unsigned char* exp,const unsigned char* packed,
    const unsigned char* rcp){
    const unsigned n=origin+queries;
    if(q.size()!=size_t(n)*4096u||k.size()!=size_t(n)*512u||v.size()!=k.size())
        throw std::runtime_error("compact fixture input extent");
    const std::vector<uint16_t> compact(q.begin()+size_t(origin)*4096u,q.end());
    Guarded dq(q.size()*2u),cq(compact.size()*2u),dk(k.size()*2u),dv(v.size()*2u),
        dt(k.size()*2u),ct(k.size()*2u),vt(v.size()*2u),dr(std::max(size_t(2u),golden.size()*2u));
    dq.put(q);cq.put(compact);dk.put(k);dv.put(v);if(!golden.empty())dr.put(golden);
    RangePrepared full(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q,k,n,origin,queries);
    RangePrepared small(cq.as<uint16_t>(),dk.as<uint16_t>(),ct.as<uint16_t>(),compact,k,n,origin,queries,origin);
    full.memory.immutable(small.expected);small.memory.immutable(full.expected);
    full.query_domain.immutable(small.expected_query);small.query_domain.immutable(full.expected_query);
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
    uint64_t checked_scores=0,cpu_dots=0,checked_context=0;
    unsigned slabs=0;
    for(unsigned offset=0;offset<queries;offset+=128u){
        const unsigned start=origin+offset,count=std::min(128u,queries-offset),stride=start+count;
        AttentionOutputs expected(n,count),actual(n,count);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        // Full original QK and scalar original PV supply comparison only.
        native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),full,
            expected,start,count,n,exp,packed,rcp,false,true);
        const auto reference=lf_reference(expected,dv.as<uint16_t>(),start,count,rcp);
        const unsigned golden_rows=unsigned(golden.size()/4096u);
        if(golden_rows){
            hipLaunchKernelGGL(captured_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
                expected.output.as<float>(),dr.as<uint16_t>(),offset,count,golden_rows,bad.as<unsigned>());
            check(hipGetLastError());finish();
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("original context differs from GB10");
            checked_context+=uint64_t(std::min(count,offset<golden_rows?golden_rows-offset:0u))*4096u;
        }
        for(unsigned variant=0;variant<2u;++variant){
            auto& prepared=variant?small:full;
            auto* query=variant?cq.as<uint16_t>():dq.as<uint16_t>();
            auto* transposed=variant?ct.as<uint16_t>():dt.as<uint16_t>();
            actual.reset();
            check(hipError_t(qrt_long_narrow_qk::launch_workspace(&prepared.workspace,query,transposed,
                actual.tensor.scores.as<float>()+guard,nullptr,start,count,stride,n)));
            lf_probability(actual,dv.as<uint16_t>(),start,count,exp,packed,rcp,true);
            finish();lf_native(actual,reference);
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp,true);finish();
            (void)lf_verify(expected,actual,reference,start,count,true,bad);
            lf_owner(query,transposed,dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
                actual,start,count,n,exp,packed,rcp,bad);
            checked_scores+=uint64_t(count)*16u*stride;
        }
        for(unsigned j=0;j<8u;++j){
            const unsigned row=j*(count-1u)/7u,head=(slabs+j*3u)%16u,key=(start+row)*j/7u;
            const float want=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                compact.data()+(size_t(offset+row)*16u+head)*256u,
                k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
            uint32_t observed=0;check(hipMemcpy(&observed,actual.tensor.scores.as<uint32_t>()+guard+
                (size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(observed!=bits(want))throw std::runtime_error("compact original CPU dot mismatch");++cpu_dots;
        }
        ++slabs;
    }
    full.verify();small.verify();dq.immutable(q);cq.immutable(compact);dk.immutable(k);dv.immutable(v);
    transpose_immutable(dt,k,n);transpose_immutable(ct,k,n);transpose_immutable(vt,v,n);
    if(!golden.empty())dr.immutable(golden);else dr.guards();
    std::printf("{\"kind\":\"compact_suffix_query\",\"query_start\":%u,\"query_count\":%u,\"family\":%u,\"slabs\":%u,\"variants\":2,\"checked_score_slots\":%llu,\"cpu_dots\":%llu,\"gb10_context_cells\":%llu,\"full_query_bytes\":%zu,\"compact_query_bytes\":%zu,\"complete_surfaces\":true,\"exact_fallback\":true,\"query_metadata_equal\":true,\"candidate_membership\":true,\"guards_inputs_tails\":true,\"nonzero_output_offset\":true,\"original_dot_order\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
        origin,queries,family,slabs,(unsigned long long)checked_scores,(unsigned long long)cpu_dots,
        (unsigned long long)checked_context,dq.bytes,cq.bytes);
}
void compact_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    struct Shape{unsigned start,count;};
    const Shape shapes[]={{17u,1u},{8191u,2u},{8192u,33u},{16384u,129u},
        {32768u,257u},{131071u,2u},{262144u,2u},{264719u,17u}};
    for(const auto shape:shapes)for(unsigned family=0;family<3u;++family){
        const unsigned n=shape.start+shape.count;
        std::vector<uint16_t> q(size_t(n)*4096u,0x7fc1u),k(size_t(n)*512u),v(k.size());
        for(unsigned r=0;r<shape.count;++r)for(unsigned f=0;f<4096u;++f){
            const size_t cell=size_t(shape.start+r)*4096u+f;
            q[cell]=uint16_t(((f*13u+r*17u)&0x807fu)|((119u+(f+r)%12u)<<7u));
            if(family==1u && r+1u==shape.count && f==4095u)q[cell]=uint16_t(1u|((r&1u)<<15u));
            if(family==2u)q[cell]=uint16_t((r+f)&1u?0x3f80u:0xbf80u);
        }
        for(size_t i=0;i<k.size();++i){
            k[i]=uint16_t(((i*31u)&0x807fu)|((117u+i%14u)<<7u));
            v[i]=uint16_t(((i*71u)&0x807fu)|((119u+i%12u)<<7u));
            if(family==1u && i==511u)k[i]=1u;
            if(family==2u){k[i]=uint16_t(i&1u?0x3f81u:0xbf81u);v[i]=uint16_t((i/512u)&1u?0x3f80u:0xbf80u);}
        }
        compact_case(q,k,v,shape.start,shape.count,family,{},exp,packed,rcp);
    }
}
void compact_capture(unsigned queries,char** argv,const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    constexpr unsigned origin=16384u,source_queries=1024u,source_tokens=17408u;
    const auto oq=read_words(argv[2],source_queries*4096u),ok=read_words(argv[3],source_tokens*512u),
        ov=read_words(argv[4],source_tokens*512u),golden=read_words(argv[5],source_queries*4096u);
    const unsigned n=origin+queries;
    std::vector<uint16_t> q(size_t(n)*4096u,0x7fc1u),k(size_t(n)*512u),v(k.size());
    for(unsigned r=0;r<queries;++r)std::copy_n(oq.data()+size_t(r%source_queries)*4096u,4096u,q.data()+size_t(origin+r)*4096u);
    for(unsigned r=0;r<n;++r){std::copy_n(ok.data()+size_t(r%source_tokens)*512u,512u,k.data()+size_t(r)*512u);
        std::copy_n(ov.data()+size_t(r%source_tokens)*512u,512u,v.data()+size_t(r)*512u);}
    compact_case(q,k,v,origin,queries,3u,golden,exp,packed,rcp);
}
}
int main(int argc,char** argv)try{
    const bool safety=argc==4&&!std::strcmp(argv[1],"safety");
    const unsigned queries=argc==8&&!std::strcmp(argv[1],"1024")?1024u:
        argc==8&&!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!safety&&!queries)throw std::runtime_error("usage: safety EXP RCP | 1024|8192 Q K V GB10_CONTEXT EXP RCP");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    const auto exp=read_table(argv[argc-2],delta::source::table_bytes),rcp=read_table(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("table layout");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("native EXP full domain mismatch");
    const auto packed=lf_read<unsigned char>(dd);
    if(safety)compact_safety(de.data(),dd.data(),dc.data());else compact_capture(queries,argv,de.data(),dd.data(),dc.data());
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"compact_suffix_query_error=%s\n",error.what());return 2;}
