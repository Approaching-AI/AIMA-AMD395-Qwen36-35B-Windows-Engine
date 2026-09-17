// Reuse the independent original-QK control, guarded owners, completion
// deadline and byte comparisons from the qualified complete-attention test.
#define QRT_COMBINED_ATTENTION_NO_MAIN
#include "combined_exact_attention_capture.cpp"
#include "../../native/providers/ck_fmha/streamed_exact_attention.h"

namespace {
void immutable_transpose(Guarded& device,const std::vector<uint16_t>& source,unsigned tokens){
    std::vector<uint16_t> expected(source.size());
    for(unsigned token=0u;token<tokens;++token)for(unsigned feature=0u;feature<512u;++feature)
        expected[size_t(feature)*tokens+token]=source[size_t(token)*512u+feature];
    device.immutable(expected);
}
void streamed(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    Prepared& prepared,AttentionOutputs& out,unsigned start,unsigned count,unsigned tokens,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,bool diagnostic){
    const unsigned stride=start+count;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* s=out.tensor.scales.as<float>()+guard;
    hipLaunchKernelGGL(qrt_streamed_exact_attention::produce,
        dim3(16u,(count+31u)/32u),dim3(256u),0u,nullptr,
        q,kt,v,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
        prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,
        p,s,out.output.as<float>(),out.error.as<float>(),out.accumulator.as<float>(),
        out.denominator.as<float>(),diagnostic?out.tensor.scores.as<float>()+guard:nullptr,
        start,count,stride,tokens,exp,packed,rcp,true);
    check(hipGetLastError());
    check(hipError_t(launch_compacted_pv_replay(v,p,s,out.output.as<float>(),start,count,0u,stride,
        rcp,out.accumulator.as<float>(),out.denominator.as<float>(),out.error.as<float>(),
        out.indices.as<unsigned>(),out.count.as<unsigned>(),nullptr,nullptr,vt,tokens,0u,nullptr,true)));
}

void compare_stream(AttentionOutputs& expected,AttentionOutputs& actual,Device& bad,bool diagnostic){
    if(diagnostic){compare(expected,actual,bad);return;}
    // A production-shaped call omits the dead global score matrix entirely.
    // Keep its original sentinel bytes and compare every other live surface.
    for(unsigned surface=1u;surface<3u;++surface){
        auto& a=surface==1u?expected.tensor.probability:expected.tensor.scales;
        auto& b=surface==1u?actual.tensor.probability:actual.tensor.scales;
        const size_t bytes=surface==1u?(expected.tensor.cells+2u*guard)*2u:
            (expected.tensor.scale_cells+2u*guard)*4u;
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
            a.as<unsigned char>(),b.as<unsigned char>(),bytes,bad.as<unsigned>());check(hipGetLastError());
    }
    for(auto pair:{std::make_pair(&expected.output,&actual.output),std::make_pair(&expected.accumulator,&actual.accumulator),
        std::make_pair(&expected.denominator,&actual.denominator),std::make_pair(&expected.error,&actual.error),
        std::make_pair(&expected.count,&actual.count)}){
        const size_t bytes=pair.first->bytes+256u;
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
            pair.first->storage.as<unsigned char>(),pair.second->storage.as<unsigned char>(),bytes,bad.as<unsigned>());
        check(hipGetLastError());
    }
    finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("streamed attention non-score surface differs");
    const auto scores=download<uint32_t>(actual.tensor.scores,actual.tensor.cells+2u*guard);
    for(auto x:scores)if(x!=0xa5a5a5a5u)throw std::runtime_error("score-free producer wrote score storage");
    actual.guards();
}

void run_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    struct Shape{unsigned tokens,start,count;};
    const Shape shapes[]={{1,0,1},{17,0,17},{33,1,32},{65,17,33},{129,1,128},
        {257,127,128},{513,385,128},{8192,8064,128}};
    unsigned cases=0u;
    for(const auto shape:shapes)for(unsigned mode=0u;mode<5u;++mode){
        const unsigned n=shape.tokens,start=shape.start,count=shape.count;
        std::vector<uint16_t> q(size_t(n)*4096u),k(size_t(n)*512u),v(k.size());
        for(size_t i=0;i<q.size();++i)q[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((123u+i%8u)<<7u));
        for(size_t i=0;i<k.size();++i){
            k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%10u)<<7u));
            v[i]=uint16_t(((i*71u+i/17u)&0x807fu)|((120u+i%12u)<<7u));
        }
        if(mode==1u){
            for(size_t i=0;i<q.size();i+=7u)q[i]=i%3u?0u:0x8000u;
            for(size_t i=0;i<k.size();i+=11u)k[i]=i%3u?0u:0x8000u;
            for(size_t i=0;i<v.size();i+=13u)v[i]=i%3u?0u:0x8000u;
        }
        if(mode==2u){
            const uint16_t edges[]={1u,0x8001u,0x007fu,0x807fu,uint16_t(63u<<7u|19u),uint16_t(192u<<7u|11u),0u,0x8000u};
            for(unsigned i=0;i<8u;++i){q[size_t(start)*4096u+i]=edges[i];k[i]=edges[7u-i];}
        }
        if(mode==3u){
            std::fill(q.begin(),q.end(),uint16_t(127u<<7u|127u));
            for(size_t i=0;i<k.size();++i)k[i]=uint16_t(127u<<7u|127u|((i/16u)&1u?0x8000u:0u));
        }
        if(mode==4u)for(size_t i=0;i<q.size();++i)q[i]=i%2u?0u:0x8000u;
        Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
        dq.put(q);dk.put(k);dv.put(v);
        Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),n);
        check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
        AttentionOutputs expected(n),actual(n);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,n,exp,nullptr,rcp,true,0u,nullptr);finish();
        for(bool diagnostic:{true,false}){
            actual.reset();streamed(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,actual,start,count,n,exp,packed,rcp,diagnostic);finish();
            compare_stream(expected,actual,bad,diagnostic);++cases;
        }
        dq.immutable(q);dk.immutable(k);dv.immutable(v);prepared.verify();
        immutable_transpose(dt,k,n);immutable_transpose(vt,v,n);expected.guards();
        std::fprintf(stderr,"STREAMED_SAFETY tokens=%u start=%u queries=%u mode=%u pass=1\n",n,start,count,mode);
    }
    std::printf("{\"kind\":\"streamed_exact_attention_safety\",\"cases\":%u,\"shapes\":8,\"data_modes\":5,\"diagnostic_and_score_free\":true,\"raw_bit_mismatches\":0,\"guards_pass\":true,\"immutable_inputs\":true}\n",cases);
}

void run_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    auto q=read_words(qfile,7169u*4096u),k=read_words(kfile,7169u*512u),v=read_words(vfile,7169u*512u);
    for(auto pair:{std::make_pair(&q,4096u),std::make_pair(&k,512u),std::make_pair(&v,512u)}){
        const auto old=*pair.first;
        pair.first->insert(pair.first->end(),old.begin(),old.begin()+size_t(tokens-7169u)*pair.second);
    }
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);
    Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),tokens);
    const auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),tokens,nullptr)));finish();
    const double transpose_ms=elapsed(begin);
    const auto reference=read_words(reference_file,7169u*4096u);Guarded dr(reference.size()*2u);dr.put(reference);
    AttentionOutputs expected(tokens),actual(tokens);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    double samples[2][3]{};uint64_t candidates[2]{},score_cells=0u;unsigned cpu_dots=0u;
    for(unsigned start=0;start<tokens;start+=query_batch){
        const unsigned count=std::min(query_batch,tokens-start),stride=start+count;
        expected.reset();attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,tokens,exp,nullptr,rcp,true,0u,nullptr);finish();
        hipLaunchKernelGGL(external_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
            expected.output.as<float>(),dr.as<uint16_t>(),start,count,bad.as<unsigned>());check(hipGetLastError());
        hipLaunchKernelGGL(tensor_tails,dim3((expected.tensor.cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
            expected.tensor.scores.as<uint32_t>(),expected.tensor.probability.as<uint16_t>(),expected.tensor.scales.as<uint32_t>(),
            expected.tensor.cells,expected.tensor.scale_cells,start,count,stride,bad.as<unsigned>());check(hipGetLastError());finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("control differs from GB10 or writes unused tails");
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<2u;++position){
            const unsigned variant=(position+start/query_batch+attempt)%2u;
            actual.reset();finish();const auto timed=std::chrono::steady_clock::now();
            if(!variant)attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
                actual,start,count,tokens,exp,packed,rcp,false,3u,nullptr);
            else streamed(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
                actual,start,count,tokens,exp,packed,rcp,!attempt);
            finish();const double ms=elapsed(timed);if(attempt)samples[variant][attempt-1u]+=ms;
            compare_stream(expected,actual,bad,!variant||!attempt);
            if(!attempt){unsigned n=0;check(hipMemcpy(&n,actual.count.data(),4u,hipMemcpyDeviceToHost));candidates[variant]+=n;}
        }
        for(unsigned s=0;s<4u;++s){
            const unsigned row=s*(count-1u)/3u,head=(start/query_batch+s*5u)%16u,key=(start+row)*s/3u;
            const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                q.data()+(size_t(start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
            uint32_t gpu=0;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+(size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(gpu!=bits(cpu))throw std::runtime_error("control differs from CPU QK");++cpu_dots;
        }
        score_cells+=size_t(count)*16u*stride;
    }
    dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);prepared.verify();
    immutable_transpose(dt,k,tokens);immutable_transpose(vt,v,tokens);
    if(candidates[0]!=candidates[1])throw std::runtime_error("original PV selection differs");
    for(unsigned variant=0;variant<2u;++variant){
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"streamed_exact_attention_capture\",\"tokens\":%u,\"source_capture_tokens\":7169,\"repeated_rows\":%u,\"variant\":%u,\"streamed_producer\":%s,\"query_batch\":128,\"score_slots\":%llu,\"output_cells\":%u,\"gb10_context_cells\":29364224,\"cpu_dots\":%u,\"pv_candidates\":%llu,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"common_preparation_ms\":%.9f,\"raw_bit_mismatches\":0,\"gb10_context_mismatches\":0,\"all_attempts_checked\":true,\"warmups_per_slab\":1,\"timed_attempts_per_slab\":3,\"original_qk_and_pv\":true,\"reference_is_compute_input\":false,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",tokens,tokens-7169u,variant,variant?"true":"false",(unsigned long long)score_cells,tokens*4096u,cpu_dots,(unsigned long long)candidates[variant],samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],prepared.ms+transpose_ms);
    }
}
} // namespace

int main(int argc,char** argv)try{
    if(argc!=4&&argc!=8)throw std::runtime_error("usage: safety EXP RCP | 7169|8192 Q K V GB10_CONTEXT EXP RCP");
    const bool safety_mode=argc==4&&!std::strcmp(argv[1],"safety");
    const unsigned tokens=argc==8&&!std::strcmp(argv[1],"7169")?7169u:argc==8&&!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!safety_mode&&!tokens)throw std::runtime_error("invalid action");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    const auto exp=read_table(argv[argc-2],delta::source::table_bytes),rcp=read_table(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("table layout");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);
    Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("complete EXP domain differs");
    std::vector<unsigned char> packed(delta::packed_bytes);check(hipMemcpy(packed.data(),dd.data(),packed.size(),hipMemcpyDeviceToHost));
    if(safety_mode)run_safety(de.data(),dd.data(),dc.data());
    else run_capture(tokens,argv[2],argv[3],argv[4],argv[5],de.data(),dd.data(),dc.data());
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"streamed_exact_attention_error=%s\n",e.what());return 2;}
