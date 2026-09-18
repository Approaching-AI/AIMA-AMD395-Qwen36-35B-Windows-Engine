// Reuse the independent original-QK control, guarded owners, completion
// deadline and byte comparisons from the qualified complete-attention test.
#define QRT_COMBINED_ATTENTION_NO_MAIN
#include "combined_exact_attention_capture.cpp"
#include "../../native/providers/ck_fmha/streamed_exact_attention.h"
#include "../../native/providers/ck_fmha/fused_probability_pv.h"
#include "../../native/providers/ck_fmha/narrow_domain_qk.h"

namespace {
void immutable_transpose(Guarded& device,const std::vector<uint16_t>& source,unsigned tokens){
    std::vector<uint16_t> expected(source.size());
    for(unsigned token=0u;token<tokens;++token)for(unsigned feature=0u;feature<512u;++feature)
        expected[size_t(feature)*tokens+token]=source[size_t(token)*512u+feature];
    device.immutable(expected);
}
struct NarrowDomain {
    Guarded query,key,statistics;std::vector<unsigned> expected_query,expected_key;double ms=0.0;
    NarrowDomain(const uint16_t* q,const uint16_t* k,const std::vector<uint16_t>& hq,
        const std::vector<uint16_t>& hk,unsigned tokens):query(size_t(tokens)*16u*4u),key(size_t(tokens)*2u*4u),
        statistics(8u),expected_query(size_t(tokens)*16u,1u),expected_key(size_t(tokens)*2u,1u){
        for(auto pair:{std::make_pair(&expected_query,&hq),std::make_pair(&expected_key,&hk)})
            for(size_t row=0;row<pair.first->size();++row)for(unsigned i=0u;i<256u;++i)
                (*pair.first)[row]&=unsigned(qrt_sm121_narrow_f32_carry::eligible((*pair.second)[row*256u+i]));
        finish();const auto begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(qrt_narrow_domain_qk::classify_rows,dim3(tokens*16u),dim3(256u),0u,nullptr,q,query.as<unsigned>(),tokens*16u);
        check(hipGetLastError());
        hipLaunchKernelGGL(qrt_narrow_domain_qk::classify_rows,dim3(tokens*2u),dim3(256u),0u,nullptr,k,key.as<unsigned>(),tokens*2u);
        check(hipGetLastError());finish();ms=elapsed(begin);verify();
    }
    void verify(){query.immutable(expected_query);key.immutable(expected_key);statistics.guards();}
};
template<unsigned Queries,unsigned Keys,unsigned Window>
void narrow_scores(Prepared& prepared,NarrowDomain& domain,float* scores,
    unsigned start,unsigned count,unsigned stride,unsigned tokens){
    const dim3 grid((stride+Keys*16u-1u)/(Keys*16u),16u,(count+Queries*16u-1u)/(Queries*16u));
    hipLaunchKernelGGL((qrt_narrow_domain_qk::scores<true,Queries,Keys,Window>),grid,dim3(256u),0u,nullptr,
        prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
        prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,
        domain.query.as<unsigned>(),domain.key.as<unsigned>(),scores,domain.statistics.as<unsigned>(),start,count,stride,tokens);
    check(hipGetLastError());
    hipLaunchKernelGGL((qrt_narrow_domain_qk::scores<false,Queries,Keys,Window>),grid,dim3(256u),0u,nullptr,
        prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
        prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,
        domain.query.as<unsigned>(),domain.key.as<unsigned>(),scores,domain.statistics.as<unsigned>(),start,count,stride,tokens);
    check(hipGetLastError());
}

void producer(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    Prepared& prepared,NarrowDomain& domain,AttentionOutputs& out,unsigned start,unsigned count,unsigned tokens,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,unsigned variant){
    const unsigned stride=start+count;
    auto* scores=out.tensor.scores.as<float>()+guard;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* s=out.tensor.scales.as<float>()+guard;
    check(hipMemsetAsync(domain.statistics.data(),0,8u,nullptr));
    if(!variant){
    hipLaunchKernelGGL((qrt_microtile_exact_qk::scores<2u,2u>),
        dim3((stride+31u)/32u,16u,(count+31u)/32u),dim3(256u),0u,nullptr,
        prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
        prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores,start,count,stride,tokens);
    check(hipGetLastError());
    }else if(variant==1u)narrow_scores<2u,2u,128u>(prepared,domain,scores,start,count,stride,tokens);
    else if(variant==2u)narrow_scores<4u,2u,64u>(prepared,domain,scores,start,count,stride,tokens);
    else if(variant==3u)narrow_scores<2u,4u,64u>(prepared,domain,scores,start,count,stride,tokens);
    else throw std::runtime_error("invalid QK variant");
    hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
        dim3((size_t(count)*16u*stride+255u)/256u),dim3(256u),0u,nullptr,
        q,kt,scores,start,count,stride,tokens);check(hipGetLastError());
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    check(hipError_t(qrt_fused_probability_pv::launch(&owner,scores,v,p,s,
        out.output.as<float>(),out.error.as<float>(),out.accumulator.as<float>(),out.denominator.as<float>(),
        start,count,0u,stride,exp,rcp,true,nullptr)));
}
void replay(const uint16_t* v,const uint16_t* vt,AttentionOutputs& out,
    unsigned start,unsigned count,unsigned tokens,const unsigned char* rcp){
    const unsigned stride=start+count;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* s=out.tensor.scales.as<float>()+guard;
    check(hipError_t(launch_compacted_pv_replay(v,p,s,out.output.as<float>(),start,count,0u,stride,
        rcp,out.accumulator.as<float>(),out.denominator.as<float>(),out.error.as<float>(),
        out.indices.as<unsigned>(),out.count.as<unsigned>(),nullptr,nullptr,vt,tokens,0u,nullptr,true)));
}

void run_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    struct Shape{unsigned tokens,start,count;};
    const Shape shapes[]={{1,0,1},{17,0,17},{33,1,32},{65,17,33},{129,1,128},
        {257,127,128},{513,385,128},{8192,8064,128}};
    unsigned cases=0u;
    for(const auto shape:shapes)for(unsigned mode=0u;mode<9u;++mode){
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
        if(mode>=5u && mode<=7u){
            const unsigned exponent=mode==5u?95u:159u;
            for(size_t i=0;i<q.size();++i)q[i]=uint16_t(exponent<<7u|127u);
            for(size_t i=0;i<k.size();++i)k[i]=uint16_t(exponent<<7u|127u|((mode==6u && (i&1u))?0x8000u:0u));
        }
        if(mode==8u){
            q[size_t(start)*4096u+255u]=uint16_t(94u<<7u|37u);
            k[511u]=uint16_t(160u<<7u|17u);
        }
        Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
        dq.put(q);dk.put(k);dv.put(v);
        Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),n);
        check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
        NarrowDomain domain(dq.as<uint16_t>(),dk.as<uint16_t>(),q,k,n);
        AttentionOutputs expected(n),native_control(n),actual(n);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,n,exp,nullptr,rcp,true,0u,nullptr);finish();
        producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,domain,native_control,start,count,n,exp,packed,rcp,0u);finish();
        for(unsigned variant=0u;variant<4u;++variant){
            actual.reset();producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,domain,actual,start,count,n,exp,packed,rcp,variant);finish();
            compare(native_control,actual,bad);
            replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp);finish();
            compare(expected,actual,bad);++cases;
        }
        native_control.guards();domain.verify();
        dq.immutable(q);dk.immutable(k);dv.immutable(v);prepared.verify();
        immutable_transpose(dt,k,n);immutable_transpose(vt,v,n);expected.guards();
        std::fprintf(stderr,"NARROW_DOMAIN_QK_SAFETY tokens=%u start=%u queries=%u mode=%u pass=1\n",n,start,count,mode);
    }
    std::printf("{\"kind\":\"narrow_domain_qk_safety\",\"cases\":%u,\"shapes\":8,\"data_modes\":9,\"pre_replay_native_surfaces_and_complete_replay\":true,\"domain_extremes_and_nearby_rejections\":true,\"raw_bit_mismatches\":0,\"guards_pass\":true,\"immutable_inputs\":true}\n",cases);
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
    NarrowDomain domain(dq.as<uint16_t>(),dk.as<uint16_t>(),q,k,tokens);
    AttentionOutputs expected(tokens),native_control(tokens),actual(tokens);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    double samples[4][3]{};uint64_t candidates[4]{},fast_tiles[4]{},slow_tiles[4]{},score_cells=0u;unsigned cpu_dots=0u;
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
        native_control.reset();producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,domain,native_control,start,count,tokens,exp,packed,rcp,0u);finish();
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<4u;++position){
            const unsigned variant=(position+start/query_batch+attempt)%4u;
            actual.reset();finish();const auto timed=std::chrono::steady_clock::now();
            producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,domain,
                actual,start,count,tokens,exp,packed,rcp,variant);
            // Only untimed warmups stop before replay. Timed calls retain the
            // original contiguous QK -> consumer -> exact-PV submission.
            if(!attempt){finish();compare(native_control,actual,bad);}
            replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,tokens,rcp);
            finish();const double ms=elapsed(timed);if(attempt)samples[variant][attempt-1u]+=ms;
            compare(expected,actual,bad);
            if(!attempt){unsigned n=0;check(hipMemcpy(&n,actual.count.data(),4u,hipMemcpyDeviceToHost));candidates[variant]+=n;
                unsigned counts[2]{};check(hipMemcpy(counts,domain.statistics.data(),8u,hipMemcpyDeviceToHost));
                slow_tiles[variant]+=counts[0];fast_tiles[variant]+=counts[1];}
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
    dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);prepared.verify();domain.verify();
    immutable_transpose(dt,k,tokens);immutable_transpose(vt,v,tokens);
    for(unsigned variant=1u;variant<4u;++variant){
        if(candidates[0]!=candidates[variant])throw std::runtime_error("original PV selection differs");
        if(!fast_tiles[variant] || !slow_tiles[variant])throw std::runtime_error("domain split not exercised");
    }
    for(unsigned variant=0;variant<4u;++variant){
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"narrow_domain_qk_capture\",\"tokens\":%u,\"source_capture_tokens\":7169,\"repeated_rows\":%u,\"variant\":%u,\"query_cells\":%u,\"key_cells\":%u,\"narrow_tiles\":%llu,\"original_tiles\":%llu,\"domain_classification_ms\":%.9f,\"pre_replay_native_surfaces_checked_on_warmup\":true,\"query_batch\":128,\"score_slots\":%llu,\"output_cells\":%u,\"gb10_context_cells\":29364224,\"cpu_dots\":%u,\"pv_candidates\":%llu,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"common_preparation_ms\":%.9f,\"raw_bit_mismatches\":0,\"gb10_context_mismatches\":0,\"all_attempts_checked\":true,\"warmups_per_slab\":1,\"timed_attempts_per_slab\":3,\"original_qk_and_pv\":true,\"reference_is_compute_input\":false,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",tokens,tokens-7169u,variant,variant==2u?4u:2u,variant==3u?4u:2u,(unsigned long long)fast_tiles[variant],(unsigned long long)slow_tiles[variant],variant?domain.ms:0.0,(unsigned long long)score_cells,tokens*4096u,cpu_dots,(unsigned long long)candidates[variant],samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],prepared.ms+transpose_ms);
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
}catch(const std::exception& e){std::fprintf(stderr,"narrow_domain_qk_error=%s\n",e.what());return 2;}
