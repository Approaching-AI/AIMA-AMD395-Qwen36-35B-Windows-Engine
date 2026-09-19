#define QRT_LONG_ATTENTION_PIPELINE_NO_MAIN
#include "long_attention_pipeline_capture.cpp"
#include "../../native/providers/ck_fmha/long_final_probability_pv.h"

int qrt_long_final_bound_selftest();

namespace {
template<class T>std::vector<T> lf_read(Guarded& memory){
    std::vector<T> values(memory.bytes/sizeof(T));
    check(hipMemcpy(values.data(),memory.data(),memory.bytes,hipMemcpyDeviceToHost));memory.guards();return values;
}
struct LfReference {
    std::vector<float> native,accumulator,denominator,error,exact,exact_accumulator,exact_denominator;
};
void lf_probability(AttentionOutputs& out,const uint16_t* value,unsigned start,unsigned count,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,bool candidate) {
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    auto call=candidate?qrt_long_final_probability_pv::launch:qrt_long_fused_probability_pv::launch;
    check(hipError_t(call(&owner,out.tensor.scores.as<float>()+guard,value,
        out.tensor.probability.as<uint16_t>()+guard,out.tensor.scales.as<float>()+guard,
        out.output.as<float>(),out.error.as<float>(),out.accumulator.as<float>(),out.denominator.as<float>(),
        start,count,0u,start+count,exp,rcp,true,nullptr)));
}
LfReference lf_reference(AttentionOutputs& out,const uint16_t* value,unsigned start,unsigned count,const unsigned char* rcp) {
    finish();LfReference result{lf_read<float>(out.output),lf_read<float>(out.accumulator),
        lf_read<float>(out.denominator),lf_read<float>(out.error),{},{},{}};
    // Independent original scalar K16 PV for every cell, never timed or used
    // by either producer/selector. This has no compacted or register replay.
    hipLaunchKernelGGL(blackwell_probability_value_kernel,dim3(16u,count),dim3(256u),0u,nullptr,
        value,out.tensor.probability.as<uint16_t>()+guard,out.tensor.scales.as<float>()+guard,
        out.output.as<float>(),start,0u,start+count,rcp,out.accumulator.as<float>(),out.denominator.as<float>(),nullptr);
    check(hipGetLastError());finish();result.exact=lf_read<float>(out.output);
    result.exact_accumulator=lf_read<float>(out.accumulator);result.exact_denominator=lf_read<float>(out.denominator);
    if(result.denominator!=result.exact_denominator)throw std::runtime_error("original PV denominator changed");
    return result;
}
void lf_metadata(AttentionOutputs& expected,AttentionOutputs& actual,Device& bad) {
    for(unsigned surface=0;surface<3u;++surface){
        auto& a=surface==0u?expected.tensor.scores:surface==1u?expected.tensor.probability:expected.tensor.scales;
        auto& b=surface==0u?actual.tensor.scores:surface==1u?actual.tensor.probability:actual.tensor.scales;
        const size_t bytes=surface==2u?(expected.tensor.scale_cells+2u*guard)*4u:(expected.tensor.cells+2u*guard)*(surface==1u?2u:4u);
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,a.as<unsigned char>(),b.as<unsigned char>(),bytes,bad.as<unsigned>());
        check(hipGetLastError());
    }
    finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("QK probability or scales changed");
}
void lf_native(AttentionOutputs& actual,const LfReference& reference){
    const auto out=lf_read<float>(actual.output),acc=lf_read<float>(actual.accumulator),den=lf_read<float>(actual.denominator);
    if(std::memcmp(out.data(),reference.native.data(),out.size()*4u)||
        std::memcmp(acc.data(),reference.accumulator.data(),acc.size()*4u)||
        std::memcmp(den.data(),reference.denominator.data(),den.size()*4u))throw std::runtime_error("native PV arithmetic changed");
}
unsigned lf_verify(AttentionOutputs& expected,AttentionOutputs& actual,const LfReference& reference,
    unsigned start,unsigned count,bool candidate,Device& bad) {
    lf_metadata(expected,actual,bad);check_tail(actual,start,count,bad);
    const unsigned cells=count*4096u;
    const auto out=lf_read<float>(actual.output),acc=lf_read<float>(actual.accumulator),den=lf_read<float>(actual.denominator),errors=lf_read<float>(actual.error);
    const auto ids=lf_read<unsigned>(actual.indices);const auto count_words=lf_read<unsigned>(actual.count);const unsigned selected=count_words[0];
    if(selected>cells)throw std::runtime_error("PV candidate capacity");std::vector<unsigned char> seen(cells,0u);
    for(unsigned i=0;i<selected;++i){if(ids[i]>=cells||seen[ids[i]])throw std::runtime_error("PV candidate permutation");seen[ids[i]]=1u;}
    for(unsigned i=selected;i<ids.size();++i)if(ids[i]!=0xa5a5a5a5u)throw std::runtime_error("PV unused candidate tail");
    for(unsigned cell=0;cell<cells;++cell){
        if(candidate?!(errors[cell]>=reference.error[cell]):bits(errors[cell])!=bits(reference.error[cell]))throw std::runtime_error("PV error dominance");
        const bool old_selected=!qrt_sm121_pv_bound::same_bf16(reference.native[cell],reference.error[cell]);
        const bool new_selected=!qrt_sm121_pv_bound::same_bf16(reference.native[cell],errors[cell]);
        if(bool(seen[cell])!=new_selected||(old_selected&&!new_selected))throw std::runtime_error("PV full candidate identity");
        if(!std::isfinite(out[cell])||qrt_sm121_pv_bound::bf16(out[cell])!=qrt_sm121_pv_bound::bf16(reference.exact[cell]))throw std::runtime_error("PV canonical BF16 differs");
        if(bits(out[cell])!=bits(seen[cell]?reference.exact[cell]:reference.native[cell])||
            bits(acc[cell])!=bits(seen[cell]?reference.exact_accumulator[cell]:reference.accumulator[cell]))throw std::runtime_error("PV raw replay differs");
        if(std::abs(double(reference.native[cell])-double(reference.exact[cell]))>double(errors[cell]))throw std::runtime_error("PV original interval undercoverage");
    }
    if(std::memcmp(den.data(),reference.exact_denominator.data(),den.size()*4u))throw std::runtime_error("PV final denominator changed");
    actual.guards();expected.guards();return selected;
}
void lf_short(AttentionOutputs& actual,const uint16_t* v,unsigned start,unsigned count,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp) {
    AttentionOutputs prior(start+count,count);
    check(hipMemcpy(prior.tensor.scores.pointer,actual.tensor.scores.pointer,(actual.tensor.cells+2u*guard)*4u,hipMemcpyDeviceToDevice));
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    check(hipError_t(qrt_fused_probability_pv::launch(&owner,prior.tensor.scores.as<float>()+guard,v,
        prior.tensor.probability.as<uint16_t>()+guard,prior.tensor.scales.as<float>()+guard,
        prior.output.as<float>(),prior.error.as<float>(),prior.accumulator.as<float>(),prior.denominator.as<float>(),
        start,count,0u,start+count,exp,rcp,true,nullptr)));finish();
    Device bad(4u);check(hipMemset(bad.pointer,0,4u));compare(prior,actual,bad);
}
void lf_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    struct Shape{unsigned start,count;};
    const Shape shapes[]={{0,1},{1,32},{1,128},{8191,2},{8192,128},{16352,33},{32768,128},{65520,17},{131041,32},{264719,17}};
    unsigned configurations=0,short_cases=0;uint64_t checked_cells=0,candidates[2]{};
    for(auto shape:shapes)for(unsigned mode=0;mode<5u;++mode){
        const unsigned start=shape.start,count=shape.count,n=start+count;
        std::vector<uint16_t> value(size_t(n)*512u);
        for(size_t i=0;i<value.size();++i){
            value[i]=uint16_t(((i*71u+i/17u)&0x807fu)|((120u+i%12u)<<7u));
            if(mode==1u)value[i]=uint16_t((i&1u)<<15u);
            if(mode==2u)value[i]=uint16_t(0x3f81u|((i/512u&1u)<<15u));
            if(mode==3u)value[i]=uint16_t(((i*173u)&0x807fu)|((1u+i%32u)<<7u));
            if(mode==4u)value[i]=uint16_t(((i*37u)&0x807fu)|((194u+i%7u)<<7u));
        }
        std::vector<float> scores(size_t(count)*16u*n);
        for(size_t i=0;i<scores.size();++i){const unsigned row=unsigned(i/n)/16u,key=unsigned(i%n);
            scores[i]=key>start+row?-INFINITY:mode==1u?0.0f:mode==4u?float(key/1024u)*0.03125f:
                float(int((key*37u+i/n*13u)%257u)-128)*0.0625f;}
        Guarded dv(value.size()*2u),vt(value.size()*2u);dv.put(value);
        check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),value.size(),n,nullptr)));finish();
        AttentionOutputs expected(n,count),actual(n,count);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        check(hipMemcpy(expected.tensor.scores.as<float>()+guard,scores.data(),scores.size()*4u,hipMemcpyHostToDevice));
        lf_probability(expected,dv.as<uint16_t>(),start,count,exp,packed,rcp,false);
        const auto reference=lf_reference(expected,dv.as<uint16_t>(),start,count,rcp);
        for(unsigned variant=0;variant<2u;++variant){
            actual.reset();check(hipMemcpy(actual.tensor.scores.as<float>()+guard,scores.data(),scores.size()*4u,hipMemcpyHostToDevice));
            lf_probability(actual,dv.as<uint16_t>(),start,count,exp,packed,rcp,bool(variant));finish();lf_native(actual,reference);
            if(variant&&n<=8192u){lf_short(actual,dv.as<uint16_t>(),start,count,exp,packed,rcp);++short_cases;}
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp,true);finish();
            candidates[variant]+=lf_verify(expected,actual,reference,start,count,bool(variant),bad);++configurations;
        }
        checked_cells+=uint64_t(count)*4096u;dv.immutable(value);transpose_immutable(vt,value,n);
        std::fprintf(stderr,"LONG_FINAL_PV_SAFETY start=%u count=%u mode=%u pass=1\n",start,count,mode);
    }
    std::printf("{\"kind\":\"long_final_pv_safety\",\"configurations\":%u,\"shapes\":10,\"modes\":5,\"short_template_regressions\":%u,\"distinct_cells\":%llu,\"old_candidates\":%llu,\"new_candidates\":%llu,\"maximum_keys\":264736,\"native_arithmetic_bitexact\":true,\"canonical_bf16_and_selected_raw_pass\":true,\"error_dominance\":true,\"complete_candidate_superset\":true,\"guards_tails_inputs_pass\":true,\"inference_acceptance\":false}\n",configurations,short_cases,(unsigned long long)checked_cells,(unsigned long long)candidates[0],(unsigned long long)candidates[1]);
}
void lf_owner(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    RangePrepared& prepared,AttentionOutputs& expected,unsigned start,unsigned count,unsigned n,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,Device& bad) {
    constexpr unsigned output_start=3u;
    const auto layout=qrt_long_attention_layout::layout(count,start+count);
    Guarded scratch(layout.elements*4u),output(size_t(count+output_start+2u)*4096u*4u);
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    unsigned observed=0u;
    SplitCompletionObserver observer{&observed,[](void* state,unsigned stage,hipStream_t stream)->int {
        auto& next=*static_cast<unsigned*>(state);if(stage!=next)return int(hipErrorInvalidValue);
        const auto status=hipStreamSynchronize(stream);if(status==hipSuccess)++next;return int(status);
    }};
    auto launch=[&](size_t extent){return qrt_long_attention_pipeline::launch(prepared.workspace,owner,
        q,kt,v,vt,output.as<float>(),start,count,output_start,n,exp,rcp,scratch.as<float>(),extent,nullptr,&observer,true);};
    if(launch(layout.elements-1u)!=int(hipErrorInvalidValue)||observed)throw std::runtime_error("undersized final-bound owner accepted");
    output.immutable(std::vector<uint32_t>(output.bytes/4u,0xa5a5a5a5u));
    scratch.immutable(std::vector<uint32_t>(scratch.bytes/4u,0xa5a5a5a5u));
    check(hipError_t(launch(layout.elements)));finish();if(observed!=5u)throw std::runtime_error("long final owner stage order");
    auto region=[&](const unsigned char* a,const unsigned char* b,size_t bytes){
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,a,b,bytes,bad.as<unsigned>());check(hipGetLastError());
    };
    const size_t cells=size_t(count)*16u*(start+count);
    region(expected.tensor.scores.as<unsigned char>()+guard*4u,scratch.data(),cells*4u);
    region(expected.tensor.probability.as<unsigned char>()+guard*2u,scratch.data()+layout.probability*4u,cells*2u);
    region(expected.tensor.scales.as<unsigned char>()+guard*4u,scratch.data()+layout.scales*4u,(layout.errors-layout.scales)*4u);
    region(expected.error.data(),scratch.data()+layout.errors*4u,size_t(count)*4096u*4u);
    region(expected.count.data(),scratch.data()+layout.count*4u,4u);
    region(expected.output.data(),output.data()+size_t(output_start)*4096u*4u,size_t(count)*4096u*4u);
    finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("contiguous deferred owner differs");
    const auto old_ids=lf_read<unsigned>(expected.indices);const unsigned selected=lf_read<unsigned>(expected.count)[0];
    std::vector<unsigned> ids(count*4096u);check(hipMemcpy(ids.data(),scratch.data()+layout.indices*4u,ids.size()*4u,hipMemcpyDeviceToHost));
    std::vector<unsigned char> old_seen(ids.size(),0u),seen(ids.size(),0u);
    for(unsigned i=0;i<selected;++i){old_seen[old_ids[i]]=1u;if(ids[i]>=ids.size()||seen[ids[i]])throw std::runtime_error("owner candidate permutation");seen[ids[i]]=1u;}
    if(seen!=old_seen)throw std::runtime_error("owner candidate membership");
    for(size_t i=selected;i<ids.size();++i)if(ids[i]!=0xa5a5a5a5u)throw std::runtime_error("owner unused candidate tail");
    const auto words=lf_read<uint32_t>(output);
    for(size_t i=0;i<words.size();++i)if((i<size_t(output_start)*4096u||i>=size_t(output_start+count)*4096u)&&words[i]!=0xa5a5a5a5u)
        throw std::runtime_error("owner output padding changed");
    scratch.guards();output.guards();
}
void lf_capture(unsigned queries,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    constexpr unsigned origin=16384u,source_queries=1024u,source_tokens=17408u,capacity=128u;
    const unsigned n=origin+queries;
    const auto oq=read_words(qfile,source_queries*4096u),ok=read_words(kfile,source_tokens*512u),ov=read_words(vfile,source_tokens*512u);
    const auto golden=read_words(reference_file,source_queries*4096u);
    std::vector<uint16_t> q(size_t(n)*4096u,0u),k(size_t(n)*512u),v(k.size());
    for(unsigned i=0;i<queries;++i)std::copy_n(oq.data()+size_t(i%source_queries)*4096u,4096u,q.data()+size_t(origin+i)*4096u);
    for(unsigned i=0;i<n;++i){std::copy_n(ok.data()+size_t(i%source_tokens)*512u,512u,k.data()+size_t(i)*512u);std::copy_n(ov.data()+size_t(i%source_tokens)*512u,512u,v.data()+size_t(i)*512u);}
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u),dr(golden.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);dr.put(golden);
    RangePrepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q,k,n,origin,queries);
    finish();auto begin=std::chrono::steady_clock::now();check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();const double transpose_ms=elapsed(begin);
    double samples[2][3]{};uint64_t candidates[2]{},score_slots=0,cpu_dots=0;
    for(unsigned offset=0;offset<queries;offset+=capacity){
        const unsigned start=origin+offset,count=std::min(capacity,queries-offset),stride=start+count;
        AttentionOutputs expected(n,count),actual(n,count);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,expected,start,count,n,exp,packed,rcp,false,true);
        const auto reference=lf_reference(expected,dv.as<uint16_t>(),start,count,rcp);
        hipLaunchKernelGGL(captured_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
            expected.output.as<float>(),dr.as<uint16_t>(),offset,count,source_queries,bad.as<unsigned>());check(hipGetLastError());check_tail(expected,start,count,bad);
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("canonical long context differs from GB10");
        unsigned counts[2]{};
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned order=0;order<2u;++order){
            const unsigned variant=(order+attempt+offset/capacity)%2u;
            actual.reset();finish();begin=std::chrono::steady_clock::now();
            check(hipError_t(qrt_long_narrow_qk::launch_workspace(&prepared.workspace,dq.as<uint16_t>(),dt.as<uint16_t>(),
                actual.tensor.scores.as<float>()+guard,nullptr,start,count,stride,n)));
            lf_probability(actual,dv.as<uint16_t>(),start,count,exp,packed,rcp,bool(variant));
            if(!attempt){finish();lf_native(actual,reference);begin=std::chrono::steady_clock::now();}
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp,true);finish();
            const double ms=elapsed(begin);if(attempt)samples[variant][attempt-1u]+=ms;
            const unsigned selected=lf_verify(expected,actual,reference,start,count,bool(variant),bad);
            if(!attempt&&variant)lf_owner(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,actual,start,count,n,exp,packed,rcp,bad);
            if(!attempt){counts[variant]=selected;candidates[variant]+=selected;}
            else if(selected!=counts[variant])throw std::runtime_error("PV candidate work changed across attempts");
        }
        for(unsigned j=0;j<4u;++j){const unsigned row=j*(count-1u)/3u,head=(offset/capacity+j*5u)%16u,key=(start+row)*j/3u;
            const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                q.data()+(size_t(start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
            uint32_t gpu=0;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+(size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(gpu!=bits(cpu))throw std::runtime_error("independent original QK CPU dot");++cpu_dots;}
        score_slots+=uint64_t(count)*16u*stride;
    }
    prepared.verify();dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(golden);transpose_immutable(dt,k,n);transpose_immutable(vt,v,n);
    std::fprintf(stderr,"LONG_FINAL_PV_OWNER_CAPTURE query_count=%u slabs=%u nonzero_output_offset=1 undersized_rejected=1 candidate_membership=1 complete_surfaces=1 stages=5\n",queries,(queries+127u)/128u);
    for(unsigned variant=0;variant<2u;++variant){auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"long_final_pv_capture\",\"query_start\":16384,\"query_count\":%u,\"key_tokens\":%u,\"source_queries\":1024,\"extended_queries\":%u,\"variant\":%u,\"query_batch\":128,\"pv_candidates\":%llu,\"score_slots\":%llu,\"cpu_dots\":%llu,\"gb10_context_cells\":4194304,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"common_preparation_ms\":%.9f,\"native_arithmetic_bitexact\":true,\"canonical_bf16_and_selected_raw_pass\":true,\"error_dominance\":true,\"complete_candidate_superset\":true,\"all_attempts_checked\":true,\"guards_tails_inputs_pass\":true,\"reference_is_compute_input\":false,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",queries,n,queries-source_queries,variant,(unsigned long long)candidates[variant],(unsigned long long)score_slots,(unsigned long long)cpu_dots,samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],prepared.common_ms+prepared.domain_ms+transpose_ms);
    }
}
} // namespace

#ifndef QRT_LONG_FINAL_PV_NO_MAIN
int main(int argc,char** argv)try {
    if(argc==2&&!std::strcmp(argv[1],"bound"))return qrt_long_final_bound_selftest();
    const bool safety=argc==4&&!std::strcmp(argv[1],"safety");
    const unsigned queries=argc==8&&!std::strcmp(argv[1],"1024")?1024u:argc==8&&!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!safety&&!queries)throw std::runtime_error("usage: safety EXP RCP | 1024|8192 Q K V GB10_CONTEXT EXP RCP");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    const auto exp=read_table(argv[argc-2],delta::source::table_bytes),rcp=read_table(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("table layout");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("native EXP full domain mismatch");
    const auto packed=lf_read<unsigned char>(dd);
    if(safety)lf_safety(de.data(),dd.data(),dc.data());else lf_capture(queries,argv[2],argv[3],argv[4],argv[5],de.data(),dd.data(),dc.data());
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"long_final_pv_error=%s\n",e.what());return 2;}
#endif
