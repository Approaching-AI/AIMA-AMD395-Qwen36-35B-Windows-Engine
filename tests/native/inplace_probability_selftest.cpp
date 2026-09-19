#define QRT_LONG_ATTENTION_PIPELINE_NO_MAIN
#include "long_attention_pipeline_capture.cpp"
#include "../../native/providers/ck_fmha/inplace_probability_pipeline.h"

namespace {
namespace ip = qrt_inplace_probability_storage;
constexpr unsigned ip_output_start = 3u;
struct IpOutputs {
    bool inplace;
    unsigned queries, stride;
    qrt_long_attention_layout::Layout layout;
    Guarded scratch, output, accumulator, denominator;
    IpOutputs(unsigned count, unsigned keys, bool alias):inplace(alias),queries(count),stride(keys),
        layout(alias?ip::layout(count,keys):qrt_long_attention_layout::layout(count,keys)),
        scratch(layout.elements*4u),output(size_t(count+ip_output_start+2u)*4096u*4u),
        accumulator(output.bytes),denominator(size_t(count+ip_output_start+2u)*16u*4u) {}
    float* scores(){return scratch.as<float>();}
    uint16_t* probability(){return reinterpret_cast<uint16_t*>(scratch.as<float>()+layout.probability);}
    float* scales(){return scratch.as<float>()+layout.scales;}
    float* errors(){return scratch.as<float>()+layout.errors;}
    unsigned* indices(){return reinterpret_cast<unsigned*>(scratch.as<float>()+layout.indices);}
    unsigned* selected(){return reinterpret_cast<unsigned*>(scratch.as<float>()+layout.count);}
    void reset(){scratch.reset();output.reset();accumulator.reset();denominator.reset();}
    void guards(){scratch.guards();output.guards();accumulator.guards();denominator.guards();}
};
__global__ void ip_verify_probability(const uint16_t* packed,const float* original_scores,
    const float* alias,unsigned start,unsigned queries,unsigned stride,unsigned* bad) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(queries)*16u*stride)return;
    const unsigned row=unsigned(i/stride),key=unsigned(i%stride),tokens=start+row/16u+1u;
    const unsigned end=min(stride,((tokens+31u)/32u)*32u);
    const uint32_t actual=__float_as_uint(alias[i]);
    if(key<end) {
        if(actual!=(ip::tag|uint32_t(packed[i])))atomicAdd(bad,1u);
    } else if(actual!=__float_as_uint(original_scores[i]))atomicAdd(bad,1u);
}
void ip_bytes(const void* expected,const void* actual,size_t bytes,Device& bad) {
    hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
        static_cast<const unsigned char*>(expected),static_cast<const unsigned char*>(actual),bytes,bad.as<unsigned>());
    check(hipGetLastError());
}
void ip_checked(Device& bad,const char* message) {
    finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error(message);
}
void ip_metadata(AttentionOutputs& expected,IpOutputs& actual,unsigned start,Device& bad) {
    const size_t cells=size_t(actual.queries)*16u*actual.stride;
    if(actual.inplace) {
        hipLaunchKernelGGL(ip_verify_probability,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
            expected.tensor.probability.as<uint16_t>()+guard,expected.tensor.scores.as<float>()+guard,
            actual.scores(),start,actual.queries,actual.stride,bad.as<unsigned>());check(hipGetLastError());
    } else {
        ip_bytes(expected.tensor.scores.as<float>()+guard,actual.scores(),cells*4u,bad);
        ip_bytes(expected.tensor.probability.as<uint16_t>()+guard,actual.probability(),cells*2u,bad);
    }
    const size_t scale_bytes=(actual.layout.errors-actual.layout.scales)*4u;
    ip_bytes(expected.tensor.scales.as<float>()+guard,actual.scales(),scale_bytes,bad);
    ip_bytes(expected.error.data(),actual.errors(),size_t(actual.queries)*4096u*4u,bad);
    ip_checked(bad,"in-place probability payload, untouched score tail, scale or bound differs");
}
void ip_padding(IpOutputs& actual,bool raw) {
    for(auto* memory:{&actual.output,&actual.accumulator,&actual.denominator}) {
        if(!raw&&memory!=&actual.output)continue;
        const size_t features=memory==&actual.denominator?16u:4096u;
        std::vector<uint32_t> before(ip_output_start*features),after(2u*features);
        check(hipMemcpy(before.data(),memory->data(),before.size()*4u,hipMemcpyDeviceToHost));
        check(hipMemcpy(after.data(),memory->data()+(ip_output_start+actual.queries)*features*4u,
            after.size()*4u,hipMemcpyDeviceToHost));
        for(const auto& values:{before,after})for(auto word:values)
            if(word!=0xa5a5a5a5u)throw std::runtime_error("in-place output padding changed");
    }
    actual.guards();
}
void ip_values(AttentionOutputs& expected,IpOutputs& actual,Device& bad,bool raw) {
    const size_t output_bytes=size_t(actual.queries)*4096u*4u,offset=size_t(ip_output_start)*4096u*4u;
    ip_bytes(expected.output.data(),actual.output.data()+offset,output_bytes,bad);
    if(raw) {
        ip_bytes(expected.accumulator.data(),actual.accumulator.data()+offset,output_bytes,bad);
        ip_bytes(expected.denominator.data(),actual.denominator.data()+size_t(ip_output_start)*16u*4u,
            size_t(actual.queries)*16u*4u,bad);
    }
    ip_checked(bad,"in-place native or original replay output/accumulator/denominator differs");
    ip_padding(actual,raw);
}
unsigned ip_candidates(AttentionOutputs& expected,IpOutputs& actual) {
    unsigned a=0,b=0;check(hipMemcpy(&a,expected.count.data(),4u,hipMemcpyDeviceToHost));
    check(hipMemcpy(&b,actual.selected(),4u,hipMemcpyDeviceToHost));
    const unsigned cells=actual.queries*4096u;
    if(a!=b||a>cells)throw std::runtime_error("in-place candidate count differs");
    std::vector<unsigned> left(cells),right(cells);
    check(hipMemcpy(left.data(),expected.indices.data(),cells*4u,hipMemcpyDeviceToHost));
    check(hipMemcpy(right.data(),actual.indices(),cells*4u,hipMemcpyDeviceToHost));
    for(unsigned i=a;i<cells;++i)if(left[i]!=0xa5a5a5a5u||right[i]!=0xa5a5a5a5u)
        throw std::runtime_error("in-place candidate tail changed");
    left.resize(a);right.resize(a);std::sort(left.begin(),left.end());std::sort(right.begin(),right.end());
    if(left!=right||std::adjacent_find(left.begin(),left.end())!=left.end()||(!left.empty()&&left.back()>=cells))
        throw std::runtime_error("in-place candidate identity or uniqueness differs");
    return a;
}
unsigned ip_component(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    RangePrepared& prepared,AttentionOutputs& expected,IpOutputs& actual,
    const std::vector<uint16_t>& hq,const std::vector<uint16_t>& hk,unsigned start,unsigned n,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,bool final_bound,Device& bad) {
    const unsigned count=actual.queries,stride=actual.stride;
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    expected.reset();actual.reset();
    check(hipError_t(qrt_long_narrow_qk::launch_workspace(&prepared.workspace,q,kt,
        expected.tensor.scores.as<float>()+guard,nullptr,start,count,stride,n)));finish();
    for(unsigned j=0;j<128u;++j) {
        const unsigned row=(j*43u)%count,head=j%16u,tokens=start+row+1u;
        const unsigned key=j==0u?0u:j==127u?tokens-1u:(j*8191u+273u)%tokens;
        const size_t qrow=size_t(start+row-prepared.workspace.query_origin)*4096u+head*256u;
        const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
            hq.data()+qrow,hk.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
        uint32_t gpu=0;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+
            (size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
        if(gpu!=bits(cpu))throw std::runtime_error("in-place original CPU QK dot differs");
    }
    const size_t score_bytes=size_t(count)*16u*stride*4u;
    check(hipMemcpy(actual.scores(),expected.tensor.scores.as<float>()+guard,score_bytes,hipMemcpyDeviceToDevice));
    auto probability=final_bound?qrt_long_final_probability_pv::launch:qrt_long_fused_probability_pv::launch;
    check(hipError_t(probability(&owner,expected.tensor.scores.as<float>()+guard,v,
        expected.tensor.probability.as<uint16_t>()+guard,expected.tensor.scales.as<float>()+guard,
        expected.output.as<float>(),expected.error.as<float>(),expected.accumulator.as<float>(),expected.denominator.as<float>(),
        start,count,0u,stride,exp,rcp,true,nullptr)));
    check(hipError_t(qrt_inplace_probability_pipeline::probability(owner,actual.scores(),v,
        actual.scales(),actual.output.as<float>(),actual.errors(),actual.accumulator.as<float>(),actual.denominator.as<float>(),
        start,count,ip_output_start,stride,exp,rcp,nullptr,final_bound)));
    ip_metadata(expected,actual,start,bad);ip_values(expected,actual,bad,true);
    exact_replay(v,vt,expected,start,count,n,rcp,true);
    check(hipError_t(qrt_long_fused_probability_pv::replay<true>(v,vt,actual.probability(),actual.scales(),
        actual.output.as<float>(),actual.errors(),actual.accumulator.as<float>(),actual.denominator.as<float>(),
        actual.indices(),actual.selected(),start,count,ip_output_start,stride,n,rcp,true,nullptr)));
    ip_metadata(expected,actual,start,bad);ip_values(expected,actual,bad,true);
    expected.guards();check_tail(expected,start,count,bad);
    return ip_candidates(expected,actual);
}
double ip_owner(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    RangePrepared& prepared,AttentionOutputs& expected,IpOutputs& actual,unsigned start,unsigned n,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,bool final_bound,
    bool observe,Device& bad) {
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    actual.reset();
    struct State{unsigned next;AttentionOutputs* expected;IpOutputs* actual;Device* bad;};
    State state{0u,&expected,&actual,&bad};
    SplitCompletionObserver observer{&state,[](void* opaque,unsigned stage,hipStream_t stream)->int {
        auto& s=*static_cast<State*>(opaque);
        if(stage!=s.next)return int(hipErrorInvalidValue);
        const auto status=hipStreamSynchronize(stream);if(status!=hipSuccess)return int(status);
        if(!stage) {
            ip_bytes(s.expected->tensor.scores.as<float>()+guard,s.actual->scores(),
                size_t(s.actual->queries)*16u*s.actual->stride*4u,*s.bad);
            ip_checked(*s.bad,"in-place owner QK differs before storage reuse");
        }
        ++s.next;return int(hipSuccess);
    }};
    auto call=[&](size_t extent,SplitCompletionObserver* observation) {
        auto launch=actual.inplace?qrt_inplace_probability_pipeline::launch:qrt_long_attention_pipeline::launch;
        return launch(prepared.workspace,owner,q,kt,v,vt,actual.output.as<float>(),start,actual.queries,
            ip_output_start,n,exp,rcp,actual.scores(),extent,nullptr,observation,final_bound);
    };
    if(observe) {
        if(call(actual.layout.elements-1u,&observer)!=int(hipErrorInvalidValue)||state.next)
            throw std::runtime_error("in-place undersized workspace accepted");
        actual.scratch.immutable(std::vector<uint32_t>(actual.scratch.bytes/4u,0xa5a5a5a5u));
    }
    finish();const auto begin=std::chrono::steady_clock::now();
    check(hipError_t(call(actual.layout.elements,observe?&observer:nullptr)));finish();const double ms=elapsed(begin);
    if(!std::isfinite(ms)||ms<=0.0)throw std::runtime_error("in-place incomplete host time");
    if(observe&&state.next!=5u)throw std::runtime_error("in-place owner stage order");
    ip_metadata(expected,actual,start,bad);ip_values(expected,actual,bad,false);ip_candidates(expected,actual);
    return ms;
}
void ip_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp) {
    struct Shape{unsigned start,count;};
    const Shape shapes[]={{0,1},{1,32},{1,128},{8191,2},{8192,128},
        {16352,33},{32768,128},{65520,17},{131041,32},{264719,17}};
    unsigned cases=0;uint64_t cells=0,score_cells=0,candidates=0;
    for(auto shape:shapes)for(unsigned mode=0;mode<3u;++mode) {
        const unsigned start=shape.start,count=shape.count,n=start+count;
        std::vector<uint16_t> q(size_t(count)*4096u),k(size_t(n)*512u),v(k.size());
        for(size_t i=0;i<q.size();++i)
            q[i]=mode==1u?uint16_t((i&1u)<<15u):uint16_t(((i*37u+i/19u)&0x807fu)|((123u+i%8u)<<7u));
        for(size_t i=0;i<k.size();++i) {
            k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%10u)<<7u));
            v[i]=uint16_t(((i*71u+i/17u)&0x807fu)|((120u+i%12u)<<7u));
            if(mode==1u)v[i]=uint16_t(0x3f81u|((i/512u&1u)<<15u));
            if(mode==2u)v[i]=uint16_t(((i*37u)&0x807fu)|((194u+i%7u)<<7u));
        }
        if(mode==2u) {
            q[0]=1u;q[1]=0x8001u;q[255]=uint16_t(94u<<7u|37u);
            k[0]=0x807fu;k[511]=uint16_t(160u<<7u|17u);
        }
        Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),kt(k.size()*2u),vt(v.size()*2u);
        dq.put(q);dk.put(k);dv.put(v);
        RangePrepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),kt.as<uint16_t>(),q,k,n,start,count,start);
        check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
        for(bool final_bound:{false,true}) {
            AttentionOutputs expected(n,count);IpOutputs actual(count,n,true),control(count,n,false);
            Device bad(4u);check(hipMemset(bad.pointer,0,4u));
            candidates+=ip_component(dq.as<uint16_t>(),kt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,expected,actual,q,k,start,n,exp,packed,rcp,final_bound,bad);
            for(auto* arm:{&control,&actual})
                ip_owner(dq.as<uint16_t>(),kt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                    prepared,expected,*arm,start,n,exp,packed,rcp,final_bound,true,bad);
            cells+=uint64_t(count)*4096u;score_cells+=uint64_t(count)*16u*n;++cases;
            std::fprintf(stderr,"INPLACE_P_SAFETY start=%u queries=%u mode=%u final_bound=%u original_cpu_dots=128 pass=1\n",
                start,count,mode,unsigned(final_bound));
        }
        prepared.verify();dq.immutable(q);dk.immutable(k);dv.immutable(v);
        transpose_immutable(kt,k,n);transpose_immutable(vt,v,n);
    }
    if(cases!=60u)throw std::runtime_error("incomplete in-place probability safety");
    std::printf("{\"kind\":\"inplace_probability_safety\",\"cases\":%u,\"original_cpu_dots\":%u,"
        "\"output_cells\":%llu,\"score_cells\":%llu,\"candidates\":%llu,\"maximum_keys\":264736,"
        "\"score_stage_before_overwrite\":true,\"payloads_and_unused_scores_bitexact\":true,"
        "\"native_and_replayed_raw_surfaces_bitexact\":true,\"candidate_identity\":true,"
        "\"both_bound_policies\":true,\"stages\":5,\"undersized_rejected\":true,"
        "\"nonzero_output_offset\":true,\"guards_and_inputs_pass\":true,\"model_acceptance\":false}\n",
        cases,cases*128u,(unsigned long long)cells,(unsigned long long)score_cells,(unsigned long long)candidates);
}
void ip_capture(unsigned queries,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp) {
    constexpr unsigned origin=16384u,source_queries=1024u,source_tokens=17408u;
    const unsigned n=origin+queries;
    const auto original_q=read_words(qfile,source_queries*4096u);
    const auto original_k=read_words(kfile,source_tokens*512u),original_v=read_words(vfile,source_tokens*512u);
    const auto reference=read_words(reference_file,source_queries*4096u);
    std::vector<uint16_t> q(size_t(queries)*4096u),k(size_t(n)*512u),v(k.size());
    for(unsigned i=0;i<queries;++i)
        std::copy_n(original_q.data()+size_t(i%source_queries)*4096u,4096u,q.data()+size_t(i)*4096u);
    for(unsigned i=0;i<n;++i) {
        std::copy_n(original_k.data()+size_t(i%source_tokens)*512u,512u,k.data()+size_t(i)*512u);
        std::copy_n(original_v.data()+size_t(i%source_tokens)*512u,512u,v.data()+size_t(i)*512u);
    }
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),kt(k.size()*2u),vt(v.size()*2u),dr(reference.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);dr.put(reference);
    RangePrepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),kt.as<uint16_t>(),q,k,n,origin,queries,origin);
    finish();const auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
    const double transpose_ms=elapsed(begin);
    double samples[2][3]{};uint64_t candidates=0,score_cells=0;unsigned slabs=0;
    size_t maximum_scratch[2]{};
    for(unsigned offset=0;offset<queries;offset+=128u) {
        const unsigned start=origin+offset,count=std::min(128u,queries-offset),stride=start+count;
        AttentionOutputs expected(stride,count);IpOutputs actual(count,stride,true),control(count,stride,false);
        Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        candidates+=ip_component(dq.as<uint16_t>(),kt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,expected,actual,q,k,start,n,exp,packed,rcp,true,bad);
        hipLaunchKernelGGL(captured_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
            expected.output.as<float>(),dr.as<uint16_t>(),offset,count,source_queries,bad.as<unsigned>());
        check(hipGetLastError());ip_checked(bad,"original captured GB10 context differs");
        for(auto* arm:{&control,&actual})
            ip_owner(dq.as<uint16_t>(),kt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,expected,*arm,start,n,exp,packed,rcp,true,true,bad);
        // One untimed warmup pair and three rotated complete samples. Copies,
        // resets, validation and row preparation are outside each host wall.
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned order=0;order<2u;++order) {
            const unsigned mode=(order+attempt)%2u;auto& arm=mode?actual:control;
            const double ms=ip_owner(dq.as<uint16_t>(),kt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,expected,arm,start,n,exp,packed,rcp,true,false,bad);
            if(attempt)samples[mode][attempt-1u]+=ms;
            maximum_scratch[mode]=std::max(maximum_scratch[mode],arm.scratch.bytes);
        }
        score_cells+=uint64_t(count)*16u*stride;++slabs;
        std::fprintf(stderr,"INPLACE_P_CAPTURE queries=%u slab=%u count=%u complete_original_surfaces=1 pass=1\n",
            queries,slabs,count);
    }
    prepared.verify();dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);
    transpose_immutable(kt,k,n);transpose_immutable(vt,v,n);
    for(unsigned mode=0;mode<2u;++mode) {
        std::vector<double> sorted(samples[mode],samples[mode]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"inplace_probability_capture\",\"mode\":%u,\"query_start\":16384,"
            "\"query_count\":%u,\"key_tokens\":%u,\"slabs\":%u,\"query_batch\":128,"
            "\"original_cpu_dots\":%u,\"score_cells\":%llu,\"candidates\":%llu,"
            "\"original_gb10_context_cells\":4194304,\"extended_queries\":%u,"
            "\"maximum_scratch_bytes\":%zu,\"completed_host_samples_ms\":[%.9f,%.9f,%.9f],"
            "\"median_completed_host_ms\":%.9f,\"common_preparation_ms\":%.9f,"
            "\"payloads_and_unused_scores_bitexact\":true,\"native_and_replayed_raw_surfaces_bitexact\":true,"
            "\"complete_candidate_identity\":true,\"owner_stage_order\":true,\"nonzero_output_offset\":true,"
            "\"undersized_rejected\":true,\"guards_and_inputs_pass\":true,\"every_attempt_checked\":true,"
            "\"reference_is_compute_input\":false,\"model_loaded\":false,\"performance_acceptance\":false}\n",
            mode,queries,n,slabs,slabs*128u,(unsigned long long)score_cells,(unsigned long long)candidates,
            queries-source_queries,maximum_scratch[mode],samples[mode][0],samples[mode][1],samples[mode][2],
            sorted[1],prepared.common_ms+prepared.domain_ms+transpose_ms);
    }
}
} // namespace

int main(int argc,char** argv)try {
    const bool safety=argc==4&&!std::strcmp(argv[1],"safety");
    const unsigned queries=argc==8&&!std::strcmp(argv[1],"1024")?1024u:argc==8&&!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!safety&&!queries)throw std::runtime_error("usage: safety EXP RCP | 1024|8192 Q K V GB10_CONTEXT EXP RCP");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    const auto exp=read_table(argv[argc-2],delta::source::table_bytes),rcp=read_table(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))
        throw std::runtime_error("table layout");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);
    Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());
    check(hipGetLastError());ip_checked(bad,"in-place native EXP table mismatch");
    std::vector<unsigned char> packed(delta::packed_bytes);
    check(hipMemcpy(packed.data(),dd.data(),packed.size(),hipMemcpyDeviceToHost));
    if(safety)ip_safety(de.data(),dd.data(),dc.data());
    else ip_capture(queries,argv[2],argv[3],argv[4],argv[5],de.data(),dd.data(),dc.data());
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"inplace_probability_error=%s\n",e.what());return 2;}
