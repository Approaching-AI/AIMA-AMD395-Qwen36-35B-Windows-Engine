#define QRT_COMBINED_ATTENTION_NO_MAIN
#include "combined_exact_attention_capture.cpp"
#include "../../native/providers/ck_fmha/long_narrow_qk.h"
#include "../../native/providers/ck_fmha/long_fused_probability_pv.h"
#include "../../native/providers/ck_fmha/long_attention_pipeline.h"
#include "../../native/providers/ck_fmha/fused_probability_pv.h"

namespace {
namespace range=qrt_prepared_decoded_qk_range;
struct RangePrepared {
    Guarded memory,query_domain,key_domain,statistics;
    std::vector<uint32_t> expected;
    std::vector<unsigned> expected_query,expected_key;
    qrt_long_narrow_qk::Workspace workspace;
    double common_ms=0.0,domain_ms=0.0;
    RangePrepared(const uint16_t* q,const uint16_t* k,uint16_t* kt,
        const std::vector<uint16_t>& hq,const std::vector<uint16_t>& hk,
        unsigned n,unsigned origin,unsigned queries):
        memory(range::workspace_words(n)*4u),query_domain(size_t(queries)*16u*4u),
        key_domain(size_t(n)*2u*4u),statistics(8u),
        expected(range::workspace_words(n),0xa5a5a5a5u),
        expected_query(size_t(queries)*16u,1u),expected_key(size_t(n)*2u,1u),
        workspace{{memory.as<uint32_t>(),range::workspace_words(n),n,origin,queries,n},
            query_domain.as<unsigned>(),key_domain.as<unsigned>(),statistics.as<unsigned>()} {
        finish();auto begin=std::chrono::steady_clock::now();
        check(hipError_t(range::prepare_workspace(q,k,kt,workspace.decoded,nullptr)));finish();common_ms=elapsed(begin);
        begin=std::chrono::steady_clock::now();
        check(hipError_t(qrt_long_narrow_qk::prepare_domain(q,k,workspace,nullptr)));finish();domain_ms=elapsed(begin);
        for(unsigned key=0;key<2u;++key) {
            const unsigned heads=key?2u:16u,rows=(key?n:queries)*heads;
            const auto* source=key?hk.data():hq.data()+size_t(origin)*4096u;
            auto& domain=key?expected_key:expected_query;
            const size_t offset=key?range::query_words:0u;
            const size_t flags=range::query_words+range::key_words(n)+(key?range::query_flag_words:0u);
            for(unsigned row=0;row<rows;++row) {
                unsigned eligible=1u;
                for(unsigned c=0;c<256u;++c) {
                    const uint16_t x=source[size_t(row)*256u+c];const unsigned e=(x>>7u)&255u;
                    eligible&=unsigned(!(x&0x7fffu)||(e>=64u&&e<=190u));
                    domain[row]&=unsigned(qrt_sm121_narrow_f32_carry::eligible(x));
                    const int exponent=(x&0x7fffu)?int(e)-127:-512;
                    const size_t index=key?(size_t(row%heads)*256u+c)*n+row/heads:size_t(row)*256u+c;
                    expected[offset+index]=(uint32_t(x)<<16u)|uint16_t(exponent);
                }
                expected[flags+row]=eligible;
            }
        }
        verify();
    }
    void verify(){memory.immutable(expected);query_domain.immutable(expected_query);key_domain.immutable(expected_key);statistics.guards();}
};
void transpose_immutable(Guarded& device,const std::vector<uint16_t>& source,unsigned tokens) {
    std::vector<uint16_t> expected(source.size());
    for(unsigned t=0;t<tokens;++t)for(unsigned c=0;c<512u;++c)expected[size_t(c)*tokens+t]=source[size_t(t)*512u+c];
    device.immutable(expected);
}
void native_producer(const uint16_t* q,const uint16_t* kt,const uint16_t* v,
    RangePrepared& prepared,AttentionOutputs& out,unsigned start,unsigned count,unsigned n,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,
    bool candidate,bool independent=false) {
    const unsigned stride=start+count;
    auto* scores=out.tensor.scores.as<float>()+guard;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* s=out.tensor.scales.as<float>()+guard;
    if(independent) {
        hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,
            dim3((stride+31u)/32u,16u,(count+7u)/8u),dim3(256u),0u,nullptr,
            q,kt,scores,start,count,stride,n);check(hipGetLastError());
    } else if(candidate) {
        check(hipError_t(qrt_long_narrow_qk::launch_workspace(&prepared.workspace,
            q,kt,scores,nullptr,start,count,stride,n)));
    } else {
        check(hipError_t(range::launch_workspace(&prepared.workspace.decoded,
            q,kt,scores,nullptr,start,count,stride,n)));
    }
    if(candidate) {
        const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
        check(hipError_t(qrt_long_fused_probability_pv::launch(&owner,scores,v,p,s,
            out.output.as<float>(),out.error.as<float>(),out.accumulator.as<float>(),out.denominator.as<float>(),
            start,count,0u,stride,exp,rcp,true,nullptr)));
    } else {
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,count),dim3(32u),0u,nullptr,
            scores,p,s,start,stride,exp,true);check(hipGetLastError());
        hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,false,true,false>),
            dim3(256u/kIntegerMatrixColumns,16u,(count+15u)/16u),dim3(256u),0u,nullptr,
            v,p,s,out.output.as<float>(),start,count,0u,stride,rcp,
            out.accumulator.as<float>(),out.denominator.as<float>(),nullptr,nullptr,out.error.as<float>());
        check(hipGetLastError());
    }
}
void exact_replay(const uint16_t* v,const uint16_t* vt,AttentionOutputs& out,
    unsigned start,unsigned count,unsigned n,const unsigned char* rcp,bool candidate) {
    if(!candidate&&count<=split_query_limit(22u,start+count)) {
        check(hipError_t(launch_compacted_pv_replay(v,out.tensor.probability.as<uint16_t>()+guard,
            out.tensor.scales.as<float>()+guard,out.output.as<float>(),start,count,0u,start+count,
            rcp,out.accumulator.as<float>(),out.denominator.as<float>(),out.error.as<float>(),
            out.indices.as<unsigned>(),out.count.as<unsigned>(),nullptr,nullptr,vt,n,0u,nullptr,false)));
    }else {
        check(hipError_t(qrt_long_fused_probability_pv::replay(v,vt,out.tensor.probability.as<uint16_t>()+guard,
            out.tensor.scales.as<float>()+guard,out.output.as<float>(),out.error.as<float>(),
            out.accumulator.as<float>(),out.denominator.as<float>(),out.indices.as<unsigned>(),out.count.as<unsigned>(),
            start,count,0u,start+count,n,rcp,candidate,nullptr)));
    }
}
__global__ void captured_context(const float* output,const uint16_t* reference,
    unsigned row_offset,unsigned count,unsigned reference_rows,unsigned* bad) {
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell<count*4096u&&row_offset+cell/4096u<reference_rows) {
        const float x=output[cell];
        if(!isfinite(x)||f32_to_bf16(x)!=reference[size_t(row_offset)*4096u+cell])atomicAdd(bad,1u);
    }
}
void check_tail(AttentionOutputs& out,unsigned start,unsigned count,Device& bad) {
    hipLaunchKernelGGL(tensor_tails,dim3((out.tensor.cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
        out.tensor.scores.as<uint32_t>(),out.tensor.probability.as<uint16_t>(),out.tensor.scales.as<uint32_t>(),
        out.tensor.cells,out.tensor.scale_cells,start,count,start+count,bad.as<unsigned>());
    check(hipGetLastError());finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("range tensor tail changed");
}
void check_pipeline_owner(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    RangePrepared& prepared,AttentionOutputs& expected,unsigned start,unsigned count,unsigned n,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,Device& bad) {
    constexpr unsigned output_start=3u;
    const auto layout=qrt_long_attention_layout::layout(count,start+count);
    Guarded scratch(layout.elements*4u),output(size_t(count+output_start+2u)*4096u*4u);
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
    unsigned observed=0u;
    SplitCompletionObserver observer{&observed,[](void* state,unsigned stage,hipStream_t stream)->int {
        auto& next=*static_cast<unsigned*>(state);
        if(stage!=next)return int(hipErrorInvalidValue);
        const auto status=hipStreamSynchronize(stream);if(status==hipSuccess)++next;return int(status);
    }};
    auto launch=[&](size_t extent) {return qrt_long_attention_pipeline::launch(prepared.workspace,owner,
        q,kt,v,vt,output.as<float>(),start,count,output_start,n,exp,rcp,scratch.as<float>(),extent,nullptr,&observer);};
    if(launch(layout.elements-1u)!=int(hipErrorInvalidValue)||observed)
        throw std::runtime_error("undersized owner was accepted");
    output.immutable(std::vector<uint32_t>(output.bytes/4u,0xa5a5a5a5u));
    check(hipError_t(launch(layout.elements)));finish();
    if(observed!=5u)throw std::runtime_error("owner stage sequence");
    auto compare_region=[&](const unsigned char* a,const unsigned char* b,size_t bytes) {
        hipLaunchKernelGGL(compare_words,dim3((bytes+255u)/256u),dim3(256u),0u,nullptr,
            a,b,bytes,bad.as<unsigned>());check(hipGetLastError());
    };
    const size_t cells=size_t(count)*16u*(start+count);
    compare_region(expected.tensor.scores.as<unsigned char>()+guard*4u,scratch.data(),cells*4u);
    compare_region(expected.tensor.probability.as<unsigned char>()+guard*2u,scratch.data()+layout.probability*4u,cells*2u);
    compare_region(expected.tensor.scales.as<unsigned char>()+guard*4u,scratch.data()+layout.scales*4u,(layout.errors-layout.scales)*4u);
    compare_region(expected.error.data(),scratch.data()+layout.errors*4u,size_t(count)*4096u*4u);
    compare_region(expected.count.data(),scratch.data()+layout.count*4u,4u);
    compare_region(expected.output.data(),output.data()+size_t(output_start)*4096u*4u,size_t(count)*4096u*4u);
    finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("contiguous long owner differs");
    std::vector<uint32_t> prefix(size_t(output_start)*4096u),suffix(2u*4096u);
    check(hipMemcpy(prefix.data(),output.data(),prefix.size()*4u,hipMemcpyDeviceToHost));
    check(hipMemcpy(suffix.data(),output.data()+size_t(output_start+count)*4096u*4u,suffix.size()*4u,hipMemcpyDeviceToHost));
    for(const auto& padding:{prefix,suffix})for(uint32_t word:padding)
        if(word!=0xa5a5a5a5u)throw std::runtime_error("owner output padding changed");
    scratch.guards();output.guards();
}
void run_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp) {
    struct Shape{unsigned n,start,count;};
    const Shape shapes[]={{1,0,1},{33,1,32},{129,1,128},{8193,8191,2},{8320,8192,128},
        {16385,16352,33},{32896,32768,128},{65537,65520,17},{131073,131041,32},{264736,264719,17}};
    unsigned configurations=0,short_regressions=0,owner_cases=0;uint64_t cells=0,scores=0;
    for(const auto shape:shapes)for(unsigned mode=0;mode<5u;++mode) {
        const unsigned n=shape.n,start=shape.start,count=shape.count;
        std::vector<uint16_t> q(size_t(n)*4096u,0u),k(size_t(n)*512u),v(k.size());
        for(size_t i=size_t(start)*4096u;i<q.size();++i)q[i]=uint16_t(((i*37u+i/19u)&0x807fu)|((123u+i%8u)<<7u));
        for(size_t i=0;i<k.size();++i){k[i]=uint16_t(((i*53u+i/23u)&0x807fu)|((121u+i%10u)<<7u));v[i]=uint16_t(((i*71u+i/17u)&0x807fu)|((120u+i%12u)<<7u));}
        if(mode==1u){for(size_t i=size_t(start)*4096u;i<q.size();++i)q[i]=i%2u?0u:0x8000u;}
        if(mode==2u){const uint16_t edges[]={1u,0x8001u,0x007fu,0x807fu,uint16_t(63u<<7u|19u),uint16_t(192u<<7u|11u),0u,0x8000u};for(unsigned i=0;i<8u;++i){q[size_t(start)*4096u+i]=edges[i];k[i]=edges[7u-i];}}
        if(mode==3u){for(size_t i=size_t(start)*4096u;i<q.size();++i)q[i]=uint16_t(159u<<7u|127u);for(size_t i=0;i<k.size();++i)k[i]=uint16_t(159u<<7u|127u|((i&1u)?0x8000u:0u));}
        if(mode==4u){q[size_t(start)*4096u+255u]=uint16_t(94u<<7u|37u);k[511u]=uint16_t(160u<<7u|17u);}
        Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
        dq.put(q);dk.put(k);dv.put(v);
        RangePrepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q,k,n,start,count);
        check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();
        AttentionOutputs expected(n,count),actual(n,count);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,expected,start,count,n,exp,packed,rcp,false,true);finish();
        for(bool candidate:{false,true}) {
            actual.reset();native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,actual,start,count,n,exp,packed,rcp,candidate);finish();
            compare(expected,actual,bad);check_tail(actual,start,count,bad);++configurations;
        }
        exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),expected,start,count,n,rcp,false);finish();
        exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp,true);finish();compare(expected,actual,bad);
        check_pipeline_owner(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,expected,start,count,n,exp,packed,rcp,bad);++owner_cases;
        if(n<=129u) {
            expected.reset();actual.reset();
            native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,expected,start,count,n,exp,packed,rcp,false,true);finish();
            hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true,false>),
                dim3(256u/kIntegerMatrixColumns,16u,(count+15u)/16u),dim3(256u),0u,nullptr,
                dv.as<uint16_t>(),expected.tensor.probability.as<uint16_t>()+guard,expected.tensor.scales.as<float>()+guard,
                expected.output.as<float>(),start,count,0u,start+count,rcp,
                expected.accumulator.as<float>(),expected.denominator.as<float>(),nullptr,nullptr,expected.error.as<float>());
            check(hipGetLastError());finish();
            check(hipMemcpy(actual.tensor.scores.pointer,expected.tensor.scores.pointer,
                (expected.tensor.cells+2u*guard)*4u,hipMemcpyDeviceToDevice));
            const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(packed),exp};
            check(hipError_t(qrt_fused_probability_pv::launch(&owner,actual.tensor.scores.as<float>()+guard,dv.as<uint16_t>(),
                actual.tensor.probability.as<uint16_t>()+guard,actual.tensor.scales.as<float>()+guard,
                actual.output.as<float>(),actual.error.as<float>(),actual.accumulator.as<float>(),actual.denominator.as<float>(),
                start,count,0u,start+count,exp,rcp,true,nullptr)));finish();compare(expected,actual,bad);
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),expected,start,count,n,rcp,false);finish();
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp,true);finish();compare(expected,actual,bad);
            ++short_regressions;
        }
        prepared.verify();dq.immutable(q);dk.immutable(k);dv.immutable(v);transpose_immutable(dt,k,n);transpose_immutable(vt,v,n);
        cells+=uint64_t(count)*4096u;scores+=uint64_t(count)*16u*(start+count);
        std::fprintf(stderr,"LONG_ATTENTION_SAFETY tokens=%u start=%u count=%u mode=%u pass=1\n",n,start,count,mode);
    }
    if(owner_cases!=50u)throw std::runtime_error("incomplete owner safety cases");
    std::fprintf(stderr,"LONG_ATTENTION_OWNER_SAFETY cases=%u nonzero_output_offset=1 undersized_rejected=1 complete_surfaces=1 stages=5\n",owner_cases);
    std::printf("{\"kind\":\"long_attention_pipeline_safety\",\"configurations\":%u,\"shapes\":10,\"data_modes\":5,\"short_default_template_regressions\":%u,\"distinct_output_cells\":%llu,\"distinct_score_slots\":%llu,\"all_native_surfaces_bitexact\":true,\"complete_replay_bitexact\":true,\"original_per_group_bounds\":true,\"range_metadata_and_guards_pass\":true,\"input_immutable\":true}\n",configurations,short_regressions,(unsigned long long)cells,(unsigned long long)scores);
}
void run_capture(unsigned queries,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp) {
    constexpr unsigned origin=16384u,source_queries=1024u,source_tokens=17408u;
    const unsigned n=origin+queries;
    const auto original_q=read_words(qfile,source_queries*4096u);
    const auto original_k=read_words(kfile,source_tokens*512u),original_v=read_words(vfile,source_tokens*512u);
    std::vector<uint16_t> q(size_t(n)*4096u,0u),k(size_t(n)*512u),v(k.size());
    for(unsigned i=0;i<queries;++i)std::copy_n(original_q.data()+size_t(i%source_queries)*4096u,4096u,q.data()+size_t(origin+i)*4096u);
    for(unsigned i=0;i<n;++i){std::copy_n(original_k.data()+size_t(i%source_tokens)*512u,512u,k.data()+size_t(i)*512u);std::copy_n(original_v.data()+size_t(i%source_tokens)*512u,512u,v.data()+size_t(i)*512u);}
    const auto reference=read_words(reference_file,source_queries*4096u);
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u),dr(reference.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);dr.put(reference);
    RangePrepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q,k,n,origin,queries);
    finish();auto begin=std::chrono::steady_clock::now();check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),n,nullptr)));finish();const double transpose_ms=elapsed(begin);
    double samples[4][3]{};uint64_t candidates[4]{},score_slots[4]{},cpu_dots[4]{};
    // Alternate complete-route order, including both slab widths. Every timed
    // slab is checked against independently generated original QK/P/PV.
    for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned order=0;order<4u;++order) {
        const unsigned variant=(order+attempt)%4u,capacity=variant<2u?32u:128u;
        const bool candidate=variant&1u;
        AttentionOutputs expected(n,capacity),actual(n,capacity);Device bad(4u);check(hipMemset(bad.pointer,0,4u));
        for(unsigned offset=0;offset<queries;offset+=capacity) {
            const unsigned start=origin+offset,count=std::min(capacity,queries-offset),stride=start+count;
            expected.reset();native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,expected,start,count,n,exp,packed,rcp,false,true);finish();
            if(!attempt) {
                actual.reset();native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,actual,start,count,n,exp,packed,rcp,candidate);finish();compare(expected,actual,bad);
            }
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),expected,start,count,n,rcp,false);finish();
            hipLaunchKernelGGL(captured_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
                expected.output.as<float>(),dr.as<uint16_t>(),offset,count,source_queries,bad.as<unsigned>());check(hipGetLastError());check_tail(expected,start,count,bad);
            if(download<unsigned>(bad,1u)[0])throw std::runtime_error("original long context differs from GB10");
            if(!attempt&&variant==3u)
                check_pipeline_owner(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                    prepared,expected,start,count,n,exp,packed,rcp,bad);
            actual.reset();finish();begin=std::chrono::steady_clock::now();
            native_producer(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),prepared,actual,start,count,n,exp,packed,rcp,candidate);
            exact_replay(dv.as<uint16_t>(),vt.as<uint16_t>(),actual,start,count,n,rcp,candidate);finish();const double ms=elapsed(begin);
            if(attempt)samples[variant][attempt-1u]+=ms;
            compare(expected,actual,bad);
            if(!attempt) {
                unsigned selected=0;check(hipMemcpy(&selected,actual.count.data(),4u,hipMemcpyDeviceToHost));candidates[variant]+=selected;
                score_slots[variant]+=uint64_t(count)*16u*stride;
                for(unsigned j=0;j<4u;++j) {
                    const unsigned row=j*(count-1u)/3u,head=(offset/capacity+j*5u)%16u,key=(start+row)*j/3u;
                    const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                        q.data()+(size_t(start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
                    uint32_t gpu=0;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+(size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
                    if(gpu!=bits(cpu))throw std::runtime_error("long QK CPU dot mismatch");++cpu_dots[variant];
                }
            }
        }
    }
    for(unsigned variant=0;variant<4u;++variant)if(candidates[variant]!=candidates[0])throw std::runtime_error("slab width changed PV candidate count");
    std::fprintf(stderr,"LONG_ATTENTION_OWNER_CAPTURE query_count=%u slabs=%u nonzero_output_offset=1 undersized_rejected=1 complete_surfaces=1 stages=5\n",queries,(queries+127u)/128u);
    prepared.verify();dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);transpose_immutable(dt,k,n);transpose_immutable(vt,v,n);
    for(unsigned variant=0;variant<4u;++variant) {
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"long_attention_pipeline_capture\",\"query_start\":16384,\"query_count\":%u,\"key_tokens\":%u,\"source_queries\":1024,\"extended_queries\":%u,\"variant\":%u,\"candidate\":%s,\"query_batch\":%u,\"pv_candidates\":%llu,\"score_slots\":%llu,\"cpu_dots\":%llu,\"gb10_context_cells\":4194304,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"domain_preparation_ms\":%.9f,\"common_preparation_ms\":%.9f,\"native_surfaces_and_exact_replay_bitexact\":true,\"all_attempts_checked\":true,\"original_per_group_bounds\":true,\"guards_and_inputs_pass\":true,\"reference_is_compute_input\":false,\"model_loaded\":false,\"performance_acceptance\":false}\n",queries,n,queries-source_queries,variant,variant&1u?"true":"false",variant<2u?32u:128u,(unsigned long long)candidates[variant],(unsigned long long)score_slots[variant],(unsigned long long)cpu_dots[variant],samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],variant&1u?prepared.domain_ms:0.0,prepared.common_ms+transpose_ms);
    }
}
} // namespace

int main(int argc,char** argv)try {
    const bool safety=argc==4&&!std::strcmp(argv[1],"safety");
    const unsigned queries=argc==8&&!std::strcmp(argv[1],"1024")?1024u:argc==8&&!std::strcmp(argv[1],"8192")?8192u:0u;
    if(!safety&&!queries)throw std::runtime_error("usage: safety EXP RCP | 1024|8192 Q K V GB10_CONTEXT EXP RCP");
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    const auto exp=read_table(argv[argc-2],delta::source::table_bytes),rcp=read_table(argv[argc-1],qrt_sm121_attention_rcp::table_bytes);
    if(!delta::source::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("table layout");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);
    Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("native EXP full domain mismatch");
    std::vector<unsigned char> packed(delta::packed_bytes);check(hipMemcpy(packed.data(),dd.data(),packed.size(),hipMemcpyDeviceToHost));
    if(safety)run_safety(de.data(),dd.data(),dc.data());else run_capture(queries,argv[2],argv[3],argv[4],argv[5],de.data(),dd.data(),dc.data());
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"long_attention_pipeline_error=%s\n",e.what());return 2;}
