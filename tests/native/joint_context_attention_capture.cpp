// Reuse the independent original-QK/PV control and guarded captured-input
// owners. The candidate below never receives those comparison values.
#define QRT_STREAMED_ATTENTION_NO_MAIN
#include "streamed_exact_attention_capture.cpp"
#include "../../native/providers/ck_fmha/joint_context_priority.h"

namespace {
namespace joint=qrt_joint_context_qk;
struct JointWorkspace {
    size_t cells,rows,scale_cells;
    Guarded score_error,lower,upper,maxima,unit_scales,center_denominator,maximum_cost,
        denominator_intervals,budgets,pending,priorities,selected,q_indices,q_count,certified;
    explicit JointWorkspace(unsigned tokens):cells(size_t(query_batch)*16u*tokens),rows(query_batch*16u),
        scale_cells(rows*((tokens+31u)/32u+1u)),score_error(cells*4u),lower(cells*4u),upper(cells*4u),
        maxima(scale_cells*4u),unit_scales(scale_cells*4u),center_denominator(rows*4u),maximum_cost(rows*4u),
        denominator_intervals(rows*8u),budgets(rows*4u),pending(rows*4u),priorities(cells),selected(cells),
        q_indices(cells*4u),q_count(4u),certified(4u){}
    void reset(){for(auto* p:{&score_error,&lower,&upper,&maxima,&unit_scales,&center_denominator,
        &maximum_cost,&denominator_intervals,&budgets,&pending,&priorities,&selected,&q_indices,&q_count,&certified})p->reset();}
    void guards(){for(auto* p:{&score_error,&lower,&upper,&maxima,&unit_scales,&center_denominator,
        &maximum_cost,&denominator_intervals,&budgets,&pending,&priorities,&selected,&q_indices,&q_count,&certified})p->guards();}
};

void joint_attention(const uint16_t* q,const uint16_t* kt,const uint16_t* v,const uint16_t* vt,
    Prepared& prepared,AttentionOutputs& out,JointWorkspace& w,unsigned start,unsigned count,unsigned tokens,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    const unsigned stride=start+count,cells=count*4096u;
    const size_t score_cells=size_t(count)*16u*stride;
    auto* scores=out.tensor.scores.as<float>()+guard;
    auto* probabilities=out.tensor.probability.as<uint16_t>()+guard;
    auto* scales=out.tensor.scales.as<float>()+guard;
    auto* pq=prepared.qp.as<uint32_t>()+guard;auto* pk=prepared.kp.as<uint32_t>()+guard;
    auto* qflags=prepared.qf.as<unsigned>()+guard;auto* kflags=prepared.kf.as<unsigned>()+guard;
    check(hipMemsetAsync(w.certified.data(),0,4u,nullptr));
    hipLaunchKernelGGL(qrt_selective_qk::native_scores,
        dim3((stride+kIntegerMatrixColumns-1u)/kIntegerMatrixColumns,16u,(count+15u)/16u),dim3(256u),0u,nullptr,
        q,kt,scores,w.score_error.as<float>(),start,count,stride,tokens);check(hipGetLastError());
    for(unsigned pass=0u;pass<2u;++pass){
        check(hipMemsetAsync(w.q_count.data(),0,4u,nullptr));
        if(!pass){
            hipLaunchKernelGGL(qrt_selective_qk::collect_maxima,dim3(16u,count),dim3(32u),0u,nullptr,
                scores,w.score_error.as<float>(),start,stride,w.q_indices.as<unsigned>(),w.q_count.as<unsigned>());
        }else{
            hipLaunchKernelGGL(joint::collect_probabilities,dim3(16u,count),dim3(32u),0u,nullptr,
                scores,w.score_error.as<float>(),w.maxima.as<float>(),start,stride,exp,packed,
                w.q_indices.as<unsigned>(),w.q_count.as<unsigned>());
        }
        check(hipGetLastError());
        hipLaunchKernelGGL((qrt_adaptive_denominator_qk::repair<false>),dim3(256u),dim3(256u),0u,nullptr,
            q,kt,pq,pk,qflags,kflags,scores,w.score_error.as<float>(),start,stride,tokens,
            w.q_indices.as<unsigned>(),w.q_count.as<unsigned>(),nullptr,nullptr,nullptr,nullptr);check(hipGetLastError());
    }
    hipLaunchKernelGGL(joint::initialize,dim3(16u,count),dim3(32u),0u,nullptr,
        scores,w.score_error.as<float>(),w.maxima.as<float>(),probabilities,scales,w.unit_scales.as<float>(),
        w.lower.as<float>(),w.upper.as<float>(),w.center_denominator.as<float>(),w.maximum_cost.as<float>(),
        w.pending.as<unsigned>(),start,stride,exp,packed);check(hipGetLastError());
    hipLaunchKernelGGL((blackwell_mantissa_value_kernel<true,false,true,true,true,true>),
        dim3(256u/kIntegerMatrixColumns,16u,(count+15u)/16u),dim3(256u),0u,nullptr,
        v,probabilities,w.unit_scales.as<float>(),out.output.as<float>(),start,count,0u,stride,rcp,
        out.accumulator.as<float>(),nullptr,nullptr,nullptr,out.error.as<float>());check(hipGetLastError());
    const auto certify=[&](bool exact_point){
        hipLaunchKernelGGL(joint::certify,dim3(16u,count),dim3(32u),0u,nullptr,
            w.lower.as<float>(),w.upper.as<float>(),out.accumulator.as<float>(),out.error.as<float>(),scales,
            w.denominator_intervals.as<float>(),out.output.as<float>(),w.pending.as<unsigned>(),
            start,stride,rcp,exact_point,w.certified.as<unsigned>());check(hipGetLastError());
    };
    const auto repair_numerators=[&](bool final){
        check(hipMemsetAsync(out.count.data(),0,4u,nullptr));
        hipLaunchKernelGGL(joint::collect_numerators,dim3((cells+255u)/256u),dim3(256u),0u,nullptr,
            out.accumulator.as<float>(),out.error.as<float>(),w.center_denominator.as<float>(),
            final?w.denominator_intervals.as<float>():nullptr,w.pending.as<unsigned>(),cells,rcp,
            out.indices.as<unsigned>(),out.count.as<unsigned>());check(hipGetLastError());
        hipLaunchKernelGGL((blackwell_compacted_pv_replay_kernel<true,false,true>),
            dim3(std::min(1024u,(cells+63u)/64u)),dim3(256u),0u,nullptr,
            v,probabilities,w.unit_scales.as<float>(),out.output.as<float>(),start,0u,stride,rcp,
            out.accumulator.as<float>(),nullptr,out.indices.as<unsigned>(),out.count.as<unsigned>(),vt,tokens,0u);
        check(hipGetLastError());
        hipLaunchKernelGGL(joint::mark_exact_numerators,dim3(256u),dim3(256u),0u,nullptr,
            out.error.as<float>(),out.indices.as<unsigned>(),out.count.as<unsigned>());check(hipGetLastError());
    };
    const auto repair_selected=[&]{
        hipLaunchKernelGGL((qrt_selected_microtile_qk::scores<2u,2u>),
            dim3((stride+31u)/32u,16u,(count+31u)/32u),dim3(256u),0u,nullptr,
            pq,pk,qflags,kflags,scores,w.score_error.as<float>(),w.selected.data(),start,count,stride,tokens);
        check(hipGetLastError());
        hipLaunchKernelGGL(qrt_deferred_qk_fallback::replay_scan,
            dim3((score_cells+255u)/256u),dim3(256u),0u,nullptr,q,kt,scores,start,count,stride,tokens);
        check(hipGetLastError());
        hipLaunchKernelGGL(joint::update_intervals,dim3((score_cells+255u)/256u),dim3(256u),0u,nullptr,
            scores,w.selected.data(),w.maxima.as<float>(),w.lower.as<float>(),w.upper.as<float>(),
            start,count,stride,exp,packed);check(hipGetLastError());
    };
    certify(false);repair_numerators(false);
    hipLaunchKernelGGL(joint::row_budget,dim3(16u,count),dim3(32u),0u,nullptr,
        out.accumulator.as<float>(),out.error.as<float>(),w.center_denominator.as<float>(),
        w.denominator_intervals.as<float>(),w.pending.as<unsigned>(),w.budgets.as<float>(),rcp);check(hipGetLastError());
    hipLaunchKernelGGL(joint::histogram_select,dim3(16u,count),dim3(32u),0u,nullptr,
        w.score_error.as<float>(),w.lower.as<float>(),w.upper.as<float>(),scales,w.maximum_cost.as<float>(),
        w.budgets.as<float>(),w.pending.as<unsigned>(),w.priorities.data(),w.selected.data(),start,stride);
    check(hipGetLastError());repair_selected();certify(false);
    hipLaunchKernelGGL(joint::select_remaining,dim3((score_cells+255u)/256u),dim3(256u),0u,nullptr,
        w.score_error.as<float>(),w.pending.as<unsigned>(),w.selected.data(),start,count,stride);
    check(hipGetLastError());repair_selected();certify(true);
    repair_numerators(true);certify(true);
}

// These observers run after completion, outside candidate computation/timing.
// The full original numerator is generated independently for comparison only.
struct JointMetrics {
    unsigned score_bound_failures=0u,probability_differences=0u,alpha_differences=0u,
        denominator_bound_failures=0u,numerator_bound_failures=0u,context_differences=0u,
        pending_rows=0u,exact_scores=0u,exact_numerators=0u;
};
__global__ void inspect_joint(const float* expected_scores,const uint16_t* expected_probability,
    const float* expected_scales,const float* expected_numerator,const float* expected_output,
    const float* scores,const float* score_errors,const uint16_t* probabilities,const float* scales,
    const float* numerator,const float* numerator_error,const float* denominators,const float* output,
    const unsigned* pending,unsigned start,unsigned count,unsigned stride,JointMetrics* metrics){
    const size_t cell=size_t(blockIdx.x)*blockDim.x+threadIdx.x,rows=size_t(count)*16u;
    bool exact_score=false,exact_numerator=false;
    if(cell<rows*stride){
        const unsigned row=unsigned(cell/stride),key=unsigned(cell%stride);
        if(key<=start+row/16u){
            const auto interval=qrt_selective_qk::score_interval(scores[cell],score_errors[cell]);
            if(!(expected_scores[cell]>=interval.low && expected_scores[cell]<=interval.high))atomicAdd(&metrics->score_bound_failures,1u);
            if(expected_probability[cell]!=probabilities[cell])atomicAdd(&metrics->probability_differences,1u);
            exact_score=score_errors[cell]==0.0f;
        }
    }
    if(cell<rows*256u){
        const auto interval=joint::numerator_interval(numerator[cell],numerator_error[cell]);
        if(!(expected_numerator[cell]>=interval.low && expected_numerator[cell]<=interval.high))atomicAdd(&metrics->numerator_bound_failures,1u);
        if(!isfinite(output[cell]) || f32_to_bf16(output[cell])!=f32_to_bf16(expected_output[cell]))atomicAdd(&metrics->context_differences,1u);
        exact_numerator=numerator_error[cell]==0.0f;
    }
    if(cell<rows){
        const unsigned tiles=(stride+31u)/32u;
        for(unsigned tile=0u;tile<(start+unsigned(cell)/16u+32u)/32u;++tile)
            if(__float_as_uint(expected_scales[cell*(tiles+1u)+tile])!=__float_as_uint(scales[cell*(tiles+1u)+tile]))atomicAdd(&metrics->alpha_differences,1u);
        const float expected=expected_scales[cell*(tiles+1u)+tiles];
        if(!(expected>=denominators[cell*2u] && expected<=denominators[cell*2u+1u]))atomicAdd(&metrics->denominator_bound_failures,1u);
        if(pending[cell])atomicAdd(&metrics->pending_rows,1u);
    }
    const unsigned score_mask=__ballot(exact_score),numerator_mask=__ballot(exact_numerator);
    if(!(threadIdx.x%32u)){
        if(score_mask)atomicAdd(&metrics->exact_scores,unsigned(__popc(score_mask)));
        if(numerator_mask)atomicAdd(&metrics->exact_numerators,unsigned(__popc(numerator_mask)));
    }
}

__global__ void inspect_context(const float* expected,const float* actual,unsigned cells,unsigned* bad){
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x;
    if(cell<cells && (!isfinite(actual[cell]) || f32_to_bf16(expected[cell])!=f32_to_bf16(actual[cell])))atomicAdd(bad,1u);
}
__global__ void unchanged_tail(const unsigned char* data,size_t begin,size_t end,unsigned* bad){
    const size_t cell=begin+size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(cell<end && data[cell]!=0xa5u)atomicAdd(bad,1u);
}
void tail(Guarded& data,size_t used,Device& bad){
    if(used>data.bytes)throw std::runtime_error("invalid used extent");
    if(used<data.bytes){
        hipLaunchKernelGGL(unchanged_tail,dim3((data.bytes-used+255u)/256u),dim3(256u),0u,nullptr,
            data.data(),used,data.bytes,bad.as<unsigned>());check(hipGetLastError());
    }
}
void check_tensor_tails(AttentionOutputs& out,unsigned start,unsigned count,Device& bad){
    hipLaunchKernelGGL(tensor_tails,dim3((out.tensor.cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
        out.tensor.scores.as<uint32_t>(),out.tensor.probability.as<uint16_t>(),out.tensor.scales.as<uint32_t>(),
        out.tensor.cells,out.tensor.scale_cells,start,count,start+count,bad.as<unsigned>());check(hipGetLastError());
    for(auto* data:{&out.output,&out.accumulator,&out.error,&out.indices})tail(*data,size_t(count)*4096u*4u,bad);
    tail(out.denominator,size_t(count)*16u*4u,bad);
}
void original_pv(const uint16_t* value,AttentionOutputs& original,Guarded& context,Guarded& numerator,
    unsigned start,unsigned count,const unsigned char* reciprocal){
    context.reset();numerator.reset();
    hipLaunchKernelGGL(blackwell_probability_value_kernel,dim3(16u,count),dim3(256u),0u,nullptr,
        value,original.tensor.probability.as<uint16_t>()+guard,original.tensor.scales.as<float>()+guard,
        context.as<float>(),start,0u,start+count,reciprocal,numerator.as<float>(),nullptr,nullptr);
    check(hipGetLastError());finish();
}
void verify_original(AttentionOutputs& original,Guarded& canonical,Guarded& raw,Guarded* gb10,
    unsigned start,unsigned count,Device& bad){
    check(hipMemset(bad.pointer,0,4u));check_tensor_tails(original,start,count,bad);
    hipLaunchKernelGGL(inspect_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
        canonical.as<float>(),original.output.as<float>(),count*4096u,bad.as<unsigned>());check(hipGetLastError());
    if(gb10){
        hipLaunchKernelGGL(external_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
            canonical.as<float>(),gb10->as<uint16_t>(),start,count,bad.as<unsigned>());check(hipGetLastError());
    }
    tail(canonical,size_t(count)*4096u*4u,bad);tail(raw,size_t(count)*4096u*4u,bad);finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("original context, GB10 or unused-tail mismatch");
    original.guards();canonical.guards();raw.guards();
}
JointMetrics verify_joint(AttentionOutputs& expected,Guarded& canonical,Guarded& raw,AttentionOutputs& actual,
    JointWorkspace& w,Guarded* gb10,unsigned start,unsigned count,Device& bad,Guarded& result){
    check(hipMemset(bad.pointer,0,4u));check(hipMemset(result.data(),0,sizeof(JointMetrics)));
    const unsigned stride=start+count,rows=count*16u,tiles=(stride+31u)/32u;
    const size_t cells=size_t(rows)*stride;
    hipLaunchKernelGGL(inspect_joint,dim3((std::max(cells,size_t(rows)*256u)+255u)/256u),dim3(256u),0u,nullptr,
        expected.tensor.scores.as<float>()+guard,expected.tensor.probability.as<uint16_t>()+guard,
        expected.tensor.scales.as<float>()+guard,raw.as<float>(),canonical.as<float>(),
        actual.tensor.scores.as<float>()+guard,w.score_error.as<float>(),actual.tensor.probability.as<uint16_t>()+guard,
        actual.tensor.scales.as<float>()+guard,actual.accumulator.as<float>(),actual.error.as<float>(),
        w.denominator_intervals.as<float>(),actual.output.as<float>(),w.pending.as<unsigned>(),start,count,stride,result.as<JointMetrics>());
    check(hipGetLastError());check_tensor_tails(actual,start,count,bad);
    if(gb10){
        hipLaunchKernelGGL(external_context,dim3((count*4096u+255u)/256u),dim3(256u),0u,nullptr,
            actual.output.as<float>(),gb10->as<uint16_t>(),start,count,bad.as<unsigned>());check(hipGetLastError());
    }
    for(auto* data:{&w.score_error,&w.lower,&w.upper,&w.q_indices})tail(*data,cells*4u,bad);
    for(auto* data:{&w.priorities,&w.selected})tail(*data,cells,bad);
    for(auto* data:{&w.center_denominator,&w.maximum_cost,&w.budgets,&w.pending})tail(*data,size_t(rows)*4u,bad);
    tail(w.maxima,size_t(rows)*tiles*4u,bad);tail(w.unit_scales,size_t(rows)*(tiles+1u)*4u,bad);
    tail(w.denominator_intervals,size_t(rows)*8u,bad);tail(actual.denominator,0u,bad);finish();
    JointMetrics metrics{};unsigned certified=0u;
    check(hipMemcpy(&metrics,result.data(),sizeof(metrics),hipMemcpyDeviceToHost));
    check(hipMemcpy(&certified,w.certified.data(),4u,hipMemcpyDeviceToHost));
    const auto errors=metrics.score_bound_failures+metrics.probability_differences+metrics.alpha_differences+
        metrics.denominator_bound_failures+metrics.numerator_bound_failures+metrics.context_differences+metrics.pending_rows;
    if(errors || certified!=rows || download<unsigned>(bad,1u)[0]){
        std::fprintf(stderr,"JOINT_MISMATCH start=%u count=%u score=%u probability=%u alpha=%u denominator=%u numerator=%u context=%u pending=%u certified=%u tail_or_gb10=%u\n",
            start,count,metrics.score_bound_failures,metrics.probability_differences,metrics.alpha_differences,
            metrics.denominator_bound_failures,metrics.numerator_bound_failures,metrics.context_differences,
            metrics.pending_rows,certified,download<unsigned>(bad,1u)[0]);
        throw std::runtime_error("joint context boundary mismatch");
    }
    actual.guards();w.guards();result.guards();return metrics;
}

void joint_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
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
        AttentionOutputs expected(n),actual(n);JointWorkspace workspace(n);Device bad(4u);
        Guarded canonical(size_t(query_batch)*4096u*4u),raw(canonical.bytes),metrics(sizeof(JointMetrics));
        attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,n,exp,nullptr,rcp,true,0u,nullptr);finish();
        original_pv(dv.as<uint16_t>(),expected,canonical,raw,start,count,rcp);
        verify_original(expected,canonical,raw,nullptr,start,count,bad);
        streamed(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            actual,start,count,n,exp,packed,rcp,true);finish();compare(expected,actual,bad);++cases;
        actual.reset();workspace.reset();joint_attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
            prepared,actual,workspace,start,count,n,exp,packed,rcp);finish();
        (void)verify_joint(expected,canonical,raw,actual,workspace,nullptr,start,count,bad,metrics);++cases;
        dq.immutable(q);dk.immutable(k);dv.immutable(v);prepared.verify();
        immutable_transpose(dt,k,n);immutable_transpose(vt,v,n);expected.guards();
        std::fprintf(stderr,"JOINT_SAFETY tokens=%u start=%u queries=%u mode=%u pass=1\n",n,start,count,mode);
    }
    std::printf("{\"kind\":\"joint_context_attention_safety\",\"cases\":%u,\"shapes\":8,\"data_modes\":5,\"interval_bound_failures\":0,\"probability_and_alpha_differences\":0,\"context_differences\":0,\"unresolved_rows\":0,\"full_original_pv_observer\":true,\"guards_pass\":true,\"unused_tails_pass\":true,\"immutable_inputs\":true}\n",cases);
}

void joint_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,const char* reference_file,
    const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp,double table_build_ms,double table_verify_ms){
    auto q=read_words(qfile,7169u*4096u),k=read_words(kfile,7169u*512u),v=read_words(vfile,7169u*512u);
    for(auto pair:{std::make_pair(&q,4096u),std::make_pair(&k,512u),std::make_pair(&v,512u)}){
        const auto old=*pair.first;pair.first->insert(pair.first->end(),old.begin(),old.begin()+size_t(tokens-7169u)*pair.second);
    }
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);
    Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),tokens);
    const auto transpose_begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),tokens,nullptr)));finish();
    const double transpose_ms=elapsed(transpose_begin);
    const auto reference=read_words(reference_file,7169u*4096u);Guarded dr(reference.size()*2u);dr.put(reference);
    AttentionOutputs expected(tokens),actual(tokens);JointWorkspace workspace(tokens);Device bad(4u);
    Guarded canonical(size_t(query_batch)*4096u*4u),raw(canonical.bytes),metrics(sizeof(JointMetrics));
    double samples[2][3]{};uint64_t control_replays=0u,exact_scores=0u,exact_numerators=0u;
    uint64_t score_cells=0u;unsigned cpu_dots=0u;
    for(unsigned start=0;start<tokens;start+=query_batch){
        const unsigned count=std::min(query_batch,tokens-start),stride=start+count;
        expected.reset();attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),prepared,
            expected,start,count,tokens,exp,nullptr,rcp,true,0u,nullptr);finish();
        original_pv(dv.as<uint16_t>(),expected,canonical,raw,start,count,rcp);
        verify_original(expected,canonical,raw,&dr,start,count,bad);
        for(unsigned attempt=0u;attempt<4u;++attempt)for(unsigned position=0u;position<2u;++position){
            const unsigned variant=(position+start/query_batch+attempt)%2u;
            actual.reset();if(variant)workspace.reset();finish();const auto begin=std::chrono::steady_clock::now();
            if(!variant)streamed(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,actual,start,count,tokens,exp,packed,rcp,true);
            else joint_attention(dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
                prepared,actual,workspace,start,count,tokens,exp,packed,rcp);
            finish();const double ms=elapsed(begin);if(attempt)samples[variant][attempt-1u]+=ms;
            if(!variant){
                compare(expected,actual,bad);if(!attempt){unsigned n=0u;check(hipMemcpy(&n,actual.count.data(),4u,hipMemcpyDeviceToHost));control_replays+=n;}
            }else{
                const auto result=verify_joint(expected,canonical,raw,actual,workspace,&dr,start,count,bad,metrics);
                if(!attempt){exact_scores+=result.exact_scores;exact_numerators+=result.exact_numerators;}
            }
        }
        for(unsigned s=0u;s<4u;++s){
            const unsigned row=s*(count-1u)/3u,head=(start/query_batch+s*5u)%16u,key=(start+row)*s/3u;
            const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                q.data()+(size_t(start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
            uint32_t gpu=0u;check(hipMemcpy(&gpu,expected.tensor.scores.as<uint32_t>()+guard+(size_t(row)*16u+head)*stride+key,4u,hipMemcpyDeviceToHost));
            if(gpu!=bits(cpu))throw std::runtime_error("original score differs from CPU");++cpu_dots;
        }
        for(unsigned row=0u;row<count;++row)score_cells+=uint64_t(start+row+1u)*16u;
    }
    dq.immutable(q);dk.immutable(k);dv.immutable(v);dr.immutable(reference);prepared.verify();
    immutable_transpose(dt,k,tokens);immutable_transpose(vt,v,tokens);expected.guards();
    for(unsigned variant=0u;variant<2u;++variant){
        auto sorted=std::vector<double>(samples[variant],samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"joint_context_attention_capture\",\"tokens\":%u,\"source_capture_tokens\":7169,\"repeated_rows\":%u,\"variant\":%u,\"joint_context_intervals\":%s,\"query_batch\":128,\"causal_score_cells\":%llu,\"output_cells\":%u,\"gb10_context_cells\":29364224,\"cpu_dots\":%u,\"original_score_replays\":%llu,\"original_pv_replays\":%llu,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"common_preparation_ms\":%.9f,\"common_exp_build_ms\":%.9f,\"common_exp_verify_ms\":%.9f,\"interval_bound_failures\":0,\"probability_and_alpha_differences\":0,\"context_differences\":0,\"gb10_context_mismatches\":0,\"unresolved_rows\":0,\"all_attempts_checked\":true,\"warmups_per_slab\":1,\"timed_attempts_per_slab\":3,\"full_original_pv_observer\":true,\"reference_is_compute_input\":false,\"full_fallback_in_timing\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens,tokens-7169u,variant,variant?"true":"false",(unsigned long long)score_cells,tokens*4096u,cpu_dots,
            (unsigned long long)(variant?exact_scores:score_cells),(unsigned long long)(variant?exact_numerators:control_replays),
            samples[variant][0],samples[variant][1],samples[variant][2],sorted[1],prepared.ms+transpose_ms,table_build_ms,table_verify_ms);
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
    if(bits(qrt_sm121_attention_rcp::evaluate(rcp.data(),1.0f))!=bits(1.0f))throw std::runtime_error("unit reciprocal is not exact");
    Guarded de(exp.size()),dc(rcp.size()),dd(delta::packed_bytes);de.put(exp);dc.put(rcp);
    Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    const auto build_begin=std::chrono::steady_clock::now();
    hipLaunchKernelGGL(delta::build,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data());check(hipGetLastError());finish();
    const double build_ms=elapsed(build_begin);const auto verify_begin=std::chrono::steady_clock::now();
    hipLaunchKernelGGL(verify_derived_exp,dim3(4096u),dim3(256u),0u,nullptr,de.data(),dd.data(),bad.as<unsigned>());check(hipGetLastError());finish();
    const double verify_ms=elapsed(verify_begin);
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("complete EXP domain differs");
    std::vector<unsigned char> packed(delta::packed_bytes);check(hipMemcpy(packed.data(),dd.data(),packed.size(),hipMemcpyDeviceToHost));
    if(safety_mode)joint_safety(de.data(),dd.data(),dc.data());
    else joint_capture(tokens,argv[2],argv[3],argv[4],argv[5],de.data(),dd.data(),dc.data(),build_ms,verify_ms);
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"joint_context_attention_error=%s\n",e.what());return 2;}
