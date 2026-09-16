// Reuse guarded shapes, original producer/replay and independent wide CPU
// recurrence helpers. The prior experiment's main is not executed.
#define main qrt_prior_coarse_pv_main
#include "coarse_pv_transfer_selftest.cpp"
#undef main
#include "../../native/providers/ck_fmha/prefix_pv_replay.h"

namespace {
namespace prefix = qrt_prefix_pv;
struct PrefixBuffer {
    size_t bytes;Device device;
    explicit PrefixBuffer(size_t n):bytes(n),device(n+512u){reset();}
    void reset(){check(hipMemset(device.pointer,0xa5,bytes+512u));}
    template<class T>T* data(){return reinterpret_cast<T*>(device.as<unsigned char>()+256u);}
    void verify_guards(){
        unsigned char data[512];check(hipMemcpy(data,device.pointer,256u,hipMemcpyDeviceToHost));
        check(hipMemcpy(data+256u,device.as<unsigned char>()+256u+bytes,256u,hipMemcpyDeviceToHost));
        for(auto b:data)if(b!=0xa5u)throw std::runtime_error("prefix PV redzone");
    }
};
struct PrefixWorkspace {
    unsigned chunk;size_t cells,rows,chunks;
    PrefixBuffer checkpoint,weight,first,second,count,decision,interval;
    PrefixWorkspace(const Shape& q,unsigned k):chunk(k),cells(size_t(q.capacity)*4096u),
        rows(size_t(q.capacity)*16u),chunks((q.maximum_stride+k-1u)/k),
        checkpoint(cells*chunks*sizeof(prefix::Checkpoint)),weight(rows*chunks*sizeof(prefix::Weight)),
        first(cells*sizeof(prefix::Entry)),second(first.bytes),count((chunks+1u)*4u),
        decision(cells*4u),interval(cells*sizeof(prefix::Interval)){}
    void reset(){for(auto* p:{&checkpoint,&weight,&first,&second,&count,&decision,&interval})p->reset();}
    size_t bytes()const{return checkpoint.bytes+weight.bytes+first.bytes+second.bytes+count.bytes+decision.bytes+interval.bytes;}
    prefix::Workspace view(){return {checkpoint.data<prefix::Checkpoint>(),weight.data<prefix::Weight>(),
        first.data<prefix::Entry>(),second.data<prefix::Entry>(),count.data<unsigned>(),decision.data<unsigned>(),
        interval.data<prefix::Interval>(),cells*chunks,rows*chunks,cells,chunks+1u,cells};}
    void guards(){for(auto* p:{&checkpoint,&weight,&first,&second,&count,&decision,&interval})p->verify_guards();}
};
int prefix_submit(Stage& s,PrefixWorkspace& w,const uint16_t* v,const uint16_t* tv,unsigned n,
    const unsigned char* rcp,prefix::Workspace workspace){
    const auto& q=s.shape;
#define PREFIX_SUBMIT(K) return prefix::launch<K>(v,s.p.as<uint16_t>()+guard,s.s.as<float>()+guard, \
    s.o.as<float>()+guard,s.e.as<float>()+guard,q.start,q.queries,q.output_start,q.stride,rcp,tv,n, \
    workspace,nullptr,s.a.as<float>()+guard,s.d.as<float>()+guard)
    if(w.chunk==64u){PREFIX_SUBMIT(64u);}
    if(w.chunk==512u){PREFIX_SUBMIT(512u);}
    if(w.chunk==1024u){PREFIX_SUBMIT(1024u);}
#undef PREFIX_SUBMIT
    return int(hipErrorInvalidValue);
}

// References only observe completed work. They never enter producer/replay.
__global__ void prefix_results(const float* native_output,const float* native_raw,
    const float* native_error,const float* expected,const float* exact_raw,
    const float* actual,const float* actual_raw,const float* actual_error,
    const prefix::Checkpoint* checkpoints,const unsigned* decisions,const prefix::Interval* intervals,
    unsigned start,unsigned queries,unsigned output_start,unsigned stride,unsigned chunk,
    unsigned long long* stats){
    const unsigned cell=blockIdx.x*blockDim.x+threadIdx.x,cells=queries*4096u;
    if(cell>=cells)return;
    const unsigned index=output_start*4096u+cell,chunks=(stride+chunk-1u)/chunk;
    const bool selected=!qrt_sm121_pv_bound::same_bf16(native_output[index],native_error[cell]);
    const unsigned decision=decisions[cell],groups=(start+cell/4096u+32u)/32u*2u;
    if(qrt_sm121_pv_bound::bf16(actual[index])!=qrt_sm121_pv_bound::bf16(expected[index]))atomicAdd(stats,1ull);
    if(__float_as_uint(native_error[cell])!=__float_as_uint(actual_error[cell])||
        __float_as_uint(checkpoints[size_t(chunks-1u)*cells+cell].center)!=__float_as_uint(native_raw[index]))atomicAdd(stats+1u,1ull);
    if(bool(decision)!=selected || (selected&&((decision&0x7fffffffu)>groups||!(decision&0x7fffffffu))))atomicAdd(stats+2u,1ull);
    if(selected){
        atomicAdd(stats+4u,1ull);atomicAdd(stats+7u,(unsigned long long)(decision&0x7fffffffu));atomicAdd(stats+8u,(unsigned long long)groups);
        if(decision&0x80000000u){
            atomicAdd(stats+6u,1ull);
            if((decision&0x7fffffffu)!=groups||__float_as_uint(actual_raw[index])!=__float_as_uint(exact_raw[index]))atomicAdd(stats+3u,1ull);
        }else{
            atomicAdd(stats+5u,1ull);const auto value=intervals[cell];const double exact=double(exact_raw[index]);
            if((decision&0x7fffffffu)*16u%chunk || !prefix::certificate::finite(exact) ||
                exact<value.lower||exact>value.upper)atomicAdd(stats+3u,1ull);
        }
    }else if(__float_as_uint(actual[index])!=__float_as_uint(native_output[index])||
        __float_as_uint(actual_raw[index])!=__float_as_uint(native_raw[index]))atomicAdd(stats+3u,1ull);
}
__device__ bool prefix_canary(const void* data,unsigned words){
    const unsigned* p=static_cast<const unsigned*>(data);
    for(unsigned i=0;i<words;++i)if(p[i]!=0xa5a5a5a5u)return false;
    return true;
}
__global__ void prefix_workspace_check(prefix::Workspace w,unsigned start,unsigned queries,
    unsigned stride,unsigned chunk,unsigned* bad){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const unsigned cells=queries*4096u,rows=queries*16u,chunks=(stride+chunk-1u)/chunk;
    if(i<w.checkpoint_slots){
        const unsigned phase=unsigned(i/cells),cell=unsigned(i%cells),end=(start+cell/4096u+32u)/32u*32u;
        const bool live=phase<chunks&&(phase==chunks-1u||(phase+1u)*chunk<end);
        if(!live&&!prefix_canary(w.checkpoints+i,2u))atomicAdd(bad,1u);
    }
    if(i<w.weight_slots){
        const unsigned phase=unsigned(i/rows),row=unsigned(i%rows),end=(start+row/16u+32u)/32u*32u;
        const bool live=phase<chunks&&(phase==chunks-1u||(phase+1u)*chunk<end);
        if(!live){if(!prefix_canary(w.weights+i,4u))atomicAdd(bad,1u);}
        else if(!prefix::certificate::valid(w.weights[i]) ||
            (phase==chunks-1u&&(w.weights[i].lower!=1.0||w.weights[i].upper!=1.0)))atomicAdd(bad,1u);
    }
    if(i<w.queue_slots){
        if(i>=w.counts[0]){if(!prefix_canary(w.first+i,2u))atomicAdd(bad,1u);}
        else if(w.first[i].cell>=cells)atomicAdd(bad,1u);
        if(i>=w.counts[1]){if(!prefix_canary(w.second+i,2u))atomicAdd(bad,1u);}
        else if(w.second[i].cell>=cells)atomicAdd(bad,1u);
    }
    if(i<w.cell_slots){
        if(i>=cells){if(w.decisions[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);}
        if((i>=cells||!w.decisions[i])&&!prefix_canary(w.intervals+i,4u))atomicAdd(bad,1u);
    }
    if(i<w.count_slots){
        if(i>chunks){if(w.counts[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);}
        else if(w.counts[i]>cells || (i&&w.counts[i]>w.counts[i-1u]) || (i==chunks&&w.counts[i]))atomicAdd(bad,1u);
    }
}
std::array<unsigned long long,5> verify_prefix(Stage& native,Stage& expected,Stage& actual,
    PrefixWorkspace& workspace,Device& stats,Device& bad){
    const auto& q=actual.shape;check(hipMemset(stats.pointer,0,9u*8u));check(hipMemset(bad.pointer,0,4u));
    hipLaunchKernelGGL(prefix_results,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
        native.o.as<float>()+guard,native.a.as<float>()+guard,native.e.as<float>()+guard,
        expected.o.as<float>()+guard,expected.a.as<float>()+guard,actual.o.as<float>()+guard,actual.a.as<float>()+guard,
        actual.e.as<float>()+guard,workspace.checkpoint.data<prefix::Checkpoint>(),workspace.decision.data<unsigned>(),
        workspace.interval.data<prefix::Interval>(),q.start,q.queries,q.output_start,q.stride,workspace.chunk,stats.as<unsigned long long>());
    check(hipGetLastError());
    hipLaunchKernelGGL(prefix_workspace_check,dim3((workspace.view().checkpoint_slots+255u)/256u),dim3(256u),0u,nullptr,
        workspace.view(),q.start,q.queries,q.stride,workspace.chunk,bad.as<unsigned>());check(hipGetLastError());
    compare_device(native.d,actual.d,q.dwords(),bad);finish();
    const auto h=download<unsigned long long>(stats,9u);const auto errors=download<unsigned>(bad,1u);
    if(h[0]||h[1]||h[2]||h[3]||errors[0]){
        std::fprintf(stderr,"prefix_verify bf16=%llu metadata=%llu membership=%llu interval_or_raw=%llu workspace=%u start=%u chunk=%u\n",
            h[0],h[1],h[2],h[3],errors[0],q.start,workspace.chunk);
        throw std::runtime_error("prefix PV verification");
    }
    unsigned selected=0;check(hipMemcpy(&selected,workspace.count.data<unsigned>(),4u,hipMemcpyDeviceToHost));
    if(selected!=h[4]||h[4]!=h[5]+h[6]||h[7]>h[8])throw std::runtime_error("prefix PV work conservation");
    workspace.guards();stage_guards(actual);return {h[4],h[5],h[6],h[7],h[8]};
}

void prefix_generated(unsigned start,unsigned queries,unsigned mode,bool vllm,
    const unsigned char* exp,const unsigned char* rcp){
    Shape q{start,queries,start+queries,queries+3u,start+queries+7u,3u};
    std::vector<float> scores(q.pwords(),12345.0f);
    std::vector<uint16_t> values(size_t(q.stride)*512u+2u*guard,0x5a5au),tv(values.size(),0x5a5au);
    for(unsigned row=0;row<queries*16u;++row)for(unsigned key=0;key<start+row/16u+1u;++key)
        scores[guard+size_t(row)*q.stride+key]=score_value(row,key,mode);
    for(unsigned key=0;key<q.stride;++key)for(unsigned feature=0;feature<512u;++feature){
        const size_t i=size_t(key)*512u+feature;
        uint16_t word=qrt_sm121_pv_bound::bf16(float(int((i*173u+i/17u)%63u)-31)/64.0f);
        if(mode==1u&&i%7u==0u)word=uint16_t(i%127u+1u);
        if(mode==2u&&i%3u==0u)word=uint16_t((i&1u?0x8000u:0u)|0x5f80u);
        if(mode==3u&&i%3u==0u)word=(i&1u)?0x8000u:0u;
        if(mode==4u)word=uint16_t((i&1u?0x8000u:0u)|(i%3u?0x0101u:0x0001u));
        if(mode==5u&&i%251u==0u)word=(i&1u)?0xffc1u:0x7fc1u;
        values[guard+i]=word;tv[guard+size_t(feature)*q.stride+key]=word;
    }
    Device ds(scores.size()*4u),dv(values.size()*2u),dvt(tv.size()*2u),bad(4u),stats(9u*8u);
    upload(ds,scores);upload(dv,values);upload(dvt,tv);
    Stage original(q),native(q),candidate(q);Parts unused(q);
    Device ids((size_t(q.capacity)*4096u+1u+2u*guard)*4u);
    hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
        ds.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,
        q.start,q.stride,exp,vllm);check(hipGetLastError());
    copy_probability(native,original);coarse_produce(0u,native,unused,dv.as<uint16_t>()+guard,rcp);
    reset_indices(ids,q);hipLaunchKernelGGL(all_indices,dim3((q.cells()+255u)/256u),dim3(256u),0u,nullptr,
        ids.as<unsigned>()+guard,ids.as<unsigned>()+guard+q.cells(),q.cells());check(hipGetLastError());
    queue_exact(original,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp);finish();
    verify_indices(ids,q);
    const auto raw=download<float>(original.a,q.owords());
    const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
    unsigned cpu=0;
    for(unsigned sample=0;sample<8u;++sample){
        const unsigned cell=sample*(q.cells()-1u)/7u,row=cell/256u,head=row%16u;
        const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,values.data()+guard,
            hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),q.start+row/16u+1u,(head/8u)*256u+cell%256u);
        if(bits(ref)!=bits(raw[guard+q.output_start*4096u+cell]))throw std::runtime_error("prefix PV CPU recurrence");++cpu;
    }
    for(unsigned chunk:{64u,512u,1024u}){
        PrefixWorkspace workspace(q,chunk);candidate.reset();copy_probability(candidate,original);
        auto short_space=workspace.view();short_space.checkpoint_slots=size_t(q.cells())*((q.stride+chunk-1u)/chunk)-1u;
        if(prefix_submit(candidate,workspace,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp,short_space)!=int(hipErrorInvalidValue))throw std::runtime_error("prefix PV short workspace accepted");
        auto null_space=workspace.view();null_space.counts=nullptr;
        if(prefix_submit(candidate,workspace,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp,null_space)!=int(hipErrorInvalidValue))throw std::runtime_error("prefix PV null workspace accepted");
        check(hipError_t(prefix_submit(candidate,workspace,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,q.stride,rcp,workspace.view())));finish();
        const auto report=verify_prefix(native,original,candidate,workspace,stats,bad);
        unchanged(candidate.p,hp);unchanged(candidate.s,hs);
        std::printf("{\"kind\":\"prefix_pv_safety\",\"start\":%u,\"queries\":%u,\"mode\":%u,\"vllm_sum\":%s,\"chunk\":%u,\"cells\":%u,\"selected\":%llu,\"early_certified\":%llu,\"full_replay\":%llu,\"replayed_groups\":%llu,\"original_groups\":%llu,\"cpu_dots\":%u,\"bf16_mismatches\":0,\"interval_undercoverage\":0,\"native_raw_and_error_parity\":true,\"complete_original_membership\":true,\"short_and_null_workspace_rejected\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"inference_acceptance\":false}\n",
            start,queries,mode,vllm?"true":"false",chunk,q.cells(),report[0],report[1],report[2],report[3],report[4],cpu);std::fflush(stdout);
    }
    unchanged(ds,scores);unchanged(dv,values);unchanged(dvt,tv);stage_guards(native);
}
void prefix_extend(std::vector<uint16_t>& values,unsigned columns,unsigned tokens){
    if(tokens<7169u||tokens>8192u)throw std::runtime_error("prefix capture extent");
    const std::vector<uint16_t> tail(values.begin(),values.begin()+size_t(tokens-7169u)*columns);
    values.insert(values.end(),tail.begin(),tail.end());
    values.insert(values.begin(),guard,0x5a5au);values.insert(values.end(),guard,0x5a5au);
}
void prefix_capture(unsigned tokens,const char* qfile,const char* kfile,const char* vfile,
    const char* reference_file,const char* exp_file,const char* rcp_file){
    constexpr unsigned batch=128u;
    auto hq=read_values<uint16_t>(qfile,size_t(7169u)*4096u);
    auto hk=read_values<uint16_t>(kfile,size_t(7169u)*512u),hv=read_values<uint16_t>(vfile,size_t(7169u)*512u);
    const auto golden=read_values<uint16_t>(reference_file,size_t(7169u)*4096u);
    const auto exp=read_values<unsigned char>(exp_file,exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(rcp_file,qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("prefix capture table layout");
    prefix_extend(hq,4096u,tokens);prefix_extend(hk,512u,tokens);prefix_extend(hv,512u,tokens);
    Device dq(hq.size()*2u),dk(hk.size()*2u),dv(hv.size()*2u),dkt(hk.size()*2u),dvt(hv.size()*2u),dex(exp.size()),drcp(rcp.size());
    upload(dq,hq);upload(dk,hk);upload(dv,hv);upload(dex,exp);upload(drcp,rcp);
    check(hipMemset(dkt.pointer,0x5a,hk.size()*2u));check(hipMemset(dvt.pointer,0x5a,hv.size()*2u));finish();
    std::vector<uint16_t> tv(hv.size(),0x5a5au);
    for(unsigned key=0;key<tokens;++key)for(unsigned f=0;f<512u;++f)tv[guard+size_t(f)*tokens+key]=hv[guard+size_t(key)*512u+f];
    auto begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,size_t(tokens)*512u,tokens,nullptr)));finish();
    const double transpose_ms=elapsed(begin);unchanged(dvt,tv);
    Prepared prepared(dq.as<uint16_t>()+guard,dk.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,hq.data()+guard,hk.data()+guard,tokens);
    const auto kt_before=download<uint16_t>(dkt,hk.size());
    Shape q{0u,batch,batch,batch,tokens,0u};Stage original(q),native(q),candidate(q);Parts unused(q);
    PrefixWorkspace w512(q,512u),w1024(q,1024u);
    Device scores(q.pwords()*4u),score_copy(q.pwords()*4u),bad(4u),stats(9u*8u);
    Device ids((size_t(batch)*4096u+1u+2u*guard)*4u);
    double samples[3][3]{},maximum[3]{},qk_ms=0.0,probability_ms=0.0;
    unsigned long long totals[3][5]{};size_t endpoints=0,score_cells=0,verified[3]{},external[3]{},cpu=0;
    for(unsigned start=0;start<tokens;start+=batch){
        q.start=start;q.queries=std::min(batch,tokens-start);q.stride=start+q.queries;
        original.shape=q;native.shape=q;candidate.shape=q;
        check(hipMemset(scores.pointer,0xa5,q.pwords()*4u));check(hipMemset(score_copy.pointer,0xa5,q.pwords()*4u));finish();
        begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL((qrt_prepared_decoded_qk::scores<128u,true,16u,16u>),
            dim3((q.stride+15u)/16u,16u,(q.queries+15u)/16u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,prepared.qp.as<uint32_t>()+guard,prepared.kp.as<uint32_t>()+guard,
            prepared.qf.as<unsigned>()+guard,prepared.kf.as<unsigned>()+guard,scores.as<float>()+guard,start,q.queries,q.stride,tokens);
        check(hipGetLastError());finish();qk_ms+=elapsed(begin);
        hipLaunchKernelGGL(blackwell_tiled_exact_scores_kernel,dim3((q.stride+31u)/32u,16u,(q.queries+7u)/8u),dim3(256u),0u,nullptr,
            dq.as<uint16_t>()+guard,dkt.as<uint16_t>()+guard,score_copy.as<float>()+guard,start,q.queries,q.stride,tokens);
        check(hipGetLastError());check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("prefix input QK raw parity");
        original.reset();native.reset();reset_indices(ids,q);finish();begin=std::chrono::steady_clock::now();
        hipLaunchKernelGGL(blackwell_online_probability_kernel,dim3(16u,q.queries),dim3(32u),0u,nullptr,
            scores.as<float>()+guard,original.p.as<uint16_t>()+guard,original.s.as<float>()+guard,q.start,q.stride,dex.as<unsigned char>(),true);
        check(hipGetLastError());finish();probability_ms+=elapsed(begin);
        copy_probability(native,original);coarse_produce(0u,native,unused,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
        coarse_produce(0u,original,unused,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
        queue_collect(original,ids);queue_exact(original,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());finish();
        const auto original_selected=verify_indices(ids,q);packed_expected(original);
        const auto expected=download<float>(original.o,q.owords()),raw=download<float>(original.a,q.owords());
        const auto hp=download<uint16_t>(original.p,q.pwords());const auto hs=download<float>(original.s,q.swords());
        unsigned long long original_groups=0;
        for(auto cell:original_selected)original_groups+=(start+cell/4096u+32u)/32u*2u;
        for(unsigned cell=0;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell)
            if(qrt_sm121_pv_bound::bf16(expected[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("prefix original GB10 context");
        const unsigned dots=unsigned(std::min(size_t(4u),original_selected.size()));
        for(unsigned sample=0;sample<dots;++sample){
            const unsigned cell=original_selected[size_t(sample)*(original_selected.size()-1u)/std::max(1u,dots-1u)],row=cell/256u,head=row%16u;
            const float ref=cpu_pv(hp.data()+guard+size_t(row)*q.stride,hv.data()+guard,
                hs.data()+guard+size_t(row)*((q.stride+31u)/32u+1u),start+row/16u+1u,(head/8u)*256u+cell%256u);
            if(bits(ref)!=bits(raw[guard+cell]))throw std::runtime_error("prefix captured CPU PV");++cpu;
        }
        std::array<unsigned long long,5> work[3]{};
        for(unsigned attempt=0;attempt<4u;++attempt)for(unsigned position=0;position<3u;++position){
            const unsigned variant=(position+start/batch+attempt)%3u;
            auto& workspace=variant==2u?w1024:w512;
            candidate.reset();copy_probability(candidate,original);workspace.reset();reset_indices(ids,q);finish();
            begin=std::chrono::steady_clock::now();
            if(!variant){
                coarse_produce(0u,candidate,unused,dv.as<uint16_t>()+guard,drcp.as<unsigned char>());
                queue_collect(candidate,ids);queue_exact(candidate,ids,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>());
            }else check(hipError_t(prefix_submit(candidate,workspace,dv.as<uint16_t>()+guard,dvt.as<uint16_t>()+guard,tokens,drcp.as<unsigned char>(),workspace.view())));
            finish();const double wall=elapsed(begin);
            if(attempt){samples[variant][attempt-1u]+=wall;maximum[variant]=std::max(maximum[variant],wall);}
            std::array<unsigned long long,5> report{};
            if(!variant){
                const auto selected=verify_indices(ids,q);if(selected!=original_selected)throw std::runtime_error("prefix original control membership");
                check(hipMemset(bad.pointer,0,4u));compare_device(original.o,candidate.o,q.owords(),bad);
                compare_device(original.a,candidate.a,q.owords(),bad);compare_device(original.d,candidate.d,q.dwords(),bad);
                compare_device(original.e,candidate.e,q.ewords(),bad);finish();
                if(download<unsigned>(bad,1u)[0])throw std::runtime_error("prefix original control raw output");
                stage_guards(candidate);report={selected.size(),0u,selected.size(),original_groups,original_groups};
            }else report=verify_prefix(native,original,candidate,workspace,stats,bad);
            if(report[0]!=original_selected.size()||report[4]!=original_groups)throw std::runtime_error("prefix original work identity");
            if(!attempt){work[variant]=report;for(unsigned i=0;i<5u;++i)totals[variant][i]+=report[i];}
            else if(work[variant]!=report)throw std::runtime_error("prefix nondeterministic replay work");
            unchanged(candidate.p,hp);unchanged(candidate.s,hs);
            const auto output=download<float>(candidate.o,q.owords());
            for(unsigned cell=0;cell<q.cells()&&size_t(start)*4096u+cell<golden.size();++cell){
                if(qrt_sm121_pv_bound::bf16(output[guard+cell])!=golden[size_t(start)*4096u+cell])throw std::runtime_error("prefix captured GB10 mismatch");
                ++external[variant];
            }
            verified[variant]+=q.cells();
        }
        check(hipMemset(bad.pointer,0,4u));compare_device(score_copy,scores,q.pwords(),bad);finish();
        if(download<unsigned>(bad,1u)[0])throw std::runtime_error("prefix immutable input scores");
        endpoints+=q.cells();score_cells+=size_t(q.queries)*16u*q.stride;
    }
    unchanged(dq,hq);unchanged(dk,hk);unchanged(dv,hv);unchanged(dkt,kt_before);unchanged(dvt,tv);
    unchanged(dex,exp);unchanged(drcp,rcp);prepared.verify();
    if(endpoints!=size_t(tokens)*4096u)throw std::runtime_error("prefix incomplete capture");
    for(unsigned variant=0;variant<3u;++variant){
        std::array<double,3> sorted{samples[variant][0],samples[variant][1],samples[variant][2]};std::sort(sorted.begin(),sorted.end());
        std::printf("{\"kind\":\"prefix_pv_capture\",\"tokens\":%u,\"variant\":%u,\"chunk\":%u,\"query_batch\":128,\"unique_output_cells\":%zu,\"verified_output_cells\":%zu,\"external_gb10_cells_checked\":%zu,\"external_unique_gb10_cells\":29364224,\"repeated_first_capture_rows\":%u,\"selected\":%llu,\"early_certified\":%llu,\"full_replay\":%llu,\"replayed_groups\":%llu,\"original_groups\":%llu,\"cpu_pv_dots\":%zu,\"probability_cells\":%zu,\"qk_ms\":%.6f,\"qk_preparation_ms\":%.6f,\"probability_ms\":%.6f,\"value_transpose_ms\":%.6f,\"producer_collect_replay_median_ms\":%.6f,\"samples_ms\":[%.6f,%.6f,%.6f],\"maximum_completed_slab_ms\":%.6f,\"workspace_bytes\":%zu,\"warmups\":1,\"samples\":3,\"bf16_mismatches\":0,\"interval_undercoverage\":0,\"native_raw_and_error_parity\":true,\"complete_original_membership\":true,\"redzones_and_unused_tails_pass\":true,\"immutable_inputs\":true,\"all_attempts_checked\":true,\"reference_is_compute_input\":false,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
            tokens,variant,variant==1u?512u:variant==2u?1024u:0u,endpoints,verified[variant],external[variant],tokens-7169u,
            totals[variant][0],totals[variant][1],totals[variant][2],totals[variant][3],totals[variant][4],cpu,score_cells,
            qk_ms,prepared.ms,probability_ms,transpose_ms,sorted[1],samples[variant][0],samples[variant][1],samples[variant][2],maximum[variant],
            variant==1u?w512.bytes():variant==2u?w1024.bytes():0u);std::fflush(stdout);
    }
}
} // namespace

int main(int argc,char** argv)try{
    hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,0));
    if(std::strncmp(prop.gcnArchName,"gfx1151",7u))throw std::runtime_error("requires gfx1151");
    if(argc==8&&(!std::strcmp(argv[1],"--q7169")||!std::strcmp(argv[1],"--q8192"))){
        prefix_capture(!std::strcmp(argv[1],"--q7169")?7169u:8192u,argv[2],argv[3],argv[4],argv[5],argv[6],argv[7]);return 0;
    }
    if(argc!=4||std::strcmp(argv[1],"--selftest"))throw std::runtime_error("use --selftest exp2 rcp");
    const auto exp=read_values<unsigned char>(argv[2],exp2_backend::table_bytes);
    const auto rcp=read_values<unsigned char>(argv[3],qrt_sm121_attention_rcp::table_bytes);
    if(!exp2_backend::valid_layout(exp.data(),exp.size())||!qrt_sm121_attention_rcp::valid_layout(rcp.data(),rcp.size()))throw std::runtime_error("prefix PV table layout");
    Device dex(exp.size()),drcp(rcp.size());upload(dex,exp);upload(drcp,rcp);
    for(auto shape:{std::pair<unsigned,unsigned>{0u,1u},{31u,2u},{17u,17u},{0u,128u},{127u,3u},{511u,17u},{8191u,1u}})
        for(unsigned mode=0;mode<6u;++mode)for(bool vllm:{false,true})prefix_generated(shape.first,shape.second,mode,vllm,dex.as<unsigned char>(),drcp.as<unsigned char>());
    unchanged(dex,exp);unchanged(drcp,rcp);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"prefix_pv_error=%s\n",e.what());return 2;}
