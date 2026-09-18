// Isolated scheduling experiment. Every numerical producer and replay is the
// qualified short-attention implementation. No provider dispatch is changed.
#define QRT_NARROW_DOMAIN_QK_NO_MAIN
#include "narrow_domain_qk_capture.cpp"
#include <memory>

namespace {
constexpr unsigned pipeline_slots=2u;
struct PipelineStreams {
    hipStream_t streams[pipeline_slots]{};
    hipEvent_t ready[pipeline_slots]{},done[pipeline_slots]{},begin{};
    PipelineStreams(){
        check(hipEventCreateWithFlags(&begin,hipEventDisableTiming));
        for(unsigned i=0;i<pipeline_slots;++i){
            check(hipStreamCreateWithFlags(&streams[i],hipStreamNonBlocking));
            check(hipEventCreateWithFlags(&ready[i],hipEventDisableTiming));
            check(hipEventCreateWithFlags(&done[i],hipEventDisableTiming));
        }
    }
    ~PipelineStreams(){
        // The externally bounded native process also covers a failed drain.
        for(unsigned i=0;i<pipeline_slots;++i){
            if(streams[i]){(void)hipStreamSynchronize(streams[i]);(void)hipStreamDestroy(streams[i]);}
            if(ready[i])(void)hipEventDestroy(ready[i]);
            if(done[i])(void)hipEventDestroy(done[i]);
        }
        if(begin)(void)hipEventDestroy(begin);
    }
    void start(){
        check(hipEventRecord(begin,nullptr));
        for(auto stream:streams)check(hipStreamWaitEvent(stream,begin,0u));
    }
    void join(){
        for(unsigned i=0;i<pipeline_slots;++i){
            check(hipEventRecord(done[i],streams[i]));
            check(hipStreamWaitEvent(nullptr,done[i],0u));
        }
    }
};
struct PipelineSlab {
    AttentionOutputs actual,expected,native;
    Guarded statistics,expected_members,actual_members;
    unsigned start=0u,count=0u;
    unsigned expected_statistics[2]{};
    explicit PipelineSlab(unsigned tokens):actual(tokens),expected(tokens),native(tokens),
        statistics(8u),expected_members(query_batch*4096u*4u),actual_members(expected_members.bytes){}
    void reset(){actual.reset();statistics.reset();}
};
struct PipelineInputs {
    const uint16_t *q,*kt,*v,*vt;
    Prepared& prepared;NarrowDomain& domain;
    unsigned tokens;
    const unsigned char *exp,*packed,*rcp;
};
void pipeline_qk(PipelineInputs& in,PipelineSlab& slab,hipStream_t stream){
    if(!slab.count)return;
    auto workspace=in.domain.workspace(in.prepared,in.tokens);
    workspace.tile_counts=slab.statistics.as<unsigned>();
    check(hipMemsetAsync(workspace.tile_counts,0,8u,stream));
    check(hipError_t(qrt_narrow_domain_qk::launch_workspace(&workspace,in.q,in.kt,
        slab.actual.tensor.scores.as<float>()+guard,stream,slab.start,slab.count,
        slab.start+slab.count,in.tokens)));
}
void pipeline_pv(PipelineInputs& in,PipelineSlab& slab,hipStream_t stream,bool selected){
    if(!slab.count)return;
    auto& out=slab.actual;const unsigned stride=slab.start+slab.count;
    auto* p=out.tensor.probability.as<uint16_t>()+guard;
    auto* scales=out.tensor.scales.as<float>()+guard;
    const qrt_native_exp2_workspace::Workspace owner{const_cast<unsigned char*>(in.packed),in.exp};
    check(hipError_t(qrt_fused_probability_pv::launch(&owner,out.tensor.scores.as<float>()+guard,
        in.v,p,scales,out.output.as<float>(),out.error.as<float>(),out.accumulator.as<float>(),
        out.denominator.as<float>(),slab.start,slab.count,0u,stride,in.exp,in.rcp,true,stream)));
    if(selected)check(hipError_t(launch_compacted_pv_replay(in.v,p,scales,out.output.as<float>(),
        slab.start,slab.count,0u,stride,in.rcp,out.accumulator.as<float>(),out.denominator.as<float>(),
        out.error.as<float>(),out.indices.as<unsigned>(),out.count.as<unsigned>(),stream,nullptr,
        in.vt,in.tokens,0u,nullptr,true)));
}
void pipeline_pair(PipelineInputs& in,PipelineSlab* const* slabs,PipelineStreams& owner,
    unsigned variant,bool selected){
    if(variant>2u)throw std::runtime_error("invalid pipeline variant");
    if(!variant){
        for(unsigned i=0;i<pipeline_slots;++i){pipeline_qk(in,*slabs[i],nullptr);pipeline_pv(in,*slabs[i],nullptr,selected);}
        return;
    }
    owner.start();
    for(unsigned i=0;i<pipeline_slots;++i){
        if(variant==1u){
            pipeline_qk(in,*slabs[i],owner.streams[i]);pipeline_pv(in,*slabs[i],owner.streams[i],selected);
        }else{
            pipeline_qk(in,*slabs[i],owner.streams[0]);
            check(hipEventRecord(owner.ready[i],owner.streams[0]));
            check(hipStreamWaitEvent(owner.streams[1],owner.ready[i],0u));
            pipeline_pv(in,*slabs[i],owner.streams[1],selected);
        }
    }
    owner.join();
}
__global__ void pipeline_members(const unsigned* indices,const unsigned* count,unsigned* members,
    unsigned live,unsigned capacity,unsigned* bad){
    const unsigned i=blockIdx.x*blockDim.x+threadIdx.x,n=*count;
    if(!i&&n>live)atomicAdd(bad,1u);
    if(i>=capacity)return;
    if(i<n){
        const unsigned index=indices[i];
        if(index>=live)atomicAdd(bad,1u);
        else if(atomicCAS(members+index,0u,1u)!=0u)atomicAdd(bad,1u);
    }else if(indices[i]!=0xa5a5a5a5u)atomicAdd(bad,1u);
}
void pipeline_compare(PipelineSlab& slab,Device& bad){
    compare(slab.expected,slab.actual,bad);
    constexpr unsigned capacity=query_batch*4096u;
    check(hipMemset(slab.expected_members.data(),0,capacity*4u));
    check(hipMemset(slab.actual_members.data(),0,capacity*4u));
    for(unsigned arm=0;arm<2u;++arm){
        auto& out=arm?slab.actual:slab.expected;
        auto& members=arm?slab.actual_members:slab.expected_members;
        hipLaunchKernelGGL(pipeline_members,dim3((capacity+255u)/256u),dim3(256u),0u,nullptr,
            out.indices.as<unsigned>(),out.count.as<unsigned>(),members.as<unsigned>(),
            slab.count*4096u,capacity,bad.as<unsigned>());check(hipGetLastError());
    }
    hipLaunchKernelGGL(compare_words,dim3((capacity*4u+255u)/256u),dim3(256u),0u,nullptr,
        slab.expected_members.data(),slab.actual_members.data(),capacity*4u,bad.as<unsigned>());
    check(hipGetLastError());finish();
    if(download<unsigned>(bad,1u)[0])throw std::runtime_error("pipeline candidate membership or tail differs");
    slab.statistics.guards();slab.expected_members.guards();slab.actual_members.guards();
}
struct PipelineResult {
    double samples[3][3]{},preparation_ms=0.0;
    uint64_t candidates[3]{},fast_tiles[3]{},slow_tiles[3]{},score_slots=0u;
    unsigned cpu_dots=0u,slabs=0u,groups=0u,attempts=0u;
};
PipelineResult pipeline_case(const std::vector<uint16_t>& q,const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& v,unsigned start,unsigned count,
    const std::vector<uint16_t>* reference,const unsigned char* exp,const unsigned char* packed,
    const unsigned char* rcp,bool timed){
    const unsigned tokens=unsigned(k.size()/512u),end=start+count;
    if(!count||end>tokens||q.size()!=size_t(tokens)*4096u||v.size()!=k.size())throw std::runtime_error("pipeline input extent");
    Guarded dq(q.size()*2u),dk(k.size()*2u),dv(v.size()*2u),dt(k.size()*2u),vt(v.size()*2u);
    dq.put(q);dk.put(k);dv.put(v);
    Prepared prepared(dq.as<uint16_t>(),dk.as<uint16_t>(),dt.as<uint16_t>(),q.data(),k.data(),tokens);
    const auto transpose_begin=std::chrono::steady_clock::now();
    check(hipError_t(transpose_keys(dv.as<uint16_t>(),vt.as<uint16_t>(),v.size(),tokens,nullptr)));finish();
    const double transpose_ms=elapsed(transpose_begin);
    NarrowDomain domain(dq.as<uint16_t>(),dk.as<uint16_t>(),q,k,prepared,tokens);
    std::unique_ptr<Guarded> reference_device;
    if(reference){reference_device=std::make_unique<Guarded>(reference->size()*2u);reference_device->put(*reference);}
    PipelineSlab first(tokens),second(tokens);PipelineSlab* slabs[]={&first,&second};
    PipelineStreams owner;Device bad(4u);check(hipMemset(bad.pointer,0,4u));
    PipelineInputs in{dq.as<uint16_t>(),dt.as<uint16_t>(),dv.as<uint16_t>(),vt.as<uint16_t>(),
        prepared,domain,tokens,exp,packed,rcp};
    PipelineResult result;result.preparation_ms=prepared.ms+transpose_ms+domain.ms;
    for(unsigned begin=start;begin<end;begin+=pipeline_slots*query_batch){
        for(unsigned slot=0;slot<pipeline_slots;++slot){
            auto& slab=*slabs[slot];slab.start=begin+slot*query_batch;
            slab.count=slab.start<end?std::min(query_batch,end-slab.start):0u;
            slab.expected.reset();slab.native.reset();
            if(!slab.count)continue;
            attention(in.q,in.kt,in.v,in.vt,prepared,slab.expected,slab.start,slab.count,tokens,
                exp,nullptr,rcp,true,0u,nullptr);finish();
            producer(in.q,in.kt,in.v,in.vt,prepared,domain,slab.native,slab.start,slab.count,
                tokens,exp,packed,rcp,3u);finish();
            check(hipMemcpy(slab.expected_statistics,domain.statistics.data(),8u,hipMemcpyDeviceToHost));
            hipLaunchKernelGGL(tensor_tails,dim3((slab.expected.tensor.cells+2u*guard+255u)/256u),dim3(256u),0u,nullptr,
                slab.expected.tensor.scores.as<uint32_t>(),slab.expected.tensor.probability.as<uint16_t>(),
                slab.expected.tensor.scales.as<uint32_t>(),slab.expected.tensor.cells,slab.expected.tensor.scale_cells,
                slab.start,slab.count,slab.start+slab.count,bad.as<unsigned>());check(hipGetLastError());
            if(reference){
                hipLaunchKernelGGL(external_context,dim3((slab.count*4096u+255u)/256u),dim3(256u),0u,nullptr,
                    slab.expected.output.as<float>(),reference_device->as<uint16_t>(),slab.start,slab.count,bad.as<unsigned>());
                check(hipGetLastError());
            }
            finish();if(download<unsigned>(bad,1u)[0])throw std::runtime_error("pipeline original reference or tails differ");
            for(unsigned sample=0;sample<4u;++sample){
                const unsigned row=sample*(slab.count-1u)/3u,head=(result.slabs+sample*5u)%16u,key=(slab.start+row)*sample/3u;
                const float cpu=qrt_q1_moe_hawkeye::accumulate_bf16_hopper_blackwell(0.0f,
                    q.data()+(size_t(slab.start+row)*16u+head)*256u,k.data()+(size_t(key)*2u+head/8u)*256u,256u)*kExactScale;
                uint32_t gpu=0u;check(hipMemcpy(&gpu,slab.expected.tensor.scores.as<uint32_t>()+guard+
                    (size_t(row)*16u+head)*(slab.start+slab.count)+key,4u,hipMemcpyDeviceToHost));
                if(gpu!=bits(cpu))throw std::runtime_error("pipeline reference CPU QK differs");++result.cpu_dots;
            }
            result.score_slots+=size_t(slab.count)*16u*(slab.start+slab.count);++result.slabs;
        }
        for(unsigned attempt=0;attempt<(timed?4u:1u);++attempt)for(unsigned position=0;position<3u;++position){
            const unsigned variant=(position+result.groups+attempt)%3u;
            for(auto* slab:slabs)slab->reset();finish();
            const auto clock=std::chrono::steady_clock::now();
            pipeline_pair(in,slabs,owner,variant,attempt!=0u);finish();
            if(attempt)result.samples[variant][attempt-1u]+=elapsed(clock);
            if(!attempt){
                for(auto* slab:slabs)if(slab->count){
                    compare(slab->native,slab->actual,bad);
                    replay(in.v,in.vt,slab->actual,slab->start,slab->count,tokens,rcp);
                }
                finish();
            }
            for(auto* slab:slabs){
                if(!slab->count){
                    compare(slab->expected,slab->actual,bad);
                    slab->actual.indices.immutable(std::vector<unsigned>(query_batch*4096u,0xa5a5a5a5u));
                    slab->statistics.immutable(std::vector<unsigned>(2u,0xa5a5a5a5u));continue;
                }
                pipeline_compare(*slab,bad);
                unsigned statistics[2]{};check(hipMemcpy(statistics,slab->statistics.data(),8u,hipMemcpyDeviceToHost));
                if(std::memcmp(statistics,slab->expected_statistics,8u))throw std::runtime_error("pipeline QK tile statistics differ");
                if(!attempt){
                    unsigned selected=0u;check(hipMemcpy(&selected,slab->actual.count.data(),4u,hipMemcpyDeviceToHost));
                    result.candidates[variant]+=selected;result.slow_tiles[variant]+=statistics[0];result.fast_tiles[variant]+=statistics[1];
                }
            }
            ++result.attempts;
        }
        ++result.groups;
    }
    for(unsigned variant=1;variant<3u;++variant)
        if(result.candidates[variant]!=result.candidates[0]||result.fast_tiles[variant]!=result.fast_tiles[0]||
            result.slow_tiles[variant]!=result.slow_tiles[0])throw std::runtime_error("pipeline aggregate work differs");
    dq.immutable(q);dk.immutable(k);dv.immutable(v);prepared.verify();domain.verify();
    immutable_transpose(dt,k,tokens);immutable_transpose(vt,v,tokens);
    if(reference)reference_device->immutable(*reference);
    return result;
}
void pipeline_safety(const unsigned char* exp,const unsigned char* packed,const unsigned char* rcp){
    struct Shape{unsigned tokens,start,count;};
    const Shape shapes[]={{1,0,1},{17,0,17},{65,1,64},{129,1,128},{257,0,257},
        {513,127,257},{8192,7935,257},{8192,8175,17}};
    unsigned configurations=0u,groups=0u,slabs=0u,cpu_dots=0u,attempts=0u;
    for(auto shape:shapes)for(unsigned mode=0;mode<7u;++mode){
        std::vector<uint16_t> q(size_t(shape.tokens)*4096u),k(size_t(shape.tokens)*512u),v(k.size());
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
            for(unsigned i=0;i<8u;++i){q[size_t(shape.start)*4096u+i]=edges[i];k[i]=edges[7u-i];}
        }
        if(mode==3u){
            std::fill(q.begin(),q.end(),uint16_t(127u<<7u|127u));
            for(size_t i=0;i<k.size();++i)k[i]=uint16_t(127u<<7u|127u|((i/16u)&1u?0x8000u:0u));
        }
        if(mode==4u)for(size_t i=0;i<q.size();++i)q[i]=i%2u?0u:0x8000u;
        if(mode==5u){
            for(size_t i=0;i<q.size();++i)q[i]=uint16_t(95u<<7u|127u);
            for(size_t i=0;i<k.size();++i)k[i]=uint16_t(159u<<7u|127u|(i&1u?0x8000u:0u));
        }
        if(mode==6u){q[size_t(shape.start)*4096u+255u]=uint16_t(94u<<7u|37u);k[511u]=uint16_t(160u<<7u|17u);}
        const auto result=pipeline_case(q,k,v,shape.start,shape.count,nullptr,exp,packed,rcp,false);
        configurations+=3u;groups+=result.groups;slabs+=result.slabs;cpu_dots+=result.cpu_dots;attempts+=result.attempts;
        std::fprintf(stderr,"PIPELINED_ATTENTION_SAFETY tokens=%u start=%u queries=%u mode=%u groups=%u pass=1\n",
            shape.tokens,shape.start,shape.count,mode,result.groups);
    }
    std::printf("{\"kind\":\"pipelined_attention_safety\",\"configurations\":%u,\"shapes\":8,\"data_modes\":7,\"variants\":3,\"groups\":%u,\"slabs\":%u,\"attempts\":%u,\"cpu_dots\":%u,\"complete_surfaces_bitexact\":true,\"native_surfaces_checked\":true,\"candidate_membership_and_tails\":true,\"inactive_slot_untouched\":true,\"guards_and_inputs_pass\":true,\"inference_acceptance\":false}\n",configurations,groups,slabs,attempts,cpu_dots);
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
    if(safety_mode)pipeline_safety(de.data(),dd.data(),dc.data());
    else{
        auto q=read_words(argv[2],7169u*4096u),k=read_words(argv[3],7169u*512u),v=read_words(argv[4],7169u*512u);
        for(auto pair:{std::make_pair(&q,4096u),std::make_pair(&k,512u),std::make_pair(&v,512u)}){
            const auto old=*pair.first;
            pair.first->insert(pair.first->end(),old.begin(),old.begin()+size_t(tokens-7169u)*pair.second);
        }
        const auto reference=read_words(argv[5],7169u*4096u);
        const auto result=pipeline_case(q,k,v,0u,tokens,&reference,de.data(),dd.data(),dc.data(),true);
        if(!result.fast_tiles[0]||!result.slow_tiles[0])throw std::runtime_error("captured narrow and original QK required");
        for(unsigned variant=0;variant<3u;++variant){
            std::vector<double> sorted(result.samples[variant],result.samples[variant]+3u);std::sort(sorted.begin(),sorted.end());
            std::printf("{\"kind\":\"pipelined_attention_capture\",\"tokens\":%u,\"source_capture_tokens\":7169,\"repeated_rows\":%u,\"variant\":%u,\"pipeline_slots\":2,\"group_queries\":256,\"slab_queries\":128,\"groups\":%u,\"slabs\":%u,\"cpu_dots\":%u,\"score_slots\":%llu,\"gb10_context_cells\":29364224,\"pv_candidates\":%llu,\"narrow_tiles\":%llu,\"original_tiles\":%llu,\"completed_attention_samples_ms\":[%.9f,%.9f,%.9f],\"median_completed_attention_ms\":%.9f,\"common_preparation_ms\":%.9f,\"complete_surfaces_bitexact\":true,\"native_surfaces_checked_on_warmup\":true,\"candidate_membership_and_tails\":true,\"all_attempts_checked\":true,\"guards_and_inputs_pass\":true,\"reference_is_compute_input\":false,\"model_loaded\":false,\"inference_acceptance\":false,\"performance_acceptance\":false}\n",
                tokens,tokens-7169u,variant,result.groups,result.slabs,result.cpu_dots,(unsigned long long)result.score_slots,
                (unsigned long long)result.candidates[variant],(unsigned long long)result.fast_tiles[variant],(unsigned long long)result.slow_tiles[variant],
                result.samples[variant][0],result.samples[variant][1],result.samples[variant][2],sorted[1],result.preparation_ms);
        }
    }
    de.immutable(exp);dc.immutable(rcp);dd.immutable(packed);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"pipelined_attention_error=%s\n",e.what());return 2;}
